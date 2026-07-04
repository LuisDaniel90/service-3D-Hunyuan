# Investigacion - Mejora de Reconstruccion 3D Facial

## Contexto

El servicio vision360-deca captura **6 fotos** del rostro del usuario desde distintos angulos
(front, left_45, right_45, left_90, right_90, up) a traves de una app React Router.
Actualmente usa **DECA** para generar una malla 3D (FLAME mesh) y exportarla como GLB.

**Problema**: La malla generada es geometricamente aceptable pero la textura es generica
y de baja calidad. No se parece a la persona real.

**Objetivo**: Que el modelo 3D quede nitido y se asemeje a las fotos reales del usuario.

---

## Modelos Evaluados

### 1. Modelos Single-Image-to-3D (NO aceptan multi-vista)

Estos modelos generan 3D a partir de UNA sola imagen. No aprovechan las 6 vistas.

| Modelo | Licencia | CPU | Peso | Descripcion |
|--------|----------|-----|------|-------------|
| **TripoSR** | MIT (libre) | Si (~30s) | ~1.2 GB | Stability AI. Genera mesh con textura desde 1 imagen. Buena calidad general. |
| **InstantMesh** | Apache 2.0 | Impractico | ~4-6 GB | TencentARC. Usa difusion interna (Zero123++) para generar vistas. Pesado. |
| **Wonder3D** | Apache 2.0 | No (GPU) | ~5-6 GB | Genera 6 vistas + normales internamente desde 1 imagen. Requiere 8+ GB VRAM. |
| **ERA3D** | Apache 2.0 | No (GPU) | ~5-7 GB | Similar a Wonder3D. Requiere 8-12 GB VRAM. |

**Conclusion**: TripoSR es el unico viable para CPU/Docker. Los demas requieren GPU potente.

---

### 2. Upgrades de DECA (misma arquitectura FLAME)

Mejoran la calidad del modelo FLAME pero NO soportan multi-vista nativo.

| Modelo | Licencia | Mejora vs DECA | CPU | Peso |
|--------|----------|----------------|-----|------|
| **EMOCA** | CC BY-NC-SA 4.0 | Mejor expresion facial | Si | ~300-500 MB |
| **MICA** | CC BY-NC-SA 4.0 | Mejor forma/identidad | Si | ~200 MB |

**Nota**: Ambos tienen licencia **no comercial**. Para uso comercial se necesita negociar
con Max Planck Institute.

**Uso combinado ideal**: MICA para shape + EMOCA para expresion = mejor resultado que DECA solo.
Sin embargo, el procesamiento multi-vista seria igual que con DECA (procesar cada imagen
independientemente y promediar parametros).

---

### 3. Modelos Multi-Vista Reales (SI aceptan multiples fotos)

Estos SI pueden recibir las 6 fotos como entrada.

| Modelo | Licencia | CPU | Descripcion |
|--------|----------|-----|-------------|
| **OpenMVS** | AGPL | Si | Multi-View Stereo clasico. Genera mesh denso desde multiples fotos. |
| **NeuS2** | Free | No (GPU) | Neural surface reconstruction. Alta calidad. |
| **MVSNet** | Free | GPU recomendado | Deep learning multi-view stereo. |

**Conclusion**: OpenMVS es la unica opcion multi-vista que corre en CPU, pero requiere
calibracion de camara (intrinsics/extrinsics) que complica la integracion.

---

### 4. APIs Externas Gratuitas

- **No existen** APIs gratuitas e ilimitadas para reconstruccion 3D facial.
- HuggingFace Spaces tiene demos interactivos de DECA/EMOCA/TripoSR pero son rate-limited
  y no estan disenados como APIs de produccion.

---

## Estrategia Recomendada

### Enfoque Hibrido: DECA (geometria) + Textura Multi-Vista (fotos reales)

Este enfoque combina lo mejor de ambos mundos:

#### Paso 1: Geometria con DECA (ya implementado)
- Procesar las 6 imagenes con DECA
- Promediar parametros `shape`, `exp`, y `detail` entre todas las vistas
- Generar la malla FLAME con la geometria promediada

#### Paso 2: Textura HD con Proyeccion Multi-Vista (a implementar)
- Usar los parametros de `pose` estimados por DECA para cada vista
  (DECA devuelve rotacion y traslacion de la cabeza)
- Con pytorch3d (ya instalado), proyectar cada foto real sobre el UV map de la malla
- Cada zona de la cara usa la foto del angulo que mejor la ve:
  - **front** -> nariz, boca, ojos, frente central
  - **left_45 / left_90** -> oreja izquierda, mejilla izquierda
  - **right_45 / right_90** -> oreja derecha, mejilla derecha
  - **up** -> frente superior, linea del cabello
- Blending ponderado por visibilidad (angulo entre normal de superficie y direccion de camara)

#### Paso 3 (Opcional): TripoSR como alternativa
- Para objetos/rostros donde DECA falle en detectar cara
- TripoSR genera mesh + textura desde 1 imagen (MIT, gratis, CPU)
- Se puede ofrecer como fallback o modo alternativo

---

## Costos

| Componente | Costo |
|------------|-------|
| DECA | Gratis (uso academico, licencia permisiva) |
| pytorch3d | Gratis (BSD) |
| TripoSR | Gratis (MIT) |
| EMOCA/MICA | Gratis solo no-comercial |
| OpenMVS | Gratis (AGPL) |
| HuggingFace Spaces (hosting) | Gratis (tier free) |

**Todo el stack recomendado es 100% gratuito.**

---

## Implementacion Tecnica

### Proyeccion de Textura Multi-Vista con pytorch3d

**Como funciona la pose de DECA:**
- DECA devuelve `pose` como un vector de 6 valores:
  - Primeros 3: rotacion global de la cabeza (axis-angle)
  - Ultimos 3: traslacion
- Se convierte axis-angle a matriz de rotacion con `batch_rodrigues(pose[:,:3])`

**Flujo completo:**

```python
from pytorch3d.renderer import (
    MeshRasterizer, RasterizationSettings,
    PerspectiveCameras
)
from pytorch3d.ops import interpolate_face_attributes
import torch

# 1. Para cada vista, DECA devuelve:
#    - codedict["pose"]  -> rotacion/traslacion de la cabeza
#    - La imagen original en alta resolucion

# 2. Convertir pose DECA a camara pytorch3d
def pose_to_camera(pose_params, device):
    """Convierte parametros de pose DECA a camara pytorch3d."""
    from pytorch3d.transforms import axis_angle_to_matrix
    rot = axis_angle_to_matrix(pose_params[:, :3])  # [B, 3, 3]
    trans = pose_params[:, 3:]  # [B, 3]
    return PerspectiveCameras(R=rot, T=trans, device=device)

# 3. Rasterizar para obtener visibilidad
raster_settings = RasterizationSettings(image_size=1024)
rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster_settings)
fragments = rasterizer(mesh)
# fragments.pix_to_face -> que triangulo es visible en cada pixel
# fragments.bary_coords -> coordenadas baricentricas

# 4. Interpolar UVs con coordenadas baricentricas
# FLAME tiene UV coords en data/FLAME_texture.npz
uv_coords = interpolate_face_attributes(
    fragments.pix_to_face,
    fragments.bary_coords,
    face_uv_coords
)

# 5. Samplear color de la foto real en esa posicion UV
# Para cada texel visible, escribir el color de la imagen

# 6. Blending ponderado por visibilidad
# weight = dot(face_normal, view_direction)
# final_color = sum(w_i * c_i) / sum(w_i)
# Descartar donde dot < 0 (backfacing) o fragmento ocluido
```

**Funciones clave de pytorch3d:**
- `MeshRasterizer` + `RasterizationSettings(image_size=1024)`
- `interpolate_face_attributes()` para interpolacion baricentrica de UVs
- `PerspectiveCameras` para configurar camara
- FLAME UV coords desde `Meshes(verts, faces, textures=TexturesUV(...))`

---

### Integracion con TripoSR

**Instalacion:**
```bash
pip install git+https://github.com/VAST-AI-Research/TripoSR.git
# Dependencias: torch, transformers, trimesh, Pillow, rembg
```

**Uso basico:**
```python
from tsr.system import TSR
from PIL import Image

# Cargar modelo (auto-descarga de HuggingFace, ~1.2GB)
model = TSR.from_pretrained("stabilityai/TripoSR", device="cpu")

# Reducir uso de memoria (trade speed for RAM)
model.renderer.set_chunk_size(8192)

# Inferencia
image = Image.open("front.png")  # PIL Image RGB
meshes = model.run_image([image], bake_texture=True)

# Exportar
meshes[0].export("output.glb")  # o "output.obj"
```

**Caracteristicas:**
- Input: PIL Image RGB (usa rembg internamente para quitar fondo)
- Output: `trimesh.Trimesh` con vertices, caras, y textura UV (con `bake_texture=True`)
- CPU: ~60 segundos por imagen, ~4-6 GB RAM
- GPU: ~3 segundos por imagen
- Genera textura pero "hallucina" zonas no visibles (la proyeccion multi-vista es superior)

**Combinacion TripoSR + DECA:**
- TripoSR puede generar geometria de cabeza completa (incluyendo pelo, orejas)
- DECA/FLAME es superior para la region facial (geometria precisa)
- Se pueden combinar: TripoSR para back-of-head, DECA para cara
- Requiere alinear ambas mallas (ICP o landmarks) y blending en la frontera

---

## Arquitectura Final Propuesta

```
React Router App (6 fotos)
        |
        v
  POST /reconstruct
        |
        v
  +------------------+
  | Para cada vista:  |
  |  DECA encode()    |
  +------------------+
        |
        v
  +------------------+
  | Promediar shape,  |
  | exp, detail       |
  +------------------+
        |
        v
  +------------------+
  | DECA decode()     |
  | -> malla FLAME    |
  +------------------+
        |
        v
  +------------------+
  | Proyectar 6 fotos |
  | sobre UV map      |
  | (pytorch3d)       |
  +------------------+
        |
        v
  +------------------+
  | Blending textura  |
  | ponderado por     |
  | visibilidad       |
  +------------------+
        |
        v
  GLB con textura HD
```

---

## Proximos Pasos

1. [ ] Implementar promedio de `exp` y `detail` (ademas de `shape`)
2. [ ] Implementar proyeccion de textura multi-vista con pytorch3d
3. [ ] Implementar blending ponderado por visibilidad
4. [ ] Evaluar integracion de TripoSR como fallback
5. [ ] Testing con fotos reales de 6 angulos
6. [ ] Optimizar tiempos de procesamiento
