---
title: Vision360 3D
emoji: 🧠
colorFrom: blue
colorTo: blue
pinned: false
---

# Vision360 3D — RunPod Serverless Worker

Worker GPU serverless que genera modelos 3D texturizados de rostros
usando Hunyuan3D-2.1.

## Input

```json
{
  "input": {
    "views": [
      { "pose": "front", "image_base64": "..." },
      { "pose": "left", "image_base64": "..." },
      { "pose": "right", "image_base64": "..." }
    ]
  }
}
```

Poses válidas: `front`, `left45`, `right45`, `left`, `right`, `eyes` (ignorada).

## Output

```json
{
  "glb_base64": "...",
  "size_bytes": 1234567
}
```

## Deploy

```bash
docker build -t <tu-registry>/vision360-3d .
docker push <tu-registry>/vision360-3d
```

En RunPod: Serverless → New Endpoint → Docker Image → seleccionar GPU (24GB+).
