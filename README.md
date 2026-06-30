---
title: Vision360 DECA
emoji: 🧠
colorFrom: blue
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Vision360 DECA — 3D Face Reconstruction API

Microservicio de reconstrucción facial 3D basado en DECA.

**Endpoints:**
- `GET /health` — estado del servicio
- `POST /reconstruct` — recibe foto en base64, retorna GLB
