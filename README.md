# Geo-Camera World Reconstructor

This project reconstructs:

- Player world position
- Camera motion + angle
- Trajectory over time

from raw gameplay video using:

- GeoGuessr-style scene anchoring
- Optical flow camera estimation
- Vision-based player detection
- Fusion-based world reconstruction

---

## Pipeline

Frame → GeoGuessr anchor → Camera motion → Player detection → Fusion → World path

---

## Run

```bash
pip install opencv-python numpy
python main.py
