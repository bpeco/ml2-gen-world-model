"""Inferencia de punta a punta residente en el dispositivo: video -> encoder VQ -> world model -> decoder VQ -> video.

Ver Docs/plan-inferencia.md. Módulos:
  codec.py       encoder y decoder del VQ-VAE de comma en fp16 con CUDA graphs (entrada y salida uint8 en el device)
  video.py       lectura del video de cámara de ejemplo de commaVQ y escritura de mp4
  stream.py      estado de inferencia por frame (observe / predict / imagine) para cada familia de modelos
  bench_codec.py fidelidad y latencia del codec en el dispositivo
  bench_e2e.py   latencia de la cadena completa (p50, p99, frames tarde)
  quality.py     calidad a 1 paso (modo en línea) y equivalencia del camino nuevo con los rollouts
  demo.py        mp4 con el video real y el predicho lado a lado
"""
