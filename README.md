# PGP course project

This project is realization of swarm of particles optimization method.

Space partition in this method is regular grid.

Target function is function of two variables. In this project this is [Ackley function](https://en.wikipedia.org/wiki/Ackley_function).

This program have visualisation via OpenGL.

Tested on Linux Mint 18.3, compi.

## Requirements to compile

To compile, one must have NVIDIA videocard with CUDA support and OpenGL libraries: 
 * OpenGL library (libgl);
 * OpenGL Utility Toolkit (GLUT);
 * libglu;
 * OpenGL Extension Wrangler Library (GLEW).
 
You also need NVIDIA CUDA Compiler (NVCC).

## How to compile

BASH way:

```bash
chmod +x compile.sh
./compile.sh
```
