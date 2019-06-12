#!/bin/bash

nvcc -std=c++11 -o course course.cu -lGL -lGLU -lglut -lGLEW -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
