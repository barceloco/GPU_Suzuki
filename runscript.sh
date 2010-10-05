#!/bin/bash

rm -f GPU_Suzuki.x
nvcc GPU_Suzuki.cu -o GPU_Suzuki.x
./GPU_Suzuki.x -V -debug -px0 5.0 -pbcx -pbcy -alpha 10000.0 -beta 1.0 -nx 500 -ny 500 -Lx 10.0 -Ly 10.0 -sx 2.0 -x0 -9.9 -dt 0.0001 -R1 0.03 -R2 0.1 -time 0.05 -target 0.94

