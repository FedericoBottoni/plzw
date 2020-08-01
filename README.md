# plzw
> Parallel LZW algorithms

## Overview
The Lempel–Ziv–Welch (LZW) is a lossless data compression algorithm capable of exploiting especially the presence of repetitive patterns, in the raw data, and acting a very effective compression.
This repository contains some parallel implementations of the algorithm in order to compare them, their performances and to quantify the requested computational resources to get the best outcome.

## Prerequisites

* [MPICH](https://www.mpich.org/)
* [OpenMP](https://www.openmp.org/)
* [Cuda toolkit](https://developer.nvidia.com/cuda-downloads)


## Get ready on Visual Studio
```sh
$ git clone https://github.com/FedericoBottoni/plzw
```
Open the solution and try to build the projects, if you can't, proceed to the installations and configuration of the dependencies:
* Install MS-MPI and set the environment configurations on the project's propriety [pages](https://docs.microsoft.com/en-us/archive/blogs/windowshpc/how-to-compile-and-run-a-simple-ms-mpi-program)
* Enable the OMP configuration option on the project's propriety [pages](https://docs.microsoft.com/en-us/cpp/build/reference/openmp-enable-openmp-2-0-support)


## Running
You can run the different projects from the Release folder "plzw\x64\Release\" (or "plzw\Release\" if your system is Win32).
WARNING: This commands are defaults to run 4 processes in parallel implementations.

### Serial
```sh
$ PLZW_Serial.exe
```

### MPI
You can edit the number of processes from CLI, changing "-np 4" in "-np {number}"
```sh
$ mpiexec -np 4 PLZW_MPI.exe
```

### OpenMP
You can edit the number of processes defining DEFAULT_NPROCS
```sh
$ PLZW_OMP.exe
```

### CUDA
Available in CUDA branch, you can edit the grid and blocks sizes defining BLOCKS_GRID and THREADS_A_BLOCK
```sh
$ PLZW_CUDA.exe
```

## Author
**Federico Bottoni** - [github.com/FedericoBottoni](https://github.com/FedericoBottoni)
