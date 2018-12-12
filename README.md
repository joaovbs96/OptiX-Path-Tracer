# OptiX Path Tracer

![alt text](ch7_edited.png "'The Next Week' Chapter 7 - 1080x1080 w/ 1000 spp")

## Overview

Project forked from Ingo Wald's OptiX version of Pete Shirley's "Ray Tracing in one Weekend" series.

I personally implemented a C++ version of all three minibooks before attempting
to deal with a GPU based version, just so I could understand the underlining 
theory. After that, I tried my luck with CUDA, based on Roger Allen's version,
and only then I tried to implement an OptiX version based on Ingo Wald's code. 
I would like to thank Peter Shirley, Roger Allen and Ingo Wald for help and tips whenever I needed.

- Ingo Wald's original OptiX code:
https://github.com/ingowald/RTOW-OptiX

- More info & tutorials:
http://ingowald.blog

- Peter Shirley's original C++ code:
https://github.com/petershirley/raytracinginoneweekend

- Peter Shirley's Ray Tracing books:
https://www.amazon.com/Ray-Tracing-Weekend-Minibooks-Book-ebook/dp/B01B5AODD8

- Roger Allen's CUDA version:
https://devblogs.nvidia.com/accelerated-ray-tracing-cuda/

## Prerequisites

To buid this project, you need

- a install of CUDA, preferably CUDA 10. Make sure to put your CUDA
  binary directory into your path.

- a install of OptiX, preferably (ie, tested with) OptiX 5.1.1. Under
  Linux, I'd suggest to put a ```export OptiX_INSTALL_DIR=...``` into your
  ```.bashrc```.

- the usual compiler and build tools - gcc, clang, etc.

- cmake, version 2.8 should do.


## Building

This project is built with cmake. On linux, simply create a build
directory, and start the build with with ccmake:

   mkdir build
   cd build
   cmake ..
   make

Assuming you have nvcc (CUDA) in your path, and have set a
```OptiX_INSTALL_DIR``` environment variable to point to the OptiX
install dir, everything should be configured automatically.

On Windows, you'll have to use the cmake gui, and make sure to set the
right paths for optix include dirs, optix paths, etc.


## Running

Run the ./OptiX-Path-Tracer binary (OptiX-Path-Tracer.exe on windows). This
should render a PNG image under the output folder(that needs to be 
created on ahead). To change image resolution (default 1200x800), 
number of samples (default 128), etc, just edit ```OptiX-Path-Tracer/main.cpp```.

On Windows, you might see a "DLL File is Missing" warning. Just copy the missing 
file from ```OptiX SDK X.X.X\SDK-precompiled-samples``` to the build folder.

## Code Overview

- The main host cost is in main.cpp. This sets up the optix
  node graph, creates and compiles all programs, etc.
  
- Host functions and constructors are separated into different header files 
under the ```host_includes/``` folder..

- All OptiX device programs are in ```OptiX-Path-Tracer/programs/```:
  - raygen.cu - ray generation program (the main render launch)
  - under ```hitables/```: sphere.cu for the sphere intersection and bounding box codes
  - under ```materials/```: metal/dielectric/lambertian.cu for the three material types
  - some other headers with device helper functions.

- The ```OptiX-Path-Tracer/CMakeLists``` scripts sets up the build; in
particular, it defines all the cmake rules for compiling the device
programs, embedding them into host object files, and linking them to
the final binary.

- Ingo Wald made two different versions of the code, a recursive one and
an iterative one(better suited to GPUs). On my fork, I decided to keep 
the iterative version only, but you can still check his original code on 
his repository.


## ChangeLog

- 11/16/18 - Initial Release by Ingo Wald;
- 11/18/18 - Iterative Version Release;
- 11/20/18 - Switched random number generator to DRand48;
- 11/26/18 - Project Forked: Added PNG output, gamma SQRT correction, changed sampling functions to non-loop versions, moved contructors to separate files under the 'header_include' folder;
- 11/27/18 - Added moving spheres;
- 12/07/18 - Added constant, checkered and Perlin noise textures;
- 12/08/18 - Added image textures;
