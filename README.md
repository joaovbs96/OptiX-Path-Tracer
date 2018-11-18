# RTOW-OptiX (my OptiX version of Pete's 'Ray Tracing in One Weekend' series)

## Overview

This project aims at providing a Optix Version of the 'final chapter'
example in Pete Shirley's "Ray Tracing in one Week-End" series.

If you haven't read that book yet, you should (have a look here:
https://www.amazon.com/Ray-Tracing-Weekend-Minibooks-Book-ebook/dp/B01B5AODD8
).

For reference, Pete's original C++ code is here:
https://github.com/petershirley/raytracinginoneweekend

Note this project covers *only* the final chapter example; Pete and I
are in parallel working on a more complete OptiX-version of Pete's
entire book series, but that might take a while yet.

## Some random notes

- I tried to follow Pete's original example wherever I could. I did
  take the freedom to clean up a few things, such as using longer (ie,
  more meaningful) variable names and caml-case to improve
  readability; or using float-constants more consistently (eg, "1.f"
  vs "1" or "1."); etc... but kept with his blueprint were
  possible. 

- There are multiple ways of realizing this example in OptiX; in
  particular the recursion found in Pete's example could be realized
  in multiple ways that would be more efficient, faster, more elegant,
  etc. However, as said before I tried to follow his example as well
  as I could, so this is my version. I'll probably provide a iterative
  version soon as well, but for now that should do.

- In terms of performance, this version "should" be way faster than
  either Pete's CPU version or Roger Allen's CUDA version (see
  https://devblogs.nvidia.com/accelerated-ray-tracing-cuda/ for that
  one); this is not because of any magic I might have done, but simply
  because OptiX will automatically build an acceleration structure
  over all those spheres, whereas their examples didn't. The magic of
  OptiX, I guess :-).


## Prerequisites

To buidl this project, you need

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

On windows, you'll have to use the cmake gui, and make sure to set the
right paths for optix include dirs, optix paths, etc.


## Running

Just run the ./finalChapter binary (finalChapter.exe on windows). This
should render a ```finalChapter.ppm``` image. To change image
resolution (default 1200x800), number of samples (default 128), etc,
just edit ```FinalChapter/finalChapter.cpp```.



## Code Overview

- The main host cost is in finalChapter.cpp. This sets up the optix
  node graph, creates and compiles all programs, etc.

- All OptiX device programs are in FinalChapter/programs/, sorted by
  raygen.cu for the ray generation program (the main render launch),
  sphere.cu for the sphere intersectoin and bounding box codes, and
  metal/dielectric/lambertian.cu for the three material types, with
  some helper code in vec.h (vector math), sampling, random number
  code, etc.

- The finalChapter/CMakeLists scripts sets up the build; in
  particular, it defines all the cmake rules for compiling the device
  programs, embedding them into host object files, and linking them to
  the final binary.

- There are two versions of the code - a 'iterative' one, in which the
  materials execute the scattering event, but the tracing of the path
  itself is done iteratively in the ray generaiton program; and a
  'recursive' variant in which the materials themselves recursively
  trace the path on. The recursive one is closer to Pete's original
  example code, but of course, requires a lot of stack per pixel,
  which isn't cheap. The iterative version is thus "better" for a GPU,
  but the recursive version shows that you can most definitely use
  recursion if you need to (just make sure to set the stack size
  accordingly).
  
I'll write a more "tutorial" - style version that explains the
high-level concepts in a separate place (likely my blog, on
http://ingowald.blog).


## ChangeLog

- 11/16/18 Initial Release
