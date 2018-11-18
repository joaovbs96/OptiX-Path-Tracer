// ======================================================================== //
// Copyright 2018 Ingo Wald                                                 //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

// ooawe
#include "programs/vec.h"
// std
#include <fstream>
#include <vector>
#include <assert.h>

/*! saving to a 'P3' type PPM file (255 values per channel).  There
  are many other, more C++-like, ways of writing this; this version is
  intentionally written as close to the RTOW version as possible */
inline void savePPM(const std::string &fileName,
                    const size_t Nx, const size_t Ny, const vec3f *pixels)
{
  std::ofstream file(fileName);
  assert(file.good());

  file << "P3\n" << Nx << " " << Ny << "\n255\n";
  for (int iy=(int)Ny-1; iy>=0; --iy)
    for (int ix=0; ix<(int)Nx; ix++) {
      int ir = int(255.99*pixels[ix+Nx*iy].x);
      int ig = int(255.99*pixels[ix+Nx*iy].y);
      int ib = int(255.99*pixels[ix+Nx*iy].z);
      file << ir << " " << ig << " " << ib << "\n";
    }
  assert(file.good());
}

