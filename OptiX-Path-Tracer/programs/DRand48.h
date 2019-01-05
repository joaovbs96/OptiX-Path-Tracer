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

#include "vec.h"
#include <stdint.h>

// Original class implementation by Ingo Wald
// Adapted to use Aras Pranckevicius' version of XorShift32
// source: https://github.com/aras-p/ToyPathTracer/blob/master/Cpp/Source/Maths.cpp#L5-L18
struct DRand48 {
  // initialize the random number generator with a new seed (usually per pixel)
  inline __device__ void init(unsigned int seed = 1) {
    state = seed;
    for (int warmUp=0; warmUp < 10; warmUp++)
      (*this)();
}

  // get the next 'random' number in the sequence
  inline __device__ float operator() () {
    unsigned int x = state;
    
    x ^= x << 13;
	  x ^= x >> 17;
	  x ^= x << 15;
	  
    state = x;

    return (x & 0xFFFFFF) / 16777216.0f;
  }

  unsigned int state;
};
