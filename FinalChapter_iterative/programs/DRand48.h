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

struct DRand48
{
  /*! initialize the random number generator with a new seed (usually
      per pixel) */
  inline __device__ void init(int seed = 0)
  {
    state = seed;
    for (int warmUp=0;warmUp<10;warmUp++)
      (*this)();
  }

  /*! get the next 'random' number in the sequence */
  inline __device__ float operator() ()
  {
    const uint64_t a = 0x5DEECE66DULL;
    const uint64_t c = 0xBULL;
    const uint64_t mask = 0xFFFFFFFFFFFFULL;
    state = a*state + c;
    return float((state & mask) / float(mask+1ULL));
  }

  uint64_t state;
};
