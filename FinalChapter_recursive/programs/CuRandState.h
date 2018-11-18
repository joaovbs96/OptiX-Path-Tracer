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
#include <curand_kernel.h>

struct CuRandState
{
  /*! initialize the random number generator with a new seed (usually
      per pixel) */
  inline __device__ void init(int seed = 0)
  {
    curand_init(1984, seed, 0, &state);
  }

  /*! get the next 'random' number in the sequence */
  inline __device__ float operator() ()
  {
    return curand_uniform(&state);
  }
  
  curandState state;
};
