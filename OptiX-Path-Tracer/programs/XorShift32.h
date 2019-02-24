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

#include <stdint.h>
#include "vec.h"

// Original class implementation by Ingo Wald
// Adapted to use Aras Pranckevicius' version of XorShift32
// source:
// https://github.com/aras-p/ToyPathTracer/blob/master/Cpp/Source/Maths.cpp#L5-L18
struct XorShift32 {
  // initialize the random number generator with a new seed (usually per pixel)
  inline __device__ void init(unsigned int s0, unsigned int s1) {
    state = s0 + WangHash(s1);
    for (int warmUp = 0; warmUp < 10; warmUp++) (*this)();
  }

  // get the next 'random' number in the sequence
  inline __device__ float operator()() {
    uint32_t x = state;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 15;

    state = x;

    return (x & 0xFFFFFF) / 16777216.f;
  }

  // Wang Hash source:
  // http://richiesams.blogspot.com/2015/03/creating-randomness-and-acummulating.html
  uint32_t __device__ WangHash(uint32_t a) {
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a;
  }

  uint32_t state;
};
