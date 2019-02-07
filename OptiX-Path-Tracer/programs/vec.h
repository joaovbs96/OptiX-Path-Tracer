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

#include "common.h"

inline __host__ __device__ float squared_length(const float3 &a) { 
    return length(a) * length(a);
}

// returns true if all components are zero
inline __host__ __device__ bool isNull(const float3 &a) {
  return (a.x == 0.f) && (a.y == 0.f) && (a.z == 0.f);
}

/*! return absolute value of each component */
inline __host__ __device__ float3 abs(const float3 &v) {
  return make_float3(fabsf(v.x), fabsf(v.y), fabsf(v.z));
}

/* return unit vector */ 
inline __host__ __device__ float3 unit_vector(const float3 &v) {
  return v / length(v);
}

/*! return mod between two vectors */
inline __host__ __device__ float3 mod(const float3 &a,const float3 &b) {
  return make_float3(fmodf(a.x,b.x), fmodf(a.y,b.y), fmodf(a.z,b.z));
}

// if (a < b), return a, else return b
inline __host__ __device__ float ffmin(const float &a, const float &b) {
  return a < b ? a : b;
}

// if (a > b), return a, else return b
inline __host__ __device__ float ffmax(const float &a, const float &b) {
  return a > b ? a : b;
}

// return pairwise min vector
inline __host__ __device__ float3 min_vec(const float3 &a, const float3 &b) {
  return make_float3(ffmin(a.x, b.x), ffmin(a.y, b.y), ffmin(a.z, b.z));
}

// return pairwise max vector
inline __host__ __device__ float3 max_vec(float3 a, float3 b) {
  return make_float3(ffmax(a.x, b.x), ffmax(a.y, b.y), ffmax(a.z, b.z));
}

// return max component of vector
inline __host__ __device__ float max_component(float3 a) {
  return ffmax(ffmax(a.x, a.y), a.z);
}

// return max component of vector
inline __host__  __device__ float min_component(float3 a) {
  return ffmin(ffmin(a.x, a.y), a.z);
}