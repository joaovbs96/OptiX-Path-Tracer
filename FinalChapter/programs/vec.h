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

// optix code:
#include <cuda.h>
#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#ifdef __CUDACC__
#else
#endif

struct vec3f 
{
  inline __device__ vec3f() {}
  inline __device__ vec3f(float f) 
    : x(f), y(f), z(f) 
  {}
  inline __device__ vec3f(const float x, const float y, const float z)
    : x(x), y(y), z(z)
  {}
  inline __device__ vec3f(const vec3f &v)
    : x(v.x), y(v.y), z(v.z)
  {}
#ifdef __CUDACC__
  inline __device__ vec3f(const float3 v)
    : x(v.x), y(v.y), z(v.z)
  {}
  inline __device__ float3 as_float3() const { return make_float3(x, y, z); }
#endif
  inline __device__ float squared_length() const { return x * x + y * y + z * z; }
  inline __device__ float length() const { return sqrtf(x * x + y * y + z * z); }
  float x, y, z;
};

inline __device__ vec3f operator*(const vec3f &a, const vec3f &b)
{
  return vec3f(a.x*b.x, a.y*b.y, a.z*b.z);
}

inline __device__ vec3f operator/(const vec3f &a, const vec3f &b)
{
  return vec3f(a.x/b.x, a.y/b.y, a.z/b.z);
}

inline __device__ vec3f operator-(const vec3f &a, const vec3f &b)
{
  return vec3f(a.x-b.x, a.y-b.y, a.z-b.z);
}

inline __device__ vec3f operator-(const vec3f &a)
{
  return vec3f(-a.x, -a.y, -a.z);
}

inline __device__ vec3f operator+(const vec3f &a, const vec3f &b)
{
  return vec3f(a.x+b.x, a.y+b.y, a.z+b.z);
}

inline __device__ vec3f &operator+=(vec3f &a, const vec3f &b)
{
  a = a + b;
  return a;
}

inline __device__ float dot(const vec3f &a, const vec3f &b)
{
  return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline __device__ vec3f cross(const vec3f &a, const vec3f &b)
{
  return vec3f(
               a.y*b.z - a.z*b.y,
               a.z*b.x - a.x*b.z,
               a.x*b.y - a.y*b.x);
}

inline __device__ vec3f normalize(const vec3f &v)
{
  return v * (1.f / sqrtf(dot(v, v)));
}


inline __device__ vec3f unit_vector(const vec3f &v)
{
  return normalize(v);
}

/*! return absolute value of each component */
inline __device__ vec3f abs(const vec3f &v)
{
  return vec3f(fabsf(v.x),fabsf(v.y),fabsf(v.z));
}

inline __device__ vec3f mod(const vec3f &a,const vec3f &b)
{
  return vec3f(fmodf(a.x,b.x),
               fmodf(a.y,b.y),
               fmodf(a.z,b.z));
}
