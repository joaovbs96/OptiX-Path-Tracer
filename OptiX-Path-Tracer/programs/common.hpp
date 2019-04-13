#pragma once

// Common defines and includes

// OptiX includes:
#include <optix.h>
#include <optix_math.h>
#include <optix_world.h>

#define RT_USE_TEMPLATED_RTCALLABLEPROGRAM 1

using namespace optix;

#define RT_FUNCTION __forceinline__ __device__

// Math defines
#ifndef PI_F
#define PI_F 3.141592654f
#endif

#ifndef PI_D
#define PI_D 3.14159265358979323846264338327950288
#endif

// Material defines
typedef enum {
  Lambertian_BRDF,
  Diffuse_Light_BRDF,
  Metal_BRDF,
  Dielectric_BRDF,
  Isotropic_BRDF,
  Normal_BRDF,
  Anisotropic_BRDF,
  Oren_Nayar_BRDF,
  Torrance_Sparrow_BRDF
} BRDFType;

#define NUMBER_OF_MATERIALS 9

// Axis type
typedef enum { X_AXIS, Y_AXIS, Z_AXIS } AXIS;

// TODO: implement a math.cuh or math.hpp file
// Common math functon defines
RT_FUNCTION __host__ float Radians(float deg) { return (PI_F / 180.f) * deg; }
RT_FUNCTION __host__ float Degrees(float rad) { return (180.f / PI_F) * rad; }
RT_FUNCTION __host__ float sqrf(float value) { return value * value; }