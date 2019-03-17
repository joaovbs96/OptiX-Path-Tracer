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
  Lambertian_Material,
  Diffuse_Light_Material,
  Metal_Material,
  Dielectric_Material,
  Isotropic_Material,
  Normal_Material,
  Anisotropic_Material,
  Oren_Nayar_Material
} MaterialType;

#define NUMBER_OF_MATERIALS 8

// Axis type
typedef enum { X_AXIS, Y_AXIS, Z_AXIS } AXIS;