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
  Isotropic_Material
} MaterialType;

#define NUMBER_OF_MATERIALS 5

// TODO: what would be a better way to do this? We use this data type to keep
// the programs generic. We can keep the PRD minimal this way. Is it faster or
// slower? Alternatives?
struct MaterialParameters {
  int textureIndex;
  MaterialType type;
  // All materials need at least a texture, so this 'index' is used by all.
  // The ones who need extra textures can specify in their sub-structs.

  struct {
    float fuzz;
  } metal;

  struct {
    float ref_idx;
    float density;
    int volumeTexIndex;
  } dielectric;
};

// Axis type
typedef enum { X_AXIS, Y_AXIS, Z_AXIS } AXIS;

// Clamp functions
RT_FUNCTION __host__ float Clamp(const float& value, const float& bottom,
                                 const float& top) {
  if (value < bottom) return bottom;
  if (value > top) return top;
  return value;
}

RT_FUNCTION __host__ float3 Clamp(const float3& c) {
  float3 temp = c;
  if (temp.x > 1.f) temp.x = 1.f;
  if (temp.y > 1.f) temp.y = 1.f;
  if (temp.z > 1.f) temp.z = 1.f;

  return temp;
}