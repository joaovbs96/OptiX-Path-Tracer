#pragma once

#include "../prd.cuh"

RT_FUNCTION int2 Get_Motion_Data(float2 motion_range, float cur_time,
                                 int num_keys, float &pt) {
  float t0 = motion_range.x;
  float t1 = motion_range.y;
  float clamped_time = clamp(cur_time, t0, t1);
  float step_size = num_keys == 1 ? 1.0f : (t1 - t0) / (num_keys - 1);
  int time_per_step = static_cast<int>((clamped_time - t0) / step_size);

  int t0_idx = fminf(num_keys - 1, time_per_step);
  int t1_idx = fminf(num_keys - 1, t0_idx + 1);
  // pt = (clamped_time - t0) / step_size - t0_idx;

  return make_int2(t0_idx, t1_idx);
}

// Offset and refine functions
// https://github.com/nvpro-samples/optix_advanced_samples/blob/21465ae85c47f3c57371c77fd2aa3bac4adabcd4/src/device_include/intersection_refinement.h#L1

// Offset the hit point using integer arithmetic
RT_FUNCTION float3 offset(const float3& hit_point, const float3& normal) {
  const float epsilon = 1.0e-4f;
  const float offset  = 4096.0f * 2.0f;

  float3 offset_point = hit_point;
  if((__float_as_int(hit_point.x) & 0x7fffffff) < __float_as_int(epsilon)) {
    offset_point.x += epsilon * normal.x;
  } else {
    offset_point.x = __int_as_float(__float_as_int(offset_point.x) + int(copysign(offset, hit_point.x) * normal.x));
  }

  if((__float_as_int(hit_point.y) & 0x7fffffff) < __float_as_int(epsilon)) {
    offset_point.y += epsilon * normal.y;
  } else {
    offset_point.y = __int_as_float(__float_as_int(offset_point.y) + int(copysign(offset, hit_point.y) * normal.y));
  }

  if((__float_as_int(hit_point.z) & 0x7fffffff) < __float_as_int(epsilon)) {
    offset_point.z += epsilon * normal.z;
  } else {
    offset_point.z = __int_as_float(__float_as_int(offset_point.z) + int(copysign(offset, hit_point.z) * normal.z));
  }

  return offset_point;
}

// Refine the hit point to be more accurate and offset it for reflection and
// refraction ray starting points.
RT_FUNCTION void refine_and_offset_hitpoint(const float3& P, // P
                                            const float3& direction,          // Wo
                                            const float3& normal,             // N
                                            const float3& P0,                  // ?
                                            float3& back_hit_point,
                                            float3& front_hit_point) {
  // Refine hit point
  float refined_t = -(dot(normal, P - P0)) / dot(normal, direction);
  float3 refined_hit_point = P + refined_t * direction;

  // Offset hit point
  if(dot(direction, normal) > 0.0f ) {
    back_hit_point  = offset(refined_hit_point,  normal);
    front_hit_point = offset(refined_hit_point, -normal);
  } else {
    back_hit_point  = offset(refined_hit_point, -normal);
    front_hit_point = offset(refined_hit_point,  normal);
  }
}