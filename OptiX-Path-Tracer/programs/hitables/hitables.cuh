#pragma once

#include "../random.cuh"
#include "../vec.hpp"

RT_FUNCTION int2 Get_Motion_Data(float2 motion_range, float cur_time,
                                 int num_keys, float& pt) {
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
