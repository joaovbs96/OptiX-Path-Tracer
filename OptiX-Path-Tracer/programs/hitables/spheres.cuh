#pragma once

#include "../prd.cuh"
#include "hitables.cuh"


/*RT_FUNCTION void Sphere_Parameters(float3& P,   // hit point
                                   float3& Wo,  // view direction
                                   float3& Ns,  // surface shading normal
                                   float3& Ng,  // surface geometric normal
                                   float t,     // hit distance
                                   float u,
                                   float v,      // surface coordinates
                                   int index) {  // material index
  // view direction
  Wo = normalize(-ray.direction);

  hit_rec.t = temp;

  hit_rec.Wo = normalize(-ray.direction);

  float3 hit_point = ray.origin + temp * ray.direction;
  hit_point = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);
  hit_rec.P = hit_point;

  float3 normal = (hit_rec.P - center) / radius;
  normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));
  hit_rec.geometric_normal = normal;
  hit_rec.shading_normal = hit_rec.geometric_normal;

  get_sphere_uv((hit_rec.P - center) / radius);

  hit_rec.index = index;
}*/