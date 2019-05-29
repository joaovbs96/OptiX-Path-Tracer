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

#include "../prd.cuh"
#include "hitables.cuh"

// OptiX Context objects
rtDeclareVariable(Ray, ray, rtCurrentRay, );

// Intersected Geometry Attributes
rtDeclareVariable(int, geo_index, attribute geo_index, );  // primitive index
rtDeclareVariable(float2, bc, attribute bc, );  // triangle barycentrics

// Primitive Parameters
rtDeclareVariable(float3, center, , );
rtDeclareVariable(float, radius, , );

// Checks if Ray intersects Sphere and computes hit distance
RT_PROGRAM void hit_sphere(int pid) {
  const float3 oc = ray.origin - center;

  // if the ray hits the sphere, the following equation has two roots:
  // tdot(B, B) + 2tdot(B,A-C) + dot(A-C,A-C) - R = 0

  // Using Bhaskara's Formula, we have:
  const float a = dot(ray.direction, ray.direction);
  const float b = dot(oc, ray.direction);
  const float c = dot(oc, oc) - radius * radius;
  const float discriminant = b * b - a * c;

  // if the discriminant is lower than zero, there's no real
  // solution and thus no hit
  if (discriminant < 0.f) return;

  // first root of the sphere equation:
  float t = (-b - sqrtf(discriminant)) / a;
  if (rtPotentialIntersection(t)) {
    geo_index = 0;
    bc = make_float2(0);
    rtReportIntersection(0);
  }

  t = (-b + sqrtf(discriminant)) / a;
  if (rtPotentialIntersection(t)) {
    geo_index = 0;
    bc = make_float2(0);
    rtReportIntersection(0);
  }
}

// Gets HitRecord parameters, given a ray, an index and a hit distance
RT_CALLABLE_PROGRAM HitRecord Get_HitRecord(int index,    // primitive index
                                            Ray ray,      // current ray
                                            float t_hit,  // intersection dist
                                            float2 bc) {  // barycentrics
  HitRecord rec;

  // view direction
  rec.Wo = normalize(-ray.direction);

  // Hit Point
  float3 hit_point = ray.origin + t_hit * ray.direction;
  rec.P = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);

  // Normal
  float3 T = (rec.P - center) / radius;
  float3 normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, T));
  rec.shading_normal = rec.geometric_normal = normal;

  // Texture coordinates
  float phi = atan2(T.z, T.x);
  float theta = asin(T.y);
  rec.u = 1.f - (phi + PI_F) / (2.f * PI_F);
  rec.v = (theta + PI_F / 2.f) / PI_F;

  // Texture Index
  rec.index = index;

  return rec;
}

// Computes Sphere bounding box attributes
RT_PROGRAM void get_bounds(int pid, float result[6]) {
  Aabb* aabb = (Aabb*)result;
  aabb->m_min = center - radius;
  aabb->m_max = center + radius;
}
