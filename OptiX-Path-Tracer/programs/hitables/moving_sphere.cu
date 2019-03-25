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

#include "hitables.cuh"

/*! the parameters that describe each individual sphere geometry */
rtDeclareVariable(float3, center0, , );
rtDeclareVariable(float3, center1, , );
rtDeclareVariable(float, radius, , );
rtDeclareVariable(float, time0, , );
rtDeclareVariable(float, time1, , );
rtDeclareVariable(int, index, , );

/*! the implicit state's ray we will intersect against */
rtDeclareVariable(Ray, ray, rtCurrentRay, );

/*! the attributes we use to communicate between intersection programs and hit
 * program */
rtDeclareVariable(HitRecord, hit_rec, attribute hit_rec, );

/*! the per ray data we operate on */
rtDeclareVariable(PerRayData, prd, rtPayload, );

RT_FUNCTION void get_sphere_uv(const float3& p) {
  float phi = atan2(p.z, p.x);
  float theta = asin(p.y);

  hit_rec.u = 1 - (phi + PI_F) / (2 * PI_F);
  hit_rec.v = (theta + PI_F / 2) / PI_F;
}

RT_FUNCTION float3 center(float time) {
  return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
}

// Program that performs the ray-sphere intersection
//
// note that this is here is a simple, but not necessarily most numerically
// stable ray-sphere intersection variant out there. There are more
// stable variants out there, but for now let's stick with the one that
// the reference code used.
RT_PROGRAM void hit_sphere(int pid) {
  float pt;
  // int2 idx = Get_Motion_Data(motionRange, curTime, vertex_buffers.size(),
  // pt);

  const float3 oc = ray.origin - center(prd.time);

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
  float temp = (-b - sqrtf(discriminant)) / a;

  // for a sphere, its normal is in (hitpoint - center)

  // if the first root was a hit,
  if (temp < ray.tmax && temp > ray.tmin) {
    if (rtPotentialIntersection(temp)) {
      hit_rec.distance = temp;

      hit_rec.view_direction = normalize(-ray.direction);

      float3 hit_point = ray.origin + temp * ray.direction;
      hit_point = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);
      hit_rec.p = hit_point;

      float3 normal = (hit_rec.p - center(prd.time)) / radius;
      normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));
      hit_rec.geometric_normal = normal;
      hit_rec.shading_normal = hit_rec.geometric_normal;

      get_sphere_uv((hit_rec.p - center(prd.time)) / radius);

      hit_rec.index = index;

      rtReportIntersection(0);
    }
  }

  // if the second root was a hit,
  temp = (-b + sqrtf(discriminant)) / a;
  if (temp < ray.tmax && temp > ray.tmin) {
    if (rtPotentialIntersection(temp)) {
      hit_rec.distance = temp;

      hit_rec.view_direction = normalize(-ray.direction);

      float3 hit_point = ray.origin + temp * ray.direction;
      hit_point = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);
      hit_rec.p = hit_point;

      float3 normal = (hit_rec.p - center(prd.time)) / radius;
      normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));
      hit_rec.geometric_normal = normal;
      hit_rec.shading_normal = hit_rec.geometric_normal;

      get_sphere_uv((hit_rec.p - center(prd.time)) / radius);

      hit_rec.index = index;

      rtReportIntersection(0);
    }
  }
}

/*! returns the bounding box of the pid'th primitive
  in this gometry. Since we only have one sphere in this
  program (we handle multiple spheres by having a different
  geometry per sphere), the'pid' parameter is ignored */
RT_PROGRAM void get_bounds(int pid, float result[6]) {
  Aabb* box0 = (Aabb*)result;
  box0->m_min = center(time0) - radius;
  box0->m_max = center(time0) + radius;

  Aabb box1;
  box1.m_min = center(time1) - radius;
  box1.m_max = center(time1) + radius;

  // extends box to include the t1 box
  box0->include(box1);
}
