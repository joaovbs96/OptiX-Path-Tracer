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

#include "../prd.h"

/*! the parameters that describe each individual sphere geometry */
rtDeclareVariable(float3, center, , );
rtDeclareVariable(float,  radius, , );

/*! the implicit state's ray we will intersect against */
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

/*! the attributes we use to communicate between intersection programs and hit program */
rtDeclareVariable(Hit_Record, hit_rec, attribute hit_rec, );

/*! the per ray data we operate on */
rtDeclareVariable(PerRayData, prd, rtPayload, );

inline __device__ void get_sphere_uv(const float3& p) {
	float phi = atan2(p.z, p.x);
	float theta = asin(p.y); 

	hit_rec.u = 1.f - (phi + PI_F) / (2.f * PI_F);
	hit_rec.v = (theta + PI_F / 2.f) / PI_F;
}

// Program that performs the ray-sphere intersection
//
// note that this is here is a simple, but not necessarily most numerically
// stable ray-sphere intersection variant out there. There are more
// stable variants out there, but for now let's stick with the one that
// the reference code used.
RT_PROGRAM void hit_sphere(int pid) {
  const float3 oc = ray.origin - center;

	// if the ray hits the sphere, the following equation has two roots:
	// tdot(B, B) + 2tdot(B,A-C) + dot(A-C,A-C) - R = 0

	// Using Bhaskara's Formula, we have:
  const float  a = dot(ray.direction, ray.direction);
  const float  b = dot(oc, ray.direction);
  const float  c = dot(oc, oc) - radius * radius;
  const float  discriminant = b * b - a * c;
  
  // if the discriminant is lower than zero, there's no real 
  // solution and thus no hit
  if (discriminant < 0.f) 
    return;

  // first root of the sphere equation:
  float temp = (-b - sqrtf(discriminant)) / a;

  // for a sphere, its normal is in (hitpoint - center)
  
  // if the first root was a hit,
  if (temp < ray.tmax && temp > ray.tmin) {
    if (rtPotentialIntersection(temp)) {
      hit_rec.distance = temp;

      float3 hit_point = ray.origin + temp * ray.direction;
      hit_point = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);
      hit_rec.p = hit_point;

      float3 normal = (hit_rec.p - center) / radius;
      normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));
      hit_rec.normal = normal;

      get_sphere_uv((hit_rec.p - center) / radius);

      hit_rec.index = 0;

      rtReportIntersection(0);
    }
  }
  
  // if the second root was a hit,
  temp = (-b + sqrtf(discriminant)) / a;
  if (temp < ray.tmax && temp > ray.tmin) {
    if (rtPotentialIntersection(temp)) {
      hit_rec.distance = temp;
      
      float3 hit_point = ray.origin + temp * ray.direction;
      hit_point = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);
      hit_rec.p = hit_point;

      float3 normal = (hit_rec.p - center) / radius;
      normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));
      hit_rec.normal = normal;

      get_sphere_uv((hit_rec.p - center) / radius);

      hit_rec.index = 0;

      rtReportIntersection(0);
    }
  }
}

/*! returns the bounding box of the pid'th primitive
  in this gometry. Since we only have one sphere in this 
  program (we handle multiple spheres by having a different
  geometry per sphere), the'pid' parameter is ignored */
RT_PROGRAM void get_bounds(int pid, float result[6]) {
  optix::Aabb* aabb = (optix::Aabb*)result;
  aabb->m_min = center - radius;
  aabb->m_max = center + radius;
}
