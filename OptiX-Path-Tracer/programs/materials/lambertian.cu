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

#include "light_sample.cuh"

///////////////////////////////////
// --- Lambertian BRDF Model --- //
///////////////////////////////////

// OptiX Context objects
rtDeclareVariable(Ray, ray, rtCurrentRay, );                // current ray
rtDeclareVariable(PerRayData, prd, rtPayload, );            // ray PRD
rtDeclareVariable(rtObject, world, , );                     // scene graph
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );  // hit distance

// Intersected Geometry Attributes
rtDeclareVariable(HitRecord_Function, Get_HitRecord, , );  // HitRecord function
rtDeclareVariable(int, geo_index, attribute geo_index, );  // primitive index
rtDeclareVariable(float2, bc, attribute bc, );  // triangle barycentrics

// Material Parameters
rtDeclareVariable(Texture_Function, sample_texture, , );

RT_FUNCTION Lambertian_Parameters Get_Parameters(const float3 &P,  // hit point
                                                 float u,  // texture coord x
                                                 float v,  // texture coord y
                                                 int index) {  // texture index
  Lambertian_Parameters surface;

  surface.color = sample_texture(u, v, P, index);

  return surface;
}

// Lambertian Material Closest Hit Program
RT_PROGRAM void closest_hit() {
  HitRecord rec = Get_HitRecord(geo_index, ray, t_hit, bc);
  int index = rec.index;          // texture index
  float3 P = rec.P;               // Hit Point
  float3 Wo = rec.Wo;             // Ray view direction
  float3 N = rec.shading_normal;  // normal

  Lambertian_Parameters surface = Get_Parameters(P, rec.u, rec.v, index);

  // Sample Direct Light
  float3 direct = Direct_Light(surface, P, Wo, N, false, prd.seed);
  prd.radiance += prd.throughput * direct;

  // Sample BRDF
  float3 Wi = Sample(surface, P, Wo, N, prd.seed);
  float pdf;  // calculated in the Evaluate function
  float3 attenuation = Evaluate(surface, P, Wo, Wi, N, pdf);

  // Assign parameters to PRD
  prd.scatterEvent = rayGotBounced;
  prd.origin = P;
  prd.direction = Wi;
  prd.throughput *= clamp(attenuation / pdf, 0.f, 1.f);
  prd.isSpecular = false;
}