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

#include "material.h"

/*! the implicit state's ray we will intersect against */
rtDeclareVariable(Ray, ray, rtCurrentRay, );

/*! the per ray data we operate on */
rtDeclareVariable(PerRayData, prd, rtPayload, );
rtDeclareVariable(rtObject, world, , );

/*! the attributes we use to communicate between intersection programs and hit
 * program */
rtDeclareVariable(HitRecord, hit_rec, attribute hit_rec, );

/*! and finally - that particular material's parameters */
rtBuffer<rtCallableProgramId<float3(float, float, float3)> > sample_texture;

RT_PROGRAM void closest_hit() {
  prd.matType = Lambertian_Material;
  prd.isSpecular = false;
  prd.scatterEvent = rayGotBounced;

  prd.origin = hit_rec.p;
  prd.normal = hit_rec.normal;

  int index = hit_rec.index;
  prd.emitted = make_float3(0.f);
  prd.attenuation = sample_texture[index](hit_rec.u, hit_rec.v, hit_rec.p);
}

RT_CALLABLE_PROGRAM float3 BRDF_Sample(PDFParams &pdf, XorShift32 &rnd) {
  float3 temp;

  cosine_sample_hemisphere(rnd(), rnd(), temp);

  Onb uvw(pdf.normal);
  uvw.inverse_transform(temp);

  pdf.direction = temp;

  return pdf.direction;
}

RT_CALLABLE_PROGRAM float BRDF_PDF(PDFParams &pdf) {
  float cosine = dot(unit_vector(pdf.direction), unit_vector(pdf.normal));

  if (cosine < 0.f) cosine = 0.f;

  return cosine / PI_F;
}

RT_CALLABLE_PROGRAM float BRDF_Evaluate(PDFParams &pdf) {
  float cosine = dot(unit_vector(pdf.normal), unit_vector(pdf.direction));

  if (cosine < 0.f) cosine = 0.f;

  return cosine / PI_F;
}