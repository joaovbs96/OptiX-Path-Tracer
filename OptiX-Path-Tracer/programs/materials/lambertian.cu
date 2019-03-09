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

#include "material.cuh"

/*! the implicit state's ray we will intersect against */
rtDeclareVariable(Ray, ray, rtCurrentRay, );

/*! the per ray data we operate on */
rtDeclareVariable(PerRayData, prd, rtPayload, );
rtDeclareVariable(rtObject, world, , );

/*! the attributes we use to communicate between intersection programs and hit
 * program */
rtDeclareVariable(HitRecord, hit_rec, attribute hit_rec, );

/*! and finally - that particular material's parameters */
rtDeclareVariable(rtCallableProgramId<float3(float, float, float3, int)>,
                  sample_texture, , );

// TODO: add geometric_normal and shading_normal params to PRD

RT_PROGRAM void closest_hit() {
  // get material params from buffer
  int texIndex = hit_rec.index;

  // assign material params to prd
  prd.matType = Lambertian_Material;
  prd.isSpecular = false;
  prd.scatterEvent = rayGotBounced;

  prd.emitted = make_float3(0.f);
  prd.attenuation = sample_texture(hit_rec.u, hit_rec.v, hit_rec.p, texIndex);

  // assign hit params to prd
  prd.origin = hit_rec.p;
  prd.geometric_normal = hit_rec.geometric_normal;
  prd.shading_normal = hit_rec.shading_normal;
}

RT_CALLABLE_PROGRAM float3 BRDF_Sample(PDFParams &pdf, uint &seed) {
  float3 temp;
  cosine_sample_hemisphere(rnd(seed), rnd(seed), temp);

  Onb uvw(pdf.normal);
  uvw.inverse_transform(temp);

  pdf.direction = temp;

  return pdf.direction;
}

RT_CALLABLE_PROGRAM float BRDF_PDF(PDFParams &pdf) {
  float cosine = dot(unit_vector(pdf.direction), unit_vector(pdf.normal));

  if (cosine < 0.f)
    return 0.f;
  else
    return cosine / PI_F;
}

RT_CALLABLE_PROGRAM float BRDF_Evaluate(PDFParams &pdf) {
  float cosine = dot(unit_vector(pdf.direction), unit_vector(pdf.normal));

  if (cosine < 0.f)
    return 0.f;
  else
    return cosine / PI_F;
}