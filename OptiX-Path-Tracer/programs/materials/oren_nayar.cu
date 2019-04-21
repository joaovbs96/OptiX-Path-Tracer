#include "material.cuh"

//////////////////////////////////////////
// --- Oren-Nayar Reflectance Model --- //
//////////////////////////////////////////

// Original Paper: "Generalization of Lambert’s Reflectance Model"
// http://www.cs.columbia.edu/CAVE/projects/oren/
// http://www1.cs.columbia.edu/CAVE/publications/pdfs/Oren_SIGGRAPH94.pdf

// Yasuhiro Fujii’s "A tiny improvement of Oren-Nayar reflectance model" variant
// http://mimosa-pudica.net/improved-oren-nayar.html

// Reference Implementations:
// https://developer.blender.org/diffusion/C/browse/master/src/kernel/closure/bsdf_oren_nayar.h
// https://github.com/mmp/pbrt-v3/blob/f7653953b2f9cc5d6a53b46acb5ce03317fd3e8b/src/core/reflection.cpp#L197-L224

// OptiX Context objects
rtDeclareVariable(Ray, ray, rtCurrentRay, );                 // current ray
rtDeclareVariable(PerRayData, prd, rtPayload, );             // ray PRD
rtDeclareVariable(HitRecord, hit_rec, attribute hit_rec, );  // from geometry

// Material Parameters
rtDeclareVariable(rtCallableProgramId<float3(float, float, float3, int)>,
                  sample_texture, , );
rtDeclareVariable(float, rA, , );
rtDeclareVariable(float, rB, , );

///////////////////////////
// --- BRDF Programs --- //
///////////////////////////

// TODO: idea, the closest hit can get the surface parameters as normal and then
// call a templated function, that then calls the sample and other functions as
// normal

// Assigns material and hit parameters to PRD
RT_PROGRAM void closest_hit() {
  prd.matType = Oren_Nayar_BRDF;
  prd.isSpecular = false;
  prd.scatterEvent = rayGotBounced;

  // Get hit params
  prd.origin = hit_rec.p;
  prd.geometric_normal = normalize(hit_rec.geometric_normal);
  prd.shading_normal = normalize(hit_rec.shading_normal);
  prd.view_direction = normalize(hit_rec.view_direction);

  // Get material color
  int index = hit_rec.index;
  float3 color = sample_texture(hit_rec.u, hit_rec.v, hit_rec.p, index);

  // Assign material parameters to PRD, to be used in the sampling programs
  prd.matParams.orenNayar.rA = rA;
  prd.matParams.orenNayar.rB = rB;
  prd.matParams.attenuation = color;
}

RT_CALLABLE_PROGRAM float3 BRDF_Sample(const BRDFParameters &surface,
                                       const float3 &P,   // next ray origin
                                       const float3 &Wo,  // prev ray direction
                                       const float3 &N,   // shading normal
                                       uint &seed) {
  float3 Wi;
  cosine_sample_hemisphere(rnd(seed), rnd(seed), Wi);

  Onb uvw(N);
  uvw.inverse_transform(Wi);

  return Wi;
}

RT_CALLABLE_PROGRAM float BRDF_PDF(const BRDFParameters &surface,
                                   const float3 &P,    // next ray origin
                                   const float3 &Wo,   // prev ray direction
                                   const float3 &Wi,   // next ray direction
                                   const float3 &N) {  // shading normal
  float cosine = dot(normalize(Wi), normalize(N));

  if (cosine < 0.f)
    return 0.f;
  else
    return cosine / PI_F;
}

// Evaluates BRDF, returning its reflectance
RT_CALLABLE_PROGRAM float3
BRDF_Evaluate(const BRDFParameters &surface,
              const float3 &P,    // next ray origin
              const float3 &Wo,   // prev ray direction
              const float3 &Wi,   // next ray direction
              const float3 &N) {  // shading normal
  float3 WiN = normalize(Wi);

  float sinThetaI = SinTheta(WiN);
  float sinThetaO = SinTheta(Wo);
  // Compute cosine term of Oren-Nayar model
  float maxCos = 0;
  if (sinThetaI > 1e-4 && sinThetaO > 1e-4) {
    float sinPhiI = SinPhi(WiN), cosPhiI = CosPhi(WiN);
    float sinPhiO = SinPhi(Wo), cosPhiO = CosPhi(Wo);
    float dCos = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
    maxCos = fmaxf(0.f, dCos);
  }

  // Compute sine and tangent terms of Oren-Nayar model
  float sinAlpha, tanBeta;
  if (AbsCosTheta(WiN) > AbsCosTheta(Wo)) {
    sinAlpha = sinThetaO;
    tanBeta = sinThetaI / AbsCosTheta(WiN);
  } else {
    sinAlpha = sinThetaI;
    tanBeta = sinThetaO / AbsCosTheta(Wo);
  }

  float rA = surface.orenNayar.rA;
  float rB = surface.orenNayar.rB;
  float3 color = surface.attenuation;

  return color * (1.f / PI_F) * (rA + rB * maxCos * sinAlpha * tanBeta);
}