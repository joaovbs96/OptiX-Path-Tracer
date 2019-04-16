#include "material.cuh"
#include "microfacets.cuh"

///////////////////////////////////////////////
// --- Torrance–Sparrow Refraction Model --- //
///////////////////////////////////////////////

// Based on PBRT code & theory
// http://www.pbr-book.org/3ed-2018/Reflection_Models/Microfacet_Models.html#TheTorrancendashSparrowModel
// https://github.com/mmp/pbrt-v3/blob/9f717d847a807793fa966cf0eaa366852efef167/src/core/reflection.h#L429

// OptiX Context objects
rtDeclareVariable(Ray, ray, rtCurrentRay, );                 // current ray
rtDeclareVariable(PerRayData, prd, rtPayload, );             // ray PRD
rtDeclareVariable(HitRecord, hit_rec, attribute hit_rec, );  // from geometry

// Material Parameters
rtDeclareVariable(rtCallableProgramId<float3(float, float, float3, int)>,
                  sample_texture, , );
rtDeclareVariable(float, nu, , );
rtDeclareVariable(float, nv, , );

///////////////////////////
// --- BRDF Programs --- //
///////////////////////////

// Assigns material and hit parameters to PRD
RT_PROGRAM void closest_hit() {
  prd.matType = Microfacet_Transmission_BRDF;
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
  prd.matParams.attenuation = color;
  prd.matParams.anisotropic.nu = nu;
  prd.matParams.anisotropic.nv = nv;
}

// TODO: change Sample return to bool(returns false when invalid)

// Samples BRDF, generating outgoing direction(Wo)
RT_CALLABLE_PROGRAM float3 BRDF_Sample(PDFParams &pdf, uint &seed) {
  // Get material params from input variable
  float nu = pdf.matParams.anisotropic.nu;
  float nv = pdf.matParams.anisotropic.nv;

  float3 Wo = pdf.view_direction;  // outgoing, to camera

  // create basis
  float3 N = normalize(pdf.geometric_normal);
  float3 T = normalize(cross(N, make_float3(0.f, 1.f, 0.f)));
  float3 B = cross(T, N);

  // random variables
  float2 random = make_float2(rnd(seed), rnd(seed));
  pdf.matParams.u = random.x;
  pdf.matParams.v = random.y;

  // get half vector and rotate it to world space
  float3 H = normalize(GGX_Sample(Wo, random, nu, nv));
  H = H.x * B + H.y * N + H.z * T; 

  float HdotI = dot(H, Wo);
  if(HdotI < 0.f) H = -H;

  float3 Wi;
  float eta = CosTheta(Wo) > 0 ? (nu / nv) : (nv / nu);

	if (!refract(Wo, H, eta, Wi)) 
		pdf.direction = make_float3(0.f);
	else
  	pdf.direction = Wi;

  return pdf.direction;
}

// Gets BRDF PDF value
RT_CALLABLE_PROGRAM float BRDF_PDF(PDFParams &pdf) {
  float3 Wo = pdf.view_direction, Wi = pdf.direction;

  // Get material params from input variable
  float nu = pdf.matParams.anisotropic.nu;
	float nv = pdf.matParams.anisotropic.nv;
	
	float eta = CosTheta(Wo) > 0 ? (nv / nu) : (nu / nv);
	float3 H = normalize(Wo + Wi * eta);

	// Compute change of variables _dwh\_dwi_ for microfacet transmission
	float denom = Square(dot(Wo, H) + eta * dot(Wi, H));
	float dwh_dwi = fabsf((eta * eta * dot(Wi, H)) / denom);

  return GGX_PDF(H, Wo, nu, nv) * dwh_dwi;
}

// Evaluates BRDF, returning its reflectance
RT_CALLABLE_PROGRAM float3 BRDF_Evaluate(PDFParams &pdf) {
  float3 Wo = pdf.view_direction, Wi = pdf.direction;
  float3 Rs = pdf.matParams.attenuation;
  float nu = pdf.matParams.anisotropic.nu;
  float nv = pdf.matParams.anisotropic.nv;
  
  float cosThetaI = CosTheta(Wi), cosThetaO = CosTheta(Wo);
  if (cosThetaI == 0.f || cosThetaO == 0.f) return make_float3(0.f);

  float eta = CosTheta(Wo) > 0 ? (nv / nu) : (nu / nv);
  float3 H = normalize(Wo + Wi * eta);
  if(H.y < 0) H *= -1;

  // *Note that this is not a symetric BRDF. The PBRT implementation take both 
  // directions into account. The code here works only on unidirectional PTs.
  // https://github.com/mmp/pbrt-v3/blob/f7653953b2f9cc5d6a53b46acb5ce03317fd3e8b/src/core/reflection.cpp#L260

  float HdotO = dot(H, Wo), HdotI = dot(H, Wi);
  float AHdotO = fabsf(HdotO), AHdotI = fabsf(HdotI);

  float denom = Square(HdotO + eta * HdotI) * cosThetaI * cosThetaO;

  float3 F = schlick(Rs, HdotI);    // Fresnel Reflectance
  float G = GGX_G(Wo, Wi, nu, nv);  // Geometric Shadowing
  float D = GGX_D(H, nu, nv);       // Normal Distribution Function(NDF)

  return (Rs * (make_float3(1.f) - F) * D * G  * AHdotI * AHdotO) / denom;
}