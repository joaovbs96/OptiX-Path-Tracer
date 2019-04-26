
#include "material.cuh"
#include "microfacets.cuh"

struct Torrance_Sparrow_Parameters {
  float3 color;
  float nu, nv;
};

RT_FUNCTION float3 Sample(const Torrance_Sparrow_Parameters &surface,
                          const float3 &P,   // next ray origin
                          const float3 &Wo,  // prev ray direction
                          const float3 &N,   // shading normal
                          uint &seed) {
  // Get material params from input variable
  float nu = surface.nu;
  float nv = surface.nv;

  // create basis
  float3 Nn = normalize(N);
  float3 T = normalize(cross(Nn, make_float3(0.f, 1.f, 0.f)));
  float3 B = cross(T, Nn);

  // random variables
  float2 random = make_float2(rnd(seed), rnd(seed));

  // get half vector and rotate it to world space
  float3 H = normalize(GGX_Sample(Wo, random, nu, nv));
  H = H.x * B + H.y * Nn + H.z * T; 

  float HdotI = dot(H, Wo);
  if(HdotI < 0.f) H = -H;

  float3 Wi;
  float eta = CosTheta(Wo) > 0 ? (nu / nv) : (nv / nu);

  if (!refract(Wo, H, eta, Wi)) 
		pdf.direction = make_float3(0.f);
	else
  	pdf.direction = Wi;

  return pdf.direction;

	// reflect
	//return normalize(-Wo + 2.f * dot(Wo, H) * H);
}

RT_FUNCTION float3 Evaluate(const Torrance_Sparrow_Parameters &surface,
                            const float3 &P,   // next ray origin
                            const float3 &Wo,  // prev ray direction
                            const float3 &Wi,  // next ray direction
                            const float3 &N,   // shading normal
                            float &pdf) {
  // Get material params from input variable
  float3 Rs = surface.color;
  float nu = surface.nu;
  float nv = surface.nv;

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
	
	// Compute change of variables _dwh\_dwi_ for microfacet transmission
	float pdf_denom = Square(dot(Wo, H) + eta * dot(Wi, H));
	float dwh_dwi = fabsf((eta * eta * dot(Wi, H)) / pdf_denom);
	pdf = GGX_PDF(H, Wo, nu, nv) * dwh_dwi;

  return (Rs * (make_float3(1.f) - F) * D * G  * AHdotI * AHdotO) / denom;
}