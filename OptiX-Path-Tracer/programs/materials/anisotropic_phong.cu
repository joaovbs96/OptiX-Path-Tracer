
#include "material.cuh"
#include "microfacets.cuh"

// TODO: rename file and material name to Ashikhmin-Shirley

////////////////////////////////////////////////////////////
// --- Ashikhmin-Shirley Anisotropic Phong BRDF Model --- //
////////////////////////////////////////////////////////////

// Original Paper & Tech Report - "An Anisotropic Phong Light Reflection Model"
// https://www.cs.utah.edu/~shirley/papers/jgtbrdf.pdf
// https://www.cs.utah.edu/docs/techreports/2000/pdf/UUCS-00-014.pdf

// Reference Implementation:
// https://developer.blender.org/diffusion/C/browse/master/src/kernel/closure/bsdf_ashikhmin_shirley.h
// FresnelBlend from PBRT
// https://github.com/mmp/pbrt-v3/blob/9f717d847a807793fa966cf0eaa366852efef167/src/core/reflection.cpp
// https://github.com/mmp/pbrt-v3/blob/9f717d847a807793fa966cf0eaa366852efef167/src/core/reflection.h

// OptiX Context objects
rtDeclareVariable(Ray, ray, rtCurrentRay, );                 // current ray
rtDeclareVariable(PerRayData, prd, rtPayload, );             // ray PRD
rtDeclareVariable(HitRecord, hit_rec, attribute hit_rec, );  // from geometry

// Material Parameters
rtDeclareVariable(rtCallableProgramId<float3(float, float, float3, int)>,
                  diffuse_color, , );
rtDeclareVariable(rtCallableProgramId<float3(float, float, float3, int)>,
                  specular_color, , );
rtDeclareVariable(float, nu, , );
rtDeclareVariable(float, nv, , );

// Blender Parameters
// I - origin - omega_out
// O - direction - omega_in

///////////////////////////
// --- BRDF Programs --- //
///////////////////////////

// Assigns material and hit parameters to PRD
RT_PROGRAM void closest_hit() {
  prd.matType = Anisotropic_BRDF;
  prd.isSpecular = false;
  prd.scatterEvent = rayGotBounced;

  // Get hit params
  prd.origin = hit_rec.p;
  prd.geometric_normal = normalize(hit_rec.geometric_normal);
  prd.shading_normal = normalize(hit_rec.shading_normal);
  prd.view_direction = normalize(hit_rec.view_direction);

  // Get material colors
  int index = hit_rec.index;
  float3 diffuse = diffuse_color(hit_rec.u, hit_rec.v, hit_rec.p, index);
  float3 specular = specular_color(hit_rec.u, hit_rec.v, hit_rec.p, index);

  // Assign material parameters to PRD, to be used in the sampling programs
  prd.matParams.anisotropic.diffuse_color = diffuse;
  prd.matParams.anisotropic.specular_color = specular;
  prd.matParams.anisotropic.nu = nu;
  prd.matParams.anisotropic.nv = nv;
}

// Samples BRDF, generating outgoing direction(Wo)
RT_CALLABLE_PROGRAM float3 BRDF_Sample(const BRDFParameters &surface,
                                       const float3 &P,   // next ray origin
                                       const float3 &Wo,  // prev ray direction
                                       const float3 &Ns,  // shading normal
                                       uint &seed) {
  // Get material params from input variable
  float nu = surface.anisotropic.nu;
  float nv = surface.anisotropic.nv;

  float3 Wi;

  // create basis
  float3 N = normalize(Ns);
  float3 T = normalize(cross(N, make_float3(0.f, 1.f, 0.f)));
  float3 B = cross(T, N);

  // random variables
  float2 random = make_float2(rnd(seed), rnd(seed));

  if (random.x < 0.5) {
    // sample diffuse term
    random.x = fminf(2 * random.x, 1.f - 1e-6f);

    // Cosine-sample the hemisphere
    cosine_sample_hemisphere(random.x, random.y, Wi);

    Onb uvw(N);
    uvw.inverse_transform(Wi);

  } else {
    // sample specular term
    random.x = fminf(2 * (random.x - 0.5f), 1.f - 1e-6f);

    float phi, cos_theta;
    if (nu == nv) {
      // sample isotropic
      phi = 2.f * PI_F * random.x;
      cos_theta = powf(random.y, 1.f / (nu + 1.f));
    } else {
      // sample anisotropic
      if (random.x < 0.25f) {  // 1st Quadrant
        float remappedRX = 4.f * random.x;
        Sample_Quadrant(nu, nv, remappedRX, random.y, phi, cos_theta);
      } else if (random.x < 0.5f) {  // 2nd Quadrant
        float remappedRX = 4.f * (0.5f - random.x);
        Sample_Quadrant(nu, nv, remappedRX, random.y, phi, cos_theta);
        phi = PI_F - phi;
      } else if (random.x < 0.75f) {  // 3rd Quadrant
        float remappedRX = 4.f * (random.x - 0.5f);
        Sample_Quadrant(nu, nv, remappedRX, random.y, phi, cos_theta);
        phi = PI_F + phi;
      } else {  // 4th Quadrant
        float remappedRX = 4.f * (1.f - random.x);
        Sample_Quadrant(nu, nv, remappedRX, random.y, phi, cos_theta);
        phi = 2.f * PI_F - phi;
      }
    }

    float sin_theta = sqrtf(fmaxf(0.f, 1.f - cos_theta * cos_theta));

    // get half vector
    float3 H = normalize(Spherical_Vector(sin_theta, cos_theta, phi));
    H = H.x * B + H.y * N + H.z * T;

    float HdotI = dot(H, Wo);
    if (HdotI < 0.f) H = -H;

    Wi = normalize(-Wo + 2.f * dot(Wo, H) * H);  // reflect(Wo, H)
  }

  return Wi;
}

// Gets BRDF PDF value
RT_CALLABLE_PROGRAM float BRDF_PDF(const BRDFParameters &surface,
                                   const float3 &P,    // next ray origin
                                   const float3 &Wo,   // prev ray direction
                                   const float3 &Wi,   // next ray direction
                                   const float3 &N) {  // shading normal
  return 1.f;
}

// Evaluates BRDF, returning its reflectance
RT_CALLABLE_PROGRAM float3
BRDF_Evaluate(const BRDFParameters &surface,
              const float3 &P,     // next ray origin
              const float3 &Wo,    // prev ray direction
              const float3 &Wi,    // next ray direction
              const float3 &Ns) {  // shading normal
  // Get material params from input variable
  float3 Rd = surface.anisotropic.diffuse_color;
  float3 Rs = surface.anisotropic.specular_color;
  float nu = surface.anisotropic.nu;
  float nv = surface.anisotropic.nv;

  // create basis
  float3 Up = make_float3(0.f, 1.f, 0.f);
  float3 N = normalize(Ns);
  float3 T = normalize(cross(N, Up));
  float3 B = cross(T, N);

  float NdotI = abs(dot(Up, Wi)), NdotO = abs(dot(Up, Wo));

  // diffuse component
  float3 diffuse_component = Rd * (28.f / (23.f * PI_F));
  diffuse_component *= (make_float3(1.f) - Rs);
  diffuse_component *= (1.f - powf(1.f - NdotI * 0.5f, 5.f));
  diffuse_component *= (1.f - powf(1.f - NdotO * 0.5f, 5.f));

  // PDF Diffuse component
  float diffuse_PDF = dot(Wi, N) / PI_F;
  if (diffuse_PDF < 0.f) return make_float3(0.f);

  // half vector = (v1 + v2) / |v1 + v2|
  float3 H = Wo + Wi;
  if (isNull(H)) return make_float3(0.f);
  H = normalize(H);
  float HdotI = abs(dot(H, Wi));  // origin or direction here
  float HdotN = abs(dot(H, N)), HdotT = abs(dot(H, T)), HdotB = abs(dot(H, B));

  float norm, lobe;
  if (nu == nv) {
    norm = (nu + 1.f) / (8.f * PI_F);
    lobe = powf(HdotN, nu);
  } else {
    norm = sqrtf((nu + 1.f) * (nv + 1.f)) / (8.f * PI_F);

    if (HdotN < 1.f)
      lobe = powf(HdotN, (nu * HdotT * HdotT + nv * HdotB * HdotB) /
                             (1.f - HdotN * HdotN));
    else
      lobe = 1.f;
  }

  // specular component
  float3 specular_component = schlick(Rs, HdotI);  // fresnel reflectance term
  specular_component *= norm;
  specular_component *= lobe / (HdotI * fmaxf(NdotI, NdotO));

  float specular_PDF = HdotI / (norm * lobe);

  return (diffuse_component / diffuse_PDF) +
         (specular_component / specular_PDF);
}