
#include "material.cuh"
#include "microfacets.cuh"

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
  prd.geometric_normal = hit_rec.geometric_normal;
  prd.shading_normal = hit_rec.shading_normal;
  prd.view_direction = hit_rec.view_direction;

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
RT_CALLABLE_PROGRAM float3 BRDF_Sample(PDFParams &pdf, uint &seed) {
  // Get material params from input variable
  float nu = pdf.matParams.anisotropic.nu;
  float nv = pdf.matParams.anisotropic.nv;

  float3 normal = pdf.geometric_normal;
  float3 hit_point = pdf.origin;
  float3 origin = pdf.view_direction;
  float3 direction;

  // random variables
  float2 random = make_float2(rnd(seed), rnd(seed));
  pdf.matParams.u = random.x;
  pdf.matParams.v = random.y;

  if (random.x < 0.5) {
    // sample diffuse term
    random.x = fminf(2 * random.x, 1.f - 1e-6f);

    // Cosine-sample the hemisphere
    cosine_sample_hemisphere(random.x, random.y, direction);

    Onb uvw(normalize(normal));
    uvw.inverse_transform(direction);

    // flip the direction if necessary
    if (!Same_Hemisphere(direction, origin)) direction *= -1;

  } else {
    // sample specular term
    random.x = fminf(2 * (random.x - 0.5f), 1.f - 1e-6f);

    // Get X & Y from normal basis
    float3 X, Y;
    if (nu == nv) {
      Make_Orthonormals(normal, X, Y);
    } else
      Make_Orthonormals_Tangent(normal, Tangent(hit_point), X, Y);

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
        float remappedRX = 4.f * random.x;
        Sample_Quadrant(nu, nv, remappedRX, random.y, phi, cos_theta);
        phi = 2.f * PI_F - phi;
      }
    }

    float sin_theta = sqrtf(fmaxf(0.f, 1.f - cos_theta * cos_theta));

    // get half vector
    float3 H = Spherical_Vector(sin_theta, cos_theta, phi);
    H = H.x * X + H.y * Y + H.z * normal;
    float HdotI = dot(H, origin);
    if (HdotI < 0.0f) H = -H;

    // reflect half-vector on incident ray
    direction = reflect(origin, H);
  }

  pdf.direction = direction;

  return pdf.direction;
}

// Gets BRDF PDF value
RT_CALLABLE_PROGRAM float BRDF_PDF(PDFParams &pdf) {
  if (!Same_Hemisphere(pdf.direction, pdf.view_direction)) return 0.0f;
  return 1.f;
  // Get material params from input variable
  float nu = pdf.matParams.anisotropic.nu;
  float nv = pdf.matParams.anisotropic.nv;

  float3 origin = pdf.view_direction;
  float3 direction = normalize(pdf.direction);

  // PDF Diffuse Term
  float diffuse_PDF = dot(direction, normalize(pdf.normal));
  if (diffuse_PDF < 0.f) diffuse_PDF = 0.f;

  // PDF Specular Term
  const float3 H = normalize(origin + direction);
  const float specular_PDF =
      GGX_PDF(H, origin, nu, nv) / (4.f * dot(origin, H));

  return (diffuse_PDF + specular_PDF) / 2.f;
}

// TODO: attempt to implement the Blender one once again
// https://developer.blender.org/diffusion/C/browse/master/src/kernel/closure/bsdf_ashikhmin_shirley.h;f403fe5d73b6eaae955df89191b452b424215a6b?as=source&blame=off

// Evaluates BRDF, returning its reflectance
RT_CALLABLE_PROGRAM float3 BRDF_Evaluate(PDFParams &pdf) {
  if (!Same_Hemisphere(pdf.direction, pdf.view_direction))
    return make_float3(0.f);

  // Get material params from input variable
  float3 Rd = pdf.matParams.anisotropic.diffuse_color;
  float3 Rs = pdf.matParams.anisotropic.specular_color;
  float nu = pdf.matParams.anisotropic.nu;
  float nv = pdf.matParams.anisotropic.nv;

  float3 origin = pdf.view_direction;
  float3 direction = normalize(pdf.direction);
  float3 normal = normalize(pdf.geometric_normal);

  float nk1 = AbsCosTheta(direction);
  float nk2 = AbsCosTheta(origin);
  if (nk1 == 0 || nk2 == 0) return make_float3(0.f);

  // half vector = (v1 + v2) / |v1 + v2|
  float3 H = origin + direction;
  if (isNull(H)) return make_float3(0.f);

  // diffuse component
  float3 diffuse_component = Rd * (28.f / (23.f * PI_F));
  diffuse_component *= (make_float3(1.f) - Rs);
  diffuse_component *= (1.f - powf(1.f - nk1 * 0.5f, 5.f));
  diffuse_component *= (1.f - powf(1.f - nk2 * 0.5f, 5.f));

  // PDF Diffuse component
  float diffuse_PDF = dot(direction, normalize(pdf.normal));
  if (diffuse_PDF < 0.f) diffuse_PDF = 0.f;

  H = normalize(H);
  float HdotI = dot(H, origin);  // origin or direction here
  float HdotN = dot(H, normal);

  // specular component
  float3 F = schlick(Rs, HdotI);  // fresnel reflectance term
  float pump = 1.f / fmaxf(1e-6f, (HdotI * fmaxf(nk1, nk2)));

  float out, specular_PDF;
  if (nu == nv) {
    float e = nu;
    float lobe = powf(HdotN, e);
    float norm = (nu + 1.f) / (8.f * PI_F);

    out = nk1 * norm * lobe * pump;
    specular_PDF = norm * lobe / HdotI;
  } else {
    // Get X & Y from normal basis
    float3 X, Y;
    if (nu == nv)
      Make_Orthonormals(normal, X, Y);
    else
      Make_Orthonormals_Tangent(normal, Tangent(pdf.origin), X, Y);

    float HdotX = dot(H, X), HdotY = dot(H, Y);
    float lobe;
    if (HdotN < 1.f) {
      float e =
          (nu * HdotX * HdotX + nv * HdotY * HdotY) / (1.f - HdotN * HdotN);
      lobe = powf(HdotN, e);
    } else {
      lobe = 1.f;
    }
    float norm = sqrtf((nu + 1.f) * (nv + 1.f)) / (8.f * PI_F);

    out = nk1 * norm * lobe * pump;
    specular_PDF = norm * lobe / HdotI;
  }

  float3 specular_component = F * out * Rs;

  return (diffuse_component + specular_component) /
         (0.5f * (diffuse_PDF + specular_PDF));
}