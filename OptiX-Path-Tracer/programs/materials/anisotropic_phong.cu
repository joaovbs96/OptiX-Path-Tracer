
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
  prd.matType = Anisotropic_Material;
  prd.isSpecular = false;
  prd.scatterEvent = rayGotBounced;

  // Get hit params
  prd.origin = hit_rec.p;
  prd.geometric_normal = hit_rec.geometric_normal;
  prd.shading_normal = hit_rec.shading_normal;

  // Get material colors
  int index = hit_rec.index;
  float3 diffuse = diffuse_color(hit_rec.u, hit_rec.v, hit_rec.p, index);
  float3 specular = specular_color(hit_rec.u, hit_rec.v, hit_rec.p, index);

  // Assign material parameters to PRD, to be used in the sampling programs
  prd.matParams.anisotropic.diffuse_color = diffuse;
  prd.matParams.anisotropic.specular_color = specular;
  prd.matParams.anisotropic.nu = Beckmann_Roughness(nu);
  prd.matParams.anisotropic.nv = Beckmann_Roughness(nv);
}

// Samples BRDF, generating outgoing direction(Wo)
RT_CALLABLE_PROGRAM float3 BRDF_Sample(PDFParams &pdf, uint &seed) {
  // Get material params from input variable
  float nu = pdf.matParams.anisotropic.nu;
  float nv = pdf.matParams.anisotropic.nv;

  // random variables
  float2 random = make_float2(rnd(seed), rnd(seed));
  pdf.matParams.u = random.x;
  pdf.matParams.v = random.y;

  float3 direction;
  if (random.x < 0.5) {
    // sample diffuse term
    random.x = fminf(2 * random.x, 1.f - 1e-6f);

    // Cosine-sample the hemisphere, flipping the direction if necessary
    cosine_sample_hemisphere(random.x, random.y, direction);

    Onb uvw(pdf.normal);
    uvw.inverse_transform(direction);

  } else {
    // sample specular term
    random.x = fminf(2 * (random.x - 0.5f), 1.f - 1e-6f);

    // get x,y basis on the surface for anisotropy
    float3 X, Y;
    if (nu == nv)
      Make_Orthonormals(pdf.normal, X, Y);
    else {
      float3 tangent = normalize(Tangent(pdf.origin));
      Make_Orthonormals_Tangent(pdf.normal, tangent, X, Y);
    }

    // sample spherical coords for h in tangent space
    float phi, cos_theta;
    if (nu == nv) {
      // isotropic sampling
      phi = 2.f * PI_F * random.x;
      cos_theta = powf(random.y, 1.f / (nu + 1.f));
    } else {
      // anisotropic sampling
      if (random.x < 0.25f) {  // first quadrant
        float remapped_randx = 4.f * random.x;
        Sample_Quadrant(nu, nv, remapped_randx, random.y, phi, cos_theta);
      } else if (random.x < 0.5f) {  // second quadrant
        float remapped_randx = 4.f * (0.5f - random.x);
        Sample_Quadrant(nu, nv, remapped_randx, random.y, phi, cos_theta);
        phi = PI_F - phi;
      } else if (random.x < 0.75f) {  // third quadrant
        float remapped_randx = 4.f * (random.x - 0.5f);
        Sample_Quadrant(nu, nv, remapped_randx, random.y, phi, cos_theta);
        phi = PI_F + phi;
      } else {  // fourth quadrant
        float remapped_randx = 4.f * (1.f - random.x);
        Sample_Quadrant(nu, nv, remapped_randx, random.y, phi, cos_theta);
        phi = 2.f * PI_F - phi;
      }
    }

    // get half vector in tangent space
    float sin_theta = sqrtf(fmaxf(0.f, 1.f - cos_theta * cos_theta));
    float cos_phi = cosf(phi);
    float sin_phi = sinf(phi);  // no sqrt(1-cos^2) here, it causes artifacts
    float3 H = make_float3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);

    float HdotI = dot(H, pdf.origin);
    if (HdotI < 0.0f) H = -H;

    // reflect I/origin on H to get omega_in/direction
    direction = reflect(pdf.origin, H);
  }

  pdf.direction = direction;

  return pdf.direction;
}

// TODO: what should I use for PDF?
// Gets BRDF PDF value
RT_CALLABLE_PROGRAM float BRDF_PDF(PDFParams &pdf) { return 1.f; }

// Evaluates BRDF, returning its reflectance
RT_CALLABLE_PROGRAM float3 BRDF_Evaluate(PDFParams &pdf) {
  // Get material params from input variable
  float3 Rd, Rs;
  Rd = pdf.matParams.anisotropic.diffuse_color;
  Rs = pdf.matParams.anisotropic.specular_color;
  float nu = pdf.matParams.anisotropic.nu;
  float nv = pdf.matParams.anisotropic.nv;

  float nk1 = fabsf(dot(normalize(pdf.direction), normalize(pdf.normal)));
  float nk2 = fabsf(dot(normalize(pdf.origin), normalize(pdf.normal)));

  // diffuse component
  float3 diffuse_component = Rd * (28.f / (23.f * PI_F));
  diffuse_component *= (make_float3(1.f) - Rs);
  diffuse_component *= (1.f - powf(1.f - nk1 * 0.5f, 5.f));
  diffuse_component *= (1.f - powf(1.f - nk2 * 0.5f, 5.f));

  float cosine = dot(unit_vector(pdf.direction), unit_vector(pdf.normal));
  float diffuse_PDF = cosine / PI_F;

  // half vector = (v1 + v2) / |v1 + v2|
  float3 H = normalize(pdf.direction + pdf.origin);
  if (isNull(H)) return diffuse_component / diffuse_PDF;

  float HdotI = fmaxf(fabsf(dot(H, normalize(pdf.direction))), 1e-6f);
  float HdotN = fmaxf(dot(H, normalize(pdf.normal)), 1e-6f);

  // specular component
  float3 specular_component = schlick(Rs, HdotI);  // fresnel reflectance term

  float specular_PDF;
  if (nu == nv) {  // isotropic
    // geometry term
    float norm = (nu + 1.f) / (8.f * PI_F);
    specular_component *= norm;

    // distribution term
    float lobe = powf(HdotN, nu);
    float D = (lobe / fmaxf(1e-6f, HdotI * fmaxf(nk1, nk2)));
    specular_component *= D;

    // PDF value
    specular_PDF = (norm * lobe) / (4.f * HdotI);

  } else {  // anisotropic
    // geometry term
    float norm = sqrtf((nu + 1.f) * (nv + 1.f)) / (8.f * PI_F);
    specular_component *= norm;

    float3 X, Y;
    float3 tangent = normalize(Tangent(pdf.origin));
    Make_Orthonormals_Tangent(pdf.normal, tangent, X, Y);

    float HdotX2 = dot(H, X), HdotY2 = dot(H, Y);
    HdotX2 *= HdotX2;  // hu^2
    HdotY2 *= HdotY2;  // hv^2

    // exponent of the distribution term
    float lobe;
    if (HdotN < 1.f)
      lobe = powf(HdotN, (nu * HdotX2 + nv * HdotY2) / (1.f - HdotN * HdotN));
    else
      lobe = 1.f;

    // distribution term
    float D = (lobe / fmaxf(1e-6f, HdotI * fmaxf(nk1, nk2)));
    specular_component *= D;

    // PDF value
    specular_PDF = (norm * lobe) / (4.f * HdotI);
  }

  return clamp((specular_component + diffuse_component) / specular_PDF, 0.f,
               1.f);
}