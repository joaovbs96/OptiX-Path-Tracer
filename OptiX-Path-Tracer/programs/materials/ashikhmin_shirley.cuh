
#include "material.cuh"
#include "microfacets.cuh"

struct Ashikhmin_Shirley_Parameters {
  float3 diffuse_color;
  float3 specular_color;
  float nu, nv;
};

RT_FUNCTION float3 Sample(const Ashikhmin_Shirley_Parameters &surface,
                          const float3 &P,   // next ray origin
                          const float3 &Wo,  // prev ray direction
                          const float3 &Ns,  // shading normal
                          uint &seed) {
  // Get material params from input variable
  float nu = surface.nu;
  float nv = surface.nv;

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

RT_FUNCTION float3 Evaluate(const Ashikhmin_Shirley_Parameters &surface,
                            const float3 &P,   // next ray origin
                            const float3 &Wo,  // prev ray direction
                            const float3 &Wi,  // next ray direction
                            const float3 &Ns,
                            float &pdf) {  // shading normal
  // Get material params from input variable
  float3 Rd = surface.diffuse_color;
  float3 Rs = surface.specular_color;
  float nu = surface.nu;
  float nv = surface.nv;

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

  // for this material, this is only used in the direct light sampling function
  pdf = (diffuse_PDF + specular_PDF) / 2.f;

  // TODO: test this
  /*diffuse_component = clamp(diffuse_component / diffuse_PDF, 0.f, 1.f);
  specular_component = clamp(specular_component / specular_PDF, 0.f, 1.f);
  return clamp(diffuse_component + specular_component, 0.f, 1.f);*/

  return (diffuse_component / diffuse_PDF) +
         (specular_component / specular_PDF);
}