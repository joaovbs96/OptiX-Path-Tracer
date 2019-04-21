#include "pdf.cuh"

// Boundary variables
rtDeclareVariable(float3, center, , );
rtDeclareVariable(float, radius, , );

// Boundary intersection function
RT_FUNCTION bool Intersect_Sphere(const float3 &P, const float3 &Wi,
                                  const float tmin, const float tmax) {
  const float3 oc = P - center;

  // if the ray hits the sphere, the following equation has two roots:
  // tdot(B, B) + 2tdot(B,A-C) + dot(A-C,A-C) - R = 0

  // Using Bhaskara's Formula, we have:
  const float a = dot(Wi, Wi);
  const float b = dot(oc, Wi);
  const float c = dot(oc, oc) - radius * radius;
  const float discriminant = b * b - a * c;

  // if the discriminant is lower than zero, there's no real
  // solution and thus no hit
  if (discriminant < 0.f) return false;

  // if the first root was a hit,
  float temp = (-b - sqrtf(discriminant)) / a;
  if (temp < tmax && temp > tmin) return true;

  // if the second root was a hit,
  temp = (-b + sqrtf(discriminant)) / a;
  if (temp < tmax && temp > tmin) return true;

  return false;
}

// Value program
RT_CALLABLE_PROGRAM float PDF(const float3 &P,    // origin of next ray
                              const float3 &Wo,   // direction of current ray
                              const float3 &Wi,   // direction of next ray
                              const float3 &N) {  // geometric normal

  if (Intersect_Sphere(P, Wi, 0.001f, FLT_MAX)) {
    float distance_squared = squared_length(center - P);
    float cos_theta_max = sqrtf(1.f - radius * radius / distance_squared);
    float solid_angle = 2.f * PI_F * (1.f - cos_theta_max);

    return 1.f / solid_angle;
  } else
    return 0.f;
}

// Sample direction relative to sphere
RT_CALLABLE_PROGRAM float3 Sample(const float3 &P,   // next ray origin
                                  const float3 &Wo,  // previous ray direction
                                  const float3 &N,   // geometric normal
                                  uint &seed) {
  float r1 = rnd(seed);
  float r2 = rnd(seed);

  float distance_squared = squared_length(center - P);
  float z = 1.f + r2 * (sqrtf(1.f - radius * radius / distance_squared) - 1.f);

  float phi = 2.f * PI_F * r1;

  float x = cosf(phi) * sqrtf(1.f - z * z);
  float y = sinf(phi) * sqrtf(1.f - z * z);

  float3 Wi = make_float3(x, y, z);

  Onb uvw(normalize(Wi));
  uvw.inverse_transform(Wi);

  return Wi;
}