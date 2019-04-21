#include "pdf.cuh"

// Boundary variables
rtDeclareVariable(float, a0, , );
rtDeclareVariable(float, a1, , );
rtDeclareVariable(float, b0, , );
rtDeclareVariable(float, b1, , );
rtDeclareVariable(float, k, , );

// Intersect X-axis aligned rectangle
RT_FUNCTION bool Intersect_X(const float3 &P, const float3 &Wi,
                             const float tmin, const float tmax, float3 &N,
                             float &t) {
  t = (k - P.x) / Wi.x;

  float a = P.y + t * Wi.y;
  float b = P.z + t * Wi.z;
  if (a < a0 || a > a1 || b < b0 || b > b1) return false;

  if (t < tmax && t > tmin) {
    N = make_float3(1.f, 0.f, 0.f);
    return true;
  }

  return false;
}

// Intersect Y-axis aligned rectangle
RT_FUNCTION bool Intersect_Y(const float3 &P, const float3 &Wi,
                             const float tmin, const float tmax, float3 &N,
                             float &t) {
  t = (k - P.y) / Wi.y;

  float a = P.x + t * Wi.x;
  float b = P.z + t * Wi.z;
  if (a < a0 || a > a1 || b < b0 || b > b1) return false;

  if (t < tmax && t > tmin) {
    N = make_float3(0.f, 1.f, 0.f);
    return true;
  }

  return false;
}

// Intersect Z-axis aligned rectangle
RT_FUNCTION bool Intersect_Z(const float3 &P, const float3 &Wi,
                             const float tmin, const float tmax, float3 &N,
                             float &t) {
  t = (k - P.z) / Wi.z;

  float a = P.x + t * Wi.x;
  float b = P.y + t * Wi.y;
  if (a < a0 || a > a1 || b < b0 || b > b1) return false;

  if (t < tmax && t > tmin) {
    N = make_float3(0.f, 0.f, 1.f);
    return true;
  }

  return false;
}

// Calculate X-axis aligned rectangle PDF
RT_CALLABLE_PROGRAM float PDF_X(const float3 &P,    // origin of next ray
                                const float3 &Wo,   // direction of current ray
                                const float3 &Wi,   // direction of next ray
                                const float3 &N) {  // geometric normal
  float t;
  float3 rectNormal;

  if (Intersect_X(P, Wi, 0.001f, FLT_MAX, rectNormal, t)) {
    float distance_squared = t * t * squared_length(Wi);
    float cosine = fabs(dot(Wi, rectNormal) / length(Wi));
    float area = (a1 - a0) * (b1 - b0);
    return distance_squared / (cosine * area);
  } else
    return 0.f;
}

// Calculate Y-axis aligned rectangle PDF
RT_CALLABLE_PROGRAM float PDF_Y(const float3 &P,    // origin of next ray
                                const float3 &Wo,   // direction of current ray
                                const float3 &Wi,   // direction of next ray
                                const float3 &N) {  // geometric normal
  float t;
  float3 rectNormal;

  if (Intersect_Y(P, Wi, 0.001f, FLT_MAX, rectNormal, t)) {
    float distance_squared = t * t * squared_length(Wi);
    float cosine = fabs(dot(Wi, rectNormal) / length(Wi));
    float area = (a1 - a0) * (b1 - b0);
    return distance_squared / (cosine * area);
  } else
    return 0.f;
}

// Calculate Z-axis aligned rectangle PDF
RT_CALLABLE_PROGRAM float PDF_Z(const float3 &P,    // origin of next ray
                                const float3 &Wo,   // direction of current ray
                                const float3 &Wi,   // direction of next ray
                                const float3 &N) {  // geometric normal
  float t;
  float3 rectNormal;

  if (Intersect_Z(P, Wi, 0.001f, FLT_MAX, rectNormal, t)) {
    float distance_squared = t * t * squared_length(Wi);
    float cosine = fabs(dot(Wi, rectNormal) / length(Wi));
    float area = (a1 - a0) * (b1 - b0);
    return distance_squared / (cosine * area);
  } else
    return 0.f;
}

// Sample X-axis aligned rectangle
RT_CALLABLE_PROGRAM float3 Sample_X(const float3 &P,   // next ray origin
                                    const float3 &Wo,  // previous ray direction
                                    const float3 &N,   // geometric normal
                                    uint &seed) {
  float3 random_point = make_float3(k,                            // X
                                    a0 + rnd(seed) * (a1 - a0),   // Y
                                    b0 + rnd(seed) * (b1 - b0));  // Z
  return random_point - P;
}

// Sample Y-axis aligned rectangle
RT_CALLABLE_PROGRAM float3 Sample_Y(const float3 &P,   // next ray origin
                                    const float3 &Wo,  // previous ray direction
                                    const float3 &N,   // geometric normal
                                    uint &seed) {
  float3 random_point = make_float3(a0 + rnd(seed) * (a1 - a0),   // X
                                    k,                            // Y
                                    b0 + rnd(seed) * (b1 - b0));  // Z
  return random_point - P;
}

// Sample Z-axis aligned rectangle
RT_CALLABLE_PROGRAM float3 Sample_Z(const float3 &P,   // next ray origin
                                    const float3 &Wo,  // previous ray direction
                                    const float3 &N,   // geometric normal
                                    uint &seed) {
  float3 random_point = make_float3(a0 + rnd(seed) * (a1 - a0),  // X
                                    b0 + rnd(seed) * (b1 - b0),  // Y
                                    k);                          // Z
  return random_point - P;
}