#include "pdf.h"

// Boundary variables
rtDeclareVariable(float, a0, , );
rtDeclareVariable(float, a1, , );
rtDeclareVariable(float, b0, , );
rtDeclareVariable(float, b1, , );
rtDeclareVariable(float, k, , );

// Boundary functions
inline __device__ bool hit_x(pdf_in &in, const float tmin, const float tmax,
                             pdf_rec &rec) {
  float t = (k - in.origin.x) / in.scattered_direction.x;

  float a = in.origin.y + t * in.scattered_direction.y;
  float b = in.origin.z + t * in.scattered_direction.z;
  if (a < a0 || a > a1 || b < b0 || b > b1) return false;

  if (t < tmax && t > tmin) {
    rec.normal = make_float3(1.f, 0.f, 0.f);
    rec.distance = t;
    return true;
  }

  return false;
}

inline __device__ bool hit_y(pdf_in &in, const float tmin, const float tmax,
                             pdf_rec &rec) {
  float t = (k - in.origin.y) / in.scattered_direction.y;

  float a = in.origin.x + t * in.scattered_direction.x;
  float b = in.origin.z + t * in.scattered_direction.z;
  if (a < a0 || a > a1 || b < b0 || b > b1) return false;

  if (t < tmax && t > tmin) {
    rec.normal = make_float3(0.f, 1.f, 0.f);
    rec.distance = t;
    return true;
  }

  return false;
}

inline __device__ bool hit_z(pdf_in &in, const float tmin, const float tmax,
                             pdf_rec &rec) {
  float t = (k - in.origin.z) / in.scattered_direction.z;

  float a = in.origin.x + t * in.scattered_direction.x;
  float b = in.origin.y + t * in.scattered_direction.y;
  if (a < a0 || a > a1 || b < b0 || b > b1) return false;

  if (t < tmax && t > tmin) {
    rec.normal = make_float3(0.f, 0.f, 1.f);
    rec.distance = t;
    return true;
  }

  return false;
}

// Value Programs
RT_CALLABLE_PROGRAM float rect_x_value(pdf_in &in) {
  pdf_rec rec;

  if (hit_x(in, 0.001f, FLT_MAX, rec)) {
    float area = (a1 - a0) * (b1 - b0);
    float distance_squared =
        rec.distance * rec.distance * squared_length(in.scattered_direction);
    float cosine = fabs(dot(in.scattered_direction, rec.normal) /
                        length(in.scattered_direction));
    return distance_squared / (cosine * area);
  } else
    return 0.f;
}

RT_CALLABLE_PROGRAM float rect_y_value(pdf_in &in) {
  pdf_rec rec;

  if (hit_y(in, 0.001f, FLT_MAX, rec)) {
    float area = (a1 - a0) * (b1 - b0);
    float distance_squared =
        rec.distance * rec.distance * squared_length(in.scattered_direction);
    float cosine = fabs(dot(in.scattered_direction, rec.normal) /
                        length(in.scattered_direction));
    return distance_squared / (cosine * area);
  } else
    return 0.f;
}

RT_CALLABLE_PROGRAM float rect_z_value(pdf_in &in) {
  pdf_rec rec;

  if (hit_z(in, 0.001f, FLT_MAX, rec)) {
    float area = (a1 - a0) * (b1 - b0);
    float distance_squared =
        rec.distance * rec.distance * squared_length(in.scattered_direction);
    float cosine = fabs(dot(in.scattered_direction, rec.normal) /
                        length(in.scattered_direction));
    return distance_squared / (cosine * area);
  } else
    return 0.f;
}

// Generate Programs
RT_CALLABLE_PROGRAM float3 rect_x_generate(pdf_in &in, XorShift32 &rnd) {
  float3 random_point =
      make_float3(k, a0 + rnd() * (a1 - a0), b0 + rnd() * (b1 - b0));
  in.scattered_direction = random_point - in.origin;
  return in.scattered_direction;
}

RT_CALLABLE_PROGRAM float3 rect_y_generate(pdf_in &in, XorShift32 &rnd) {
  float3 random_point =
      make_float3(a0 + rnd() * (a1 - a0), k, b0 + rnd() * (b1 - b0));
  in.scattered_direction = random_point - in.origin;
  return in.scattered_direction;
}

RT_CALLABLE_PROGRAM float3 rect_z_generate(pdf_in &in, XorShift32 &rnd) {
  float3 random_point =
      make_float3(a0 + rnd() * (a1 - a0), b0 + rnd() * (b1 - b0), k);
  in.scattered_direction = random_point - in.origin;
  return in.scattered_direction;
}