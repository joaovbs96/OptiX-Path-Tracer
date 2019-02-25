#include "pdf.cuh"

// Boundary variables
rtDeclareVariable(float, a0, , );
rtDeclareVariable(float, a1, , );
rtDeclareVariable(float, b0, , );
rtDeclareVariable(float, b1, , );
rtDeclareVariable(float, k, , );

// Boundary functions
RT_FUNCTION bool hit_x(PDFParams &in, const float tmin, const float tmax,
                       PDFRecord &rec) {
  float t = (k - in.origin.x) / in.direction.x;

  float a = in.origin.y + t * in.direction.y;
  float b = in.origin.z + t * in.direction.z;
  if (a < a0 || a > a1 || b < b0 || b > b1) return false;

  if (t < tmax && t > tmin) {
    rec.normal = make_float3(1.f, 0.f, 0.f);
    rec.distance = t;
    return true;
  }

  return false;
}

RT_FUNCTION bool hit_y(PDFParams &in, const float tmin, const float tmax,
                       PDFRecord &rec) {
  float t = (k - in.origin.y) / in.direction.y;

  float a = in.origin.x + t * in.direction.x;
  float b = in.origin.z + t * in.direction.z;
  if (a < a0 || a > a1 || b < b0 || b > b1) return false;

  if (t < tmax && t > tmin) {
    rec.normal = make_float3(0.f, 1.f, 0.f);
    rec.distance = t;
    return true;
  }

  return false;
}

RT_FUNCTION bool hit_z(PDFParams &in, const float tmin, const float tmax,
                       PDFRecord &rec) {
  float t = (k - in.origin.z) / in.direction.z;

  float a = in.origin.x + t * in.direction.x;
  float b = in.origin.y + t * in.direction.y;
  if (a < a0 || a > a1 || b < b0 || b > b1) return false;

  if (t < tmax && t > tmin) {
    rec.normal = make_float3(0.f, 0.f, 1.f);
    rec.distance = t;
    return true;
  }

  return false;
}

// Value Programs
RT_CALLABLE_PROGRAM float rect_x_value(PDFParams &in) {
  PDFRecord rec;

  if (hit_x(in, 0.001f, FLT_MAX, rec)) {
    float area = (a1 - a0) * (b1 - b0);
    float distance_squared =
        rec.distance * rec.distance * squared_length(in.direction);
    float cosine = fabs(dot(in.direction, rec.normal) / length(in.direction));
    return distance_squared / (cosine * area);
  } else
    return 0.f;
}

RT_CALLABLE_PROGRAM float rect_y_value(PDFParams &in) {
  PDFRecord rec;

  if (hit_y(in, 0.001f, FLT_MAX, rec)) {
    float area = (a1 - a0) * (b1 - b0);
    float distance_squared =
        rec.distance * rec.distance * squared_length(in.direction);
    float cosine = fabs(dot(in.direction, rec.normal) / length(in.direction));
    return distance_squared / (cosine * area);
  } else
    return 0.f;
}

RT_CALLABLE_PROGRAM float rect_z_value(PDFParams &in) {
  PDFRecord rec;

  if (hit_z(in, 0.001f, FLT_MAX, rec)) {
    float area = (a1 - a0) * (b1 - b0);
    float distance_squared =
        rec.distance * rec.distance * squared_length(in.direction);
    float cosine = fabs(dot(in.direction, rec.normal) / length(in.direction));
    return distance_squared / (cosine * area);
  } else
    return 0.f;
}

// Generate Programs
RT_CALLABLE_PROGRAM float3 rect_x_generate(PDFParams &in, uint &seed) {
  float3 random_point =
      make_float3(k, a0 + rnd(seed) * (a1 - a0), b0 + rnd(seed) * (b1 - b0));
  in.direction = random_point - in.origin;
  return in.direction;
}

RT_CALLABLE_PROGRAM float3 rect_y_generate(PDFParams &in, uint &seed) {
  float3 random_point =
      make_float3(a0 + rnd(seed) * (a1 - a0), k, b0 + rnd(seed) * (b1 - b0));
  in.direction = random_point - in.origin;
  return in.direction;
}

RT_CALLABLE_PROGRAM float3 rect_z_generate(PDFParams &in, uint &seed) {
  float3 random_point =
      make_float3(a0 + rnd(seed) * (a1 - a0), b0 + rnd(seed) * (b1 - b0), k);
  in.direction = random_point - in.origin;
  return in.direction;
}