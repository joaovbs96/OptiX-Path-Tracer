#include "pdf.h"

// Boundary variables
rtDeclareVariable(float3, center, , );
rtDeclareVariable(float,  radius, , );

// Boundary intersection function
inline __device__ bool hit_boundary(pdf_in &in, const float tmin, const float tmax, pdf_rec &rec) {
    const vec3f oc = in.origin - vec3f(center);
    
    // if the ray hits the sphere, the following equation has two roots:
    // tdot(B, B) + 2tdot(B,A-C) + dot(A-C,A-C) - R = 0
    
    // Using Bhaskara's Formula, we have:
    const float  a = dot(in.scattered_direction, in.scattered_direction);
    const float  b = dot(oc, in.scattered_direction);
    const float  c = dot(oc, oc) - radius * radius;
    const float  discriminant = b * b - a * c;
    
    // if the discriminant is lower than zero, there's no real 
    // solution and thus no hit
    if (discriminant < 0.f) 
        return false;
    
    // first root of the sphere equation:
    float temp = (-b - sqrtf(discriminant)) / a;
    
    // for a sphere, its normal is in (hitpoint - center)
    
    // if the first root was a hit,
    if (temp < tmax && temp > tmin) {
        rec.distance = temp;
        rec.normal = ((in.origin + temp * in.scattered_direction) - center) / radius;
        return true;
    }
    
    // if the second root was a hit,
    temp = (-b + sqrtf(discriminant)) / a;
    if (temp < tmax && temp > tmin) {
        rec.distance = temp;
        rec.normal = ((in.origin + temp * in.scattered_direction) - center) / radius;
        return true;
    }
    
    return false;
}

// Value program
RT_CALLABLE_PROGRAM float sphere_value(pdf_in &in) {
    pdf_rec rec;

    if(hit_boundary(in, 0.001f, FLT_MAX, rec)) {
        float cos_theta_max = sqrtf(1.f - radius * radius / vec3f(center - in.origin).squared_length());
        float solid_angle = 2.f * CUDART_PI_F * (1.f - cos_theta_max);
        return 1.f / solid_angle;
    }
    else
        return 0.f;
}

// Utility function: generate random directions towards the sphere
inline __device__ float3 random_to_sphere(float distance_squared, DRand48 &rnd) {
    float r1 = rnd();
    float r2 = rnd();
    
    float z = 1.f + r2 * (sqrtf(1.f - radius * radius / distance_squared) - 1.f);
    
    float phi = 2.f * CUDART_PI_F * r1;

    float x = cosf(phi) * sqrtf(1.f - z * z);
    float y = sinf(phi) * sqrtf(1.f - z * z);

    return make_float3(x, y, z);
}

// Generate program: generate directions relative to the sphere
RT_CALLABLE_PROGRAM float3 sphere_generate(pdf_in &in, DRand48 &rnd) {
    vec3f direction(center - in.origin);
    float distance_squared = direction.squared_length();
    
    in.uvw.build_from_w(direction);

    vec3f temp(random_to_sphere(distance_squared, rnd));
    in.scattered_direction = in.uvw.local(temp);

    return in.scattered_direction.as_float3();
}