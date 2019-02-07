
#include "vec.h"

// TODO: there's a built in optix::onb, use it rather than our own

struct onb {
    __device__ onb() {}

    __device__ float3 local(const float a, const float b, const float c) {
        return a * u + b * v + c * w;
    }

    __device__ float3 local(const float3 &a) {
        return a.x * u + a.y * v + a.z * w;
    }

    __device__ void build_from_w(const float3& n) {
        w = unit_vector(n);
        
        float3 a;
        if(fabsf(w.x) > 0.9f)
            a = make_float3(0.f, 1.f, 0.f);
        else
            a = make_float3(1.f, 0.f, 0.f);
        
        v = unit_vector(cross(w, a));
        u = cross(w, v);
    }

    float3 u, v, w;
};