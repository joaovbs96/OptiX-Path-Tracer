#include <optix.h>
#include <optix_world.h>

#include "vec.h"

struct onb {
    __device__ onb() {}

    __device__ vec3f local(float a, float b, float c) const {
        return a * u + b * v + c * w;
    }

    __device__ vec3f local(const vec3f &a) const {
        return a.x * u + a.y * v + a.z * w;
    }

    __device__ void onb::build_from_w(const vec3f& n){
        w = unit_vector(n);
        
        vec3f a;
        if(fabsf(w.x) > 0.9)
            a = vec3f(0.f, 1.f, 0.f);
        else
            a = vec3f(1.f, 0.f, 0.f);
        
        v = unit_vector(cross(w, a));
        u = cross(w, v);
    }

    vec3f axis[3], u, v, w;
};