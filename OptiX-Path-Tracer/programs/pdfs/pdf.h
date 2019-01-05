#pragma once

#include <optix.h>

#define RT_USE_TEMPLATED_RTCALLABLEPROGRAM 1
#include <optix_world.h>

#include "../vec.h"
#include "../onb.h"
#include "../sampling.h"

// communication between hit functions and the value programs
struct pdf_rec {
    float distance;
    vec3f normal;
};

// input structs for the PDF programs
struct pdf_in {
    __device__ pdf_in(const vec3f o, const vec3f d, const vec3f n) 
                                : origin(o), direction(d), normal(n) {
        uvw.build_from_w(normal);
    }

    const vec3f origin;
    const vec3f direction;
    const vec3f normal;
    vec3f light_direction;
    onb uvw;
};