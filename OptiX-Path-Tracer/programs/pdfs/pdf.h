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
    __device__ pdf_in(const vec3f o, const vec3f n) : origin(o), normal(n) {}

    const vec3f origin;
    const vec3f normal;
    vec3f scattered_direction;
    onb uvw;
};