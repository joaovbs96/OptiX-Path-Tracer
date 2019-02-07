#pragma once

// Common defines and includes

// OptiX includes:
#define RT_USE_TEMPLATED_RTCALLABLEPROGRAM 1
#include <optix.h>
#include <optix_world.h>
#include <optix_math.h>

using namespace optix;

#ifndef PI_F
#define PI_F 3.141592654f
#endif

#ifndef PI_D
#define PI_D 3.14159265358979323846264338327950288
#endif