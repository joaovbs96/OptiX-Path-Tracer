#ifndef CAMERAH
#define CAMERAH

#define _USE_MATH_DEFINES 1
#include <math.h>
#include <cmath>

#include "../programs/vec.hpp"

struct Camera {
  Camera() {}

  Camera(const float3 &lookfrom, const float3 &lookat, const float3 &vup,
         float vfov, float aspect, float aperture, float focus_dist, float t0,
         float t1) {
    // vfov is top to bottom in degrees
    lens_radius = aperture / 2.f;

    // shutter is open between t0 and t1
    time0 = t0;
    time1 = t1;

    float theta = vfov * ((float)M_PI) / 180.f;
    float half_height = tan(theta / 2.f);
    float half_width = aspect * half_height;

    // where camera is looking from
    origin = lookfrom;

    w = unit_vector(lookfrom - lookat);
    u = unit_vector(cross(vup, w));
    v = cross(w, u);

    lower_left_corner = origin - half_width * focus_dist * u -
                        half_height * focus_dist * v - focus_dist * w;
    horizontal = 2.f * half_width * focus_dist * u;
    vertical = 2.f * half_height * focus_dist * v;
  }

  void set(Context &g_context) {
    g_context["camera_lower_left_corner"]->set3fv(&lower_left_corner.x);
    g_context["camera_horizontal"]->set3fv(&horizontal.x);
    g_context["camera_vertical"]->set3fv(&vertical.x);
    g_context["camera_origin"]->set3fv(&origin.x);
    g_context["camera_u"]->set3fv(&u.x);
    g_context["camera_v"]->set3fv(&v.x);
    g_context["camera_w"]->set3fv(&w.x);
    g_context["time0"]->setFloat(time0);
    g_context["time1"]->setFloat(time1);
    g_context["camera_lens_radius"]->setFloat(lens_radius);
  }

  float3 origin;
  float3 lower_left_corner;
  float3 horizontal;
  float3 vertical;
  float3 u, v, w;
  float time0, time1;
  float lens_radius;
};

#endif