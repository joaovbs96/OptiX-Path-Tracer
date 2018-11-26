#ifndef SCENESH
#define SCENESH

#include <optix.h>
#include <optixu/optixpp.h>

#include "../programs/vec.h"
#include "materials.h"
#include "hitables.h"

#include <random>

float rnd() {
  // static std::random_device rd;  //Will be used to obtain a seed for the random number engine
  static std::mt19937 gen(0); //Standard mersenne_twister_engine seeded with rd()
  static std::uniform_real_distribution<float> dis(0.f, 1.f);
  return dis(gen);
}

optix::GeometryGroup InOneWeekend(optix::Context &g_context) { 
  // first, create all geometry instances (GIs), and, for now,
  // store them in a std::vector. For ease of reference, I'll
  // stick wit the 'd_list' and 'd_world' names used in the
  // reference C++ and CUDA codes.
  std::vector<optix::GeometryInstance> d_list;

  d_list.push_back(createSphere(vec3f(0.f, -1000.0f, -1.f), 1000.f, Lambertian(vec3f(0.5f, 0.5f, 0.5f)), g_context));

  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      float choose_mat = rnd();
      vec3f center(a + rnd(), 0.2f, b + rnd());
      if (choose_mat < 0.8f) {
        d_list.push_back(createSphere(center, 0.2f, Lambertian(vec3f(rnd()*rnd(), rnd()*rnd(), rnd()*rnd())), g_context));
      }
      else if (choose_mat < 0.95f) {
        d_list.push_back(createSphere(center, 0.2f, Metal(vec3f(0.5f*(1.0f + rnd()), 0.5f*(1.0f + rnd()), 0.5f*(1.0f + rnd())), 0.5f*rnd()), g_context));
      }
      else {
        d_list.push_back(createSphere(center, 0.2f, Dielectric(1.5f), g_context));
      }
    }
  }
  d_list.push_back(createSphere(vec3f(0.f, 1.f, 0.f), 1.f, Dielectric(1.5f), g_context));
  d_list.push_back(createSphere(vec3f(-4.f, 1.f, 0.f), 1.f, Lambertian(vec3f(0.4f, 0.2f, 0.1f)), g_context));
  d_list.push_back(createSphere(vec3f(4.f, 1.f, 0.f), 1.f, Metal(vec3f(0.7f, 0.6f, 0.5f), 0.0f), g_context));
  
  // now, create the optix world that contains all these GIs
  optix::GeometryGroup d_world = g_context->createGeometryGroup();
  d_world->setAcceleration(g_context->createAcceleration("Bvh"));
  d_world->setChildCount((int)d_list.size());
  for (int i = 0; i < d_list.size(); i++)
    d_world->setChild(i, d_list[i]);

  // that all we have to do, the rest is up to optix
  return d_world;
}

optix::GeometryGroup MovingSpheres(optix::Context &g_context) { 
  // first, create all geometry instances (GIs), and, for now,
  // store them in a std::vector. For ease of reference, I'll
  // stick wit the 'd_list' and 'd_world' names used in the
  // reference C++ and CUDA codes.
  std::vector<optix::GeometryInstance> d_list;

  d_list.push_back(createSphere(vec3f(0.f, -1000.0f, -1.f), 1000.f, Lambertian(vec3f(0.5f, 0.5f, 0.5f)), g_context));

  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      float choose_mat = rnd();
      vec3f center(a + rnd(), 0.2f, b + rnd());
      if (choose_mat < 0.8f) {
        d_list.push_back(createSphere(center, 0.2f, Lambertian(vec3f(rnd()*rnd(), rnd()*rnd(), rnd()*rnd())), g_context));
      }
      else if (choose_mat < 0.95f) {
        d_list.push_back(createSphere(center, 0.2f, Metal(vec3f(0.5f*(1.0f + rnd()), 0.5f*(1.0f + rnd()), 0.5f*(1.0f + rnd())), 0.5f*rnd()), g_context));
      }
      else {
        d_list.push_back(createSphere(center, 0.2f, Dielectric(1.5f), g_context));
      }
    }
  }
  d_list.push_back(createSphere(vec3f(0.f, 1.f, 0.f), 1.f, Dielectric(1.5f), g_context));
  d_list.push_back(createSphere(vec3f(-4.f, 1.f, 0.f), 1.f, Lambertian(vec3f(0.4f, 0.2f, 0.1f)), g_context));
  d_list.push_back(createSphere(vec3f(4.f, 1.f, 0.f), 1.f, Metal(vec3f(0.7f, 0.6f, 0.5f), 0.0f), g_context));
  
  // now, create the optix world that contains all these GIs
  optix::GeometryGroup d_world = g_context->createGeometryGroup();
  d_world->setAcceleration(g_context->createAcceleration("Bvh"));
  d_world->setChildCount((int)d_list.size());
  for (int i = 0; i < d_list.size(); i++)
    d_world->setChild(i, d_list[i]);

  // that all we have to do, the rest is up to optix
  return d_world;
}

#endif