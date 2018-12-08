#ifndef SCENESH
#define SCENESH

#include <optix.h>
#include <optixu/optixpp.h>

#include "../programs/vec.h"
#include "camera.h"
#include "materials.h"
#include "hitables.h"
#include "textures.h"

optix::GeometryGroup InOneWeekend(optix::Context &g_context, Camera &camera, int Nx, int Ny) { 
  // first, create all geometry instances (GIs), and, for now,
  // store them in a std::vector. For ease of reference, I'll
  // stick wit the 'd_list' and 'd_world' names used in the
  // reference C++ and CUDA codes.
  std::vector<optix::GeometryInstance> d_list;

  Texture *checker = new Checker_Texture(new Constant_Texture(vec3f(0.2, 0.3, 0.1)), new Constant_Texture(vec3f(0.9, 0.9, 0.9)));

  d_list.push_back(createSphere(vec3f(0.f, -1000.0f, -1.f), 1000.f, Lambertian(checker), g_context));

  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      float choose_mat = rnd();
      vec3f center(a + rnd(), 0.2f, b + rnd());
      if (choose_mat < 0.8f) {
        d_list.push_back(createSphere(center, 0.2f, Lambertian(new Constant_Texture(vec3f(rnd()*rnd(), rnd()*rnd(), rnd()*rnd()))), g_context));
      }
      else if (choose_mat < 0.95f) {
        d_list.push_back(createSphere(center, 0.2f, Metal(new Constant_Texture(vec3f(0.5f*(1.0f + rnd()), 0.5f*(1.0f + rnd()), 0.5f*(1.0f + rnd()))), 0.5f*rnd()), g_context));
      }
      else {
        d_list.push_back(createSphere(center, 0.2f, Dielectric(1.5f), g_context));
      }
    }
  }
  d_list.push_back(createSphere(vec3f(-4.f, 1.f, 0.f), 1.f, Dielectric(1.5f), g_context));
  d_list.push_back(createSphere(vec3f(4.f, 1.f, 0.f), 1.f, Lambertian(new Image_Texture("assets/map.jpg")), g_context));
  d_list.push_back(createSphere(vec3f(0.f, 1.f, 0.f), 1.f, Metal(new Constant_Texture(vec3f(0.7f, 0.6f, 0.5f)), 0.0f), g_context));
  d_list.push_back(createZRect(3.f, 5.f, 1.f, 3.f, -2.f, false, Diffuse_Light(new Constant_Texture(vec3f(4.f, 4.f, 4.f))), g_context));
  //createBox(vec3f(3.f, 5.f, -2.f), vec3f(1.f, 5.f, 2.f), Lambertian(new Constant_Texture(vec3f(rnd()*rnd(), rnd()*rnd(), rnd()*rnd()))), g_context, d_list);
  
  // now, create the optix world that contains all these GIs
  optix::GeometryGroup d_world = g_context->createGeometryGroup();
  d_world->setAcceleration(g_context->createAcceleration("Bvh"));
  d_world->setChildCount((int)d_list.size());
  for (int i = 0; i < d_list.size(); i++)
    d_world->setChild(i, d_list[i]);
  
  // configure camera
  const vec3f lookfrom(13, 2, 3);
  const vec3f lookat(0, 0, 0);
  const vec3f up(0, 1, 0);
  const float fovy(20.0);
  const float aspect(float(Nx) / float(Ny));
  const float aperture(0.1f);
  const float dist(10.f);
  camera = Camera(lookfrom, lookat, up, fovy, aspect, aperture, dist, 0.0, 1.0);

  // configure background color
  g_context["light"]->setInt(false);

  // that all we have to do, the rest is up to optix
  return d_world;
}

optix::GeometryGroup MovingSpheres(optix::Context &g_context, Camera &camera, int Nx, int Ny) { 
  // first, create all geometry instances (GIs), and, for now,
  // store them in a std::vector. For ease of reference, I'll
  // stick wit the 'd_list' and 'd_world' names used in the
  // reference C++ and CUDA codes.
  std::vector<optix::GeometryInstance> d_list;

  d_list.push_back(createSphere(vec3f(0.f, -1000.0f, -1.f), 1000.f, Lambertian(new Constant_Texture(vec3f(0.5f, 0.5f, 0.5f))), g_context));

  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      float choose_mat = rnd();
      vec3f center(a + rnd(), 0.2f, b + rnd());
      if (choose_mat < 0.8f) {
        d_list.push_back(createMovingSphere(center, center + vec3f(0.f, 0.5f * rnd(), 0.f), 0.f, 1.f, 0.2f, Lambertian(new Constant_Texture(vec3f(rnd()*rnd(), rnd()*rnd(), rnd()*rnd()))), g_context));
      }
      else if (choose_mat < 0.95f) {
        d_list.push_back(createSphere(center, 0.2f, Metal(new Constant_Texture(vec3f(0.5f*(1.0f + rnd()), 0.5f*(1.0f + rnd()), 0.5f*(1.0f + rnd()))), 0.5f*rnd()), g_context));
      }
      else {
        d_list.push_back(createSphere(center, 0.2f, Dielectric(1.5f), g_context));
      }
    }
  }
  d_list.push_back(createSphere(vec3f(0.f, 1.f, 0.f), 1.f, Dielectric(1.5f), g_context));
  d_list.push_back(createSphere(vec3f(-4.f, 1.f, 0.f), 1.f, Lambertian(new Constant_Texture(vec3f(0.4f, 0.2f, 0.1f))), g_context));
  d_list.push_back(createSphere(vec3f(4.f, 1.f, 0.f), 1.f, Metal(new Constant_Texture(vec3f(0.7f, 0.6f, 0.5f)), 0.0f), g_context));
  createBox(vec3f(3.f, 5.f, -2.f), vec3f(1.f, 5.f, 2.f), Lambertian(new Constant_Texture(vec3f(rnd()*rnd()*rnd(), rnd()*rnd()*rnd(), rnd()*rnd()*rnd()))), g_context, d_list);
  
  // now, create the optix world that contains all these GIs
  optix::GeometryGroup d_world = g_context->createGeometryGroup();
  d_world->setAcceleration(g_context->createAcceleration("Bvh"));
  d_world->setChildCount((int)d_list.size());
  for (int i = 0; i < d_list.size(); i++)
    d_world->setChild(i, d_list[i]);

  // configure camera
  const vec3f lookfrom(13, 2, 3);
  const vec3f lookat(0, 0, 0);
  const vec3f up(0, 1, 0);
  const float fovy(20.0);
  const float aspect(float(Nx) / float(Ny));
  const float aperture(0.1f);
  const float dist(10.f);
  camera = Camera(lookfrom, lookat, up, fovy, aspect, aperture, dist, 0.0, 1.0);

  // configure background color
  g_context["light"]->setInt(true);

  return d_world;
}

#endif