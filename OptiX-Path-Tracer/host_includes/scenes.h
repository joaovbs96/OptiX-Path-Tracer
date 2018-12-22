#ifndef SCENESH
#define SCENESH

#include <optix.h>
#include <optixu/optixpp.h>

#include "../programs/vec.h"
#include "camera.h"
#include "materials.h"
#include "transforms.h"
#include "hitables.h"
#include "textures.h"

optix::Group InOneWeekend(optix::Context &g_context, Camera &camera, int Nx, int Ny) { 
  optix::Group group = g_context->createGroup();
  group->setAcceleration(g_context->createAcceleration("Trbvh"));

  Texture *checker = new Checker_Texture(new Constant_Texture(vec3f(0.2f, 0.3f, 0.1f)), new Constant_Texture(vec3f(0.9f, 0.9f, 0.9f)));

  addChild(createSphere(vec3f(0.f, -1000.0f, -1.f), 1000.f, Lambertian(checker), g_context), group, g_context);

  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      float choose_mat = rnd();
      vec3f center(a + rnd(), 0.2f, b + rnd());
      if (choose_mat < 0.8f) {
        addChild(createSphere(center, 0.2f, Lambertian(new Constant_Texture(vec3f(rnd()*rnd(), rnd()*rnd(), rnd()*rnd()))), g_context), group, g_context);
      }
      else if (choose_mat < 0.95f) {
        addChild(createSphere(center, 0.2f, Metal(new Constant_Texture(vec3f(0.5f*(1.0f + rnd()), 0.5f*(1.0f + rnd()), 0.5f*(1.0f + rnd()))), 0.5f*rnd()), g_context), group, g_context);
      }
      else {
        addChild(createSphere(center, 0.2f, Dielectric(1.5f), g_context), group, g_context);
      }
    }
  }
  addChild(createSphere(vec3f(-4.f, 1.f, 0.f), 1.f, Dielectric(1.5f), g_context), group, g_context);
  addChild(createSphere(vec3f(4.f, 1.f, 0.f), 1.f, Lambertian(new Noise_Texture(0.1f)), g_context), group, g_context);
  addChild(createSphere(vec3f(0.f, 1.f, 0.f), 1.f, Metal(new Constant_Texture(vec3f(0.7f, 0.6f, 0.5f)), 0.0f), g_context), group, g_context);
  addChild(createZRect(3.f, 5.f, 1.f, 3.f, -2.f, false, Diffuse_Light(new Constant_Texture(vec3f(4.f, 4.f, 4.f))), g_context), group, g_context);
  
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

  return group;
}

optix::Group MovingSpheres(optix::Context &g_context, Camera &camera, int Nx, int Ny) { 
  optix::Group group = g_context->createGroup();
  group->setAcceleration(g_context->createAcceleration("Trbvh"));

  addChild(createSphere(vec3f(0.f, -1000.0f, -1.f), 1000.f, Lambertian(new Constant_Texture(vec3f(0.5f, 0.5f, 0.5f))), g_context), group, g_context);

  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      float choose_mat = rnd();
      vec3f center(a + rnd(), 0.2f, b + rnd());
      if (choose_mat < 0.8f) {
        addChild(createMovingSphere(center, center + vec3f(0.f, 0.5f * rnd(), 0.f), 0.f, 1.f, 0.2f, Lambertian(new Constant_Texture(vec3f(rnd()*rnd(), rnd()*rnd(), rnd()*rnd()))), g_context), group, g_context);
      }
      else if (choose_mat < 0.95f) {
        addChild(createSphere(center, 0.2f, Metal(new Constant_Texture(vec3f(0.5f*(1.0f + rnd()), 0.5f*(1.0f + rnd()), 0.5f*(1.0f + rnd()))), 0.5f*rnd()), g_context), group, g_context);
      }
      else {
        addChild(createSphere(center, 0.2f, Dielectric(1.5f), g_context), group, g_context);
      }
    }
  }
  addChild(createSphere(vec3f(0.f, 1.f, 0.f), 1.f, Dielectric(1.5f), g_context), group, g_context);
  addChild(createSphere(vec3f(-4.f, 1.f, 0.f), 1.f, Lambertian(new Constant_Texture(vec3f(0.4f, 0.2f, 0.1f))), g_context), group, g_context);
  addChild(createSphere(vec3f(4.f, 1.f, 0.f), 1.f, Metal(new Constant_Texture(vec3f(0.7f, 0.6f, 0.5f)), 0.0f), g_context), group, g_context);

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

  return group;
}

optix::Group Cornell(optix::Context &g_context, Camera &camera, int Nx, int Ny) { 
  optix::Group group = g_context->createGroup();
  group->setAcceleration(g_context->createAcceleration("Trbvh"));

  Material *red = new Lambertian(new Constant_Texture(vec3f(0.65f, 0.05f, 0.05f)));
  Material *white = new Lambertian(new Constant_Texture(vec3f(0.73f, 0.73f, 0.73f)));
  Material *green = new Lambertian(new Constant_Texture(vec3f(0.12f, 0.45f, 0.15f)));
  Material *light = new Diffuse_Light(new Constant_Texture(vec3f(7.f, 7.f, 7.f)));
  Material *black_fog = new Isotropic(new Constant_Texture(vec3f(0.f)));
  Material *white_fog = new Isotropic(new Constant_Texture(vec3f(1.f)));

  addChild(createXRect(0.f, 555.f, 0.f, 555.f, 555.f, true, *green, g_context), group, g_context); // left wall
  addChild(createXRect(0.f, 555.f, 0.f, 555.f, 0.f, false, *red, g_context), group, g_context); // right wall
  addChild(createYRect(113.f, 443.f, 127.f, 432.f, 554.f, false, *light, g_context), group, g_context); // light
  addChild(createYRect(0.f, 555.f, 0.f, 555.f, 555.f, true, *white, g_context), group, g_context); // roof
  addChild(createYRect(0.f, 555.f, 0.f, 555.f, 0.f, false, *white, g_context), group, g_context); // ground
  addChild(createZRect(0.f, 555.f, 0.f, 555.f, 555.f, true, *white, g_context), group, g_context); // back walls
  
  /*// big box
  addChild(translate(rotateY(createBox(vec3f(0.f), vec3f(165.f, 330.f, 165.f), *white, g_context),
                                                                                 15.f, g_context), 
                                                             vec3f(265.f, 0.f, 295.f), g_context),
                                                                                group, g_context);

  // small box
  addChild(translate(rotateY(createBox(vec3f(0.f), vec3f(130.f, 0.f, 65.f), *white, g_context),
                                                                             -18.f, g_context), 
                                                        vec3f(165.f, 165.f, 165.f), g_context),
                                                                             group, g_context);*/

  // big box
  addChild(translate(rotateY(createVolumeBox(vec3f(0.f), vec3f(165.f, 330.f, 165.f), 0.01f, *black_fog, g_context),
                                                                                                 15.f, g_context),
                                                                             vec3f(265.f, 0.f, 295.f), g_context),
                                                                                                group, g_context);

  // small box
  addChild(translate(rotateY(createVolumeBox(vec3f(0.f), vec3f(165.f, 165.f, 165.f), 0.01f, *white_fog, g_context),
                                                                                                 -18.f, g_context), 
                                                                               vec3f(130.f, 0.f, 65.f), g_context),
                                                                                                group, g_context);
  
  // configure camera
  const vec3f lookfrom(278.f, 278.f, -800.f);
  const vec3f lookat(278.f, 278.f, 0.f);
  const vec3f up(0.f, 1.f, 0.f);
  const float fovy(40.0f);
  const float aspect(float(Nx) / float(Ny));
  const float aperture(0.f);
  const float dist(10.f);
  camera = Camera(lookfrom, lookat, up, fovy, aspect, aperture, dist, 0.0, 1.0);

  // configure background color
  g_context["light"]->setInt(false);

  return group;
}

optix::Group Final_Next_Week(optix::Context &g_context, Camera &camera, int Nx, int Ny) { 
  optix::Group group = g_context->createGroup();
  group->setAcceleration(g_context->createAcceleration("Trbvh"));

  // ground
  Material *ground = new Lambertian(new Constant_Texture(vec3f(0.48f, 0.83f, 0.53f)));
  for(int i = 0; i < 20; i++){
    for(int j = 0; j < 20; j++){
      float w = 100.f;
      float x0 = -1000 + i * w;
      float z0 = -1000 + j * w;
      float y0 = 0.f;
      float x1 = x0 + w;
      float y1 = 100 * (rnd() + 0.01f);
      float z1 = z0 + w;
      addChild(createBox(vec3f(x0, y0, z0), vec3f(x1, y1, z1), *ground, g_context), group, g_context);
    }
  }

  // light
  Material *light = new Diffuse_Light(new Constant_Texture(vec3f(7.f, 7.f, 7.f)));
  addChild(createYRect(113.f, 443.f, 127.f, 432.f, 554.f, false, *light, g_context), group, g_context);

  // brown moving sphere
  vec3f center(400.f, 400.f, 200.f);
  Material *brown = new Lambertian(new Constant_Texture(vec3f(0.7f, 0.3f, 0.1f)));
  addChild(createMovingSphere(center, center + vec3f(30.f, 0.f, 0.f), 0.f, 1.f, 50.f, *brown, g_context), group, g_context);

  // glass sphere
  addChild(createSphere(vec3f(260.f, 150.f, 45.f), 50.f, Dielectric(1.5), g_context), group, g_context);

  // metal sphere
  addChild(createSphere(vec3f(0.f, 150.f, 145.f), 50.f, Metal(new Constant_Texture(vec3f(0.8f, 0.8f, 0.9f)), 10.f), g_context), group, g_context);

  // blue sphere
  // glass sphere
  addChild(createSphere(vec3f(360.f, 150.f, 45.f), 70.f, Dielectric(1.5), g_context), group, g_context);
  // blue fog
  Material *blue_fog = new Isotropic(new Constant_Texture(vec3f(0.2f, 0.4f, 0.9f)));
  addChild(createVolumeSphere(vec3f(360.f, 150.f, 45.f), 70.f, 0.2f, *blue_fog, g_context), group, g_context);

  // fog
  Material *fog = new Isotropic(new Constant_Texture(vec3f(1.f)));
  addChild(createVolumeSphere(vec3f(0.f), 5000.f, 0.0001f, *fog, g_context), group, g_context);

  // earth
  addChild(createSphere(vec3f(400.f, 200.f, 400.f), 100.f, Lambertian(new Image_Texture("assets/map.jpg")), g_context), group, g_context);

  // Perlin sphere
  addChild(createSphere(vec3f(220.f, 280.f, 300.f), 80.f, Lambertian(new Noise_Texture(0.1f)), g_context), group, g_context);

  // group of small spheres
  Material *white = new Lambertian(new Constant_Texture(vec3f(0.73f, 0.73f, 0.73f)));
  std::vector<optix::GeometryInstance> d_list;
  for(int j = 0; j < 1000; j++) {
    d_list.push_back(createSphere(vec3f(165 * rnd(), 165 * rnd(), 165 * rnd()), 10.f, *white, g_context));
  }
  optix::GeometryGroup box = g_context->createGeometryGroup();
  box->setAcceleration(g_context->createAcceleration("Trbvh"));
  box->setChildCount((int)d_list.size());
  for (int i = 0; i < d_list.size(); i++)
    box->setChild(i, d_list[i]);
  addChild(translate(rotateY(box, 15.f, g_context), vec3f(-100.f, 270.f, 395.f), g_context), group, g_context);
  
  // configure camera
  const vec3f lookfrom(478.f, 278.f, -600.f);
  const vec3f lookat(278.f, 278.f, 0.f);
  const vec3f up(0.f, 1.f, 0.f);
  const float fovy(40.0f);
  const float aspect(float(Nx) / float(Ny));
  const float aperture(0.f);
  const float dist(10.f);
  camera = Camera(lookfrom, lookat, up, fovy, aspect, aperture, dist, 0.0, 1.0);

  // configure background color
  g_context["light"]->setInt(false);

  return group;
}

#endif