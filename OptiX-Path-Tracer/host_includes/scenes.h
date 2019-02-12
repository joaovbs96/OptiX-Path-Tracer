#ifndef SCENESH
#define SCENESH

#include <chrono>

#include "../programs/vec.h"
#include "camera.h"
#include "hitables.h"
#include "materials.h"
#include "pdfs.h"
#include "programs.h"
#include "textures.h"
#include "transforms.h"

void InOneWeekend(Context& g_context, int Nx, int Ny) {
  auto t0 = std::chrono::system_clock::now();

  // configure sampling
  Mixture_PDF mixture(new Cosine_PDF(),
                      new Sphere_PDF(make_float3(4.f, 1.f, 0.f), 1.f));

  // add material PDFs
  Buffer material_pdfs =
      g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 2);
  callableProgramId<int(int)>* f_data =
      static_cast<callableProgramId<int(int)>*>(material_pdfs->map());
  f_data[0] = callableProgramId<int(int)>(Lambertian_PDF(g_context)->getId());
  f_data[1] =
      callableProgramId<int(int)>(Diffuse_Light_PDF(g_context)->getId());
  material_pdfs->unmap();

  // Set the exception, ray generation and miss shader programs
  setRayGenerationProgram(g_context, mixture, material_pdfs);
  setMissProgram(g_context, SKY);
  setExceptionProgram(g_context);

  // Set acceleration structure
  Group group = g_context->createGroup();
  group->setAcceleration(g_context->createAcceleration("Trbvh"));

  // Texture *checker = new Checker_Texture(new
  // Constant_Texture(make_float3(0.2f, 0.3f, 0.1f)), new
  // Constant_Texture(make_float3(0.9f, 0.9f, 0.9f)));

  addChild(
      Sphere(make_float3(0.f, -1000.f, -1.f), 1000.f,
             Lambertian(new Constant_Texture(make_float3(0.5f))), g_context),
      group, g_context);

  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      float choose_mat = rnd();
      float3 center = make_float3(a + rnd(), 0.2f, b + rnd());
      if (choose_mat < 0.8f) {
        addChild(Sphere(center, 0.2f,
                        Lambertian(new Constant_Texture(make_float3(
                            rnd() * rnd(), rnd() * rnd(), rnd() * rnd()))),
                        g_context),
                 group, g_context);
      } else if (choose_mat < 0.95f) {
        addChild(Sphere(center, 0.2f,
                        Metal(new Constant_Texture(make_float3(
                                  0.5f * (1.f + rnd()), 0.5f * (1.f + rnd()),
                                  0.5f * (1.f + rnd()))),
                              0.5f * rnd()),
                        g_context),
                 group, g_context);
      } else {
        addChild(Sphere(center, 0.2f, Dielectric(1.5f), g_context), group,
                 g_context);
      }
    }
  }
  addChild(
      Sphere(make_float3(4.f, 1.f, 0.f), 1.f,
             Metal(new Constant_Texture(make_float3(0.7f, 0.6f, 0.5f)), 0.f),
             g_context),
      group, g_context);
  addChild(
      Sphere(make_float3(0.f, 1.f, 0.5f), 1.f, Dielectric(1.5f), g_context),
      group, g_context);
  addChild(
      Sphere(make_float3(-4.f, 1.f, 1.f), 1.f,
             Lambertian(new Constant_Texture(make_float3(0.4f, 0.2f, 0.1f))),
             g_context),
      group, g_context);

  // configure camera
  const float3 lookfrom = make_float3(13.f, 2.f, 3.f);
  const float3 lookat = make_float3(0.f, 0.f, 0.f);
  const float3 up = make_float3(0.f, 1.f, 0.f);
  const float fovy(20.0);
  const float aspect(float(Nx) / float(Ny));
  const float aperture(0.1f);
  const float dist(10.f);
  Camera camera(lookfrom, lookat, up, fovy, aspect, aperture, dist, 0.0, 1.0);
  camera.set(g_context);

  g_context["world"]->set(group);

  auto t1 = std::chrono::system_clock::now();
  auto sceneTime = std::chrono::duration<float>(t1 - t0).count();
  printf("Done assigning scene data, which took %.2f seconds.\n", sceneTime);
}

void MovingSpheres(Context& g_context, int Nx, int Ny) {
  auto t0 = std::chrono::system_clock::now();

  // configure sampling
  std::vector<PDF*> buffer;
  buffer.push_back(new Rectangle_PDF(3.f, 5.f, 1.f, 3.f, -2.f, Z_AXIS));
  buffer.push_back(new Sphere_PDF(make_float3(4.f, 1.f, 0.f), 1.f));
  Mixture_PDF mixture(new Cosine_PDF(), new Buffer_PDF(buffer));

  // add material PDFs
  Buffer material_pdfs =
      g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 2);
  callableProgramId<int(int)>* f_data =
      static_cast<callableProgramId<int(int)>*>(material_pdfs->map());
  f_data[0] = callableProgramId<int(int)>(Lambertian_PDF(g_context)->getId());
  f_data[1] =
      callableProgramId<int(int)>(Diffuse_Light_PDF(g_context)->getId());
  material_pdfs->unmap();

  // Set the exception, ray generation and miss shader programs
  setRayGenerationProgram(g_context, mixture, material_pdfs);
  setMissProgram(g_context, DARK);
  setExceptionProgram(g_context);

  Group group = g_context->createGroup();
  group->setAcceleration(g_context->createAcceleration("Trbvh"));

  Texture* checker =
      new Checker_Texture(new Constant_Texture(make_float3(0.2f, 0.3f, 0.1f)),
                          new Constant_Texture(make_float3(0.9f, 0.9f, 0.9f)));

  addChild(Sphere(make_float3(0.f, -1000.f, -1.f), 1000.f, Lambertian(checker),
                  g_context),
           group, g_context);

  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      float choose_mat = rnd();
      float3 center = make_float3(a + rnd(), 0.2f, b + rnd());
      if (choose_mat < 0.8f) {
        addChild(
            Moving_Sphere(center, center + make_float3(0.f, 0.5f * rnd(), 0.f),
                          0.f, 1.f, 0.2f,
                          Lambertian(new Constant_Texture(make_float3(
                              rnd() * rnd(), rnd() * rnd(), rnd() * rnd()))),
                          g_context),
            group, g_context);
      } else if (choose_mat < 0.95f) {
        addChild(Sphere(center, 0.2f,
                        Metal(new Constant_Texture(make_float3(
                                  0.5f * (1.f + rnd()), 0.5f * (1.f + rnd()),
                                  0.5f * (1.f + rnd()))),
                              0.5f * rnd()),
                        g_context),
                 group, g_context);
      } else {
        addChild(Sphere(center, 0.2f, Dielectric(1.5f), g_context), group,
                 g_context);
      }
    }
  }
  addChild(Sphere(make_float3(4.f, 1.f, 1.f), 1.f, Dielectric(1.5f), g_context),
           group, g_context);
  addChild(Sphere(make_float3(0.f, 1.f, 1.5f), 1.f,
                  Metal(new Noise_Texture(4.f), 0.f), g_context),
           group, g_context);
  addChild(
      Sphere(make_float3(-4.f, 1.f, 2.f), 1.f,
             Lambertian(new Image_Texture("assets/other_textures/map.jpg")),
             g_context),
      group, g_context);
  addChild(
      Rectangle(3.f, 5.f, 1.f, 3.f, -0.5f, false, Z_AXIS,
                Diffuse_Light(new Constant_Texture(make_float3(4.f, 4.f, 4.f))),
                g_context),
      group, g_context);
  g_context["world"]->set(group);

  // configure camera
  const float3 lookfrom = make_float3(13, 2, 3);
  const float3 lookat = make_float3(0, 0, 0);
  const float3 up = make_float3(0, 1, 0);
  const float fovy(20.0);
  const float aspect(float(Nx) / float(Ny));
  const float aperture(0.1f);
  const float dist(10.f);
  Camera camera(lookfrom, lookat, up, fovy, aspect, aperture, dist, 0.0, 1.0);
  camera.set(g_context);

  auto t1 = std::chrono::system_clock::now();
  auto sceneTime = std::chrono::duration<float>(t1 - t0).count();
  printf("Done assigning scene data, which took %.2f seconds.\n", sceneTime);
}

void Cornell(Context& g_context, int Nx, int Ny) {
  auto t0 = std::chrono::system_clock::now();

  // configure sampling
  std::vector<PDF*> buffer;
  buffer.push_back(
      new Rectangle_PDF(213.f, 343.f, 227.f, 332.f, 554.f, Y_AXIS));
  buffer.push_back(new Sphere_PDF(make_float3(190.f, 90.f, 190.f), 90.f));
  Mixture_PDF mixture(new Cosine_PDF(), new Buffer_PDF(buffer));

  // add material PDFs
  Buffer material_pdfs =
      g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 2);
  callableProgramId<int(int)>* f_data =
      static_cast<callableProgramId<int(int)>*>(material_pdfs->map());
  f_data[0] = callableProgramId<int(int)>(Lambertian_PDF(g_context)->getId());
  f_data[1] =
      callableProgramId<int(int)>(Diffuse_Light_PDF(g_context)->getId());
  material_pdfs->unmap();

  // Set the exception, ray generation and miss shader programs
  setRayGenerationProgram(g_context, mixture, material_pdfs);
  setMissProgram(g_context, DARK);
  setExceptionProgram(g_context);

  Group group = g_context->createGroup();
  group->setAcceleration(g_context->createAcceleration("Trbvh"));

  Materials* red =
      new Lambertian(new Constant_Texture(make_float3(0.65f, 0.05f, 0.05f)));
  Materials* white =
      new Lambertian(new Constant_Texture(make_float3(0.73f, 0.73f, 0.73f)));
  Materials* green =
      new Lambertian(new Constant_Texture(make_float3(0.12f, 0.45f, 0.15f)));
  Materials* light =
      new Diffuse_Light(new Constant_Texture(make_float3(7.f, 7.f, 7.f)));
  Materials* aluminium =
      new Metal(new Constant_Texture(make_float3(0.8f, 0.85f, 0.88f)), 0.0);
  Materials* glass = new Dielectric(1.5f);
  Materials* black_fog = new Isotropic(new Constant_Texture(make_float3(0.f)));
  Materials* white_fog = new Isotropic(new Constant_Texture(make_float3(1.f)));

  addChild(
      Rectangle(0.f, 555.f, 0.f, 555.f, 555.f, true, X_AXIS, *green, g_context),
      group,
      g_context);  // left wall
  addChild(
      Rectangle(0.f, 555.f, 0.f, 555.f, 0.f, false, X_AXIS, *red, g_context),
      group,
      g_context);  // right wall
  addChild(Rectangle(213.f, 343.f, 227.f, 332.f, 554.f, true, Y_AXIS, *light,
                     g_context),
           group,
           g_context);  // light
  addChild(
      Rectangle(0.f, 555.f, 0.f, 555.f, 555.f, true, Y_AXIS, *white, g_context),
      group,
      g_context);  // roof
  addChild(
      Rectangle(0.f, 555.f, 0.f, 555.f, 0.f, false, Y_AXIS, *white, g_context),
      group,
      g_context);  // ground
  addChild(
      Rectangle(0.f, 555.f, 0.f, 555.f, 555.f, true, Z_AXIS, *white, g_context),
      group,
      g_context);  // back walls
  addChild(Sphere(make_float3(190.f, 90.f, 190.f), 90.f, *glass, g_context),
           group,
           g_context);  // glass sphere

  // big box
  addChild(
      translate(rotate(Box(make_float3(0.f), make_float3(165.f, 330.f, 165.f),
                           *aluminium, g_context),
                       15.f, Y_AXIS, g_context),
                make_float3(265.f, 0.f, 295.f), g_context),
      group, g_context);

  // small box
  /*
  addChild(translate(rotate(Box(make_float3(0.f), make_float3(165.f, 165.f,
  165.f), *white, g_context), -18.f, Y_AXIS, g_context), make_float3(130.f,
  0.f, 65.f), g_context), group, g_context);
  */

  // big box
  /*
  addChild(translate(rotate(Volume_Box(make_float3(0.f), make_float3(165.f,
  330.f, 165.f), 0.01f, *black_fog, g_context), 15.f, Y_AXIS, g_context),
                                                                        make_float3(265.f,
  0.f, 295.f), g_context), group, g_context);

  // small box
  addChild(translate(rotate(Volume_Box(make_float3(0.f), make_float3(165.f,
  165.f, 165.f), 0.01f, *white_fog, g_context), -18.f, Y_AXIS, g_context),
                                                                        make_float3(130.f,
  0.f, 65.f), g_context), group, g_context);
  */
  g_context["world"]->set(group);

  // configure camera
  const float3 lookfrom = make_float3(278.f, 278.f, -800.f);
  const float3 lookat = make_float3(278.f, 278.f, 0.f);
  const float3 up = make_float3(0.f, 1.f, 0.f);
  const float fovy(40.f);
  const float aspect(float(Nx) / float(Ny));
  const float aperture(0.f);
  const float dist(10.f);
  Camera camera(lookfrom, lookat, up, fovy, aspect, aperture, dist, 0.0, 1.0);
  camera.set(g_context);

  auto t1 = std::chrono::system_clock::now();
  auto sceneTime = std::chrono::duration<float>(t1 - t0).count();
  printf("Done assigning scene data, which took %.2f seconds.\n", sceneTime);
}

void Final_Next_Week(Context& g_context, int Nx, int Ny) {
  auto t0 = std::chrono::system_clock::now();

  // configure sampling
  std::vector<PDF*> buffer;
  buffer.push_back(
      new Rectangle_PDF(113.f, 443.f, 127.f, 432.f, 554.f, Y_AXIS));
  buffer.push_back(new Sphere_PDF(make_float3(260.f, 150.f, 45.f), 50.f));
  Mixture_PDF mixture(new Cosine_PDF(), new Buffer_PDF(buffer));

  // add material PDFs
  Buffer material_pdfs =
      g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 2);
  callableProgramId<int(int)>* f_data =
      static_cast<callableProgramId<int(int)>*>(material_pdfs->map());
  f_data[0] = callableProgramId<int(int)>(Lambertian_PDF(g_context)->getId());
  f_data[1] =
      callableProgramId<int(int)>(Diffuse_Light_PDF(g_context)->getId());
  material_pdfs->unmap();

  // Set the exception, ray generation and miss shader programs
  setRayGenerationProgram(g_context, mixture, material_pdfs);
  setMissProgram(g_context, DARK);
  setExceptionProgram(g_context);

  Group group = g_context->createGroup();
  group->setAcceleration(g_context->createAcceleration("Trbvh"));

  // ground
  Materials* ground =
      new Lambertian(new Constant_Texture(make_float3(0.48f, 0.83f, 0.53f)));
  for (int i = 0; i < 20; i++) {
    for (int j = 0; j < 20; j++) {
      float w = 100.f;
      float x0 = -1000 + i * w;
      float z0 = -1000 + j * w;
      float y0 = 0.f;
      float x1 = x0 + w;
      float y1 = 100 * (rnd() + 0.01f);
      float z1 = z0 + w;
      addChild(Box(make_float3(x0, y0, z0), make_float3(x1, y1, z1), *ground,
                   g_context),
               group, g_context);
    }
  }

  // light
  Materials* light =
      new Diffuse_Light(new Constant_Texture(make_float3(7.f, 7.f, 7.f)));
  addChild(Rectangle(113.f, 443.f, 127.f, 432.f, 554.f, true, Y_AXIS, *light,
                     g_context),
           group, g_context);

  // brown moving sphere
  float3 center = make_float3(400.f, 400.f, 200.f);
  Materials* brown =
      new Lambertian(new Constant_Texture(make_float3(0.7f, 0.3f, 0.1f)));
  addChild(Moving_Sphere(center, center + make_float3(30.f, 0.f, 0.f), 0.f, 1.f,
                         50.f, *brown, g_context),
           group, g_context);

  // glass sphere
  addChild(
      Sphere(make_float3(260.f, 150.f, 45.f), 50.f, Dielectric(1.5), g_context),
      group, g_context);

  // metal sphere
  addChild(
      Sphere(make_float3(0.f, 150.f, 145.f), 50.f,
             Metal(new Constant_Texture(make_float3(0.8f, 0.8f, 0.9f)), 10.f),
             g_context),
      group, g_context);

  // blue sphere
  // glass sphere
  addChild(
      Sphere(make_float3(360.f, 150.f, 45.f), 70.f, Dielectric(1.5), g_context),
      group, g_context);
  // blue fog
  Materials* blue_fog =
      new Isotropic(new Constant_Texture(make_float3(0.2f, 0.4f, 0.9f)));
  addChild(Volume_Sphere(make_float3(360.f, 150.f, 45.f), 70.f, 0.2f, *blue_fog,
                         g_context),
           group, g_context);

  // fog
  Materials* fog = new Isotropic(new Constant_Texture(make_float3(1.f)));
  addChild(Volume_Sphere(make_float3(0.f), 5000.f, 0.0001f, *fog, g_context),
           group, g_context);

  // earth
  addChild(
      Sphere(make_float3(400.f, 200.f, 400.f), 100.f,
             Lambertian(new Image_Texture("assets/other_textures/map.jpg")),
             g_context),
      group, g_context);

  // Perlin sphere
  addChild(Sphere(make_float3(220.f, 280.f, 300.f), 80.f,
                  Lambertian(new Noise_Texture(0.1f)), g_context),
           group, g_context);

  // group of small spheres
  Materials* white =
      new Lambertian(new Constant_Texture(make_float3(0.73f, 0.73f, 0.73f)));
  std::vector<GeometryInstance> d_list;
  for (int j = 0; j < 1000; j++) {
    d_list.push_back(Sphere(make_float3(165 * rnd(), 165 * rnd(), 165 * rnd()),
                            10.f, *white, g_context));
  }

  GeometryGroup box = g_context->createGeometryGroup();
  box->setAcceleration(g_context->createAcceleration("Trbvh"));
  for (int i = 0; i < d_list.size(); i++) box->addChild(d_list[i]);
  addChild(translate(rotate(box, 15.f, Y_AXIS, g_context),
                     make_float3(-100.f, 270.f, 395.f), g_context),
           group, g_context);

  g_context["world"]->set(group);

  // configure camera
  const float3 lookfrom = make_float3(478.f, 278.f, -600.f);
  const float3 lookat = make_float3(278.f, 278.f, 0.f);
  const float3 up = make_float3(0.f, 1.f, 0.f);
  const float fovy(40.f);
  const float aspect(float(Nx) / float(Ny));
  const float aperture(0.f);
  const float dist(10.f);
  Camera camera(lookfrom, lookat, up, fovy, aspect, aperture, dist, 0.0, 1.0);
  camera.set(g_context);

  auto t1 = std::chrono::system_clock::now();
  auto sceneTime = std::chrono::duration<float>(t1 - t0).count();
  printf("Done assigning scene data, which took %.2f seconds.\n", sceneTime);
}

void Test_Scene(Context& g_context, int Nx, int Ny) {
  auto t0 = std::chrono::system_clock::now();

  // configure sampling
  Mixture_PDF mixture(new Cosine_PDF(),
                      new Sphere_PDF(make_float3(300.f, -300.f, 300.f), 300.f));

  // add material PDFs
  Buffer material_pdfs =
      g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 2);
  callableProgramId<int(int)>* f_data =
      static_cast<callableProgramId<int(int)>*>(material_pdfs->map());
  f_data[0] = callableProgramId<int(int)>(Lambertian_PDF(g_context)->getId());
  f_data[1] =
      callableProgramId<int(int)>(Diffuse_Light_PDF(g_context)->getId());
  material_pdfs->unmap();

  // Set the ray generation and miss programs
  setRayGenerationProgram(g_context, mixture, material_pdfs);
  setMissProgram(g_context, HDR, "assets/hdr/fireplace.hdr");

  // Set the exception program
  setExceptionProgram(g_context);

  Group group = g_context->createGroup();
  group->setAcceleration(g_context->createAcceleration("Trbvh"));

  Materials* white = new Lambertian(new Constant_Texture(make_float3(0.73f)));
  Materials* aluminium =
      new Metal(new Constant_Texture(make_float3(0.8f, 0.85f, 0.88f)), 0.0);
  Materials* black = new Lambertian(new Constant_Texture(make_float3(0.f)));
  Materials* glass = new Dielectric(1.5f);

  // Test model
  /*float scale_factor = 1400.f;
  GeometryInstance model =
      Mesh("nam.obj", g_context, *black, true, "assets/nam/");
  if (model == NULL)
    system("PAUSE");
  else
    addChild(
        translate(rotate(scale(model, make_float3(scale_factor), g_context),
                         180.f, Y_AXIS, g_context),
                  make_float3(0.f, -300.f, 0.f), g_context),
        group, g_context);*/

  // Lucy
  /*float scale_factor = 150.f;
  GeometryInstance model =
      Mesh("Lucy1M.obj", g_context, *aluminium, true, "assets/lucy/");
  if (model == NULL)
    system("PAUSE");
  else
    addChild(translate(scale(model, make_float3(scale_factor), g_context),
                       make_float3(0.f, -550.f, 0.f), g_context),
             group, g_context);*/

  // Dragon
  /*float scale_factor = 350.f;
  GeometryInstance model =
      Mesh("dragon_cubic.obj", g_context, *white, true, "assets/dragon/");
  if (model == NULL)
    system("PAUSE");
  else
    addChild(
        translate(rotate(scale(model, make_float3(scale_factor), g_context),
                         180.f, Y_AXIS, g_context),
                  make_float3(0.f, -500.f, 200.f), g_context),
        group, g_context);*/

  addChild(Sphere(make_float3(300.f, -300.f, 300.f), 300.f, *glass, g_context),
           group, g_context);
  addChild(
      Sphere(make_float3(-300.f, -300.f, 150.f), 300.f, *aluminium, g_context),
      group, g_context);
  addChild(Rectangle(-1000.f, 1000.f, -500.f, 500.f, -500.f, false, Y_AXIS,
                     *white, g_context),
           group,
           g_context);  // ground

  // configure camera
  const float3 lookfrom = make_float3(0.f, 0.f, -800.f);
  const float3 lookat = make_float3(0.f, 0.f, 0.f);
  const float3 up = make_float3(0.f, 1.f, 0.f);
  const float fovy(100.f);
  const float aspect(float(Nx) / float(Ny));
  const float aperture(0.f);
  const float dist(0.8f);
  Camera camera(lookfrom, lookat, up, fovy, aspect, aperture, dist, 0.0, 1.0);
  camera.set(g_context);

  // Sponza
  /*float scale = 0.5f;
  GeometryInstance model = Mesh("sponza.obj", g_context, *white, false,
  "assets/sponza/", scale); addChild(translate(rotate(model, 90.f, Y_AXIS,
  g_context), make_float3(300.f, 5.f, -400.f), g_context), group, g_context);*/

  // configure camera
  /*const float3 lookfrom = make_float3(278.f, 278.f, -800.f);
  const float3 lookat = make_float3(278.f, 278.f, 0.f);
  const float3 up = make_float3(0.f, 1.f, 0.f);
  const float fovy(40.f);
  const float aspect(float(Nx) / float(Ny));
  const float aperture(0.f);
  const float dist(10.f);
  Camera camera(lookfrom, lookat, up, fovy, aspect, aperture, dist, 0.0, 1.0);
  camera.set(g_context);*/

  g_context["world"]->set(group);

  auto t1 = std::chrono::system_clock::now();
  auto sceneTime = std::chrono::duration<float>(t1 - t0).count();
  printf("Done assigning scene data, which took %.2f seconds.\n", sceneTime);
}

#endif