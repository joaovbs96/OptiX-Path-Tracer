#ifndef SCENESH
#define SCENESH

// scene.hpp: Define test scene creation functions

#include <chrono>

#include "camera.hpp"
#include "hitables.hpp"
#include "mesh.hpp"
#include "pdfs.hpp"

// TODO: convert pointers to smart/shared pointers

// Assumptions taken on scene functions:
// - each Geometry(or GeometryTriangle) has only one material
// - materials have one or more textures assigned to them
// - For GeometryTriangles, as it's done in the Mesh class, make use of the
// Vector_Texture.

void InOneWeekend(Context& g_context, int Nx, int Ny) {
  auto t0 = std::chrono::system_clock::now();

  // add BRDF programs
  BRDF_Sampler brdf;
  for (int i = 0; i < NUMBER_OF_MATERIALS; i++) {
    brdf.sample.push_back(getSampleProgram(MaterialType(i), g_context));
    brdf.pdf.push_back(getPDFProgram(MaterialType(i), g_context));
    brdf.eval.push_back(getEvaluateProgram(MaterialType(i), g_context));
  }

  // add light parameters and programs
  Light_Sampler lights;

  // Set the exception, ray generation and miss shader programs
  setRayGenerationProgram(g_context, brdf, lights);
  setMissProgram(g_context, GRADIENT,            // gradient sky pattern
                 make_float3(1.f),               // white
                 make_float3(0.5f, 0.7f, 1.f));  // light blue
  setExceptionProgram(g_context);

  // Set acceleration structure
  Group group = g_context->createGroup();
  group->setAcceleration(g_context->createAcceleration("Trbvh"));

  // create geometries
  Texture_List txt;
  Hitable_List list;
  int groundTx = txt.push(new Constant_Texture(0.5f));
  Host_Material* ground = new Lambertian(txt[groundTx]);
  list.push(new Sphere(make_float3(0.f, -1000.f, -1.f), 1000.f, ground));

  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      float choose_mat = rnd();
      float3 center = make_float3(a + rnd(), 0.2f, b + rnd());
      if (choose_mat < 0.8f) {
        Texture* tx = new Constant_Texture(rnd(), rnd(), rnd());
        Host_Material* mt = new Lambertian(tx);
        list.push(new Sphere(center, 0.2f, mt));
        txt.push(tx);
      } else if (choose_mat < 0.95f) {
        Texture* tx = new Constant_Texture(
            0.5f * (1.f + rnd()), 0.5f * (1.f + rnd()), 0.5f * (1.f + rnd()));
        Host_Material* mt = new Metal(tx, 0.5f * rnd());
        list.push(new Sphere(center, 0.2f, mt));
        txt.push(tx);
      } else {
        Texture* tx1 = new Constant_Texture(1.f);
        Texture* tx2 = new Constant_Texture(rnd(), rnd(), rnd());
        Host_Material* mt = new Dielectric(tx1, tx2, 1.5, 0.f);
        list.push(new Sphere(center, 0.2f, mt));
        txt.push(tx1);
        txt.push(tx2);
      }
    }
  }

  Texture* tx1 = new Constant_Texture(1.f);
  Texture* tx2 = new Constant_Texture(1.f, 1.f, rnd());
  Host_Material* mt1 = new Dielectric(tx1, tx2, 1.5, 0.f);
  list.push(new Sphere(make_float3(-4.f, 1.f, 1.f), 1.f, mt1));
  txt.push(tx1);
  txt.push(tx2);

  Texture* tx4 = new Constant_Texture(1.f);
  Host_Material* mt0 = new Lambertian(tx4);
  list.push(new Sphere(make_float3(0.f, 1.f, 0.5f), 1.f, mt0));

  Texture* tx3 = new Constant_Texture(0.f);
  Host_Material* mt2 = new Anisotropic(tx4, tx3, 10000.f, 10000.f);
  list.push(new Sphere(make_float3(4.f, 1.f, 0.f), 1.f, mt2));
  txt.push(tx3);

  // transforms list elements, one by one, and adds them to the graph
  list.addChildren(group, g_context);
  g_context["world"]->set(group);

  // configure camera
  const float3 lookfrom = make_float3(13.f, 26.f, 3.f);
  const float3 lookat = make_float3(0.f, 0.f, 0.f);
  const float3 up = make_float3(0.f, 1.f, 0.f);
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

void MovingSpheres(Context& g_context, int Nx, int Ny) {
  auto t0 = std::chrono::system_clock::now();

  // add BRDF programs
  BRDF_Sampler brdf;
  for (int i = 0; i < NUMBER_OF_MATERIALS; i++) {
    brdf.sample.push_back(getSampleProgram(MaterialType(i), g_context));
    brdf.pdf.push_back(getPDFProgram(MaterialType(i), g_context));
    brdf.eval.push_back(getEvaluateProgram(MaterialType(i), g_context));
  }

  // add light parameters and programs
  Light_Sampler lights;
  Rectangle_PDF rect_pdf(3.f, 5.f, 1.f, 3.f, -0.5f, Z_AXIS);
  lights.pdf.push_back(rect_pdf.createPDF(g_context));
  lights.sample.push_back(rect_pdf.createSample(g_context));
  lights.emissions.push_back(make_float3(4.f));

  // Set the exception, ray generation and miss shader programs
  setRayGenerationProgram(g_context, brdf, lights);
  setMissProgram(g_context, CONSTANT);  // dark background
  setExceptionProgram(g_context);

  // Set acceleration structure
  Group group = g_context->createGroup();
  group->setAcceleration(g_context->createAcceleration("Trbvh"));

  // create scene
  Texture_List txt;
  Material_List mat;
  Hitable_List list;
  Texture* ck1 = new Constant_Texture(0.2f, 0.3f, 0.1f);
  Texture* ck2 = new Constant_Texture(0.9f, 0.9f, 0.9f);
  int groundTx = txt.push(new Checker_Texture(ck1, ck2));
  Host_Material* ground = new Lambertian(txt[groundTx]);
  list.push(new Sphere(make_float3(0.f, -1000.f, -1.f), 1000.f, ground));

  // Small spheres
  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      float choose_mat = rnd();
      float3 center = make_float3(a + rnd(), 0.2f, b + rnd());
      float3 center2 = center + make_float3(0.f, 0.5f * rnd(), 0.f);
      if (choose_mat < (1.f / 3)) {
        int mtx = txt.push(new Constant_Texture(
            0.5f * (1.f + rnd()), 0.5f * (1.f + rnd()), 0.5f * (1.f + rnd())));
        int lmt = mat.push(new Lambertian(txt[mtx]));
        list.push(new Sphere(center, 0.2f, mat[lmt]));
      } else if (choose_mat < (2.f / 3)) {
        int mtx = txt.push(new Constant_Texture(
            0.5f * (1.f + rnd()), 0.5f * (1.f + rnd()), 0.5f * (1.f + rnd())));
        int lmt = mat.push(new Metal(txt[mtx], 0.5f * rnd()));
        list.push(new Sphere(center, 0.2f, mat[lmt]));
      } else {
        int mtx = txt.push(new Constant_Texture(
            0.5f * (1.f + rnd()), 0.5f * (1.f + rnd()), 0.5f * (1.f + rnd())));
        int lmt = mat.push(new Dielectric(txt[mtx], txt[mtx], 1.5, 0.f));
        list.push(new Sphere(center, 0.2f, mat[lmt]));
      }
    }
  }

  // Earth
  int etx =
      txt.push(new Image_Texture("../../../assets/other_textures/map.jpg"));
  Host_Material* emt = new Lambertian(txt[etx]);
  list.push(new Sphere(make_float3(-4.f, 1.f, 2.f), 1.f, emt));

  // Glass Sphere
  int gtx1 = txt.push(new Constant_Texture(1.f));
  int gtx2 = txt.push(new Constant_Texture(rnd(), rnd(), rnd()));
  Host_Material* gmt = new Dielectric(txt[gtx1], txt[gtx2], 1.5, 0.f);
  list.push(new Sphere(make_float3(4.f, 1.f, 1.f), 1.f, gmt));

  // 'rusty' Metal Sphere
  int mtx = txt.push(new Noise_Texture(4.f));
  Host_Material* mmt = new Metal(txt[mtx], 0.f);
  list.push(new Sphere(make_float3(0.f, 1.f, 1.5f), 1.f, mmt));

  // Light
  int ltx = txt.push(new Constant_Texture(4.f));
  Host_Material* lmt = new Diffuse_Light(txt[ltx]);
  list.push(new AARect(3.f, 5.f, 1.f, 3.f, -0.5f, false, Z_AXIS, lmt));

  // transforms list elements, one by one, and adds them to the graph
  list.addChildren(group, g_context);
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

  // configure materials
  BRDF_Sampler brdf;
  for (int i = 0; i < NUMBER_OF_MATERIALS; i++) {
    brdf.sample.push_back(getSampleProgram(MaterialType(i), g_context));
    brdf.pdf.push_back(getPDFProgram(MaterialType(i), g_context));
    brdf.eval.push_back(getEvaluateProgram(MaterialType(i), g_context));
  }

  // add light parameters and programs
  Light_Sampler lights;
  Rectangle_PDF rect_pdf(213.f, 343.f, 227.f, 332.f, 554.f, Y_AXIS);
  lights.pdf.push_back(rect_pdf.createPDF(g_context));
  lights.sample.push_back(rect_pdf.createSample(g_context));
  lights.emissions.push_back(make_float3(7.f));

  // Set the exception, ray generation and miss shader programs
  setRayGenerationProgram(g_context, brdf, lights);
  setMissProgram(g_context, CONSTANT);  // dark background
  setExceptionProgram(g_context);

  // create scene group
  Group group = g_context->createGroup();
  group->setAcceleration(g_context->createAcceleration("Trbvh"));

  // create textures
  Texture_List textures;
  int redTx = textures.push(new Constant_Texture(0.65f, 0.05f, 0.05f));
  int whiteTx = textures.push(new Constant_Texture(0.73f));
  int greenTx = textures.push(new Constant_Texture(0.12f, 0.45f, 0.15f));
  int lightTx = textures.push(new Constant_Texture(7.f));
  int alumTx = textures.push(new Constant_Texture(0.8f, 0.85f, 0.88f));
  int pWhiteTx = textures.push(new Constant_Texture(1.f));
  int pBlackTx = textures.push(new Constant_Texture(0.f));
  int testTx = textures.push(new Constant_Texture(0.5f));

  // create materials
  Material_List mats;
  int redMt = mats.push(new Lambertian(textures[redTx]));
  int whiteMt = mats.push(new Lambertian(textures[whiteTx]));
  int greenMt = mats.push(new Lambertian(textures[greenTx]));
  int lightMt = mats.push(new Diffuse_Light(textures[lightTx]));
  int alumMt = mats.push(new Metal(textures[alumTx], 0.0));
  int glassMt =
      mats.push(new Dielectric(textures[pWhiteTx], textures[redTx], 1.5f, 0.f));
  int blackSmokeMt = mats.push(new Isotropic(textures[pBlackTx]));
  int oren = mats.push(new Oren_Nayar(textures[whiteTx], 1.f));

  Texture* tx3 = new Constant_Texture(1.f);
  Texture* tx4 = new Constant_Texture(0.f);
  int aniso = mats.push(new Anisotropic(tx3, tx4, 10000.f, 10000.f));

  // create geometries/hitables
  Hitable_List list;
  list.push(
      new AARect(0.f, 555.f, 0.f, 555.f, 555.f, true, X_AXIS, mats[redMt]));
  list.push(
      new AARect(0.f, 555.f, 0.f, 555.f, 0.f, false, X_AXIS, mats[greenMt]));
  list.push(new AARect(213.f, 343.f, 227.f, 332.f, 554.f, true, Y_AXIS,
                       mats[lightMt]));
  list.push(
      new AARect(0.f, 555.f, 0.f, 555.f, 555.f, true, Y_AXIS, mats[whiteMt]));
  list.push(
      new AARect(0.f, 555.f, 0.f, 555.f, 0.f, false, Y_AXIS, mats[whiteMt]));
  list.push(
      new AARect(0.f, 555.f, 0.f, 555.f, 555.f, true, Z_AXIS, mats[whiteMt]));
  list.push(new Sphere(make_float3(555.f / 2.f, 90.f, 555.f / 2.f), 90.f,
                       mats[aniso]));
  /*list.push(
      new Sphere(make_float3(555 / 3.f, 90.f, 555 / 2.f), 90.f, mats[oren]));
  list.push(new Sphere(make_float3(2 * 555 / 3.f, 90.f, 555 / 2.f), 90.f,
                       mats[whiteMt]));*/
  /*list.push(new AARect(-1000.f, 1000.f, -1000.f, 1000.f, 0.f, false, Y_AXIS,
                       mats[testMt]));*/

  // Aluminium box
  Box box =
      Box(make_float3(0.f), make_float3(165.f, 330.f, 165.f), mats[whiteMt]);
  box.translate(make_float3(265.f, 0.f, 295.f));
  box.rotate(15.f, Y_AXIS);
  list.push(&box);

  Box box2 =
      Box(make_float3(0.f), make_float3(165.f, 165.f, 165.f), mats[whiteMt]);
  box2.translate(make_float3(130.f, 0.f, 65.f));
  box2.rotate(-18.f, Y_AXIS);
  list.push(&box2);

  // transforms list elements, one by one, and adds them to the graph
  list.addChildren(group, g_context);
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

  // add BRDF programs
  BRDF_Sampler brdf;
  for (int i = 0; i < NUMBER_OF_MATERIALS; i++) {
    brdf.sample.push_back(getSampleProgram(MaterialType(i), g_context));
    brdf.pdf.push_back(getPDFProgram(MaterialType(i), g_context));
    brdf.eval.push_back(getEvaluateProgram(MaterialType(i), g_context));
  }

  // TODO: find a way to add these automatically once we create a diffuse light
  // Maybe we could create lights separately, not with the other geometry
  // add light parameters and programs
  Light_Sampler lights;
  Rectangle_PDF rect_pdf(113.f, 443.f, 127.f, 432.f, 554.f, Y_AXIS);
  lights.pdf.push_back(rect_pdf.createPDF(g_context));
  lights.sample.push_back(rect_pdf.createSample(g_context));
  lights.emissions.push_back(make_float3(7.f));

  // Set the exception, ray generation and miss shader programs
  setRayGenerationProgram(g_context, brdf, lights);
  setMissProgram(g_context, CONSTANT);  // dark background
  setExceptionProgram(g_context);

  Group group = g_context->createGroup();
  group->setAcceleration(g_context->createAcceleration("Trbvh"));

  Texture_List txt;
  Material_List mat;
  Hitable_List list;

  int groundTx = txt.push(new Constant_Texture(0.48f, 0.83f, 0.53f));
  Host_Material* ground = new Lambertian(txt[groundTx]);

  // ground
  for (int i = 0; i < 20; i++) {
    for (int j = 0; j < 20; j++) {
      float w = 100.f;
      float x0 = -1000 + i * w;
      float z0 = -1000 + j * w;
      float y0 = 0.f;
      float x1 = x0 + w;
      float y1 = 100 * (rnd() + 0.01f);
      float z1 = z0 + w;
      float3 p0 = make_float3(x0, y0, z0);
      float3 p1 = make_float3(x1, y1, z1);
      list.push(new Box(p0, p1, ground));
    }
  }

  // light
  int lightTx = txt.push(new Constant_Texture(7.f));
  Host_Material* light = new Diffuse_Light(txt[lightTx]);
  list.push(new AARect(113.f, 443.f, 127.f, 432.f, 554.f, true, Y_AXIS, light));

  // brown sphere
  float3 center = make_float3(400.f, 400.f, 200.f);
  int brownTx = txt.push(new Constant_Texture(0.7f, 0.3f, 0.1f));
  Host_Material* brown = new Lambertian(txt[brownTx]);
  list.push(new Sphere(center, 50.f, brown));

  // glass sphere
  int glassTx1 = txt.push(new Constant_Texture(1.f));
  Host_Material* glass = new Dielectric(txt[glassTx1], txt[glassTx1], 1.5f);
  list.push(new Sphere(make_float3(260.f, 150.f, 45.f), 50.f, glass));

  // metal sphere
  int metalTx = txt.push(new Constant_Texture(0.8f, 0.8f, 0.9f));
  Host_Material* metal = new Metal(txt[metalTx], 10.f);
  list.push(new Sphere(make_float3(0.f, 150.f, 145.f), 50.f, metal));

  // blue sphere
  // glass sphere
  list.push(new Sphere(make_float3(360.f, 150.f, 45.f), 70.f, glass));
  // blue fog
  int blueTx = txt.push(new Constant_Texture(0.2f, 0.4f, 0.9f));
  Host_Material* blueFog = new Isotropic(txt[blueTx]);
  list.push(new Volumetric_Sphere(make_float3(360.f, 150.f, 45.f), 70.f, 0.2f,
                                  blueFog));

  // white fog
  Host_Material* whiteFog = new Isotropic(txt[glassTx1]);
  list.push(new Volumetric_Sphere(make_float3(0.f), 5000.f, 0.0001f, whiteFog));

  // earth
  int etx =
      txt.push(new Image_Texture("../../../assets/other_textures/map.jpg"));
  Host_Material* emt = new Lambertian(txt[etx]);
  list.push(new Sphere(make_float3(400.f, 200.f, 400.f), 100.f, emt));

  // Perlin sphere
  int perlinTx = txt.push(new Noise_Texture(0.1f));
  Host_Material* noise = new Lambertian(txt[perlinTx]);
  list.push(new Sphere(make_float3(220.f, 280.f, 300.f), 80.f, noise));

  // group of small spheres
  int whiteTx = txt.push(new Constant_Texture(0.73f));
  Host_Material* whiteMt = new Lambertian(txt[whiteTx]);
  Hitable_List spheres;
  for (int j = 0; j < 1000; j++) {
    center = make_float3(165 * rnd(), 165 * rnd(), 165 * rnd());
    spheres.push(new Sphere(center, 10.f, whiteMt));
  }
  spheres.translate(make_float3(-100.f, 270.f, 395.f));
  spheres.rotate(15.f, Y_AXIS);
  spheres.addList(group, g_context);

  // transforms list elements, one by one, and adds them to the graph
  list.addChildren(group, g_context);
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

void Test_Scene(Context& g_context, int Nx, int Ny, int modelID) {
  auto t0 = std::chrono::system_clock::now();

  // configure BRDF programs
  BRDF_Sampler brdf;
  for (int i = 0; i < NUMBER_OF_MATERIALS; i++) {
    brdf.sample.push_back(getSampleProgram(MaterialType(i), g_context));
    brdf.pdf.push_back(getPDFProgram(MaterialType(i), g_context));
    brdf.eval.push_back(getEvaluateProgram(MaterialType(i), g_context));
  }

  // add light parameters and programs
  Light_Sampler lights;

  // Set the exception, ray generation and miss shader programs
  setRayGenerationProgram(g_context, brdf, lights);
  setMissProgram(g_context, HDR, "../../../assets/hdr/fireplace.hdr");
  setExceptionProgram(g_context);

  // create scene group
  Group group = g_context->createGroup();
  group->setAcceleration(g_context->createAcceleration("Trbvh"));

  // create textures
  Texture_List txts;
  int whiteTx = txts.push(new Constant_Texture(0.73f));
  int blackTx = txts.push(new Constant_Texture(0.f));
  int alumTx = txts.push(new Constant_Texture(0.8f, 0.85f, 0.88f));
  int noiseTx = txts.push(new Noise_Texture(0.01f));
  int blueTx = txts.push(new Constant_Texture(0.2f, 0.4f, 0.9f));
  int perlinXTx = txts.push(new Noise_Texture(0.01f, X_AXIS));
  int perlinYTx = txts.push(new Noise_Texture(0.01f, Y_AXIS));
  int perlinZTx = txts.push(new Noise_Texture(0.01f, Z_AXIS));
  int pWhiteTx = txts.push(new Constant_Texture(1.f));

  // create materials
  Material_List mats;
  int whiteMt = mats.push(new Lambertian(txts[whiteTx]));
  int blackMt = mats.push(new Lambertian(txts[blackTx]));
  int alumMt = mats.push(new Metal(txts[alumTx], 0.0));
  int glassMt =
      mats.push(new Dielectric(txts[noiseTx], txts[blueTx], 1.5f, 0.f));
  int blueMt =
      mats.push(new Dielectric(txts[alumTx], txts[whiteTx], 1.5f, 0.f));
  int normalMt = mats.push(new Normal_Shader());
  int shadingMt = mats.push(new Normal_Shader(true));
  int perlinXMt = mats.push(new Lambertian(txts[perlinXTx]));
  int perlinYMt = mats.push(new Lambertian(txts[perlinYTx]));
  int perlinZMt = mats.push(new Lambertian(txts[perlinZTx]));
  int whiteIso = mats.push(new Isotropic(txts[blueTx]));

  // create geometries
  Hitable_List list;

  // Test model
  if (modelID == 0) {
    Mesh_List meshList;

    Mesh model1 = Mesh("nam.obj", "../../../assets/nam/", mats[shadingMt]);
    model1.scale(make_float3(1400.f));
    model1.rotate(180.f, Y_AXIS);
    model1.translate(make_float3(-300.f, -600.f, 0.f));
    meshList.push(&model1);

    Mesh model2 = Mesh("nam.obj", "../../../assets/nam/", mats[normalMt]);
    model2.scale(make_float3(1400.f));
    model2.rotate(180.f, Y_AXIS);
    model2.translate(make_float3(300.f, -600.f, 0.f));
    meshList.push(&model2);

    meshList.addChildren(group, g_context);

    list.push(new AARect(-1000.f, 1000.f, -500.f, 500.f, -600.f, false, Y_AXIS,
                         mats[normalMt]));
  }

  // lucy
  else if (modelID == 1) {
    Mesh model = Mesh("Lucy1M.obj", "../../../assets/lucy/");
    model.scale(make_float3(150.f));
    model.translate(make_float3(0.f, -550.f, 0.f));
    model.addToScene(group, g_context);

    list.push(new AARect(-1000.f, 1000.f, -500.f, 500.f, -600.f, false, Y_AXIS,
                         mats[whiteMt]));
  }

  // Dragon
  else if (modelID == 2) {
    Mesh model = Mesh("dragon_cubic.obj", "../../../assets/dragon/");
    model.scale(make_float3(350.f));
    model.rotate(180.f, Y_AXIS);
    model.translate(make_float3(0.f, -500.f, 200.f));
    model.addToScene(group, g_context);

    list.push(new AARect(-1000.f, 1000.f, -500.f, 500.f, -600.f, false, Y_AXIS,
                         mats[whiteMt]));
  }

  // spheres
  else if (modelID == 3) {
    /*list.push(
        new Sphere(make_float3(-350.f, -300.f, 0.f), 150.f, mats[perlinXMt]));*/
    list.push(new Sphere(make_float3(0.f, -450.f, 0.f), 150.f, mats[whiteIso]));
    /*list.push(
        new Sphere(make_float3(350.f, -300.f, 0.f), 150.f, mats[perlinZMt]));*/
    list.push(new AARect(-1000.f, 1000.f, -500.f, 500.f, -600.f, false, Y_AXIS,
                         mats[whiteMt]));
  }

  // pie
  else if (modelID == 4) {
    Mesh model = Mesh("pie.obj", "../../../assets/pie/");
    model.scale(make_float3(150.f));
    model.translate(make_float3(0.f, -550.f, 0.f));
    model.addToScene(group, g_context);

    list.push(new AARect(-1000.f, 1000.f, -500.f, 500.f, -600.f, false, Y_AXIS,
                         mats[whiteMt]));
  }

  // sponza
  else {
    Mesh model = Mesh("sponza.obj", "../../../assets/sponza/");
    model.scale(make_float3(0.5f));
    model.rotate(90.f, Y_AXIS);
    model.translate(make_float3(300.f, 5.f, -400.f));
    model.addToScene(group, g_context);
  }

  // transforms list elements, one by one, and adds them to the graph
  list.addChildren(group, g_context);
  g_context["world"]->set(group);

  // configure camera
  if ((modelID >= 0) && (modelID < 5)) {
    const float3 lookfrom = make_float3(0.f, 0.f, -800.f);
    const float3 lookat = make_float3(0.f, 0.f, 0.f);
    const float3 up = make_float3(0.f, 1.f, 0.f);
    const float fovy(100.f);
    const float aspect(float(Nx) / float(Ny));
    const float aperture(0.f);
    const float dist(0.8f);
    Camera camera(lookfrom, lookat, up, fovy, aspect, aperture, dist, 0.0, 1.0);
    camera.set(g_context);
  }

  // for sponza
  else if (modelID == 5) {
    const float3 lookfrom = make_float3(278.f, 278.f, -800.f);
    const float3 lookat = make_float3(278.f, 278.f, 0.f);
    const float3 up = make_float3(0.f, 1.f, 0.f);
    const float fovy(40.f);
    const float aspect(float(Nx) / float(Ny));
    const float aperture(0.f);
    const float dist(10.f);
    Camera camera(lookfrom, lookat, up, fovy, aspect, aperture, dist, 0.0, 1.0);
    camera.set(g_context);
  }

  auto t1 = std::chrono::system_clock::now();
  auto sceneTime = std::chrono::duration<float>(t1 - t0).count();
  printf("Done assigning scene data, which took %.2f seconds.\n", sceneTime);
}

#endif