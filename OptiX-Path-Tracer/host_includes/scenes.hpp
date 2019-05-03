#ifndef SCENESH
#define SCENESH

// scene.hpp: Define test scene creation functions

#include <chrono>

#include "camera.hpp"
#include "hitables.hpp"
#include "mesh.hpp"
#include "pdfs.hpp"

// TODO: convert pointers to smart/shared pointers
// TODO: add lights separately from raygen, and after all materials are
// created(we may need to add back the material type to the materials)

void InOneWeekend(Context& g_context, int Nx, int Ny) {
  auto t0 = std::chrono::system_clock::now();

  // add light parameters and programs
  Light_Sampler lights;

  // Set the exception, ray generation and miss shader programs
  setRayGenerationProgram(g_context, lights);
  setMissProgram(g_context, GRADIENT,            // gradient sky pattern
                 make_float3(1.f),               // white
                 make_float3(0.5f, 0.7f, 1.f));  // light blue
  setExceptionProgram(g_context);

  // Set acceleration structure
  Group group = g_context->createGroup();
  group->setAcceleration(g_context->createAcceleration("Trbvh"));

  // create geometries
  Hitable_List list;
  Texture* groundTx = new Constant_Texture(0.5f);
  BRDF* ground = new Lambertian(groundTx);

  list.push(new Sphere(make_float3(0.f, -1000.f, -1.f), 1000.f, ground));

  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      float choose_mat = rnd();
      float3 center = make_float3(a + rnd(), 0.2f, b + rnd());
      if (choose_mat < 0.8f) {
        Texture* tx = new Constant_Texture(rnd(), rnd(), rnd());
        BRDF* mt = new Lambertian(tx);
        list.push(new Sphere(center, 0.2f, mt));
      } else if (choose_mat < 0.95f) {
        Texture* tx = new Constant_Texture(
            0.5f * (1.f + rnd()), 0.5f * (1.f + rnd()), 0.5f * (1.f + rnd()));
        BRDF* mt = new Metal(tx, 0.5f * rnd());
        list.push(new Sphere(center, 0.2f, mt));
      } else {
        Texture* tx1 = new Constant_Texture(1.f);
        Texture* tx2 = new Constant_Texture(rnd(), rnd(), rnd());
        BRDF* mt = new Dielectric(tx1, tx2, 1.5, 0.f);
        list.push(new Sphere(center, 0.2f, mt));
      }
    }
  }

  Texture* tx1 = new Constant_Texture(1.f);
  BRDF* mt0 = new Dielectric(tx1, tx1, 1.5, 0.f);
  list.push(new Sphere(make_float3(4.f, 1.f, 0.f), 1.f, mt0));

  Texture* tx2 = new Constant_Texture(0.4f, 0.2f, 0.1f);
  BRDF* mt2 = new Lambertian(tx2);
  list.push(new Sphere(make_float3(0.f, 1.f, 0.5f), 1.f, mt2));

  Texture* tx3 = new Constant_Texture(0.7f, 0.6f, 0.5f);
  BRDF* mt3 = new Metal(tx3, 0.f);
  list.push(new Sphere(make_float3(-4.f, 1.f, 1.f), 1.f, mt3));

  // transforms list elements, one by one, and adds them to the graph
  list.addElementsTo(group, g_context);
  g_context["world"]->set(group);

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

  auto t1 = std::chrono::system_clock::now();
  auto sceneTime = std::chrono::duration<float>(t1 - t0).count();
  printf("Done assigning scene data, which took %.2f seconds.\n", sceneTime);
}

void MovingSpheres(Context& g_context, int Nx, int Ny) {
  auto t0 = std::chrono::system_clock::now();

  // add light parameters and programs
  Light_Sampler lights;
  Rectangle_PDF rect_pdf(3.f, 5.f, 1.f, 3.f, -0.5f, Z_AXIS);
  lights.pdf.push_back(rect_pdf.createPDF(g_context));
  lights.sample.push_back(rect_pdf.createSample(g_context));
  lights.emissions.push_back(make_float3(4.f));

  // Set the exception, ray generation and miss shader programs
  setRayGenerationProgram(g_context, lights);
  setMissProgram(g_context, CONSTANT);  // dark background
  setExceptionProgram(g_context);

  // Set acceleration structure
  Group group = g_context->createGroup();
  group->setAcceleration(g_context->createAcceleration("Trbvh"));

  // create scene
  Hitable_List list;
  Texture* ck1 = new Constant_Texture(0.2f, 0.3f, 0.1f);
  Texture* ck2 = new Constant_Texture(0.9f, 0.9f, 0.9f);
  Texture* groundTx = new Checker_Texture(ck1, ck2);
  BRDF* ground = new Lambertian(groundTx);
  list.push(new Sphere(make_float3(0.f, -1000.f, -1.f), 1000.f, ground));

  // Small spheres
  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      float choose_mat = rnd();
      float3 center = make_float3(a + rnd(), 0.2f, b + rnd());
      float3 center2 = center + make_float3(0.f, 0.5f * rnd(), 0.f);
      if (choose_mat < (1.f / 3)) {
        Texture* mtx = new Constant_Texture(
            0.5f * (1.f + rnd()), 0.5f * (1.f + rnd()), 0.5f * (1.f + rnd()));
        BRDF* lmt = new Lambertian(mtx);
        list.push(new Sphere(center, 0.2f, lmt));
      } else if (choose_mat < (2.f / 3)) {
        Texture* mtx = new Constant_Texture(
            0.5f * (1.f + rnd()), 0.5f * (1.f + rnd()), 0.5f * (1.f + rnd()));
        BRDF* lmt = new Metal(mtx, 0.5f * rnd());
        list.push(new Sphere(center, 0.2f, lmt));
      } else {
        Texture* mtx = new Constant_Texture(
            0.5f * (1.f + rnd()), 0.5f * (1.f + rnd()), 0.5f * (1.f + rnd()));
        BRDF* lmt = new Dielectric(mtx, mtx, 1.5, 0.f);
        list.push(new Sphere(center, 0.2f, lmt));
      }
    }
  }

  // Earth
  Texture* etx = new Image_Texture("../../../assets/other_textures/map.jpg");
  BRDF* emt = new Lambertian(etx);
  list.push(new Sphere(make_float3(-4.f, 1.f, 2.f), 1.f, emt));

  // Glass Sphere
  Texture* gtx1 = new Constant_Texture(1.f);
  Texture* gtx2 = new Constant_Texture(rnd(), rnd(), rnd());
  BRDF* gmt = new Dielectric(gtx1, gtx2, 1.5, 0.f);
  list.push(new Sphere(make_float3(4.f, 1.f, 1.f), 1.f, gmt));

  // 'rusty' Metal Sphere
  Texture* mtx = new Noise_Texture(4.f);
  BRDF* mmt = new Metal(mtx, 0.f);
  list.push(new Sphere(make_float3(0.f, 1.f, 1.5f), 1.f, mmt));

  // Light
  Texture* ltx = new Constant_Texture(4.f);
  BRDF* lmt = new Diffuse_Light(ltx);
  list.push(new AARect(3.f, 5.f, 1.f, 3.f, -0.5f, false, Z_AXIS, lmt));

  // transforms list elements, one by one, and adds them to the graph
  list.addElementsTo(group, g_context);
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

  // add light parameters and programs
  Light_Sampler lights;
  Rectangle_PDF rect_pdf(213.f, 343.f, 227.f, 332.f, 554.f, Y_AXIS);
  lights.pdf.push_back(rect_pdf.createPDF(g_context));
  lights.sample.push_back(rect_pdf.createSample(g_context));
  lights.emissions.push_back(make_float3(7.f));

  /*Sphere_PDF sph_pdf(make_float3(555.f - 100.f, 100.f, 100.f), 40.f);
  lights.pdf.push_back(sph_pdf.createPDF(g_context));
  lights.sample.push_back(sph_pdf.createSample(g_context));
  lights.emissions.push_back(make_float3(7.f));*/

  // Set the exception, ray generation and miss shader programs
  setRayGenerationProgram(g_context, lights);
  setMissProgram(g_context, CONSTANT);  // dark background
  setExceptionProgram(g_context);

  // create scene group
  Group group = g_context->createGroup();
  group->setAcceleration(g_context->createAcceleration("Trbvh"));

  // create textures
  Texture* redTx = new Constant_Texture(0.65f, 0.05f, 0.05f);
  Texture* whiteTx = new Constant_Texture(0.73f);
  Texture* greenTx = new Constant_Texture(0.12f, 0.45f, 0.15f);
  Texture* lightTx = new Constant_Texture(7.f);
  Texture* alumTx = new Constant_Texture(0.8f, 0.85f, 0.88f);
  Texture* pWhiteTx = new Constant_Texture(1.f);
  Texture* pBlackTx = new Constant_Texture(0.f);
  Texture* tx1 = new Constant_Texture(1.f);
  Texture* tx2 = new Constant_Texture(1.f, 1.f, rnd());
  Texture* tx4 = new Constant_Texture(0.f);
  Texture* tx3 = new Constant_Texture(0.4f);
  Texture* glass = new Constant_Texture(0.1f, 0.603f, 0.3f);

  // create materials
  BRDF* redMt = new Lambertian(redTx);
  BRDF* whiteMt = new Lambertian(whiteTx);
  BRDF* greenMt = new Lambertian(greenTx);
  BRDF* lightMt = new Diffuse_Light(lightTx);
  BRDF* alumMt = new Metal(pWhiteTx, 0.0);
  BRDF* glassMt = new Dielectric(pWhiteTx, glass, 1.5f, 0.f);
  BRDF* blackSmokeMt = new Isotropic(pBlackTx);
  BRDF* oren = new Oren_Nayar(whiteTx, 1.f);
  BRDF* mt2 = new Ashikhmin_Shirley(tx1, tx3, 10000, 10);
  BRDF* mt5 = new Torrance_Sparrow(tx1, 0.1f, 0.1f);
  BRDF* mt6 = new Oren_Nayar(tx1, 1.f);

  // create geometries/hitables
  Hitable_List list;
  list.push(new AARect(0.f, 555.f, 0.f, 555.f, 555.f, true, X_AXIS, redMt));
  list.push(new AARect(0.f, 555.f, 0.f, 555.f, 0.f, false, X_AXIS, greenMt));
  list.push(
      new AARect(213.f, 343.f, 227.f, 332.f, 554.f, true, Y_AXIS, lightMt));
  list.push(new AARect(0.f, 555.f, 0.f, 555.f, 555.f, true, Y_AXIS, whiteMt));
  list.push(new AARect(0.f, 555.f, 0.f, 555.f, 0.f, false, Y_AXIS, whiteMt));
  list.push(new AARect(0.f, 555.f, 0.f, 555.f, 555.f, true, Z_AXIS, whiteMt));
  // list.push(new Sphere(make_float3(150.f, 90.f, 150.f), 90.f, glassMt));
  list.push(new Sphere(make_float3(555.f - 150.f, 90.f, 555.f - 150.f), 90.f,
                       alumMt));
  /*list.push(new Sphere(make_float3(555 / 3.f, 90.f, 555 / 2.f), 90.f, mt5));
  list.push(new Sphere(make_float3(2 * 555 / 3.f, 90.f, 555 / 2.f), 90.f,
  mt6));*/

  // Aluminium box
  /*Box box =
      Box(make_float3(0.f), make_float3(165.f, 330.f, 165.f), whiteMt);
  box.translate(make_float3(265.f, 0.f, 295.f));
  box.rotate(15.f, Y_AXIS);
  list.push(&box);*/

  /*list.push(
      new Sphere(make_float3(555.f - 100.f, 100.f, 100.f), 40.f,
     alumMt));*/

  /*Box box2 =
      Box(make_float3(0.f), make_float3(165.f, 165.f, 165.f), whiteMt);
  box2.translate(make_float3(130.f, 0.f, 65.f));
  box2.rotate(-18.f, Y_AXIS);
  list.push(&box2);*/

  Mesh model2 = Mesh("teapot.obj", "../../../assets/teapot/", glassMt);
  model2.scale(make_float3(10.f));
  model2.rotate(180.f, Y_AXIS);
  model2.translate(make_float3(150.f, 0.f, 0.f));
  model2.addTo(group, g_context);

  // transforms list elements, one by one, and adds them to the scene graph
  list.addElementsTo(group, g_context);
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

  Light_Sampler lights;
  Rectangle_PDF rect_pdf(113.f, 443.f, 127.f, 432.f, 554.f, Y_AXIS);
  lights.pdf.push_back(rect_pdf.createPDF(g_context));
  lights.sample.push_back(rect_pdf.createSample(g_context));
  lights.emissions.push_back(make_float3(7.f));

  // Set the exception, ray generation and miss shader programs
  setRayGenerationProgram(g_context, lights);
  setMissProgram(g_context, CONSTANT);  // dark background
  setExceptionProgram(g_context);

  Group group = g_context->createGroup();
  group->setAcceleration(g_context->createAcceleration("Trbvh"));

  Hitable_List list;

  Texture* groundTx = new Constant_Texture(0.48f, 0.83f, 0.53f);
  BRDF* ground = new Lambertian(groundTx);

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
  Texture* lightTx = new Constant_Texture(7.f);
  BRDF* light = new Diffuse_Light(lightTx);
  list.push(new AARect(113.f, 443.f, 127.f, 432.f, 554.f, true, Y_AXIS, light));

  // brown sphere
  float3 center = make_float3(400.f, 400.f, 200.f);
  Texture* brownTx = new Constant_Texture(0.7f, 0.3f, 0.1f);
  BRDF* brown = new Lambertian(brownTx);
  list.push(new Sphere(center, 50.f, brown));

  // glass sphere
  Texture* glassTx1 = new Constant_Texture(1.f);
  BRDF* glass = new Dielectric(glassTx1, glassTx1, 1.5f);
  list.push(new Sphere(make_float3(260.f, 150.f, 45.f), 50.f, glass));

  // metal sphere
  Texture* metalTx = new Constant_Texture(0.8f, 0.8f, 0.9f);
  BRDF* metal = new Metal(metalTx, 10.f);
  list.push(new Sphere(make_float3(0.f, 150.f, 145.f), 50.f, metal));

  // blue sphere
  // glass sphere
  list.push(new Sphere(make_float3(360.f, 150.f, 45.f), 70.f, glass));
  // blue fog
  Texture* blueTx = new Constant_Texture(0.2f, 0.4f, 0.9f);
  BRDF* blueFog = new Isotropic(blueTx);
  list.push(new Volumetric_Sphere(make_float3(360.f, 150.f, 45.f), 70.f, 0.2f,
                                  blueFog));

  // white fog
  BRDF* whiteFog = new Isotropic(glassTx1);
  list.push(new Volumetric_Sphere(make_float3(0.f), 5000.f, 0.0001f, whiteFog));

  // earth
  Texture* etx = new Image_Texture("../../../assets/other_textures/map.jpg");
  BRDF* emt = new Lambertian(etx);
  list.push(new Sphere(make_float3(400.f, 200.f, 400.f), 100.f, emt));

  // Perlin sphere
  Texture* perlinTx = new Noise_Texture(0.1f);
  BRDF* noise = new Lambertian(perlinTx);
  list.push(new Sphere(make_float3(220.f, 280.f, 300.f), 80.f, noise));

  // group of small spheres
  Hitable_List spheres;
  Texture* whiteTx = new Constant_Texture(0.73f);
  BRDF* whiteMt = new Lambertian(whiteTx);
  for (int j = 0; j < 1000; j++) {
    center = make_float3(165 * rnd(), 165 * rnd(), 165 * rnd());
    spheres.push(new Sphere(center, 10.f, whiteMt));
  }
  spheres.translate(make_float3(-100.f, 270.f, 395.f));
  spheres.rotate(15.f, Y_AXIS);
  spheres.addListTo(group, g_context);

  // transforms list elements, one by one, and adds them to the graph
  list.addElementsTo(group, g_context);
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

  // add light parameters and programs
  Light_Sampler lights;

  // Set the exception, ray generation and miss shader programs
  setRayGenerationProgram(g_context, lights);
  setMissProgram(g_context, HDR, "../../../assets/hdr/ennis.hdr");
  /*setMissProgram(g_context, GRADIENT,            // gradient sky pattern
                 make_float3(1.f),               // white
                 make_float3(0.5f, 0.7f, 1.f));  // light blue*/
  setExceptionProgram(g_context);

  // create scene group
  Group group = g_context->createGroup();
  group->setAcceleration(g_context->createAcceleration("Trbvh"));

  // create textures
  Texture* whiteTx = new Constant_Texture(0.73f);
  Texture* blackTx = new Constant_Texture(0.f);
  Texture* alumTx = new Constant_Texture(0.8f, 0.85f, 0.88f);
  Texture* noiseTx = new Noise_Texture(0.01f);
  Texture* blueTx = new Constant_Texture(0.2f, 0.4f, 0.9f);
  Texture* perlinXTx = new Noise_Texture(0.01f, X_AXIS);
  Texture* perlinYTx = new Noise_Texture(0.01f, Y_AXIS);
  Texture* perlinZTx = new Noise_Texture(0.01f, Z_AXIS);
  Texture* pWhiteTx = new Constant_Texture(1.f);
  Texture* glass = new Constant_Texture(0.1f, 0.603f, 0.3f);
  Texture* glassbase = new Constant_Texture(0.2f);

  // create materials
  BRDF* whiteMt = new Lambertian(whiteTx);
  BRDF* blackMt = new Lambertian(blackTx);
  BRDF* alumMt = new Metal(alumTx, 0.0);
  BRDF* normalMt = new Normal_Shader();
  BRDF* shadingMt = new Normal_Shader(true);
  BRDF* perlinXMt = new Lambertian(perlinXTx);
  BRDF* perlinYMt = new Lambertian(perlinYTx);
  BRDF* perlinZMt = new Lambertian(perlinZTx);
  BRDF* whiteIso = new Isotropic(blueTx);

  // create geometries
  Hitable_List list;

  // Test model
  if (modelID == 0) {
    Mesh_List meshList;

    list.push(new Sphere(make_float3(0.f, -400.f, 0.f), 150.f, perlinXMt));

    Texture* tx4 = new Constant_Texture(1.f);
    Texture* tx3 = new Constant_Texture(0.3f);
    BRDF* mt2 = new Torrance_Sparrow(tx4, 0.01f, 0.02f);
    BRDF* mt3 = new Ashikhmin_Shirley(tx3, tx4, 10000, 10000);

    BRDF* glassMt = new Dielectric(glass, glassbase, 1.5f, 0.f);
    Mesh model2 = Mesh("thin.obj", "../../../assets/teapot/", glassMt);
    model2.scale(make_float3(40.f));
    // model2.rotate(-90.f, X_AXIS);
    model2.translate(make_float3(150.f, -600.f, 150.f));
    meshList.push(&model2);

    // meshList.addElementsTo(group, g_context);

    list.push(new AARect(-1000.f, 1000.f, -500.f, 500.f, -600.f, false, Y_AXIS,
                         whiteMt));
  }

  // lucy
  else if (modelID == 1) {
    Mesh model = Mesh("Lucy1M.obj", "../../../assets/lucy/");
    model.scale(make_float3(150.f));
    model.translate(make_float3(0.f, -550.f, 0.f));
    model.addTo(group, g_context);

    list.push(new AARect(-1000.f, 1000.f, -500.f, 500.f, -600.f, false, Y_AXIS,
                         whiteMt));
  }

  // Dragon
  else if (modelID == 2) {
    Mesh model = Mesh("dragon_cubic.obj", "../../../assets/dragon/");
    model.scale(make_float3(350.f));
    model.rotate(180.f, Y_AXIS);
    model.translate(make_float3(0.f, -500.f, 200.f));
    model.addTo(group, g_context);

    list.push(new AARect(-1000.f, 1000.f, -500.f, 500.f, -600.f, false, Y_AXIS,
                         whiteMt));
  }

  // spheres
  else if (modelID == 3) {
    /*list.push(
        new Sphere(make_float3(-350.f, -300.f, 0.f), 150.f, perlinXMt));*/
    list.push(new Sphere(make_float3(0.f, -450.f, 0.f), 150.f, whiteIso));
    /*list.push(
        new Sphere(make_float3(350.f, -300.f, 0.f), 150.f, perlinZMt));*/
    list.push(new AARect(-1000.f, 1000.f, -500.f, 500.f, -600.f, false, Y_AXIS,
                         whiteMt));
  }

  // pie
  else if (modelID == 4) {
    Mesh model = Mesh("pie.obj", "../../../assets/pie/");
    model.scale(make_float3(150.f));
    model.translate(make_float3(0.f, -550.f, 0.f));
    model.addTo(group, g_context);

    list.push(new AARect(-1000.f, 1000.f, -500.f, 500.f, -600.f, false, Y_AXIS,
                         whiteMt));
  }

  // sponza
  else {
    Mesh model = Mesh("sponza.obj", "../../../assets/sponza/");
    model.scale(make_float3(0.5f));
    model.rotate(90.f, Y_AXIS);
    model.translate(make_float3(300.f, 5.f, -400.f));
    model.addTo(group, g_context);
  }

  // transforms list elements, one by one, and adds them to the graph
  list.addElementsTo(group, g_context);
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