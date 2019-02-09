#ifndef PARSERH
#define PARSERH

#include "../programs/vec.h"
#include "camera.h"
#include "hitables.h"
#include "materials.h"
#include "pdfs.h"
#include "programs.h"
#include "textures.h"
#include "transforms.h"

#include "../lib/RSJparser.tcc"

/*PDF* treatSampling(Context &g_context, RSJobject::iterator &data) {
  RSJobject::iterator it = data->second.as_object().begin();

  std::cout << it->first << std::endl;

  if (data->first == "mixture") {
    return new Mixture_PDF(treatSampling(g_context, it++),
treatSampling(g_context, it));
  }
  
  else if (data->first == "cosine") {
    return new Cosine_PDF;
  }
  

  else if (data->first == "rect") {
    std::string axis = (it++)->second.as<std::string>();
    float a0 = (it++)->second.as<float>();
    float a1 = (it++)->second.as<float>();
    float b0 = (it++)->second.as<float>();
    float b1 = (it++)->second.as<float>();
    float k = it->second.as<float>();

    if(axis == "X") {
      return new Rect_X_PDF(a0, a1, b0, b1, k);
    }
    else if(axis == "Y") {
      return new Rect_Y_PDF(a0, a1, b0, b1, k);
    }
    else if(axis == "Z") {
      return new Rect_Z_PDF(a0, a1, b0, b1, k);
    }
    else{
      printf("Error: Incorrect Rect PDF Axis.\n");
      return NULL;
    }
  }
  






  else if (data->first == "sphere") {
    std::vector<float> center((it++)->second.as_vector<float>());
    float radius = it->second.as<float>();

    return new Sphere_PDF(vec3f(center[0], center[1], center[2]), radius);
  }
  






  else if (data->first == "buffer") {
    std::vector<PDF*> buffer;

    for( ; it != data->second.as_object().end(); it++)
      buffer.push_back(treatSampling(g_context, it++));

    return new Buffer_PDF(buffer);
  }
  






  else {
    printf("Error: Incorrect Sampling PDF or not yet implemented.\n");
    return NULL;
  }
}

Group Parser(Context &g_context, std::string filename) {
  std::ifstream infile(filename);
  std::string contents((std::istreambuf_iterator<char>(infile)),
std::istreambuf_iterator<char>());

  RSJresource data(contents);

  int width = data["width"].as<int>();
  int height = data["height"].as<int>();
  int samples = data["samples"].as<int>();

  RSJobject::iterator sampling = data["sampling_pdfs"].as_object().begin();
  PDF *sampling_pdf = treatSampling(g_context, sampling);
  system("PAUSE");

  exit(0);

  // configure sampling
  std::vector<PDF*> buffer;
  buffer.push_back(new Rect_Y_PDF(213.f, 343.f, 227.f, 332.f, 554.f));
  buffer.push_back(new Sphere_PDF(vec3f(190.f, 90.f, 190.f), 90.f));
  Mixture_PDF mixture(new Cosine_PDF(), new Buffer_PDF(buffer));

  // add material PDFs
  Buffer material_pdfs = g_context->createBuffer(RT_BUFFER_INPUT,
RT_FORMAT_PROGRAM_ID, 2); callableProgramId<int(int)>* f_data =
static_cast<callableProgramId<int(int)>*>(material_pdfs->map()); f_data[
0 ] = callableProgramId<int(int)>(Lambertian_PDF(g_context)->getId());
  f_data[ 1 ] =
callableProgramId<int(int)>(Diffuse_Light_PDF(g_context)->getId());
  material_pdfs->unmap();

  // Set the exception, ray generation and miss shader programs
  setRayGenerationProgram(g_context, mixture, material_pdfs);
  setMissProgram(g_context, DARK);
  setExceptionProgram(g_context);

  Group group = g_context->createGroup();
  group->setAcceleration(g_context->createAcceleration("Trbvh"));

  Materials *red = new Lambertian(new Constant_Texture(vec3f(0.65f, 0.05f,
0.05f))); Materials *white = new Lambertian(new Constant_Texture(vec3f(0.73f,
0.73f, 0.73f))); Materials *green = new Lambertian(new
Constant_Texture(vec3f(0.12f, 0.45f, 0.15f))); Materials *light = new
Diffuse_Light(new Constant_Texture(vec3f(7.f, 7.f, 7.f))); Materials *aluminium
= new Metal(new Constant_Texture(vec3f(0.8f, 0.85f, 0.88f)), 0.0); Materials
*glass = new Dielectric(1.5f);
  //Materials *black_fog = new Isotropic(new Constant_Texture(vec3f(0.f)));
  //Materials *white_fog = new Isotropic(new Constant_Texture(vec3f(1.f)));

  addChild(createXRect(0.f, 555.f, 0.f, 555.f, 555.f, true, *green, g_context),
group, g_context); // left wall addChild(createXRect(0.f, 555.f, 0.f, 555.f,
0.f, false, *red, g_context), group, g_context); // right wall
  addChild(createYRect(213.f, 343.f, 227.f, 332.f, 554.f, true, *light,
g_context), group, g_context); // light addChild(createYRect(0.f, 555.f, 0.f,
555.f, 555.f, true, *white, g_context), group, g_context); // roof
  addChild(createYRect(0.f, 555.f, 0.f, 555.f, 0.f, false, *white, g_context),
group, g_context); // ground addChild(createZRect(0.f, 555.f, 0.f, 555.f, 555.f,
true, *white, g_context), group, g_context); // back walls
  addChild(createSphere(vec3f(190.f, 90.f, 190.f), 90.f, *glass, g_context),
group, g_context);// glass sphere
  






  // big box
  addChild(translate(rotateY(createBox(vec3f(0.f), vec3f(165.f, 330.f, 165.f),
*aluminium, g_context), 15.f, g_context), vec3f(265.f, 0.f, 295.f), g_context),
                                                                                group, g_context);

  // small box
  /*addChild(translate(rotateY(createBox(vec3f(0.f), vec3f(165.f, 165.f, 165.f),
*white, g_context), -18.f, g_context), vec3f(130.f, 0.f, 65.f), g_context),
                                                                                group, g_context);*/

// big box
/*addChild(translate(rotateY(createVolumeBox(vec3f(0.f), vec3f(165.f, 330.f,
  165.f), 0.01f, *black_fog, g_context), 15.f, g_context), vec3f(265.f, 0.f,
  295.f), g_context), group, g_context);

  // small box
  addChild(translate(rotateY(createVolumeBox(vec3f(0.f), vec3f(165.f, 165.f,
  165.f), 0.01f, *white_fog, g_context), -18.f, g_context), vec3f(130.f,
  0.f, 65.f), g_context), group, g_context);*/

// configure camera
/*const vec3f lookfrom(278.f, 278.f, -800.f);
  const vec3f lookat(278.f, 278.f, 0.f);
  const vec3f up(0.f, 1.f, 0.f);
  const float fovy(40.f);
  const float aspect(float(width) / float(height));
  const float aperture(0.f);
  const float dist(10.f);
  Camera camera(lookfrom, lookat, up, fovy, aspect, aperture, dist, 0.0, 1.0);
  camera.set(g_context);

  return group;
}*/

#endif