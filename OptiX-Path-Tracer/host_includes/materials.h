#ifndef MATERIALSH
#define MATERIALSH

#include "../programs/vec.h"
#include "textures.h"

/*! The precompiled programs code (in ptx) that our cmake script
will precompile (to ptx) and link to the generated executable */
extern "C" const char metal_programs[];
extern "C" const char dielectric_programs[];
extern "C" const char lambertian_programs[];
extern "C" const char diffuse_light_programs[];
extern "C" const char isotropic_programs[];

/*! abstraction for a material that can create, and parameterize,
  a newly created GI's material and closest hit program */
struct Materials {
  virtual Program assignTo(GeometryInstance gi, Context &g_context,
                           int index = 0) const = 0;
};

/*! host side code for the "Lambertian" material; the actual
  sampling code is in the programs/lambertian.cu closest hit program */
struct Lambertian : public Materials {
  Lambertian(const Texture *t) : texture(t) {}

  /* create optix material, and assign mat and mat values to geom instance */
  virtual Program assignTo(GeometryInstance gi, Context &g_context,
                           int index = 0) const override {
    Material mat = g_context->createMaterial();

    Program closest = g_context->createProgramFromPTXString(lambertian_programs,
                                                            "closest_hit");
    mat->setClosestHitProgram(0, closest);

    gi->setMaterial(index, mat);
    return texture->assignTo(g_context);
  }
  const Texture *texture;
};

Program Lambertian_PDF(Context &g_context) {
  return g_context->createProgramFromPTXString(lambertian_programs,
                                               "scattering_pdf");
}

/*! host side code for the "Metal" material; the actual
  sampling code is in the programs/metal.cu closest hit program */
struct Metal : public Materials {
  Metal(const Texture *t, const float fuzz) : texture(t), fuzz(fuzz) {}

  /* create optix material, and assign mat and mat values to geom instance */
  virtual Program assignTo(GeometryInstance gi, Context &g_context,
                           int index = 0) const override {
    Material mat = g_context->createMaterial();

    mat->setClosestHitProgram(0, g_context->createProgramFromPTXString(
                                     metal_programs, "closest_hit"));

    gi->setMaterial(index, mat);

    // fuzz <= 1
    if (fuzz < 1.f)
      gi["fuzz"]->setFloat(fuzz);
    else
      gi["fuzz"]->setFloat(1.f);

    return texture->assignTo(g_context);
  }
  const Texture *texture;
  const float fuzz;
};

/*! host side code for the "Dielectric" material; the actual
  sampling code is in the programs/dielectric.cu closest hit program */
struct Dielectric : public Materials {
  Dielectric(const float ref_idx) : ref_idx(ref_idx) {}

  /* create optix material, and assign mat and mat values to geom instance */
  virtual Program assignTo(GeometryInstance gi, Context &g_context,
                           int index = 0) const override {
    Material mat = g_context->createMaterial();

    mat->setClosestHitProgram(0, g_context->createProgramFromPTXString(
                                     dielectric_programs, "closest_hit"));

    gi->setMaterial(index, mat);
    gi["ref_idx"]->setFloat(ref_idx);

    Constant_Texture foo(make_float3(0.f));
    return foo.assignTo(g_context);
  }

  const float ref_idx;
};

/*! host side code for the "Diffuse Light" material; the actual
  sampling code is in the programs/diffuse_light.cu closest hit program */
struct Diffuse_Light : public Materials {
  Diffuse_Light(const Texture *t) : texture(t) {}

  /* create optix material, and assign mat and mat values to geom instance */
  virtual Program assignTo(GeometryInstance gi, Context &g_context,
                           int index = 0) const override {
    Material mat = g_context->createMaterial();

    mat->setClosestHitProgram(0, g_context->createProgramFromPTXString(
                                     diffuse_light_programs, "closest_hit"));

    gi->setMaterial(index, mat);
    return texture->assignTo(g_context);  // return texture callable program to
                                          // add to a buffer outside
  }
  const Texture *texture;
};

Program Diffuse_Light_PDF(Context &g_context) {
  return g_context->createProgramFromPTXString(diffuse_light_programs,
                                               "scattering_pdf");
}

/*! host side code for the "Diffuse Light" material; the actual
  sampling code is in the programs/diffuse_light.cu closest hit program */
struct Isotropic : public Materials {
  Isotropic(const Texture *t) : texture(t) {}

  /* create optix material, and assign mat and mat values to geom instance */
  virtual Program assignTo(GeometryInstance gi, Context &g_context,
                           int index = 0) const override {
    Material mat = g_context->createMaterial();

    mat->setClosestHitProgram(0, g_context->createProgramFromPTXString(
                                     isotropic_programs, "closest_hit"));

    gi->setMaterial(index, mat);
    return texture->assignTo(g_context);
  }
  const Texture *texture;
};

#endif