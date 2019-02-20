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
extern "C" const char hit_program[];

/*! abstraction for a material that can create, and parameterize,
  a newly created GI's material and closest hit program */
struct Materials {
  virtual Program assignTo(GeometryInstance gi, Context &g_context,
                           int index = 0) const = 0;

  virtual Program getAnyHitProgram(Context &g_context) const {
    Program any = g_context->createProgramFromPTXString(hit_program, "any_hit");
    any["is_light"]->setInt(false);
    return any;
  }
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
    mat->setAnyHitProgram(1, getAnyHitProgram(g_context));

    gi->setMaterial(index, mat);
    return texture->assignTo(g_context);
  }

  static Program Sample(Context &g_context) {
    return g_context->createProgramFromPTXString(lambertian_programs,
                                                 "BRDF_Sample");
  }

  static Program PDF(Context &g_context) {
    return g_context->createProgramFromPTXString(lambertian_programs,
                                                 "BRDF_PDF");
  }

  static Program Evaluate(Context &g_context) {
    return g_context->createProgramFromPTXString(lambertian_programs,
                                                 "BRDF_Evaluate");
  }

  const Texture *texture;
};

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
    mat->setAnyHitProgram(1, getAnyHitProgram(g_context));

    gi->setMaterial(index, mat);

    // fuzz <= 1
    if (fuzz < 1.f)
      gi["fuzz"]->setFloat(fuzz);
    else
      gi["fuzz"]->setFloat(1.f);

    return texture->assignTo(g_context);
  }

  static Program Sample(Context &g_context) {
    return g_context->createProgramFromPTXString(metal_programs, "BRDF_Sample");
  }

  static Program PDF(Context &g_context) {
    return g_context->createProgramFromPTXString(metal_programs, "BRDF_PDF");
  }

  static Program Evaluate(Context &g_context) {
    return g_context->createProgramFromPTXString(metal_programs,
                                                 "BRDF_Evaluate");
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
    mat->setAnyHitProgram(1, getAnyHitProgram(g_context));

    gi->setMaterial(index, mat);
    gi["ref_idx"]->setFloat(ref_idx);

    Constant_Texture foo(make_float3(0.f));
    return foo.assignTo(g_context);
  }

  static Program Sample(Context &g_context) {
    return g_context->createProgramFromPTXString(dielectric_programs,
                                                 "BRDF_Sample");
  }

  static Program PDF(Context &g_context) {
    return g_context->createProgramFromPTXString(dielectric_programs,
                                                 "BRDF_PDF");
  }

  static Program Evaluate(Context &g_context) {
    return g_context->createProgramFromPTXString(dielectric_programs,
                                                 "BRDF_Evaluate");
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
    Program any = g_context->createProgramFromPTXString(hit_program, "any_hit");
    any["is_light"]->setInt(true);
    mat->setAnyHitProgram(1, any);

    gi->setMaterial(index, mat);
    return texture->assignTo(g_context);  // return texture callable program to
                                          // add to a buffer outside
  }

  static Program Sample(Context &g_context) {
    return g_context->createProgramFromPTXString(diffuse_light_programs,
                                                 "BRDF_Sample");
  }

  static Program PDF(Context &g_context) {
    return g_context->createProgramFromPTXString(diffuse_light_programs,
                                                 "BRDF_PDF");
  }

  static Program Evaluate(Context &g_context) {
    return g_context->createProgramFromPTXString(diffuse_light_programs,
                                                 "BRDF_Evaluate");
  }

  const Texture *texture;
};

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
    mat->setAnyHitProgram(1, getAnyHitProgram(g_context));

    gi->setMaterial(index, mat);
    return texture->assignTo(g_context);
  }

  static Program Sample(Context &g_context) {
    return g_context->createProgramFromPTXString(isotropic_programs,
                                                 "BRDF_Sample");
  }

  static Program PDF(Context &g_context) {
    return g_context->createProgramFromPTXString(isotropic_programs,
                                                 "BRDF_PDF");
  }

  static Program Evaluate(Context &g_context) {
    return g_context->createProgramFromPTXString(isotropic_programs,
                                                 "BRDF_Evaluate");
  }

  const Texture *texture;
};

Program getPDFProgram(MaterialType type, Context &g_context) {
  switch (type) {
    case Lambertian_Material:
      return Lambertian::PDF(g_context);

    case Metal_Material:
      return Metal::PDF(g_context);

    case Dielectric_Material:
      return Dielectric::PDF(g_context);

    case Diffuse_Light_Material:
      return Diffuse_Light::PDF(g_context);

    case Isotropic_Material:
      return Isotropic::PDF(g_context);

    default:
      printf("Error: Material doesn't exist.\n");
      exit(0);
  };
}

Program getSampleProgram(MaterialType type, Context &g_context) {
  switch (type) {
    case Lambertian_Material:
      return Lambertian::Sample(g_context);

    case Metal_Material:
      return Metal::Sample(g_context);

    case Dielectric_Material:
      return Dielectric::Sample(g_context);

    case Diffuse_Light_Material:
      return Diffuse_Light::Sample(g_context);

    case Isotropic_Material:
      return Isotropic::Sample(g_context);

    default:
      printf("Error: Material doesn't exist.\n");
      exit(0);
  };
}

Program getEvaluateProgram(MaterialType type, Context &g_context) {
  switch (type) {
    case Lambertian_Material:
      return Lambertian::Evaluate(g_context);

    case Metal_Material:
      return Metal::Evaluate(g_context);

    case Dielectric_Material:
      return Dielectric::Evaluate(g_context);

    case Diffuse_Light_Material:
      return Diffuse_Light::Evaluate(g_context);

    case Isotropic_Material:
      return Isotropic::Evaluate(g_context);

    default:
      printf("Error: Material doesn't exist.\n");
      exit(0);
  };
}

#endif