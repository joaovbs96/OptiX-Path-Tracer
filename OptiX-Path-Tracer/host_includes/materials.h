#ifndef MATERIALSH
#define MATERIALSH

#include <optix.h>
#include <optixu/optixpp.h>

#include "../programs/vec.h"
#include "textures.h"

/*! the precompiled programs/raygen.cu code (in ptx) that our
  cmake magic will precompile (to ptx) and link to the generated
  executable (ie, we can simply declare and use this here as
  'extern'.  */
extern "C" const char embedded_metal_programs[];
extern "C" const char embedded_dielectric_programs[];
extern "C" const char embedded_lambertian_programs[];
extern "C" const char embedded_diffuse_light_programs[];

/*! abstraction for a material that can create, and parameterize,
  a newly created GI's material and closest hit program */
struct Material {
  virtual void assignTo(optix::GeometryInstance gi, optix::Context &g_context) const = 0;
};

/*! host side code for the "Lambertian" material; the actual
  sampling code is in the programs/lambertian.cu closest hit program */
struct Lambertian : public Material {
  Lambertian(const Texture *t) : texture(t) {}
  
  /* create optix material, and assign mat and mat values to geom instance */
  virtual void assignTo(optix::GeometryInstance gi, optix::Context &g_context) const override {
    optix::Material mat = g_context->createMaterial();
    
    mat->setClosestHitProgram(0, g_context->createProgramFromPTXString
                              (embedded_lambertian_programs, "closest_hit"));

    gi->setMaterial(/*ray type:*/0, mat);
    texture->assignTo(gi, g_context);
  }
  const Texture* texture;
};

/*! host side code for the "Metal" material; the actual
  sampling code is in the programs/metal.cu closest hit program */
struct Metal : public Material {
  Metal(const Texture *t, const float fuzz) : texture(t), fuzz(fuzz) {}
  
  /* create optix material, and assign mat and mat values to geom instance */
  virtual void assignTo(optix::GeometryInstance gi, optix::Context &g_context) const override {
    optix::Material mat = g_context->createMaterial();
    
    mat->setClosestHitProgram(0, g_context->createProgramFromPTXString
                              (embedded_metal_programs, "closest_hit"));
    
    gi->setMaterial(/*ray type:*/0, mat);
    texture->assignTo(gi, g_context);
    
    // fuzz <= 1
    if(fuzz < 1.f)
      gi["fuzz"]->setFloat(fuzz);
    else
      gi["fuzz"]->setFloat(1.f);
  }
  const Texture* texture;
  const float fuzz;
};

/*! host side code for the "Dielectric" material; the actual
  sampling code is in the programs/dielectric.cu closest hit program */
struct Dielectric : public Material {
  Dielectric(const float ref_idx) : ref_idx(ref_idx) {}
  
  /* create optix material, and assign mat and mat values to geom instance */
  virtual void assignTo(optix::GeometryInstance gi, optix::Context &g_context) const override {
    optix::Material mat = g_context->createMaterial();
  
    mat->setClosestHitProgram(0, g_context->createProgramFromPTXString
                              (embedded_dielectric_programs, "closest_hit"));
  
    gi->setMaterial(/*ray type:*/0, mat);
    gi["ref_idx"]->setFloat(ref_idx);
  }
  
  const float ref_idx;
};

/*! host side code for the "Diffuse Light" material; the actual
  sampling code is in the programs/diffuse_light.cu closest hit program */
struct Diffuse_Light : public Material {
  Diffuse_Light(const Texture *t) : texture(t) {}
  
  /* create optix material, and assign mat and mat values to geom instance */
  virtual void assignTo(optix::GeometryInstance gi, optix::Context &g_context) const override {
    optix::Material mat = g_context->createMaterial();
    
    mat->setClosestHitProgram(0, g_context->createProgramFromPTXString
                              (embedded_diffuse_light_programs, "closest_hit"));

    gi->setMaterial(/*ray type:*/0, mat);
    texture->assignTo(gi, g_context);
  }
  const Texture* texture;
};

#endif