#ifndef TEXTURESH
#define TEXTURESH

#include <optix.h>
#include <optixu/optixpp.h>

#include "../programs/vec.h"

/*! the precompiled programs/raygen.cu code (in ptx) that our
  cmake magic will precompile (to ptx) and link to the generated
  executable (ie, we can simply declare and use this here as
  'extern'.  */
extern "C" const char embedded_constant_texture_programs[];

struct Texture {
    virtual void assignTo(optix::GeometryInstance gi, optix::Context &g_context) const = 0;
};

struct Constant_Texture : public Texture{
    Constant_Texture(const vec3f &c) : color(c) {}
    
    virtual void assignTo(optix::GeometryInstance gi, optix::Context &g_context) const override {
        optix::Program textProg = g_context->createProgramFromPTXString(embedded_constant_texture_programs, "sample_texture");
        
        textProg["color"]->set3fv(&color.x);
        gi["sample_texture"]->setProgramId(textProg);
    }
    
    const vec3f color;
};

#endif