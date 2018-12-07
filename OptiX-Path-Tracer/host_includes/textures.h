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
extern "C" const char embedded_checker_texture_programs[];

struct Texture {
    virtual optix::Program assignTo(optix::GeometryInstance gi, optix::Context &g_context) const = 0;
};

struct Constant_Texture : public Texture{
    Constant_Texture(const vec3f &c) : color(c) {}
    
    virtual optix::Program assignTo(optix::GeometryInstance gi, optix::Context &g_context) const override {
        optix::Program textProg = g_context->createProgramFromPTXString(embedded_constant_texture_programs, "sample_texture");
        
        textProg["color"]->set3fv(&color.x);
        gi["sample_texture"]->setProgramId(textProg);

        return textProg;
    }
    
    const vec3f color;
};

struct Checker_Texture : public Texture{
    Checker_Texture(const Texture *o, const Texture *e) : odd(o), even(e) {}
    
    virtual optix::Program assignTo(optix::GeometryInstance gi, optix::Context &g_context) const override {
        optix::Program textProg = g_context->createProgramFromPTXString(embedded_checker_texture_programs, "sample_texture");

        // this defines how the secondary texture programs will be named
        // in thecheckered program context. They might be named "sample_texture"
        // in their own source code and context, but will be invoked by the names 
        // defined here.
        textProg["odd"]->setProgramId(odd->assignTo(gi, g_context));
        textProg["even"]->setProgramId(even->assignTo(gi, g_context));
        
        // this "replaces" the previous gi->setProgramId assgined to the geometry
        // by the "odd" and "even" assignTo() calls. In practice, this assigns the
        // actual checkered_program sample_texture to the material.
        gi["sample_texture"]->setProgramId(textProg);

        return textProg;
    }
    
    const Texture* odd;
    const Texture* even;
};

#endif