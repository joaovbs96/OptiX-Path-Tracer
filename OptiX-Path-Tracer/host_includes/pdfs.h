#ifndef PDFSH
#define PDFSH

#include <optix.h>
#include <optixu/optixpp.h>
#include <random>

#include "../programs/vec.h"

extern "C" const char embedded_rect_pdf_programs[];
//extern "C" const char embedded_sphere_pdf_programs[];
extern "C" const char embedded_cosine_pdf_programs[];
extern "C" const char embedded_mixture_pdf_programs[];

struct PDF {
    virtual optix::Program assignGenerate(optix::Context &g_context) const = 0;
    virtual optix::Program assignValue(optix::Context &g_context) const = 0;
};

struct Cosine_PDF : public PDF{
    Cosine_PDF() {}

    virtual optix::Program assignGenerate(optix::Context &g_context) const override {
        optix::Program generate = g_context->createProgramFromPTXString(embedded_cosine_pdf_programs, "cosine_generate");
        
        return generate;
    }

    virtual optix::Program assignValue(optix::Context &g_context) const override {
        optix::Program value = g_context->createProgramFromPTXString(embedded_cosine_pdf_programs, "cosine_value");

        return value;
    }
};

struct Rect_Y_PDF : public PDF {
    Rect_Y_PDF(const float aa0, const float aa1, const float bb0, const float bb1, const float kk)
                : a0(aa0), a1(aa1), b0(bb0), b1(bb1), k(kk) {}

    virtual optix::Program assignGenerate(optix::Context &g_context) const override {
        optix::Program generate = g_context->createProgramFromPTXString(embedded_rect_pdf_programs, "rect_y_generate");

        generate["a0"]->setFloat(a0);
        generate["a1"]->setFloat(a1);
        generate["b0"]->setFloat(b0);
        generate["b1"]->setFloat(b1);
        generate["k"]->setFloat(k);

        return generate;
    }

    virtual optix::Program assignValue(optix::Context &g_context) const override {
        optix::Program value = g_context->createProgramFromPTXString(embedded_rect_pdf_programs, "rect_y_value");

        value["a0"]->setFloat(a0);
        value["a1"]->setFloat(a1);
        value["b0"]->setFloat(b0);
        value["b1"]->setFloat(b1);
        value["k"]->setFloat(k);
        
        return value;
    }

    float a0, a1, b0, b1, k;
};

struct Mixture_PDF : public PDF {
    Mixture_PDF(const PDF *p00, const PDF *p11) : p0(p00), p1(p11) {}

    virtual optix::Program assignGenerate(optix::Context &g_context) const override {
        optix::Program generate = g_context->createProgramFromPTXString(embedded_mixture_pdf_programs, "mixture_generate");

        generate["p0_generate"]->setProgramId(p0->assignGenerate(g_context));
        generate["p1_generate"]->setProgramId(p1->assignGenerate(g_context));

        return generate;
    }

    virtual optix::Program assignValue(optix::Context &g_context) const override {
        optix::Program value = g_context->createProgramFromPTXString(embedded_mixture_pdf_programs, "mixture_value");

        value["p0_value"]->setProgramId(p0->assignValue(g_context));
        value["p1_value"]->setProgramId(p1->assignValue(g_context));
        
        return value;
    }

    const PDF* p0;
    const PDF* p1;
};

#endif