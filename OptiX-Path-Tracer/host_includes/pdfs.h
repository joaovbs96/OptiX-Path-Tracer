#ifndef PDFSH
#define PDFSH

#include <random>

#include "../programs/vec.h"

/*! The precompiled programs code (in ptx) that our cmake script 
will precompile (to ptx) and link to the generated executable */
extern "C" const char embedded_rect_pdf_programs[];
extern "C" const char embedded_sphere_pdf_programs[];
extern "C" const char embedded_cosine_pdf_programs[];
extern "C" const char embedded_mixture_pdf_programs[];
extern "C" const char embedded_buffer_pdf_programs[];

struct PDF {
    virtual optix::Program assignGenerate(optix::Context &g_context) const = 0;
    virtual optix::Program assignValue(optix::Context &g_context) const = 0;
};

struct Cosine_PDF : public PDF{
    Cosine_PDF() {}

    virtual optix::Program assignGenerate(optix::Context &g_context) const override {
        // create PDF generate callable program
        optix::Program generate = g_context->createProgramFromPTXString(embedded_cosine_pdf_programs, "cosine_generate");
        
        return generate;
    }

    virtual optix::Program assignValue(optix::Context &g_context) const override {
        // create PDF value callable program
        optix::Program value = g_context->createProgramFromPTXString(embedded_cosine_pdf_programs, "cosine_value");

        return value;
    }
};

struct Rectangle_PDF : public PDF {
    Rectangle_PDF(const float aa0, const float aa1, 
                  const float bb0, const float bb1, 
                  const float kk, const AXIS aax) :
                  a0(aa0), a1(aa1), 
                  b0(bb0), b1(bb1), 
                  k(kk), ax(aax) {}

    virtual optix::Program assignGenerate(optix::Context &g_context) const override {
        optix::Program generate;

        // assign PDF callable program according to given axis
        switch(ax) {
          case X_AXIS:
            generate = g_context->createProgramFromPTXString(embedded_rect_pdf_programs, "rect_x_generate");
            break;
          case Y_AXIS:
            generate = g_context->createProgramFromPTXString(embedded_rect_pdf_programs, "rect_y_generate");
            break;
          case Z_AXIS:
            generate = g_context->createProgramFromPTXString(embedded_rect_pdf_programs, "rect_x_generate");
            break;
        }

        // Basic parameters
        generate["a0"]->setFloat(a0);
        generate["a1"]->setFloat(a1);
        generate["b0"]->setFloat(b0);
        generate["b1"]->setFloat(b1);
        generate["k"]->setFloat(k);

        return generate;
    }

    virtual optix::Program assignValue(optix::Context &g_context) const override {
        optix::Program value;

        // assign PDF callable program according to given axis
        switch(ax) {
          case X_AXIS:
            value = g_context->createProgramFromPTXString(embedded_rect_pdf_programs, "rect_x_value");
            break;
          case Y_AXIS:
            value = g_context->createProgramFromPTXString(embedded_rect_pdf_programs, "rect_y_value");
            break;
          case Z_AXIS:
            value = g_context->createProgramFromPTXString(embedded_rect_pdf_programs, "rect_z_value");
            break;
        }

        // Basic parameters
        value["a0"]->setFloat(a0);
        value["a1"]->setFloat(a1);
        value["b0"]->setFloat(b0);
        value["b1"]->setFloat(b1);
        value["k"]->setFloat(k);
        
        return value;
    }

    float a0, a1, b0, b1, k;
    AXIS ax;
};

struct Mixture_PDF : public PDF {
    Mixture_PDF(const PDF *p00, const PDF *p11) : p0(p00), p1(p11) {}

    virtual optix::Program assignGenerate(optix::Context &g_context) const override {
        // create PDF generate callable program 
        optix::Program generate = g_context->createProgramFromPTXString(embedded_mixture_pdf_programs, "mixture_generate");

        // assign children callable programs to mixture program
        generate["p0_generate"]->setProgramId(p0->assignGenerate(g_context));
        generate["p1_generate"]->setProgramId(p1->assignGenerate(g_context));

        return generate;
    }

    virtual optix::Program assignValue(optix::Context &g_context) const override {
        // create PDF value callable program
        optix::Program value = g_context->createProgramFromPTXString(embedded_mixture_pdf_programs, "mixture_value");

        // assign children callable programs to mixture program
        value["p0_value"]->setProgramId(p0->assignValue(g_context));
        value["p1_value"]->setProgramId(p1->assignValue(g_context));
        
        return value;
    }

    const PDF* p0;
    const PDF* p1;
};

struct Sphere_PDF : public PDF {
    Sphere_PDF(const float3 c, const float r) : center(c), radius(r) {}

    virtual optix::Program assignGenerate(optix::Context &g_context) const override {
        // create PDF generate callable program
        optix::Program generate = g_context->createProgramFromPTXString(embedded_sphere_pdf_programs, "sphere_generate");

        // Basic parameters
        generate["center"]->setFloat(center.x, center.y, center.z);
        generate["radius"]->setFloat(radius);

        return generate;
    }

    virtual optix::Program assignValue(optix::Context &g_context) const override {
        // create PDF value callable program
        optix::Program value = g_context->createProgramFromPTXString(embedded_sphere_pdf_programs, "sphere_value");

        // Basic parameters
        value["center"]->setFloat(center.x, center.y, center.z);
        value["radius"]->setFloat(radius);
        
        return value;
    }

    float radius;
    float3 center;
};

struct Buffer_PDF : public PDF {
    Buffer_PDF(const std::vector<PDF*> &b) : buffer_vector(b) {}

    virtual optix::Program assignGenerate(optix::Context &g_context) const override {
        // create PDF generate callable program
        optix::Program generate = g_context->createProgramFromPTXString(embedded_buffer_pdf_programs, "buffer_generate");

        // create buffer of callable programs
        optix::Buffer pdf_buffer = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, buffer_vector.size());
        optix::callableProgramId<int(int)>* buffer_data = static_cast<optix::callableProgramId<int(int)>*>(pdf_buffer->map());
        
        // assign buffer of PDF callable programs
        for(int i = 0; i < buffer_vector.size(); i++)
            buffer_data[i] = optix::callableProgramId<int(int)>(buffer_vector[i]->assignGenerate(g_context)->getId());
        
        pdf_buffer->unmap();

        // Basic parameters
        generate["generators"]->setBuffer(pdf_buffer);
        generate["size"]->setInt((int)buffer_vector.size());

        return generate;
    }

    virtual optix::Program assignValue(optix::Context &g_context) const override {
        // create PDF value callable program
        optix::Program value = g_context->createProgramFromPTXString(embedded_buffer_pdf_programs, "buffer_value");

        // create buffer of callable programs
        optix::Buffer pdf_buffer = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, buffer_vector.size());
        optix::callableProgramId<int(int)>* buffer_data = static_cast<optix::callableProgramId<int(int)>*>(pdf_buffer->map());
        
        // assign buffer of PDF callable programs
        for(int i = 0; i < buffer_vector.size(); i++)
            buffer_data[i] = optix::callableProgramId<int(int)>(buffer_vector[i]->assignValue(g_context)->getId());
        
        pdf_buffer->unmap();

        // Basic parameters
        value["values"]->setBuffer(pdf_buffer);
        value["size"]->setInt((int)buffer_vector.size());
        
        return value;
    }

    std::vector<PDF*> buffer_vector;
};

#endif