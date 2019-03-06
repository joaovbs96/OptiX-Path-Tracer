#ifndef PDFSH
#define PDFSH

#include "host_common.hpp"
#include "programs.hpp"

/*! The precompiled programs code (in ptx) that our cmake script
will precompile (to ptx) and link to the generated executable */
extern "C" const char rect_pdf_programs[];
extern "C" const char sphere_pdf_programs[];
extern "C" const char cosine_pdf_programs[];
extern "C" const char mixture_pdf_programs[];
extern "C" const char buffer_pdf_programs[];

struct PDF {
  virtual Program createSample(Context &g_context) const = 0;
  virtual Program createPDF(Context &g_context) const = 0;
};

struct Cosine_PDF : public PDF {
  Cosine_PDF() {}

  // create PDF generate callable program
  virtual Program createSample(Context &g_context) const override {
    return createProgram(cosine_pdf_programs, "generate", g_context);
  }

  // create PDF value callable program
  virtual Program createPDF(Context &g_context) const override {
    return createProgram(cosine_pdf_programs, "value", g_context);
  }
};

struct Rectangle_PDF : public PDF {
  Rectangle_PDF(const float aa0, const float aa1, const float bb0,
                const float bb1, const float kk, const AXIS aax)
      : a0(aa0), a1(aa1), b0(bb0), b1(bb1), k(kk), ax(aax) {}

  virtual Program createSample(Context &g_context) const override {
    Program generate;

    // assign PDF callable program according to given axis
    switch (ax) {
      case X_AXIS:
        generate = createProgram(rect_pdf_programs, "generate_x", g_context);
        break;

      case Y_AXIS:
        generate = createProgram(rect_pdf_programs, "generate_y", g_context);
        break;

      case Z_AXIS:
        generate = createProgram(rect_pdf_programs, "generate_z", g_context);
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

  virtual Program createPDF(Context &g_context) const override {
    Program value;

    // assign PDF callable program according to given axis
    switch (ax) {
      case X_AXIS:
        value = createProgram(rect_pdf_programs, "value_x", g_context);
        break;

      case Y_AXIS:
        value = createProgram(rect_pdf_programs, "value_y", g_context);
        break;

      case Z_AXIS:
        value = createProgram(rect_pdf_programs, "value_z", g_context);
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

  virtual Program createSample(Context &g_context) const override {
    // create PDF generate callable program
    Program generate =
        createProgram(mixture_pdf_programs, "generate", g_context);

    // assign children callable programs to mixture program
    generate["p0_generate"]->setProgramId(p0->createSample(g_context));
    generate["p1_generate"]->setProgramId(p1->createSample(g_context));

    return generate;
  }

  virtual Program createPDF(Context &g_context) const override {
    // create PDF value callable program
    Program value = createProgram(mixture_pdf_programs, "value", g_context);

    // assign children callable programs to mixture program
    value["p0_value"]->setProgramId(p0->createPDF(g_context));
    value["p1_value"]->setProgramId(p1->createPDF(g_context));

    return value;
  }

  const PDF *p0;
  const PDF *p1;
};

struct Sphere_PDF : public PDF {
  Sphere_PDF(const float3 c, const float r) : center(c), radius(r) {}

  virtual Program createSample(Context &g_context) const override {
    // create PDF generate callable program
    Program generate =
        createProgram(sphere_pdf_programs, "generate", g_context);

    // Basic parameters
    generate["center"]->setFloat(center.x, center.y, center.z);
    generate["radius"]->setFloat(radius);

    return generate;
  }

  virtual Program createPDF(Context &g_context) const override {
    // create PDF value callable program
    Program value = createProgram(sphere_pdf_programs, "value", g_context);

    // Basic parameters
    value["center"]->setFloat(center.x, center.y, center.z);
    value["radius"]->setFloat(radius);

    return value;
  }

  float radius;
  float3 center;
};

struct Buffer_PDF : public PDF {
  Buffer_PDF(const std::vector<PDF *> &b) : buffer_vector(b) {}

  virtual Program createSample(Context &g_context) const override {
    // create PDF generate callable program
    Program generate =
        createProgram(buffer_pdf_programs, "generate", g_context);

    // create buffer of callable programs
    std::vector<Program> buffer;
    for (int i = 0; i < buffer_vector.size(); i++)
      buffer.push_back(buffer_vector[i]->createSample(g_context));

    generate["generators"]->setBuffer(createBuffer(buffer, g_context));
    generate["size"]->setInt((int)buffer_vector.size());

    return generate;
  }

  virtual Program createPDF(Context &g_context) const override {
    // create PDF value callable program
    Program value = createProgram(buffer_pdf_programs, "value", g_context);

    // create buffer of callable programs
    std::vector<Program> buffer;
    for (int i = 0; i < buffer_vector.size(); i++)
      buffer.push_back(buffer_vector[i]->createPDF(g_context));

    value["values"]->setBuffer(createBuffer(buffer, g_context));
    value["size"]->setInt((int)buffer_vector.size());

    return value;
  }

  std::vector<PDF *> buffer_vector;
};

#endif