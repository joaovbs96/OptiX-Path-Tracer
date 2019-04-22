#ifndef PDFSH
#define PDFSH

#include "host_common.hpp"
#include "programs.hpp"

/*! The precompiled programs code (in ptx) that our cmake script
will precompile (to ptx) and link to the generated executable */
extern "C" const char Rect_PDF_PTX[];
extern "C" const char Sphere_PDF_PTX[];

struct PDF {
  virtual Program createSample(Context &g_context) const = 0;
  virtual Program createPDF(Context &g_context) const = 0;
};

struct Rectangle_PDF : public PDF {
  Rectangle_PDF(const float aa0, const float aa1, const float bb0,
                const float bb1, const float kk, const AXIS aax)
      : a0(aa0), a1(aa1), b0(bb0), b1(bb1), k(kk), ax(aax) {}

  virtual Program createSample(Context &g_context) const override {
    Program sample;

    // assign PDF callable program according to given axis
    switch (ax) {
      case X_AXIS:
        sample = createProgram(Rect_PDF_PTX, "Sample_X", g_context);
        break;

      case Y_AXIS:
        sample = createProgram(Rect_PDF_PTX, "Sample_Y", g_context);
        break;

      case Z_AXIS:
        sample = createProgram(Rect_PDF_PTX, "Sample_Z", g_context);
        break;
    }

    // Basic parameters
    sample["a0"]->setFloat(a0);
    sample["a1"]->setFloat(a1);
    sample["b0"]->setFloat(b0);
    sample["b1"]->setFloat(b1);
    sample["k"]->setFloat(k);

    return sample;
  }

  virtual Program createPDF(Context &g_context) const override {
    Program pdf;

    // assign PDF callable program according to given axis
    switch (ax) {
      case X_AXIS:
        pdf = createProgram(Rect_PDF_PTX, "PDF_X", g_context);
        break;

      case Y_AXIS:
        pdf = createProgram(Rect_PDF_PTX, "PDF_Y", g_context);
        break;

      case Z_AXIS:
        pdf = createProgram(Rect_PDF_PTX, "PDF_Z", g_context);
        break;
    }

    // Basic parameters
    pdf["a0"]->setFloat(a0);
    pdf["a1"]->setFloat(a1);
    pdf["b0"]->setFloat(b0);
    pdf["b1"]->setFloat(b1);
    pdf["k"]->setFloat(k);

    return pdf;
  }

  float a0, a1, b0, b1, k;
  AXIS ax;
};

struct Sphere_PDF : public PDF {
  Sphere_PDF(const float3 c, const float r) : center(c), radius(r) {}

  virtual Program createSample(Context &g_context) const override {
    // create PDF generate callable program
    Program sample = createProgram(Sphere_PDF_PTX, "Sample", g_context);

    // Basic parameters
    sample["center"]->setFloat(center.x, center.y, center.z);
    sample["radius"]->setFloat(radius);

    return sample;
  }

  virtual Program createPDF(Context &g_context) const override {
    // create PDF value callable program
    Program pdf = createProgram(Sphere_PDF_PTX, "PDF", g_context);

    // Basic parameters
    pdf["center"]->setFloat(center.x, center.y, center.z);
    pdf["radius"]->setFloat(radius);

    return pdf;
  }

  float radius;
  float3 center;
};

#endif