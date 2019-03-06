#ifndef MATERIALSH
#define MATERIALSH

#include "host_common.hpp"
#include "textures.hpp"

/*! The precompiled programs code (in ptx) that our cmake script
will precompile (to ptx) and link to the generated executable */
extern "C" const char metal_programs[];
extern "C" const char dielectric_programs[];
extern "C" const char lambertian_programs[];
extern "C" const char light_programs[];
extern "C" const char isotropic_programs[];
extern "C" const char hit_program[];

// TODO: add a material parameters in the PDFParams and PRD of the device side
// -> this info will be needed in the BRDF programs

/*! abstraction for a material that can create, and parameterize,
  a newly created GI's material and closest hit program */
struct Host_Material {
  Host_Material(MaterialType matType) : matType(matType) {}

  virtual Material assignTo(Context &g_context) const = 0;

  virtual MaterialType type() const { return matType; }

  virtual Program getAnyHitProgram(Context &g_context) const {
    return createProgram(hit_program, "any_hit", g_context);
  }

  static Program getBRDFProgram(Context &g_context, const MaterialType matType,
                                const std::string name) {
    switch (matType) {
      case Lambertian_Material:
        return createProgram(lambertian_programs, name, g_context);

      case Metal_Material:
        return createProgram(metal_programs, name, g_context);

      case Diffuse_Light_Material:
        return createProgram(light_programs, name, g_context);

      case Isotropic_Material:
        return createProgram(isotropic_programs, name, g_context);

      case Dielectric_Material:
        return createProgram(dielectric_programs, name, g_context);

      default:
        throw "Invalid Material";
    }
  }

  const MaterialType matType;
};

struct Lambertian : public Host_Material {
  Lambertian(const Texture *t)
      : texture(t), Host_Material(Lambertian_Material) {}

  virtual Material assignTo(Context &g_context) const override {
    Material mat = g_context->createMaterial();

    Program hit = createProgram(lambertian_programs, "closest_hit", g_context);
    hit["sample_texture"]->setProgramId(texture->assignTo(g_context));

    mat->setClosestHitProgram(0, hit);
    mat->setAnyHitProgram(1, getAnyHitProgram(g_context));

    return mat;
  }

  const Texture *texture;
};

struct Metal : public Host_Material {
  Metal(const Texture *t, const float fuzz)
      : texture(t), fuzz(fuzz), Host_Material(Metal_Material) {}

  virtual Material assignTo(Context &g_context) const override {
    Material mat = g_context->createMaterial();

    Program hit = createProgram(metal_programs, "closest_hit", g_context);
    hit["sample_texture"]->setProgramId(texture->assignTo(g_context));
    hit["fuzz"]->setFloat(fuzz);

    mat->setClosestHitProgram(0, hit);
    mat->setAnyHitProgram(1, getAnyHitProgram(g_context));

    return mat;
  }

  const Texture *texture;
  const float fuzz;
};

struct Dielectric : public Host_Material {
  Dielectric(const Texture *baseTex, const Texture *volTex, const float ref_idx,
             const float density = 0.f)
      : baseTex(baseTex),
        volTex(volTex),
        ref_idx(ref_idx),
        density(density),
        Host_Material(Dielectric_Material) {}

  virtual Material assignTo(Context &g_context) const override {
    Material mat = g_context->createMaterial();

    Program hit = createProgram(dielectric_programs, "closest_hit", g_context);
    hit["base_texture"]->setProgramId(baseTex->assignTo(g_context));
    hit["volume_texture"]->setProgramId(volTex->assignTo(g_context));
    hit["ref_idx"]->setFloat(ref_idx);
    hit["density"]->setFloat(density);

    mat->setClosestHitProgram(0, hit);
    mat->setAnyHitProgram(1, getAnyHitProgram(g_context));

    return mat;
  }

  const Texture *baseTex, *volTex;
  const float ref_idx;
  const float density;
};

struct Diffuse_Light : public Host_Material {
  Diffuse_Light(const Texture *t)
      : texture(t), Host_Material(Diffuse_Light_Material) {}

  virtual Material assignTo(Context &g_context) const override {
    Material mat = g_context->createMaterial();

    Program hit = createProgram(light_programs, "closest_hit", g_context);
    hit["sample_texture"]->setProgramId(texture->assignTo(g_context));

    mat->setClosestHitProgram(0, hit);
    mat->setAnyHitProgram(1, getAnyHitProgram(g_context));

    return mat;
  }

  virtual Program getAnyHitProgram(Context &g_context) const override {
    Program any = createProgram(hit_program, "any_hit", g_context);
    any["is_light"]->setInt(true);
    return any;
  }

  const Texture *texture;
};

struct Isotropic : public Host_Material {
  Isotropic(const Texture *t) : texture(t), Host_Material(Isotropic_Material) {}

  virtual Material assignTo(Context &g_context) const override {
    Material mat = g_context->createMaterial();

    Program hit = createProgram(isotropic_programs, "closest_hit", g_context);
    hit["sample_texture"]->setProgramId(texture->assignTo(g_context));

    mat->setClosestHitProgram(0, hit);
    mat->setAnyHitProgram(1, getAnyHitProgram(g_context));

    return mat;
  }

  const Texture *texture;
};

Program getSampleProgram(MaterialType type, Context &g_context) {
  return Host_Material::getBRDFProgram(g_context, type, "BRDF_Sample");
}

Program getPDFProgram(MaterialType type, Context &g_context) {
  return Host_Material::getBRDFProgram(g_context, type, "BRDF_PDF");
}

Program getEvaluateProgram(MaterialType type, Context &g_context) {
  return Host_Material::getBRDFProgram(g_context, type, "BRDF_Evaluate");
}

// Material 'container'
struct Material_List {
  Material_List() {}

  // Appends a geometry to the list and returns its index
  int push(Host_Material *m) {
    int index = (int)matList.size();

    matList.push_back(m);

    return index;
  }

  // returns the element of index 'i'
  Host_Material *operator[](const int i) { return matList[i]; }

  std::vector<Host_Material *> matList;
};

#endif