#ifndef MATERIALSH
#define MATERIALSH

// materials.hpp: define host-side material related classes and functions

#include "host_common.hpp"
#include "textures.hpp"

/*! The precompiled programs code (in ptx) that our cmake script
will precompile (to ptx) and link to the generated executable */
extern "C" const char Metal_PTX[];
extern "C" const char Dielectric_PTX[];
extern "C" const char Lambertian_PTX[];
extern "C" const char Light_PTX[];
extern "C" const char Isotropic_PTX[];
extern "C" const char Normal_PTX[];
extern "C" const char Ashikhmin_PTX[];
extern "C" const char Oren_Nayar_PTX[];
extern "C" const char Torrance_PTX[];
extern "C" const char Hit_PTX[];

//////////////////////////////////
//  Host-side Material Classes  //
//////////////////////////////////

// Creates base Host Material class
struct BRDF {
  virtual Material assignTo(Context &g_context) const = 0;

  // Creates device material object
  static Material createMaterial(Program &closest,      // cloests hit program
                                 Program &any,          // any hit program
                                 Context &g_context) {  // context object
    Material mat = g_context->createMaterial();
    mat->setClosestHitProgram(0, closest);
    mat->setAnyHitProgram(1, any);

    return mat;
  }
};

// Create Lambertian material
struct Lambertian : public BRDF {
  Lambertian(const Texture *t) : texture(t) {}

  // Assign host side Lambertian material to device Material object
  virtual Material assignTo(Context &g_context) const override {
    // Creates closest hit programs and assigns variables and textures
    Program hit = createProgram(Lambertian_PTX, "closest_hit", g_context);
    hit["sample_texture"]->setProgramId(texture->assignTo(g_context));

    Program any = createProgram(Hit_PTX, "any_hit", g_context);

    return createMaterial(hit, any, g_context);
  }

  const Texture *texture;
};

// Create Metal material
struct Metal : public BRDF {
  Metal(const Texture *t, const float fuzz) : texture(t), fuzz(fuzz) {}

  // Assign host side Metal material to device Material object
  virtual Material assignTo(Context &g_context) const override {
    // Creates closest hit programs and assigns variables and textures
    Program hit = createProgram(Metal_PTX, "closest_hit", g_context);
    hit["sample_texture"]->setProgramId(texture->assignTo(g_context));
    hit["fuzz"]->setFloat(fuzz);

    Program any = createProgram(Hit_PTX, "any_hit", g_context);

    return createMaterial(hit, any, g_context);
  }

  const Texture *texture;
  const float fuzz;
};

// Create Dielectric material
struct Dielectric : public BRDF {
  Dielectric(const Texture *baseTex, const Texture *extTex, const float ref_idx,
             const float density = 0.f)
      : baseTex(baseTex), extTex(extTex), ref_idx(ref_idx), density(density) {}

  // Assign host side Dielectric material to device Material object
  virtual Material assignTo(Context &g_context) const override {
    // Creates closest hit programs and assigns variables and textures
    Program hit = createProgram(Dielectric_PTX, "closest_hit", g_context);
    hit["base_texture"]->setProgramId(baseTex->assignTo(g_context));
    hit["extinction_texture"]->setProgramId(extTex->assignTo(g_context));
    hit["ref_idx"]->setFloat(ref_idx);
    hit["density"]->setFloat(density);

    Program any = createProgram(Hit_PTX, "any_hit", g_context);

    return createMaterial(hit, any, g_context);
  }

  const Texture *baseTex, *extTex;
  const float ref_idx;
  const float density;
};

// Create Diffuse Light material
struct Diffuse_Light : public BRDF {
  Diffuse_Light(const Texture *t) : texture(t) {}

  // Assign host side Diffuse Light material to device Material object
  virtual Material assignTo(Context &g_context) const override {
    // Creates closest hit programs and assigns variables and textures
    Program hit = createProgram(Light_PTX, "closest_hit", g_context);
    hit["sample_texture"]->setProgramId(texture->assignTo(g_context));

    Program any = createProgram(Hit_PTX, "any_hit", g_context);
    any["is_light"]->setInt(true);

    return createMaterial(hit, any, g_context);
  }

  const Texture *texture;
};

// Creates Isotropic material
struct Isotropic : public BRDF {
  Isotropic(const Texture *t) : texture(t) {}

  // Assign host side Isotropic material to device Material object
  virtual Material assignTo(Context &g_context) const override {
    // Creates closest hit programs and assigns variables and textures
    Program hit = createProgram(Isotropic_PTX, "closest_hit", g_context);
    hit["sample_texture"]->setProgramId(texture->assignTo(g_context));

    Program any = createProgram(Hit_PTX, "any_hit", g_context);

    return createMaterial(hit, any, g_context);
  }

  const Texture *texture;
};

// Creates Normal Shader material
struct Normal_Shader : public BRDF {
  Normal_Shader(const bool useShadingNormal = false)
      : useShadingNormal(useShadingNormal) {}

  // Assign host side Normal Shader material to device Material object
  virtual Material assignTo(Context &g_context) const override {
    // Creates closest hit programs and assigns variables
    Program hit = createProgram(Normal_PTX, "closest_hit", g_context);
    hit["useShadingNormal"]->setInt(useShadingNormal);

    Program any = createProgram(Hit_PTX, "any_hit", g_context);

    return createMaterial(hit, any, g_context);
  }

  const bool useShadingNormal;
};

// Creates Ashikhmin_Shirley Anisotropic material
struct Ashikhmin_Shirley : public BRDF {
  Ashikhmin_Shirley(const Texture *diffuse_tex, const Texture *specular_tex,
                    const float nu, const float nv)
      : diffuse_tex(diffuse_tex), specular_tex(specular_tex), nu(nu), nv(nv) {}

  // Assign host side Anisotropic material to device Material object
  virtual Material assignTo(Context &g_context) const override {
    // Creates closest hit programs and assigns variables and textures
    Program hit = createProgram(Ashikhmin_PTX, "closest_hit", g_context);
    hit["diffuse_color"]->setProgramId(diffuse_tex->assignTo(g_context));
    hit["specular_color"]->setProgramId(specular_tex->assignTo(g_context));
    hit["nu"]->setFloat(fmaxf(1.f, nu));
    hit["nv"]->setFloat(fmaxf(1.f, nv));

    Program any = createProgram(Hit_PTX, "any_hit", g_context);

    return createMaterial(hit, any, g_context);
  }

  float roughnessToAlpha(float roughness) const {
    return 2.0f / (roughness * roughness) - 2.0f;
  }

  const Texture *specular_tex;
  const Texture *diffuse_tex;
  const float nu, nv;
};

// Creates Oren-Nayar material
struct Oren_Nayar : public BRDF {
  Oren_Nayar(const Texture *t, const float R) : texture(t) {
    // converts roughness to internal parameters
    float sigma = saturate(R);  // [0, 1]
    float div = 1.f / (PI_F + ((3.f * PI_F - 4.f) / 6.f) * sigma);
    rA = 1.0f * div;
    rB = sigma * div;
  }

  // Assign host side Oren-Nayar material to device Material object
  virtual Material assignTo(Context &g_context) const override {
    // Creates closest hit programs and assigns variables and textures
    Program hit = createProgram(Oren_Nayar_PTX, "closest_hit", g_context);
    hit["sample_texture"]->setProgramId(texture->assignTo(g_context));
    hit["rA"]->setFloat(rA);
    hit["rB"]->setFloat(rB);

    Program any = createProgram(Hit_PTX, "any_hit", g_context);

    return createMaterial(hit, any, g_context);
  }

  const Texture *texture;
  float rA, rB;
};

// Creates Torrance-Sparrow material
struct Torrance_Sparrow : public BRDF {
  Torrance_Sparrow(const Texture *texture, const float nu, const float nv)
      : texture(texture), nu(nu), nv(nv) {}

  // Assign host side Torrance-Sparrow material to device Material object
  virtual Material assignTo(Context &g_context) const override {
    // Creates closest hit programs and assigns variables and textures
    Program hit = createProgram(Torrance_PTX, "closest_hit", g_context);
    hit["sample_texture"]->setProgramId(texture->assignTo(g_context));
    hit["nu"]->setFloat(roughnessToAlpha(nu));
    hit["nv"]->setFloat(roughnessToAlpha(nv));

    Program any = createProgram(Hit_PTX, "any_hit", g_context);

    return createMaterial(hit, any, g_context);
  }

  float roughnessToAlpha(float roughness) const {
    float R = fmaxf(roughness, 1e-3f);
    return R * R;
  }

  const Texture *texture;
  const float nu, nv;
};

#endif