#ifndef MATERIALSH
#define MATERIALSH

// materials.hpp: define host-side material related classes and functions

#include "host_common.hpp"
#include "textures.hpp"

/*! The precompiled programs code (in ptx) that our cmake script
will precompile (to ptx) and link to the generated executable */
extern "C" const char metal_programs[];
extern "C" const char dielectric_programs[];
extern "C" const char lambertian_programs[];
extern "C" const char light_programs[];
extern "C" const char isotropic_programs[];
extern "C" const char normal_programs[];
extern "C" const char anisotropic_programs[];
extern "C" const char oren_nayar_programs[];
extern "C" const char torrance_sparrow_programs[];
extern "C" const char disney_programs[];
extern "C" const char hit_program[];

//////////////////////////////////
//  Host-side Material Classes  //
//////////////////////////////////

// Creates base Host Material class
struct Host_Material {
  Host_Material(BRDFType matType) : matType(matType) {}

  virtual Material assignTo(Context &g_context) const = 0;

  virtual void assignParams(Program &program, Context &g_context) const = 0;

  virtual BRDFType type() const { return matType; }

  virtual Program getAnyHitProgram(Context &g_context) const {
    return createProgram(hit_program, "any_hit", g_context);
  }

  static Program getBRDFProgram(Context &g_context, const BRDFType matType,
                                const std::string name) {
    switch (matType) {
      case Lambertian_BRDF:
        return createProgram(lambertian_programs, name, g_context);

      case Metal_BRDF:
        return createProgram(metal_programs, name, g_context);

      case Diffuse_Light_BRDF:
        return createProgram(light_programs, name, g_context);

      case Isotropic_BRDF:
        return createProgram(isotropic_programs, name, g_context);

      case Dielectric_BRDF:
        return createProgram(dielectric_programs, name, g_context);

      case Normal_BRDF:
        return createProgram(normal_programs, name, g_context);

      case Anisotropic_BRDF:
        return createProgram(anisotropic_programs, name, g_context);

      case Oren_Nayar_BRDF:
        return createProgram(oren_nayar_programs, name, g_context);

      case Torrance_Sparrow_BRDF:
        return createProgram(torrance_sparrow_programs, name, g_context);

      case Disney_BRDF:
        return createProgram(disney_programs, name, g_context);

      default:
        throw "Invalid Material";
    }
  }

  // Creates device material object
  static Material createMaterial(Program &closest, Program &any,
                                 Context &g_context) {
    Material mat = g_context->createMaterial();
    mat->setClosestHitProgram(0, closest);
    mat->setAnyHitProgram(1, any);

    return mat;
  }

  const BRDFType matType;
};

// Create Lambertian material
struct Lambertian : public Host_Material {
  Lambertian(const Texture *t) : texture(t), Host_Material(Lambertian_BRDF) {}

  // Assign host side Lambertian material to device Material object
  virtual Material assignTo(Context &g_context) const override {
    // Creates closest hit programs and assigns variables and textures
    Program hit = createProgram(lambertian_programs, "closest_hit", g_context);
    assignParams(hit, g_context);

    return createMaterial(hit, getAnyHitProgram(g_context), g_context);
  }

  // Assigns variables to Lambertian programs
  virtual void assignParams(Program &program,
                            Context &g_context) const override {
    program["sample_texture"]->setProgramId(texture->assignTo(g_context));
  }

  const Texture *texture;
};

// Create Metal material
struct Metal : public Host_Material {
  Metal(const Texture *t, const float fuzz)
      : texture(t), fuzz(fuzz), Host_Material(Metal_BRDF) {}

  // Assign host side Metal material to device Material object
  virtual Material assignTo(Context &g_context) const override {
    // Creates closest hit programs and assigns variables and textures
    Program hit = createProgram(metal_programs, "closest_hit", g_context);
    assignParams(hit, g_context);

    return createMaterial(hit, getAnyHitProgram(g_context), g_context);
  }

  // Assigns variables to Metal programs
  virtual void assignParams(Program &program,
                            Context &g_context) const override {
    program["sample_texture"]->setProgramId(texture->assignTo(g_context));
    program["fuzz"]->setFloat(fuzz);
  }

  const Texture *texture;
  const float fuzz;
};

// Create Dielectric material
struct Dielectric : public Host_Material {
  Dielectric(const Texture *baseTex, const Texture *volTex, const float ref_idx,
             const float density = 0.f)
      : baseTex(baseTex),
        volTex(volTex),
        ref_idx(ref_idx),
        density(density),
        Host_Material(Dielectric_BRDF) {}

  // Assign host side Dielectric material to device Material object
  virtual Material assignTo(Context &g_context) const override {
    // Creates closest hit programs and assigns variables and textures
    Program hit = createProgram(dielectric_programs, "closest_hit", g_context);
    assignParams(hit, g_context);

    return createMaterial(hit, getAnyHitProgram(g_context), g_context);
  }

  // Assigns variables to Dielectric programs
  virtual void assignParams(Program &program,
                            Context &g_context) const override {
    program["base_texture"]->setProgramId(baseTex->assignTo(g_context));
    program["volume_texture"]->setProgramId(volTex->assignTo(g_context));
    program["ref_idx"]->setFloat(ref_idx);
    program["density"]->setFloat(density);
  }

  const Texture *baseTex, *volTex;
  const float ref_idx;
  const float density;
};

// Create Diffuse Light material
struct Diffuse_Light : public Host_Material {
  Diffuse_Light(const Texture *t)
      : texture(t), Host_Material(Diffuse_Light_BRDF) {}

  // Assign host side Diffuse Light material to device Material object
  virtual Material assignTo(Context &g_context) const override {
    // Creates closest hit programs and assigns variables and textures
    Program hit = createProgram(light_programs, "closest_hit", g_context);
    assignParams(hit, g_context);

    return createMaterial(hit, getAnyHitProgram(g_context), g_context);
  }

  // Get any hit program
  virtual Program getAnyHitProgram(Context &g_context) const override {
    Program any = createProgram(hit_program, "any_hit", g_context);
    any["is_light"]->setInt(true);
    return any;
  }

  // Assigns variables to Diffuse Light programs
  virtual void assignParams(Program &program,
                            Context &g_context) const override {
    program["sample_texture"]->setProgramId(texture->assignTo(g_context));
  }

  const Texture *texture;
};

// Creates Isotropic material
struct Isotropic : public Host_Material {
  Isotropic(const Texture *t) : texture(t), Host_Material(Isotropic_BRDF) {}

  // Assign host side Isotropic material to device Material object
  virtual Material assignTo(Context &g_context) const override {
    // Creates closest hit programs and assigns variables and textures
    Program hit = createProgram(isotropic_programs, "closest_hit", g_context);
    assignParams(hit, g_context);

    return createMaterial(hit, getAnyHitProgram(g_context), g_context);
  }

  // Assigns variables to Isotropic programs
  virtual void assignParams(Program &program,
                            Context &g_context) const override {
    program["sample_texture"]->setProgramId(texture->assignTo(g_context));
  }

  const Texture *texture;
};

// Creates Normal Shader material
struct Normal_Shader : public Host_Material {
  Normal_Shader(const bool useShadingNormal = false)
      : useShadingNormal(useShadingNormal), Host_Material(Normal_BRDF) {}

  // Assign host side Normal Shader material to device Material object
  virtual Material assignTo(Context &g_context) const override {
    // Creates closest hit programs and assigns variables
    Program hit = createProgram(normal_programs, "closest_hit", g_context);
    assignParams(hit, g_context);

    return createMaterial(hit, getAnyHitProgram(g_context), g_context);
  }

  // Assigns variables to Normal Shader programs
  virtual void assignParams(Program &program,
                            Context &g_context) const override {
    program["useShadingNormal"]->setInt(useShadingNormal);
  }

  const bool useShadingNormal;
};

// Creates Anisotropic-Phong material
struct Anisotropic : public Host_Material {
  Anisotropic(const Texture *diffuse_tex, const Texture *specular_tex,
              const float nu, const float nv)
      : diffuse_tex(diffuse_tex),
        specular_tex(specular_tex),
        nu(nu),
        nv(nv),
        Host_Material(Anisotropic_BRDF) {}

  // Assign host side Anisotropic material to device Material object
  virtual Material assignTo(Context &g_context) const override {
    // Creates closest hit programs and assigns variables and textures
    Program hit = createProgram(anisotropic_programs, "closest_hit", g_context);
    assignParams(hit, g_context);

    return createMaterial(hit, getAnyHitProgram(g_context), g_context);
  }

  // Assigns variables to Anisotropic-Phong programs
  virtual void assignParams(Program &program,
                            Context &g_context) const override {
    program["diffuse_color"]->setProgramId(diffuse_tex->assignTo(g_context));
    program["specular_color"]->setProgramId(specular_tex->assignTo(g_context));
    program["nu"]->setFloat(fmaxf(1.f, nu));
    program["nv"]->setFloat(fmaxf(1.f, nv));
  }

  float roughnessToAlpha(float roughness) const {
    return 2.0f / (roughness * roughness) - 2.0f;
  }

  const Texture *specular_tex;
  const Texture *diffuse_tex;
  const float nu, nv;
};

// Creates Oren-Nayar material
struct Oren_Nayar : public Host_Material {
  Oren_Nayar(const Texture *t, const float R)
      : texture(t), Host_Material(Oren_Nayar_BRDF) {
    // converts roughness to internal parameters
    float sigma = saturate(R);  // [0, 1]
    float div = 1.f / (PI_F + ((3.f * PI_F - 4.f) / 6.f) * sigma);
    rA = 1.0f * div;
    rB = sigma * div;
  }

  // Assign host side Oren-Nayar material to device Material object
  virtual Material assignTo(Context &g_context) const override {
    // Creates closest hit programs and assigns variables and textures
    Program hit = createProgram(oren_nayar_programs, "closest_hit", g_context);
    assignParams(hit, g_context);

    return createMaterial(hit, getAnyHitProgram(g_context), g_context);
  }

  // Assigns variables to Anisotropic-Phong programs
  virtual void assignParams(Program &program,
                            Context &g_context) const override {
    program["sample_texture"]->setProgramId(texture->assignTo(g_context));
    program["rA"]->setFloat(rA);  // roughness parameters
    program["rB"]->setFloat(rB);
  }

  const Texture *texture;
  float rA, rB;
};

// Creates Torrance-Sparrow material
struct Torrance_Sparrow : public Host_Material {
  Torrance_Sparrow(const Texture *texture, const float nu, const float nv)
      : texture(texture),
        nu(nu),
        nv(nv),
        Host_Material(Torrance_Sparrow_BRDF) {}

  // Assign host side Torrance-Sparrow material to device Material object
  virtual Material assignTo(Context &g_context) const override {
    // Creates closest hit programs and assigns variables and textures
    Program hit =
        createProgram(torrance_sparrow_programs, "closest_hit", g_context);
    assignParams(hit, g_context);

    return createMaterial(hit, getAnyHitProgram(g_context), g_context);
  }

  // Assigns variables to Torrance-Sparrow programs
  virtual void assignParams(Program &program,
                            Context &g_context) const override {
    program["sample_texture"]->setProgramId(texture->assignTo(g_context));
    program["nu"]->setFloat(roughnessToAlpha(nu));
    program["nv"]->setFloat(roughnessToAlpha(nv));
  }

  float roughnessToAlpha(float roughness) const {
    float R = fmaxf(roughness, 1e-3f);
    return R * R;
  }

  const Texture *texture;
  const float nu, nv;
};

/*// Creates Torrance-Sparrow material
struct Disney : public Host_Material {
  Disney(const Texture *texture, const float nu, const float nv)
      : texture(texture),
        nu(nu),
        nv(nv),
        Host_Material(Disney_BRDF) {}

  // Assign host side Torrance-Sparrow material to device Material object
  virtual Material assignTo(Context &g_context) const override {
    // Creates closest hit programs and assigns variables and textures
    Program hit =
        createProgram(torrance_sparrow_programs, "closest_hit", g_context);
    assignParams(hit, g_context);

    return createMaterial(hit, getAnyHitProgram(g_context), g_context);
  }

  // Assigns variables to Torrance-Sparrow programs
  virtual void assignParams(Program &program,
                            Context &g_context) const override {
    program["sample_texture"]->setProgramId(texture->assignTo(g_context));
    program["nu"]->setFloat(roughnessToAlpha(nu));
    program["nv"]->setFloat(roughnessToAlpha(nv));
  }

  float roughnessToAlpha(float roughness) const {
    float R = fmaxf(roughness, 1e-3f);
    return R * R;
  }

  const Texture *texture;
  const float nu, nv;
};*/

// List of Host_Materials
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

///////////////////////////////////////
//  BRDF Program Creation Functions  //
///////////////////////////////////////

// Returns a BRDF_Sample program object of the given material type
Program getSampleProgram(BRDFType type, Context &g_context) {
  return Host_Material::getBRDFProgram(g_context, type, "BRDF_Sample");
}

// Returns a BRDF_PDF program object of the given material type
Program getPDFProgram(BRDFType type, Context &g_context) {
  return Host_Material::getBRDFProgram(g_context, type, "BRDF_PDF");
}

// Returns a BRDF_Evaluate program object of the given material type
Program getEvaluateProgram(BRDFType type, Context &g_context) {
  return Host_Material::getBRDFProgram(g_context, type, "BRDF_Evaluate");
}

#endif