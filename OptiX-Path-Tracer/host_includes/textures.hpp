#ifndef TEXTURESH
#define TEXTURESH

#include "buffers.hpp"
#include "host_common.hpp"

/*! The precompiled programs code (in ptx) that our cmake script
will precompile (to ptx) and link to the generated executable */
extern "C" const char Color_PTX[];
extern "C" const char Checker_PTX[];
extern "C" const char Noise_PTX[];
extern "C" const char Image_PTX[];
extern "C" const char Gradient_PTX[];
extern "C" const char Vector_Tex_PTX[];

struct Texture {
  virtual Program assignTo(Context &g_context) const = 0;
};

struct Constant_Texture : public Texture {
  Constant_Texture(const float3 &c) : color(c) {}

  Constant_Texture(const float &rgb) : color(make_float3(rgb)) {}

  Constant_Texture(const float &r, const float &g, const float &b)
      : color(make_float3(r, g, b)) {}

  virtual Program assignTo(Context &g_context) const override {
    Program prog = createProgram(Color_PTX, "sample_texture", g_context);

    prog["color"]->set3fv(&color.x);

    return prog;
  }

  const float3 color;
};

struct Checker_Texture : public Texture {
  Checker_Texture(const Texture *o, const Texture *e) : odd(o), even(e) {}

  virtual Program assignTo(Context &g_context) const override {
    Program textProg = createProgram(Checker_PTX, "sample_texture", g_context);

    textProg["odd"]->setProgramId(odd->assignTo(g_context));
    textProg["even"]->setProgramId(even->assignTo(g_context));

    return textProg;
  }

  const Texture *odd;
  const Texture *even;
};

// Creates Perlin Noise Texture, sampled on the chosen axis
struct Noise_Texture : public Texture {
  Noise_Texture(const float s, const AXIS ax = X_AXIS) : scale(s), ax(ax) {}

  virtual float3 unit_float3(float x, float y, float z) const {
    float l = sqrt(x * x + y * y + z * z);
    return make_float3(x / l, y / l, z / l);
  }

  void permute(int *p) const {
    for (int i = 256 - 1; i > 0; i--) {
      int target = int(rnd() * (i + 1));
      int tmp = p[i];

      p[i] = p[target];
      p[target] = tmp;
    }
  }

  void perlin_generate_perm(Buffer &perm_buffer, Context &g_context) const {
    perm_buffer = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT, 256);
    int *perm_map = static_cast<int *>(perm_buffer->map());

    for (int i = 0; i < 256; i++) perm_map[i] = i;
    permute(perm_map);
    perm_buffer->unmap();
  }

  virtual Program assignTo(Context &g_context) const override {
    Buffer ranvec =
        g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 256);
    float3 *ranvec_map = static_cast<float3 *>(ranvec->map());

    for (int i = 0; i < 256; ++i)
      ranvec_map[i] =
          unit_float3(-1 + 2 * rnd(), -1 + 2 * rnd(), -1 + 2 * rnd());
    ranvec->unmap();

    Buffer perm_x, perm_y, perm_z;
    perlin_generate_perm(perm_x, g_context);
    perlin_generate_perm(perm_y, g_context);
    perlin_generate_perm(perm_z, g_context);

    Program textProg = createProgram(Noise_PTX, "sample_texture", g_context);

    textProg["ranvec"]->set(ranvec);
    textProg["perm_x"]->set(perm_x);
    textProg["perm_y"]->set(perm_y);
    textProg["perm_z"]->set(perm_z);
    textProg["scale"]->setFloat(scale);
    textProg["axis"]->setInt(ax);

    return textProg;
  }

  const float scale;
  const AXIS ax;
};

struct Image_Texture : public Texture {
  Image_Texture(const std::string f) : fileName(f) {}

  TextureSampler loadTexture(Context context,
                             const std::string fileName) const {
    int nx, ny, nn;
    unsigned char *tex_data =
        stbi_load((char *)fileName.c_str(), &nx, &ny, &nn, 0);

    if (!tex_data) {
      printf("Image is invalid or hasn't been found.\n");
      system("PAUSE");
      exit(0);
    }

    TextureSampler sampler = context->createTextureSampler();
    sampler->setWrapMode(0, RT_WRAP_REPEAT);
    sampler->setWrapMode(1, RT_WRAP_REPEAT);
    sampler->setWrapMode(2, RT_WRAP_REPEAT);
    sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
    sampler->setMaxAnisotropy(1.f);
    sampler->setMipLevelCount(1u);
    sampler->setArraySize(1u);

    Buffer buffer = context->createBuffer(RT_BUFFER_INPUT,
                                          RT_FORMAT_UNSIGNED_BYTE4, nx, ny);
    unsigned char *buffer_data = static_cast<unsigned char *>(buffer->map());

    for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j) {
        int bindex = (j * nx + i) * 4;
        int iindex = ((ny - j - 1) * nx + i) * nn;

        buffer_data[bindex + 0] = tex_data[iindex + 0];
        buffer_data[bindex + 1] = tex_data[iindex + 1];
        buffer_data[bindex + 2] = tex_data[iindex + 2];

        if (nn == 4)
          buffer_data[bindex + 3] = tex_data[iindex + 3];
        else  // 3-channel images
          buffer_data[bindex + 3] = (unsigned char)1.f;
      }

    buffer->unmap();
    sampler->setBuffer(0u, 0u, buffer);
    sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR,
                               RT_FILTER_NONE);

    return sampler;
  }

  virtual Program assignTo(Context &g_context) const override {
    Program textProg = createProgram(Image_PTX, "sample_texture", g_context);

    textProg["data"]->setTextureSampler(loadTexture(g_context, fileName));

    return textProg;
  }

  const std::string fileName;
};

struct HDR_Texture : public Texture {
  HDR_Texture(const std::string f) : fileName(f) {}

  TextureSampler loadHDRTexture(Context context,
                                const std::string fileName) const {
    TextureSampler sampler = context->createTextureSampler();
    sampler->setWrapMode(0, RT_WRAP_REPEAT);
    sampler->setWrapMode(1, RT_WRAP_REPEAT);
    sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
    sampler->setMaxAnisotropy(1.f);
    sampler->setMipLevelCount(1u);
    sampler->setArraySize(1u);

    HDRImage HDRresult;
    if (!HDRLoader::load((char *)fileName.c_str(), HDRresult)) {
      printf("HDR Image is invalid or hasn't been found.\n");
      system("PAUSE");
      exit(0);
    }

    Buffer buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4,
                                          HDRresult.width, HDRresult.height);
    float *buffer_data = static_cast<float *>(buffer->map());

    for (int i = 0; i < HDRresult.width; i++)
      for (int j = 0; j < HDRresult.height; j++) {
        int bindex = (j * HDRresult.width + i) * 4;
        int iindex = (j * HDRresult.width + i) * 3;

        buffer_data[bindex + 0] = HDRresult.colors[iindex + 0];
        buffer_data[bindex + 1] = HDRresult.colors[iindex + 1];
        buffer_data[bindex + 2] = HDRresult.colors[iindex + 2];
        buffer_data[bindex + 3] = 0.f;
      }

    buffer->unmap();
    sampler->setBuffer(0u, 0u, buffer);
    sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR,
                               RT_FILTER_NONE);

    return sampler;
  }

  virtual Program assignTo(Context &g_context) const override {
    Program textProg = createProgram(Image_PTX, "sample_texture", g_context);

    textProg["data"]->setTextureSampler(loadHDRTexture(g_context, fileName));

    return textProg;
  }

  const std::string fileName;
};

// Gradient Texture
struct Gradient_Texture : public Texture {
  Gradient_Texture(const float3 &cA, const float3 &cB, const float3 &cC)
      : colorA(cA), colorB(cB), colorC(cC) {}

  virtual Program assignTo(Context &g_context) const override {
    Program textProg = createProgram(Gradient_PTX, "sample_texture", g_context);

    textProg["colorA"]->set3fv(&colorA.x);
    textProg["colorB"]->set3fv(&colorB.x);
    textProg["colorC"]->set3fv(&colorC.x);

    return textProg;
  }

  const float3 colorA;
  const float3 colorB;
  const float3 colorC;
};

// TODO: what happens if a vector texture takes a vector texture?
struct Vector_Texture : public Texture {
  Vector_Texture(const std::vector<Texture *> &tv) : texture_vector(tv) {}

  virtual Program assignTo(Context &g_context) const override {
    Program prog = createProgram(Vector_Tex_PTX, "sample_texture", g_context);

    std::vector<Program> programs;
    for (int i = 0; i < texture_vector.size(); i++) {
      programs.push_back(texture_vector[i]->assignTo(g_context));
    }

    prog["size"]->setInt((int)programs.size());
    prog["texture_vector"]->setBuffer(createBuffer(programs, g_context));

    return prog;
  }

  const std::vector<Texture *> texture_vector;
};

// Texture 'container'
struct Texture_List {
  Texture_List() {}

  // Appends a geometry to the list and returns its index
  int push(Texture *t) {
    int index = (int)texList.size();

    texList.push_back(t);

    return index;
  }

  // returns the element of index 'i'
  Texture *operator[](const int i) { return texList[i]; }

  std::vector<Texture *> texList;
};

#endif