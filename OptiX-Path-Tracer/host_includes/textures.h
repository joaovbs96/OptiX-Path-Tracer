#ifndef TEXTURESH
#define TEXTURESH

#include <random>

#include "../programs/vec.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"

#include "../lib/hdr_reader.h"

// TODO: code cleanup and documentation

float rnd() {
  static std::mt19937 gen(0); //Standard mersenne_twister_engine seeded with rd()
  static std::uniform_real_distribution<float> dis(0.f, 1.f);
  return dis(gen);
}

/*! The precompiled programs code (in ptx) that our cmake script 
will precompile (to ptx) and link to the generated executable */
extern "C" const char embedded_constant_texture_programs[];
extern "C" const char embedded_checker_texture_programs[];
extern "C" const char embedded_noise_texture_programs[];
extern "C" const char embedded_image_texture_programs[];

struct Texture {
  virtual optix::Program assignTo(optix::Context &g_context) const = 0;
};

struct Constant_Texture : public Texture{
  Constant_Texture(const float3 &c) : color(c) {}
  
  virtual optix::Program assignTo(optix::Context &g_context) const override {
    optix::Program textProg = g_context->createProgramFromPTXString(embedded_constant_texture_programs, "sample_texture");
    
    textProg["color"]->set3fv(&color.x);

    return textProg;
  }
  
  const float3 color;
};

struct Checker_Texture : public Texture{
  Checker_Texture(const Texture *o, const Texture *e) : odd(o), even(e) {}
  
  virtual optix::Program assignTo(optix::Context &g_context) const override {
    optix::Program textProg = g_context->createProgramFromPTXString(embedded_checker_texture_programs, "sample_texture");

    textProg["odd"]->setProgramId(odd->assignTo(g_context));
    textProg["even"]->setProgramId(even->assignTo(g_context));
    
    return textProg;
  }
  
  const Texture* odd;
  const Texture* even;
};

struct Noise_Texture : public Texture{
  Noise_Texture(const float s) : scale(s) {}

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

  void perlin_generate_perm(optix::Buffer &perm_buffer, optix::Context &g_context) const {
    perm_buffer = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT, 256);
    int *perm_map = static_cast<int*>(perm_buffer->map());
    
    for (int i = 0; i < 256; i++)
    perm_map[i] = i;
    permute(perm_map);
    perm_buffer->unmap();
  }
    
  virtual optix::Program assignTo(optix::Context &g_context) const override {
    optix::Buffer ranvec = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 256);
    float3 *ranvec_map = static_cast<float3*>(ranvec->map());

    for (int i = 0; i < 256; ++i)
        ranvec_map[i] = unit_float3(-1 + 2 * rnd(), -1 + 2 * rnd(), -1 + 2 * rnd());
    ranvec->unmap();

    optix::Buffer perm_x, perm_y, perm_z;
    perlin_generate_perm(perm_x, g_context);
    perlin_generate_perm(perm_y, g_context);
    perlin_generate_perm(perm_z, g_context);
    
    optix::Program textProg = g_context->createProgramFromPTXString(embedded_noise_texture_programs, "sample_texture");

    textProg["ranvec"]->set(ranvec);
    textProg["perm_x"]->set(perm_x);
    textProg["perm_y"]->set(perm_y);
    textProg["perm_z"]->set(perm_z);
    textProg["scale"]->setFloat(scale);

    return textProg;
  }
    
  const float scale;
};

struct Image_Texture : public Texture{
  Image_Texture(const std::string f) : fileName(f) {}

  optix::TextureSampler loadTexture(optix::Context context, const std::string fileName) const {
    int nx, ny, nn;
    unsigned char *tex_data = stbi_load((char*)fileName.c_str(), &nx, &ny, &nn, 0);

    optix::TextureSampler sampler = context->createTextureSampler();
    sampler->setWrapMode(0, RT_WRAP_REPEAT);
    sampler->setWrapMode(1, RT_WRAP_REPEAT);
    sampler->setWrapMode(2, RT_WRAP_REPEAT);
    sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
    sampler->setMaxAnisotropy(1.0f);
    sampler->setMipLevelCount(1u);
    sampler->setArraySize(1u);
    
    optix::Buffer buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE4, nx, ny);
    unsigned char * buffer_data = static_cast<unsigned char *>(buffer->map());
    
    for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j) {
        int bindex = (j * nx + i) * 4;
        int iindex = ((ny - j - 1) * nx + i) * nn;

        buffer_data[bindex + 0] = tex_data[iindex + 0];
        buffer_data[bindex + 1] = tex_data[iindex + 1];
        buffer_data[bindex + 2] = tex_data[iindex + 2];
        
        if(nn == 4)
          buffer_data[bindex + 3] = tex_data[iindex + 3];
        else//3-channel images
          buffer_data[bindex + 3] = (unsigned char)1.f;
      }

    buffer->unmap();
    sampler->setBuffer(0u, 0u, buffer);
    sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
    
    return sampler;
  }
    
  virtual optix::Program assignTo(optix::Context &g_context) const override {
    optix::Program textProg = g_context->createProgramFromPTXString(embedded_image_texture_programs, "sample_texture");

    textProg["data"]->setTextureSampler(loadTexture(g_context, fileName));

    return textProg;
  }
    
  const std::string fileName;
};

// FIXME: still needs proper tone mapping to be useable?
struct HDR_Texture : public Texture{
  HDR_Texture(const std::string f) : fileName(f) {}

  optix::TextureSampler loadHDRTexture(optix::Context context, const std::string fileName) const {
    optix::TextureSampler sampler = context->createTextureSampler();
    sampler->setWrapMode(0, RT_WRAP_REPEAT);
    sampler->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
    sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
    sampler->setMaxAnisotropy(1.0f);
    sampler->setMipLevelCount(1u);
    sampler->setArraySize(1u);
    
    HdrInfo info;
    unsigned char * tex_data = loadHdr((char*)fileName.c_str(), &info, /*convertToFloat=*/true);

    optix::Buffer buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, info.width, info.height);
    float* buffer_data = static_cast<float*>( buffer->map() );

    for (int i = 0; i < info.width; ++i)
      for (int j = 0; j < info.height; ++j) {
        int bindex = (j * info.width + i) * 4;
        int iindex = ((info.height - j - 1) * info.width + i) * 4;

        buffer_data[bindex + 0] = ((float)tex_data[iindex + 0]);
        buffer_data[bindex + 1] = ((float)tex_data[iindex + 1]);
        buffer_data[bindex + 2] = ((float)tex_data[iindex + 2]);
        buffer_data[bindex + 3] = ((float)tex_data[iindex + 3]);
        std::cout << buffer_data[bindex + 0] << ";"
                  << buffer_data[bindex + 1] << ";"
                  << buffer_data[bindex + 2] << ";"
                  << buffer_data[bindex + 3] << std::endl;
      }

    buffer->unmap();
    sampler->setBuffer(0u, 0u, buffer);
    sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
    
    return sampler;
  }
  
  virtual optix::Program assignTo(optix::Context &g_context) const override {
    optix::Program textProg = g_context->createProgramFromPTXString(embedded_image_texture_programs, "sample_texture");

    textProg["data"]->setTextureSampler(loadHDRTexture(g_context, fileName));

    return textProg;
  }
  
  const std::string fileName;
};

#endif