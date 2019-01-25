#ifndef TEXTURESH
#define TEXTURESH

#include <optix.h>
#include <optixu/optixpp.h>
#include <random>

#include "../programs/vec.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"

float rnd() {
  // static std::random_device rd;  //Will be used to obtain a seed for the random number engine
  static std::mt19937 gen(0); //Standard mersenne_twister_engine seeded with rd()
  static std::uniform_real_distribution<float> dis(0.f, 1.f);
  return dis(gen);
}

/*! the precompiled programs/raygen.cu code (in ptx) that our
  cmake magic will precompile (to ptx) and link to the generated
  executable (ie, we can simply declare and use this here as
  'extern'.  */
extern "C" const char embedded_constant_texture_programs[];
extern "C" const char embedded_checker_texture_programs[];
extern "C" const char embedded_noise_texture_programs[];
extern "C" const char embedded_image_texture_programs[];

struct Texture {
    virtual optix::Program assignTo(optix::Context &g_context) const = 0;
};

struct Constant_Texture : public Texture{
    Constant_Texture(const vec3f &c) : color(c) {}
    
    virtual optix::Program assignTo(optix::Context &g_context) const override {
        optix::Program textProg = g_context->createProgramFromPTXString(embedded_constant_texture_programs, "sample_texture");
        
        textProg["color"]->set3fv(&color.x);

        return textProg;
    }
    
    const vec3f color;
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
        unsigned char * data = static_cast<unsigned char *>(buffer->map());
        
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                int bindex = (j * nx + i) * 4;
                int iindex = ((ny - j - 1) * nx + i) * nn;

                data[bindex + 0] = tex_data[iindex + 0];
                data[bindex + 1] = tex_data[iindex + 1];
                data[bindex + 2] = tex_data[iindex + 2];
                
                if(nn == 4)
                    data[bindex + 3] = tex_data[iindex + 3];
                else//3-channel images
                    data[bindex + 3] = (unsigned char)1.f;
            }
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

// FIXME: isn't properly working yet. Check HDR loader from sutil.
struct HDR_Texture : public Texture{
    HDR_Texture(const std::string f) : fileName(f) {}

    optix::TextureSampler loadHDRTexture(optix::Context context, const std::string fileName) const {
        optix::TextureSampler sampler = context->createTextureSampler();
        sampler->setWrapMode(0, RT_WRAP_REPEAT);
        sampler->setWrapMode(1, RT_WRAP_REPEAT);
        sampler->setWrapMode(2, RT_WRAP_REPEAT);
        sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
        sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
        sampler->setMaxAnisotropy(1.0f);
        sampler->setMipLevelCount(1u);
        sampler->setArraySize(1u);
        
        int nx, ny, nn;
        float *tex_data = stbi_loadf((char*)fileName.c_str(), &nx, &ny, &nn, 0);

        optix::Buffer buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, nx, ny);
        float* buffer_data = static_cast<float*>( buffer->map() );

        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                int bindex = (j * nx + i) * 4;
                int iindex = ((ny - j - 1) * nx + i) * 4;

                const int HDR_EXPON_BIAS = 128;
                float s = (float)ldexp(1.0, (int(tex_data[iindex + 3]) - (HDR_EXPON_BIAS + 8)));
                s *= 1.0f / 1.f;

                buffer_data[bindex + 0] = (tex_data[iindex + 0] + 0.5f)*s;
                buffer_data[bindex + 1] = (tex_data[iindex + 1] + 0.5f)*s;
                buffer_data[bindex + 2] = (tex_data[iindex + 2] + 0.5f)*s;
                buffer_data[bindex + 3] = tex_data[iindex + 3];
            }
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