#ifndef IMAGESAVEHPP
#define IMAGESAVEHPP

#include "gui.hpp"

// Save OptiX output buffer to .PNG file
int Save_PNG(App_State &app, Buffer &buffer) {
  unsigned char *arr;
  arr = (unsigned char *)malloc(app.W * app.H * 3 * sizeof(unsigned char));

  const float4 *cols = (const float4 *)buffer->map();

  for (int j = app.H - 1; j >= 0; j--)
    for (int i = 0; i < app.W; i++) {
      int index = app.W * j + i;
      int pixel_index = 3 * (app.W * j + i);

      // average & gamma correct output color
      float3 col = make_float3(cols[index].x, cols[index].y, cols[index].z);
      col = sqrt(col / float(app.samples));

      // Clamp and convert to [0, 255]
      col = 255.99f * clamp(col, 0.f, 1.f);

      // Copy int values to array
      arr[pixel_index + 0] = (int)col.x;  // R
      arr[pixel_index + 1] = (int)col.y;  // G
      arr[pixel_index + 2] = (int)col.z;  // B
    }

  buffer->unmap();

  // Save .PNG file
  app.fileName += ".png";
  const char *name = (char *)app.fileName.c_str();
  return stbi_write_png(name, app.W, app.H, 3, arr, 0);
}

// Save OptiX output buffer to .HDR file
int Save_HDR(App_State &app, Buffer &buffer) {
  float *arr;
  arr = (float *)malloc(app.W * app.H * 3 * sizeof(float));

  const float4 *cols = (const float4 *)buffer->map();

  for (int j = app.W - 1; j >= 0; j--)
    for (int i = 0; i < app.W; i++) {
      int index = app.W * j + i;
      int pixel_index = 3 * (app.W * j + i);

      // average output color
      float3 col = make_float3(cols[index].x, cols[index].y, cols[index].z);
      col = col / float(app.samples);

      // Apply Reinhard style tone mapping
      // Eq (3) from 'Photographic Tone Reproduction for Digital Images'
      // http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.164.483&rep=rep1&type=pdf
      col = col / (make_float3(1.f) + col);

      arr[pixel_index + 0] = col.x;  // R
      arr[pixel_index + 1] = col.y;  // G
      arr[pixel_index + 2] = col.z;  // B
    }

  buffer->unmap();

  // Save .HDR file
  app.fileName += ".hdr";
  const char *name = (char *)app.fileName.c_str();
  return stbi_write_hdr(name, app.W, app.H, 3, arr);

  return 0;
}

#endif