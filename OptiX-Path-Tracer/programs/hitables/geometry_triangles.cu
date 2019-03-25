/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "hitables.cuh"

// Original code from the OptiX 6.0 SDK samples

rtDeclareVariable(Ray, ray, rtCurrentRay, );

rtDeclareVariable(HitRecord, hit_rec, attribute hit_rec, );

rtBuffer<float3> vertex_buffer;
rtBuffer<float3> normal_buffer;
rtBuffer<float2> texcoord_buffer;
rtBuffer<int3> index_buffer;
rtBuffer<int> material_buffer;

RT_PROGRAM void attributes() {
  const int3 v_idx = index_buffer[rtGetPrimitiveIndex()];

  hit_rec.view_direction = normalize(-ray.direction);

  // Get triangle vertex
  float3 a = vertex_buffer[v_idx.x];
  float3 b = vertex_buffer[v_idx.y];
  float3 c = vertex_buffer[v_idx.z];

  // Get geometric normal
  float3 e1 = b - a;
  float3 e2 = c - a;
  float3 normal = cross(e1, e2);
  normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normalize(normal)));
  hit_rec.geometric_normal = normal;

  // Get hit distance
  float t = (dot(normal, ray.origin) + dot(normal, e1));
  t /= dot(normal, ray.direction);
  hit_rec.distance = t;

  // Get triangle hit point
  float3 hit_point = ray.origin + t * ray.direction;
  hit_point = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);
  hit_rec.p = hit_point;

  // Get shading normal, if possible
  float2 barycentrics = rtGetTriangleBarycentrics();
  if (normal_buffer.size() == 0) {
    hit_rec.shading_normal = hit_rec.geometric_normal;
  } else {
    normal = normal_buffer[v_idx.y] * barycentrics.x;
    normal += normal_buffer[v_idx.z] * barycentrics.y;
    normal += normal_buffer[v_idx.x] * (1.0f - barycentrics.x - barycentrics.y);
    normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));
    hit_rec.shading_normal = normal;
  }

  // Get texture coordinate, if possible
  if (texcoord_buffer.size() == 0) {
    hit_rec.u = 0.f;
    hit_rec.v = 0.f;
  } else {
    float u = barycentrics.x;
    float v = barycentrics.y;

    float2 a_uv = texcoord_buffer[v_idx.x];
    float2 b_uv = texcoord_buffer[v_idx.y];
    float2 c_uv = texcoord_buffer[v_idx.z];

    hit_rec.u = (a_uv.x * (1.0 - u - v) + b_uv.x * u + c_uv.x * v);
    hit_rec.v = (a_uv.y * (1.0 - u - v) + b_uv.y * u + c_uv.y * v);
  }

  // Get material index
  hit_rec.index = material_buffer[rtGetPrimitiveIndex()];
}