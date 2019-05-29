#pragma once

#include "../prd.cuh"
#include "hitables.cuh"

// OptiX Context objects
rtDeclareVariable(Ray, ray, rtCurrentRay, );

// TODO: make callable program parameter functions and call them from the
// closest_hit programs. Put the callable programs in
// their respective .cu files. Assign the programs to the geometryinstances.

/*
// Surface Parameters Callable Programs
typedef rtCallableProgramId<void(float3&,         // P
                                 float3&,         // Wo
                                 float3&,         // Ns
                                 float3&,         // NG
                                 float&,          // t
                                 float&, float&,  // texcoords
                                 int&,            // index
                                 HitRecord)>
    Surface_Function;
*/

// Triangle Parameters
rtBuffer<float3> vertex_buffer;
rtBuffer<float3> normal_buffer;
rtBuffer<float2> texcoord_buffer;
rtBuffer<int3> index_buffer;
rtBuffer<int> material_buffer;

RT_CALLABLE_PROGRAM void Triangle_Parameters(
    float3& P,   // hit point
    float3& Wo,  // view direction
    float3& Ns,  // surface shading normal
    float3& Ng,  // surface geometric normal
    float& t,    // hit distance
    float& u,
    float& v,       // surface coordinates
    int& matIndex,  // material index
    HitRecord hit_rec) {
  // view direction
  Wo = normalize(-ray.direction);

  // Triangle Index
  const int3 v_idx = index_buffer[hit_rec.index];

  // Triangle Vertex
  float3 a = vertex_buffer[v_idx.x];
  float3 b = vertex_buffer[v_idx.y];
  float3 c = vertex_buffer[v_idx.z];

  float3 e1 = rtTransformPoint(RT_OBJECT_TO_WORLD, b - a);
  float3 e2 = rtTransformPoint(RT_OBJECT_TO_WORLD, c - a);

  // Geometric Normal
  Ng = cross(e1, e2);
  float area = length(Ng) / 2.f;
  Ng /= 2.f * area;
  Ng = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normalize(Ng)));

  // Triangle Barycentrics
  float2 bc = hit_rec.bc;
  float b0 = 1.f - bc.x - bc.y, b1 = bc.x, b2 = bc.y;

  // Hit Point
  float3 hit_point = a * b0 + b * b1 + c * b2;
  P = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);

  // Hit Distance
  t = length(P - ray.origin);

  // Shading Normal
  if (normal_buffer.size() == 0) {
    Ns = Ng;
  } else {
    float3 a_n = normal_buffer[v_idx.x];
    float3 b_n = normal_buffer[v_idx.y];
    float3 c_n = normal_buffer[v_idx.z];

    Ns = a_n * b0 + b_n * b1 + c_n * b2;
    Ns = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, Ns));
  }

  // Texture Coordinates
  if (texcoord_buffer.size() == 0) {
    u = 0.f;
    v = 0.f;
  } else {
    float2 a_uv = texcoord_buffer[v_idx.x];
    float2 b_uv = texcoord_buffer[v_idx.y];
    float2 c_uv = texcoord_buffer[v_idx.z];

    u = a_uv.x * b0 + b_uv.x * b1 + c_uv.x * b2;
    v = a_uv.y * b0 + b_uv.y * b1 + c_uv.y * b2;
  }

  // Material Index
  matIndex = material_buffer[hit_rec.index];
}