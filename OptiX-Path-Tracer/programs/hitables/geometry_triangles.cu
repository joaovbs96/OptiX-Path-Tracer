
#include "../prd.cuh"
#include "hitables.cuh"

// Triangle Parameters
rtBuffer<float3> vertex_buffer;
rtBuffer<float3> normal_buffer;
rtBuffer<float2> texcoord_buffer;
rtBuffer<int3> index_buffer;
rtBuffer<int> material_buffer;

rtDeclareVariable(int, geo_index, attribute geo_index, );
rtDeclareVariable(float2, bc, attribute bc, );

// Attribute Program (for GeometryTriangles)
RT_PROGRAM void TriangleAttributes() {
  geo_index = rtGetPrimitiveIndex();  // texture index
  bc = rtGetTriangleBarycentrics();   // get barycentric coordinates
}

RT_CALLABLE_PROGRAM HitRecord Get_HitRecord(int index,    // primitive index
                                            Ray ray,      // current ray
                                            float t_hit,  // intersection dist
                                            float2 bc) {  // barycentrics
  HitRecord rec;

  // view direction
  rec.Wo = normalize(-ray.direction);

  // Triangle Index
  const int3 v_idx = index_buffer[index];

  // Triangle Vertex
  float3 a = vertex_buffer[v_idx.x];
  float3 b = vertex_buffer[v_idx.y];
  float3 c = vertex_buffer[v_idx.z];

  float3 e1 = rtTransformPoint(RT_OBJECT_TO_WORLD, b - a);
  float3 e2 = rtTransformPoint(RT_OBJECT_TO_WORLD, c - a);

  // Triangle Barycentrics
  float b0 = 1.f - bc.x - bc.y, b1 = bc.x, b2 = bc.y;

  // Hit Point
  float3 hit_point = a * b0 + b * b1 + c * b2;
  rec.P = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);

  // Geometric Normal
  float3 Ng = cross(e1, e2);
  float area = length(Ng) / 2.f;
  Ng /= 2.f * area;
  Ng = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normalize(Ng)));
  rec.geometric_normal = Ng;

  // Shading Normal
  if (normal_buffer.size() == 0) {
    rec.shading_normal = Ng;
  } else {
    float3 a_n = normal_buffer[v_idx.x];
    float3 b_n = normal_buffer[v_idx.y];
    float3 c_n = normal_buffer[v_idx.z];

    float3 Ns = a_n * b0 + b_n * b1 + c_n * b2;
    Ns = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, Ns));
    rec.shading_normal = Ns;
  }

  // Texture Coordinates
  if (texcoord_buffer.size() == 0) {
    rec.u = 0.f;
    rec.v = 0.f;
  } else {
    float2 a_uv = texcoord_buffer[v_idx.x];
    float2 b_uv = texcoord_buffer[v_idx.y];
    float2 c_uv = texcoord_buffer[v_idx.z];

    rec.u = a_uv.x * b0 + b_uv.x * b1 + c_uv.x * b2;
    rec.v = a_uv.y * b0 + b_uv.y * b1 + c_uv.y * b2;
  }

  // Texture Index
  rec.index = material_buffer[index];

  return rec;
}