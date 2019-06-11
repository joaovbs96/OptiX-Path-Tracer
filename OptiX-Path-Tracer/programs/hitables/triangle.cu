#include "../prd.cuh"
#include "hitables.cuh"

// OptiX Context objects
rtDeclareVariable(Ray, ray, rtCurrentRay, );

// Intersected Geometry Attributes
rtDeclareVariable(int, geo_index, attribute geo_index, );  // primitive index
rtDeclareVariable(float2, bc, attribute bc, );  // triangle barycentrics

// Triangle Parameters
rtBuffer<float3> vertex_buffer;
rtBuffer<float3> normal_buffer;
rtBuffer<float2> texcoord_buffer;
rtBuffer<int3> index_buffer;
rtBuffer<int> material_buffer;

// Intersection program from McGuire's Graphics Codex -
// https://graphicscodex.com/ Program that performs the ray-triangle
// intersection
RT_PROGRAM void Intersect(int pid) {
  // Triangle Index
  const int3 v_idx = index_buffer[pid];

  // Triangle Vertex
  float3 a = vertex_buffer[v_idx.x];
  float3 b = vertex_buffer[v_idx.y];
  float3 c = vertex_buffer[v_idx.z];
  
  float3 e1 = rtTransformPoint(RT_OBJECT_TO_WORLD, b - a);
  float3 e2 = rtTransformPoint(RT_OBJECT_TO_WORLD, c - a);

  float3 P = cross(ray.direction, e2);
  float A = dot(P, e1);

  // Backfacing / nearly parallel, or close to the limit of precision?
  if (abs(A) < 1E-8) return;

  float3 R = ray.origin - a;
  float u = dot(P, R) / A;
  if (u < 0.0 || u > 1.0) return;

  float3 Q = cross(R, e1);
  float v = dot(Q, ray.direction) / A;
  if (v < 0.0 || u + v > 1.0) return;

  float t = dot(Q, e2) / A;
  if (rtPotentialIntersection(t)) {
    geo_index = pid;
    bc = make_float2(u, v);
    rtReportIntersection(0);
  }
}

/*! returns the bounding box of the pid'th primitive in this gometry. */
RT_PROGRAM void Get_Bounds(int pid, float result[6]) {
  Aabb* aabb = (Aabb*)result;

  // Triangle Vertex
  const int3 v_idx = index_buffer[pid];
  float3 a = vertex_buffer[v_idx.x];
  float3 b = vertex_buffer[v_idx.y];
  float3 c = vertex_buffer[v_idx.z];

  // find min and max iterating through vertices
  // min(minX, minY, minZ)
  float minX = fminf(fminf(a.x, b.x), c.x);
  float minY = fminf(fminf(a.y, b.y), c.y);
  float minZ = fminf(fminf(a.z, b.z), c.z);

  // max(maxX, maxY, maxZ)
  float maxX = fmaxf(fmaxf(a.x, b.x), c.x);
  float maxY = fmaxf(fmaxf(a.y, b.y), c.y);
  float maxZ = fmaxf(fmaxf(a.z, b.z), c.z);

  aabb->m_min = make_float3(minX - 0.0001f, minY - 0.0001f, minZ - 0.0001f);
  aabb->m_max = make_float3(maxX + 0.0001f, maxY + 0.0001f, maxZ + 0.0001f);
}
