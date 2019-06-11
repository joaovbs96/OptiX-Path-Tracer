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

// Attribute Program (for GeometryTriangles)
RT_PROGRAM void Attributes() {
  geo_index = rtGetPrimitiveIndex();  // texture index
  bc = rtGetTriangleBarycentrics();   // get barycentric coordinates
}

// Triangle intersection program from McGuire's Graphics Codex
// https://graphicscodex.com/
RT_PROGRAM void Intersect(int pid) {
  // Triangle Index
  const int3 v_idx = index_buffer[pid];

  // Triangle Vertex
  float3 a = vertex_buffer[v_idx.x];
  float3 b = vertex_buffer[v_idx.y];
  float3 c = vertex_buffer[v_idx.z];

  float3 e1 = b - a;
  float3 e2 = c - a;

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
