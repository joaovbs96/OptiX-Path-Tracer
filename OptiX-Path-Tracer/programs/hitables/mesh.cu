#include "../prd.h"

// Triangle intersection function from McGuire's Graphics Codex:
// http://graphicscodex.com

// Shading Normal interpolation:
// https://computergraphics.stackexchange.com/q/5486
// https://computergraphics.stackexchange.com/q/5006

// mesh buffers
rtDeclareVariable(int, single_mat, , );
rtBuffer<float3, 1> vertex_buffer;      // x3 number of faces
rtBuffer<float3, 1> e_buffer;           // x2 number of faces
rtBuffer<float3, 1> normal_buffer;      // = number of faces
rtBuffer<float2, 1> texcoord_buffer;    // x3 number of faces
rtBuffer<float, 1> material_id_buffer;  // = number of faces

// the implicit state's ray we will intersect against
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

// the attributes we use to communicate between intersection programs and hit
// program
rtDeclareVariable(Hit_Record, hit_rec, attribute hit_rec, );

// the per ray data we operate on
rtDeclareVariable(PerRayData, prd, rtPayload, );

// Program that performs the ray-mesh intersection
RT_PROGRAM void mesh_intersection(int index) {
  // int index = index_buffer[pid];
  float3 a = vertex_buffer[3 * index];
  float3 b = vertex_buffer[3 * index + 1];
  float3 c = vertex_buffer[3 * index + 2];

  float3 e1 = e_buffer[2 * index];
  float3 e2 = e_buffer[2 * index + 1];

  float2 a_uv = texcoord_buffer[3 * index];
  float2 b_uv = texcoord_buffer[3 * index + 1];
  float2 c_uv = texcoord_buffer[3 * index + 2];

  float3 pvec = cross(ray.direction, e2);
  float aNum(dot(pvec, e1));

  // Backfacing / nearly parallel, or close to the limit of precision ?
  if (abs(aNum) < 1E-8) return;

  float3 tvec = ray.origin - a;
  float u(dot(pvec, tvec) / aNum);
  if (u < 0.0 || u > 1.0) return;

  float3 qVec = cross(tvec, e1);
  float v(dot(qVec, ray.direction) / aNum);
  if (v < 0.0 || u + v > 1.0) return;

  float t(dot(qVec, e2) / aNum);
  if (t < ray.tmax && t > ray.tmin) {
    if (rtPotentialIntersection(t)) {
      hit_rec.distance = t;

      float3 hit_point = ray.origin + t * ray.direction;
      hit_point = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);
      hit_rec.p = hit_point;

      // Interpolate shading normal
      float3 a_n = normal_buffer[3 * index];
      float3 b_n = normal_buffer[3 * index + 1];
      float3 c_n = normal_buffer[3 * index + 2];
      float3 normal = (a_n * (1.0 - u - v) + b_n * u + c_n * v);

      hit_rec.normal =
          optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));

      hit_rec.u = (a_uv.x * (1.0 - u - v) + b_uv.x * u + c_uv.x * v);
      hit_rec.v = (a_uv.y * (1.0 - u - v) + b_uv.y * u + c_uv.y * v);

      if (single_mat)
        hit_rec.index = 0;
      else
        hit_rec.index = (int)material_id_buffer[index];

      rtReportIntersection(hit_rec.index);
    }
  }
}

// returns the bounding box of the pid'th primitive in this geometry.
RT_PROGRAM void mesh_bounds(int index, float result[6]) {
  optix::Aabb* aabb = (optix::Aabb*)result;
  float3 a = vertex_buffer[3 * index];
  float3 b = vertex_buffer[3 * index + 1];
  float3 c = vertex_buffer[3 * index + 2];

  float3 e1 = e_buffer[2 * index];
  float3 e2 = e_buffer[2 * index + 1];
  const float area = length(cross(e1, e2));

  if (area > 0.0f && !isinf(area)) {
    aabb->m_min = fminf(fminf(a, b), c);
    aabb->m_max = fmaxf(fmaxf(a, b), c);
  } else {
    aabb->invalidate();
  }
}
