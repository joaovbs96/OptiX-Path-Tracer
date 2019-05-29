
#include "../prd.cuh"
#include "hitables.cuh"

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(HitRecord, hit_rec, attribute hit_rec, );

// mesh buffers
rtBuffer<float3> vertex_buffer;
rtBuffer<float3> normal_buffer;
rtBuffer<float2> texcoord_buffer;
rtBuffer<uint3> index_buffer;
rtBuffer<int> material_buffer;

// Program that performs the ray-mesh intersection
RT_PROGRAM void intersection(int index) {
  const uint3 v_idx = index_buffer[index];

  float3 a = vertex_buffer[v_idx.x];
  float3 b = vertex_buffer[v_idx.y];
  float3 c = vertex_buffer[v_idx.z];

  float3 e1 = b - a;
  float3 e2 = c - a;

  float3 pvec = cross(ray.direction, e2);
  float aNum = dot(pvec, e1);

  // Backfacing / nearly parallel, or close to the limit of precision ?
  if (abs(aNum) < 1e-8) return;

  float3 tvec = ray.origin - a;
  float u = dot(pvec, tvec) / aNum;
  if (u < 0.f || u > 1.f) return;

  float3 qVec = cross(ray.origin - a, e1);
  float v = dot(qVec, ray.direction) / aNum;
  if (v < 0.f || u + v > 1.f) return;

  float t =
      dot(cross(ray.origin - a, e1), e2) / dot(cross(ray.direction, e2), e1);
  if (t < ray.tmax && t > ray.tmin) {
    if (rtPotentialIntersection(t)) {
      // Camera/View Direction
      hit_rec.Wo = normalize(-ray.direction);

      // Hit Distance
      hit_rec.t = t;

      // Hit Point
      float3 hit_point = ray.origin + t * ray.direction;
      hit_rec.P = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);

      // Geometric Normal
      float3 normal = normalize(cross(e1, e2));
      normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));
      hit_rec.geometric_normal = normal;

      // Shading Normal
      if (normal_buffer.size() == 0) {
        hit_rec.shading_normal = normal;
      } else {
        float3 a_n = normal_buffer[v_idx.x];
        float3 b_n = normal_buffer[v_idx.y];
        float3 c_n = normal_buffer[v_idx.z];

        // interpolate vertex normals
        normal = (a_n * (1.0 - u - v) + b_n * u + c_n * v);
        normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));
        hit_rec.shading_normal = normal;
      }

      // Get texture coordinate, if possible
      if (texcoord_buffer.size() == 0) {
        hit_rec.u = 0.f;
        hit_rec.v = 0.f;
      } else {
        float2 a_uv = texcoord_buffer[v_idx.x];
        float2 b_uv = texcoord_buffer[v_idx.y];
        float2 c_uv = texcoord_buffer[v_idx.z];

        // Get texture coordinates from barycentrics
        hit_rec.u = (a_uv.x * (1.0 - u - v) + b_uv.x * u + c_uv.x * v);
        hit_rec.v = (a_uv.y * (1.0 - u - v) + b_uv.y * u + c_uv.y * v);
      }

      // Get material index
      hit_rec.index = material_buffer.size() == 0 ? 0 : material_buffer[index];

      rtReportIntersection(index);
    }
  }
}

// returns the bounding box of the pid'th primitive in this geometry.
RT_PROGRAM void bounds(int index, float result[6]) {
  const uint3 v_idx = index_buffer[index];
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

  Aabb* aabb = (Aabb*)result;
  aabb->m_min = make_float3(minX - 0.0001f, minY - 0.0001f, minZ - 0.0001f);
  aabb->m_max = make_float3(maxX + 0.0001f, maxY + 0.0001f, maxZ + 0.0001f);
}