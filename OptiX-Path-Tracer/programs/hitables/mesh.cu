#include <optix_world.h>
#include "../prd.h"

// mesh buffers
rtDeclareVariable(int, single_mat, , );
rtBuffer<float3, 1>  vertex_buffer; // x3 number of faces
rtBuffer<float3, 1>  e_buffer; // x2 number of faces
rtBuffer<float3, 1>  normal_buffer; // = number of faces
rtBuffer<float2, 1>  texcoord_buffer; // x3 number of faces
rtBuffer<float, 1>  material_id_buffer; // = number of faces

// the implicit state's ray we will intersect against
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

// the attributes we use to communicate between intersection programs and hit program
rtDeclareVariable(Hit_Record, hit_rec, attribute hit_rec, );

// the per ray data we operate on
rtDeclareVariable(PerRayData, prd, rtPayload, );

// Program that performs the ray-mesh intersection
RT_PROGRAM void mesh_intersection(int index) {
  //int index = index_buffer[pid];
  float3 a = vertex_buffer[3 * index];
  float3 b = vertex_buffer[3 * index + 1];
  float3 c = vertex_buffer[3 * index + 2];

  float3 e1 = e_buffer[2 * index];
  float3 e2 = e_buffer[2 * index + 1];

  float2 a_uv = texcoord_buffer[3 * index];
  float2 b_uv = texcoord_buffer[3 * index + 1];
  float2 c_uv = texcoord_buffer[3 * index + 2];

  vec3f pvec(cross(ray.direction, e2));
	float aNum(dot(pvec, e1));

	// Backfacing / nearly parallel, or close to the limit of precision ?
	if (abs(aNum) < 1E-8)
		return;

	vec3f tvec(ray.origin - a);
	float u(dot(pvec, tvec) / aNum);
	if (u < 0.0 || u > 1.0) 
		return;

	vec3f qVec(cross(tvec, e1));
	float v(dot(qVec, ray.direction) / aNum);
	if (v < 0.0 || u + v > 1.0) 
		return;

	float t(dot(qVec, e2) / aNum);
  if (t < ray.tmax && t > ray.tmin) {
    if (rtPotentialIntersection(t)) {
      hit_rec.distance = t;

      float3 hit_point = ray.origin + t * ray.direction;
      hit_point = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);
      hit_rec.p = hit_point;

      hit_rec.normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal_buffer[index]));

      hit_rec.u = (a_uv.x * (1.0 - u - v) + b_uv.x * u + c_uv.x * v);
      hit_rec.v = (a_uv.y * (1.0 - u - v) + b_uv.y * u + c_uv.y * v);

      if(single_mat)
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
  //int index = index_buffer[pid];
  float3 a = vertex_buffer[3 * index];
  float3 b = vertex_buffer[3 * index + 1];
  float3 c = vertex_buffer[3 * index + 2];

  // find min and max iterating through vertices
  // min(minX, minY, minZ)
  float minX = ffmin(ffmin(a.x, b.x), c.x);
  float minY = ffmin(ffmin(a.y, b.y), c.y);
  float minZ = ffmin(ffmin(a.z, b.z), c.z);
    
  // max(maxX, maxY, maxZ)
  float maxX = ffmax(ffmax(a.x, b.x), c.x);
  float maxY = ffmax(ffmax(a.y, b.y), c.y);
  float maxZ = ffmax(ffmax(a.z, b.z), c.z);

  aabb->m_min = make_float3(minX - 0.0001f, minY - 0.0001f, minZ - 0.0001f);
  aabb->m_max = make_float3(maxX + 0.0001f, maxY + 0.0001f, maxZ + 0.0001f);
}
