#include "../prd.h"

/*! the parameters that describe each individual triangle geometry */
rtDeclareVariable(float3, a, , );
rtDeclareVariable(float2, a_uv, , );
rtDeclareVariable(float3, b, , );
rtDeclareVariable(float2, b_uv, , );
rtDeclareVariable(float3, c, , );
rtDeclareVariable(float2, c_uv, , );

// precomputed variables
rtDeclareVariable(float3, e1, , );
rtDeclareVariable(float3, e2, , );
rtDeclareVariable(float3, normal, , );

rtDeclareVariable(float,  scale, , );

/*! the implicit state's ray we will intersect against */
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

/*! the attributes we use to communicate between intersection programs and hit program */
rtDeclareVariable(Hit_Record, hit_rec, attribute hit_rec, );

/*! the per ray data we operate on */
rtDeclareVariable(PerRayData, prd, rtPayload, );

// Intersection program from McGuire's Graphics Codex - https://graphicscodex.com/
// Program that performs the ray-triangle intersection
RT_PROGRAM void hit_triangle(int pid) {
  float3 pvec = cross(ray.direction, e2);
	float aNum(dot(pvec, e1));

	// Backfacing / nearly parallel, or close to the limit of precision ?
	if (abs(aNum) < 1E-8)
		return;

	float3 tvec = ray.origin - a;
	float u(dot(pvec, tvec) / aNum);
	if (u < 0.0 || u > 1.0) 
		return;

	float3 qVec = cross(tvec, e1);
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

      hit_rec.normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));

      hit_rec.u = (a_uv.x * (1.0 - u - v) + b_uv.x * u + c_uv.x * v);
      hit_rec.v = (a_uv.y * (1.0 - u - v) + b_uv.y * u + c_uv.y * v);

      hit_rec.index = 0;

      rtReportIntersection(0);
    }
  }

  return;
}

/*! returns the bounding box of the pid'th primitive in this gometry. */
RT_PROGRAM void get_bounds(int pid, float result[6]) {
  optix::Aabb* aabb = (optix::Aabb*)result;

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
