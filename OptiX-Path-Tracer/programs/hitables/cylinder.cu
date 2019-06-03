#include "../prd.cuh"
#include "hitables.cuh"
#include "../math/math_commons.cuh"

//////////////////////
// --- Cylinder --- //
//////////////////////

// http://www.iquilezles.org/www/articles/diskbbox/diskbbox.htm
// https://github.com/spinatelli/raytracer/blob/master/Cylinder.cpp
// http://hugi.scene.org/online/hugi24/coding%20graphics%20chris%20dragan%20raytracing%20shapes.htm

// OptiX Context objects
rtDeclareVariable(Ray, ray, rtCurrentRay, );

// Intersected Geometry Attributes
rtDeclareVariable(int, geo_index, attribute geo_index, );  // primitive index
rtDeclareVariable(float2, bc, attribute bc, );             // triangle barycentrics

// Primitive Parameters
rtDeclareVariable(float3, O, , );     // origin
rtDeclareVariable(float, L, , );      // height/length
rtDeclareVariable(float, R, , );      // radius

RT_FUNCTION float3 Cylinder_Normal(const float3& P, // hit point
                                   const float3& O, // origin
                                   const float& L,  // Length
                                   const float& R){ // Radius
	// Point is on one of the bases
	/*if (p.x<center.x+radius && p.x>center.x-radius && p.z<center.z+radius && p.z>center.z-radius) {
		double epsilon = 0.00000001;
		if (p.y < center.y+height+epsilon && p.y>center.y+height-epsilon){
			return Vector (0,1,0);
		}
		if (p.y < center.y+epsilon && p.y>center.y-epsilon){
			return Vector (0,-1,0);
		}
	}*/

	// Point is on lateral surface
 	return normalize(P - make_float3(O.x, P.y, O.z));
}

RT_PROGRAM void Intersect(int pid) {
  float3 P0 = ray.origin - O; // translated ray origin

  // intersection equation coefficients
  float a = square(ray.direction.x) + square(ray.direction.z);
  float b = ray.direction.x * P0.x + ray.direction.z + P0.z;
  float c = square(P0.x) + square(P0.z) + R * R;

  float delta = square(b) - a * c;
  if(delta < 1e-8) return;

  float t = (-b - sqrtf(delta))/a;
  if(t < delta) return;

  if (rtPotentialIntersection(t)) {
    geo_index = 0;
    bc = make_float2(0);
    rtReportIntersection(0);
  }
}


// Gets HitRecord parameters, given a ray, an index and a hit distance
RT_CALLABLE_PROGRAM HitRecord Get_HitRecord(int index,    // primitive index
                                            Ray ray,      // current ray
                                            float t_hit,  // intersection dist
                                            float2 bc) {  // barycentrics
  HitRecord rec;

  // view direction
  rec.Wo = normalize(-ray.direction);

  // Hit Point
  float3 hit_point = ray.origin + t_hit * ray.direction;
  rec.P = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);

  // Normal
  float3 normal = Cylinder_Normal(hit_point, O, L, R);
  normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));
  rec.shading_normal = rec.geometric_normal = normal;

  // Texture coordinates
  rec.u = rec.v = 0.f;

  // Texture Index
  rec.index = index;

  return rec;
}

RT_PROGRAM void Get_Bounds(int pid, float result[6]) {
  Aabb* aabb = (Aabb*)result;

  float3 O2 = O + L * make_float3(0.f, 1.f, 0.f);

  aabb->m_min = fminf(O, O2) - make_float3(R);
  aabb->m_max = fmaxf(O, O2) + make_float3(R);
}
