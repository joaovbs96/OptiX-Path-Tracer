#ifndef HITABLESH
#define HITABLESH

#include <optix.h>
#include <optixu/optixpp.h>

#include "../programs/vec.h"
#include "materials.h"

/*! the precompiled programs/raygen.cu code (in ptx) that our
  cmake magic will precompile (to ptx) and link to the generated
  executable (ie, we can simply declare and use this here as
  'extern'.  */
extern "C" const char embedded_sphere_programs[];
extern "C" const char embedded_moving_sphere_programs[];
extern "C" const char embedded_aarect_programs[];

enum Axis { X_AXIS, Y_AXIS, Z_AXIS };

// Sphere constructor
optix::GeometryInstance createSphere(const vec3f &center, const float radius, const Material &material, optix::Context &g_context) {
  optix::Geometry geometry = g_context->createGeometry();
  
  geometry->setPrimitiveCount(1);
  geometry->setBoundingBoxProgram(g_context->createProgramFromPTXString(embedded_sphere_programs, "get_bounds"));
  geometry->setIntersectionProgram(g_context->createProgramFromPTXString(embedded_sphere_programs, "hit_sphere"));
  
  geometry["center"]->setFloat(center.x,center.y,center.z);
  geometry["radius"]->setFloat(radius);

  optix::GeometryInstance gi = g_context->createGeometryInstance();
  gi->setGeometry(geometry);
  gi->setMaterialCount(1);
  material.assignTo(gi, g_context);
  
  return gi;
}

// Moving Sphere constructor
optix::GeometryInstance createMovingSphere(const vec3f &center0, const vec3f &center1, const float t0, const float t1, const float radius, const Material &material, optix::Context &g_context) {
  optix::Geometry geometry = g_context->createGeometry();
  
  geometry->setPrimitiveCount(1);
  geometry->setBoundingBoxProgram(g_context->createProgramFromPTXString(embedded_moving_sphere_programs, "get_bounds"));
  geometry->setIntersectionProgram(g_context->createProgramFromPTXString(embedded_moving_sphere_programs, "hit_sphere"));
  
  geometry["center0"]->setFloat(center0.x,center0.y,center0.z);
  geometry["center1"]->setFloat(center1.x,center1.y,center1.z);
  geometry["radius"]->setFloat(radius);
  geometry["time0"]->setFloat(t0);
  geometry["time1"]->setFloat(t1);

  optix::GeometryInstance gi = g_context->createGeometryInstance();
  gi->setGeometry(geometry);
  gi->setMaterialCount(1);
  material.assignTo(gi, g_context);
  
  return gi;
}

// Axis-alligned Rectangle constructor
optix::GeometryInstance createXRect(const float a0, const float a1, const float b0, const float b1, const float k, const bool flip, const Material &material, optix::Context &g_context) {
  optix::Geometry geometry = g_context->createGeometry();
  
  geometry->setPrimitiveCount(1);
  geometry->setBoundingBoxProgram(g_context->createProgramFromPTXString(embedded_aarect_programs, "get_bounds_X"));
  geometry->setIntersectionProgram(g_context->createProgramFromPTXString(embedded_aarect_programs, "hit_rect_X"));
  
  geometry["a0"]->setFloat(a0);
  geometry["a1"]->setFloat(a1);
  geometry["b0"]->setFloat(b0);
  geometry["b1"]->setFloat(b1);
  geometry["k"]->setFloat(k);
  geometry["flip"]->setInt(flip);

  optix::GeometryInstance gi = g_context->createGeometryInstance();
  gi->setGeometry(geometry);
  gi->setMaterialCount(1);
  material.assignTo(gi, g_context);
  
  return gi;
}

optix::GeometryInstance createYRect(const float a0, const float a1, const float b0, const float b1, const float k, const bool flip, const Material &material, optix::Context &g_context) {
  optix::Geometry geometry = g_context->createGeometry();
  
  geometry->setPrimitiveCount(1);
  geometry->setBoundingBoxProgram(g_context->createProgramFromPTXString(embedded_aarect_programs, "get_bounds_Y"));
  geometry->setIntersectionProgram(g_context->createProgramFromPTXString(embedded_aarect_programs, "hit_rect_Y"));
  
  geometry["a0"]->setFloat(a0);
  geometry["a1"]->setFloat(a1);
  geometry["b0"]->setFloat(b0);
  geometry["b1"]->setFloat(b1);
  geometry["k"]->setFloat(k);
  geometry["flip"]->setInt(flip);

  optix::GeometryInstance gi = g_context->createGeometryInstance();
  gi->setGeometry(geometry);
  gi->setMaterialCount(1);
  material.assignTo(gi, g_context);
  
  return gi;
}

optix::GeometryInstance createZRect(const float a0, const float a1, const float b0, const float b1, const float k, const bool flip, const Material &material, optix::Context &g_context) {
  optix::Geometry geometry = g_context->createGeometry();
  
  geometry->setPrimitiveCount(1);
  geometry->setBoundingBoxProgram(g_context->createProgramFromPTXString(embedded_aarect_programs, "get_bounds_Z"));
  geometry->setIntersectionProgram(g_context->createProgramFromPTXString(embedded_aarect_programs, "hit_rect_Z"));
  
  geometry["a0"]->setFloat(a0);
  geometry["a1"]->setFloat(a1);
  geometry["b0"]->setFloat(b0);
  geometry["b1"]->setFloat(b1);
  geometry["k"]->setFloat(k);
  geometry["flip"]->setInt(flip);

  optix::GeometryInstance gi = g_context->createGeometryInstance();
  gi->setGeometry(geometry);
  gi->setMaterialCount(1);
  material.assignTo(gi, g_context);
  
  return gi;
}

void createBox(const vec3f& p0, const vec3f& p1, Material &material, optix::Context &g_context, std::vector<optix::GeometryInstance> &d_list){
  d_list.push_back(createZRect(p0.x, p1.x, p0.y, p1.y, p0.z, true, material, g_context));
  d_list.push_back(createZRect(p0.x, p1.x, p0.y, p1.y, p1.z, false, material, g_context));

  d_list.push_back(createYRect(p0.x, p1.x, p0.z, p1.z, p0.y, true, material, g_context));
  d_list.push_back(createYRect(p0.x, p1.x, p0.z, p1.z, p1.y, false, material, g_context));
  
  d_list.push_back(createXRect(p0.y, p1.y, p0.z, p1.z, p0.x, true, material, g_context));
  d_list.push_back(createXRect(p0.y, p1.y, p0.z, p1.z, p1.x, false, material, g_context));
}

#endif