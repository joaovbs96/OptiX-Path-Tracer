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

// Sphere constructor
optix::GeometryInstance createMovingSphere(const vec3f &center, const float radius, const Material &material, const float t0, const float t1, optix::Context &g_context) {
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

#endif