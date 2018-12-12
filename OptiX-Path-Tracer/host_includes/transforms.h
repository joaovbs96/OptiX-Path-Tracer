#ifndef TRANSFORMSH
#define TRANSFORMSH

#include <optix.h>
#include <optixu/optixpp.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

#include "../programs/vec.h"
#include "materials.h"

// functions to add transforms, groups and primitives to the hierarchy
void addChild(optix::GeometryInstance gi, optix::Group &d_world, optix::Context &g_context){
  optix::GeometryGroup test = g_context->createGeometryGroup();
  test->setAcceleration(g_context->createAcceleration("Trbvh"));
  test->setChildCount(1);
  test->setChild(0, gi);

  int i = d_world->getChildCount();
  d_world->setChildCount(i + 1);
  d_world->setChild(i, test);
  d_world->getAcceleration()->markDirty();
}

void addChild(optix::GeometryGroup gg, optix::Group &d_world, optix::Context &g_context){
  int i = d_world->getChildCount();
  d_world->setChildCount(i + 1);
  d_world->setChild(i, gg);
  d_world->getAcceleration()->markDirty();
}

void addChild(optix::Transform gi, optix::Group &d_world, optix::Context &g_context){
  int i = d_world->getChildCount();
  d_world->setChildCount(i + 1);
  d_world->setChild(i, gi);
  d_world->getAcceleration()->markDirty();
}

// translate functions
optix::Matrix4x4 translateMatrix(vec3f offset){
  float floatM[16] = {
      1.0f, 0.0f, 0.0f, offset.x,
      0.0f, 1.0f, 0.0f, offset.y,
      0.0f, 0.0f, 1.0f, offset.z,
      0.0f, 0.0f, 0.0f, 1.0f
    };
  optix::Matrix4x4 mm(floatM);

  return mm;
}

optix::Transform translate(optix::GeometryInstance gi, vec3f& translate, optix::Context &g_context){
  optix::Matrix4x4 matrix = translateMatrix(translate);

  optix::GeometryGroup d_world = g_context->createGeometryGroup();
  d_world->setAcceleration(g_context->createAcceleration("Trbvh"));
  d_world->setChildCount(1);
  d_world->setChild(0, gi);

  optix::Transform translateTransform = g_context->createTransform();
  translateTransform->setChild(d_world);
  translateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

  return translateTransform;
}

optix::Transform translate(optix::GeometryGroup gi, vec3f& translate, optix::Context &g_context){
  optix::Matrix4x4 matrix = translateMatrix(translate);

  optix::Transform translateTransform = g_context->createTransform();
  translateTransform->setChild(gi);
  translateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

  return translateTransform;
}

optix::Transform translate(optix::Transform gi, vec3f& translate, optix::Context &g_context){
  optix::Matrix4x4 matrix = translateMatrix(translate);

  optix::Transform translateTransform = g_context->createTransform();
  translateTransform->setChild(gi);
  translateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());
  
  return translateTransform;
}


// rotateAboutPoint
optix::Matrix4x4 rotateAboutPointMatrix(float angle, vec3f offset){
  float floatM[16] = {
       cos(angle), 0.0f, -sin(angle), offset.x - cos(angle) * offset.x + sin(angle) * offset.z,
             0.0f, 1.0f,        0.0f,                                                      0.f,
       sin(angle), 0.0f,  cos(angle), offset.z - sin(angle) * offset.x - cos(angle) * offset.z,
             0.0f, 0.0f,        0.0f,                                                     1.0f
    };
  optix::Matrix4x4 mm(floatM);

  return mm;
}

// it's *really* slow.
optix::Transform rotateAboutPoint(optix::GeometryInstance gi, float angle, vec3f& translate, optix::Context &g_context){
  optix::Matrix4x4 matrix = rotateAboutPointMatrix(-angle * CUDART_PI_F / 180.f, translate);

  optix::GeometryGroup d_world = g_context->createGeometryGroup();
  d_world->setAcceleration(g_context->createAcceleration("Trbvh"));
  d_world->setChildCount(1);
  d_world->setChild(0, gi);

  optix::Transform translateTransform = g_context->createTransform();
  translateTransform->setChild(d_world);
  translateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

  return translateTransform;
}

// rotateX functions
optix::Matrix4x4 rotateMatrixX(float angle){
  float floatM[16] = {
             1.0f,        0.0f,       0.0f, 0.0f,
             0.0f,  cos(angle), sin(angle), 0.0f,
             0.0f, -sin(angle), cos(angle), 0.0f,
             0.0f,        0.0f,       0.0f, 1.0f
    };
  optix::Matrix4x4 mm(floatM);

  return mm;
}

optix::Transform rotateX(optix::GeometryInstance gi, float angle, optix::Context &g_context){
  optix::Matrix4x4 matrix = rotateMatrixX(-angle * CUDART_PI_F / 180.f);

  optix::GeometryGroup d_world = g_context->createGeometryGroup();
  d_world->setAcceleration(g_context->createAcceleration("Trbvh"));
  d_world->setChildCount(1);
  d_world->setChild(0, gi);

  optix::Transform translateTransform = g_context->createTransform();
  translateTransform->setChild(d_world);
  translateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());
    
  return translateTransform;
}

optix::Transform rotateX(optix::GeometryGroup gi, float angle, optix::Context &g_context){
  optix::Matrix4x4 matrix = rotateMatrixX(-angle * CUDART_PI_F / 180.f);

  optix::Transform translateTransform = g_context->createTransform();
  translateTransform->setChild(gi);
  translateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());
    
  return translateTransform;
}

optix::Transform rotateX(optix::Transform gi, float angle, optix::Context &g_context){
  optix::Matrix4x4 matrix = rotateMatrixX(-angle * CUDART_PI_F / 180.f);

  optix::Transform translateTransform = g_context->createTransform();
  translateTransform->setChild(gi);
  translateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());
    
  return translateTransform;
}

// rotat Y functions
optix::Matrix4x4 rotateMatrixY(float angle){
  float floatM[16] = {
       cos(angle), 0.0f, -sin(angle), 0.0f,
             0.0f, 1.0f,        0.0f, 0.0f,
       sin(angle), 0.0f,  cos(angle), 0.0f,
             0.0f, 0.0f,        0.0f, 1.0f
    };
  optix::Matrix4x4 mm(floatM);

  return mm;
}

optix::Transform rotateY(optix::GeometryInstance gi, float angle, optix::Context &g_context){
  optix::Matrix4x4 matrix = rotateMatrixY(-angle * CUDART_PI_F / 180.f);

  optix::GeometryGroup d_world = g_context->createGeometryGroup();
  d_world->setAcceleration(g_context->createAcceleration("Trbvh"));
  d_world->setChildCount(1);
  d_world->setChild(0, gi);

  optix::Transform translateTransform = g_context->createTransform();
  translateTransform->setChild(d_world);
  translateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());
    
  return translateTransform;
}

optix::Transform rotateY(optix::GeometryGroup gi, float angle, optix::Context &g_context){
  optix::Matrix4x4 matrix = rotateMatrixY(-angle * CUDART_PI_F / 180.f);

  optix::Transform transf = g_context->createTransform();
  transf->setChild(gi);
  transf->setMatrix(false, matrix.getData(), matrix.inverse().getData());
    
  return transf;
}

optix::Transform rotateY(optix::Transform gi, float angle, optix::Context &g_context){
  optix::Matrix4x4 matrix = rotateMatrixY(-angle * CUDART_PI_F / 180.f);

  optix::Transform transf = g_context->createTransform();
  transf->setChild(gi);
  transf->setMatrix(false, matrix.getData(), matrix.inverse().getData());
    
  return transf;
}


// rotateZ functions
optix::Matrix4x4 rotateMatrixZ(float angle){
  float floatM[16] = {
       cos(angle), sin(angle), 0.0f, 0.0f,
      -sin(angle), cos(angle), 0.0f, 0.0f,
             0.0f,       0.0f, 1.0f, 0.0f,
             0.0f,       0.0f, 0.0f, 1.0f
    };
  optix::Matrix4x4 mm(floatM);

  return mm;
}

optix::Transform rotateZ(optix::GeometryInstance gi, float angle, optix::Context &g_context){
  optix::Matrix4x4 matrix = rotateMatrixZ(-angle * CUDART_PI_F / 180.f);

  optix::GeometryGroup d_world = g_context->createGeometryGroup();
  d_world->setAcceleration(g_context->createAcceleration("Trbvh"));
  d_world->setChildCount(1);
  d_world->setChild(0, gi);

  optix::Transform transf = g_context->createTransform();
  transf->setChild(d_world);
  transf->setMatrix(false, matrix.getData(), matrix.inverse().getData());
    
  return transf;
}

optix::Transform rotateZ(optix::GeometryGroup gi, float angle, optix::Context &g_context){
  optix::Matrix4x4 matrix = rotateMatrixZ(-angle * CUDART_PI_F / 180.f);

  optix::Transform rotateTransform = g_context->createTransform();
  rotateTransform->setChild(gi);
  rotateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());
    
  return rotateTransform;
}

optix::Transform rotateZ(optix::Transform gi, float angle, optix::Context &g_context){
  optix::Matrix4x4 matrix = rotateMatrixZ(-angle * CUDART_PI_F / 180.f);

  optix::Transform rotateTransform = g_context->createTransform();
  rotateTransform->setChild(gi);
  rotateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());
    
  return rotateTransform;
}

#endif