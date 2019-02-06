#ifndef TRANSFORMSH
#define TRANSFORMSH

#include <optix.h>
#include <optixu/optixpp.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

#include "../programs/vec.h"
#include "materials.h"

// Axis type
typedef enum{
  X_AXIS,
  Y_AXIS,
  Z_AXIS
} AXIS;

// debug functions to check if children are NULL
void check_child(optix::GeometryInstance gi) {
  if(!gi) { // if NULL
    printf("Error: Assigned GeometryInstance is NULL.\n");
    system("PAUSE");
  }
}

void check_child(optix::GeometryGroup gg) {
  if(!gg) { // if NULL
    printf("Error: Assigned GeometryGroup is NULL.\n");
    system("PAUSE");
  }
}

void check_child(optix::Transform gi) {
  if(!gi) { // if NULL
    printf("Error: Assigned Transform is NULL.\n");
    system("PAUSE");
  }
}

// functions to add transforms, groups and primitives to the scene graph
void addChild(optix::GeometryInstance gi, optix::Group &d_world, optix::Context &g_context) {
  check_child(gi); // check if child is NULL

  // add GeometryInstance to GeometryGroup
  optix::GeometryGroup group = g_context->createGeometryGroup();
  group->setAcceleration(g_context->createAcceleration("Trbvh"));
  group->addChild(gi);

  // add GeometryGroup to Group
  d_world->addChild(group);
  d_world->getAcceleration()->markDirty();
}

void addChild(optix::GeometryGroup gg, optix::Group &d_world, optix::Context &g_context) {
  check_child(gg); // check if child is NULL

  // add GeometryGroup to Group
  d_world->addChild(gg);
  d_world->getAcceleration()->markDirty();
}

void addChild(optix::Transform tr, optix::Group &d_world, optix::Context &g_context) {
  check_child(tr); // check if child is NULL

  // add Transform to Group
  d_world->addChild(tr);
  d_world->getAcceleration()->markDirty();
}

// translate functions
optix::Transform translate(optix::GeometryInstance gi, vec3f& pos, optix::Context &g_context) {
  check_child(gi); // check if child is NULL

  // get translate matrix
  optix::Matrix4x4 matrix = optix::Matrix4x4::translate(make_float3(pos.x, pos.y, pos.z));

  // add GeometryInstance to GeometryGroup
  optix::GeometryGroup group = g_context->createGeometryGroup();
  group->setAcceleration(g_context->createAcceleration("Trbvh"));
  group->addChild(gi);

  // apply Transform to GeometryGroup
  optix::Transform transform = g_context->createTransform();
  transform->setChild(group);
  transform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

  return transform;
}

optix::Transform translate(optix::GeometryGroup gg, vec3f& pos, optix::Context &g_context) {
  check_child(gg); // check if child is NULL

  // get translate matrix
  optix::Matrix4x4 matrix = optix::Matrix4x4::translate(make_float3(pos.x, pos.y, pos.z));

  // apply Transform to GeometryGroup
  optix::Transform transform = g_context->createTransform();
  transform->setChild(gg);
  transform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

  return transform;
}

optix::Transform translate(optix::Transform tr, vec3f& pos, optix::Context &g_context) {
  check_child(tr); // check if child is NULL

  // get translate matrix
  optix::Matrix4x4 matrix = optix::Matrix4x4::translate(make_float3(pos.x, pos.y, pos.z));

  // apply Transform to Transform
  optix::Transform transform = g_context->createTransform();
  transform->setChild(tr);
  transform->setMatrix(false, matrix.getData(), matrix.inverse().getData());
  
  return transform;
}

// rotate functions
optix::Transform rotate(optix::GeometryInstance gi, float angle, AXIS ax, optix::Context &g_context) {
  check_child(gi); // check if child is NULL

  // check selected axis
  float3 axis;
  switch(ax){
    case X_AXIS:
      axis = make_float3(1.f, 0.f, 0.f);
      break;
    
    case Y_AXIS:
      axis = make_float3(0.f, 1.f, 0.f);
      break;
    
    case Z_AXIS:
      axis = make_float3(0.f, 0.f, 1.f);
      break;
  }

  // get rotation matrix
  optix::Matrix4x4 matrix = optix::Matrix4x4::rotate(angle * PI_F / 180.f, axis);

  // add GeometryInstance to GeometryGroup
  optix::GeometryGroup group = g_context->createGeometryGroup();
  group->setAcceleration(g_context->createAcceleration("Trbvh"));
  group->addChild(gi);

  // apply Transform to GeometryGroup
  optix::Transform transform = g_context->createTransform();
  transform->setChild(group);
  transform->setMatrix(false, matrix.getData(), matrix.inverse().getData());
    
  return transform;
}

optix::Transform rotate(optix::GeometryGroup gg, float angle, AXIS ax, optix::Context &g_context) {
  check_child(gg); // check if child is NULL

  // check selected axis
  float3 axis;
  switch(ax){
    case X_AXIS:
      axis = make_float3(1.f, 0.f, 0.f);
      break;
    
    case Y_AXIS:
      axis = make_float3(0.f, 1.f, 0.f);
      break;
    
    case Z_AXIS:
      axis = make_float3(0.f, 0.f, 1.f);
      break;
  }

  // get rotation matrix
  optix::Matrix4x4 matrix = optix::Matrix4x4::rotate(angle * PI_F / 180.f, axis);

  // apply Transform to GeometryGroup
  optix::Transform transform = g_context->createTransform();
  transform->setChild(gg);
  transform->setMatrix(false, matrix.getData(), matrix.inverse().getData());
    
  return transform;
}

optix::Transform rotate(optix::Transform tr, float angle, AXIS ax, optix::Context &g_context) {
  check_child(tr); // check if child is NULL

  // check selected axis
  float3 axis;
  switch(ax){
    case X_AXIS:
      axis = make_float3(1.f, 0.f, 0.f);
      break;
    
    case Y_AXIS:
      axis = make_float3(0.f, 1.f, 0.f);
      break;
    
    case Z_AXIS:
      axis = make_float3(0.f, 0.f, 1.f);
      break;
  }

  // get rotation matrix
  optix::Matrix4x4 matrix = optix::Matrix4x4::rotate(angle * PI_F / 180.f, axis);

  // apply Transform to Transform
  optix::Transform transform = g_context->createTransform();
  transform->setChild(tr);
  transform->setMatrix(false, matrix.getData(), matrix.inverse().getData());
    
  return transform;
}

optix::Matrix4x4 scaleMatrix(vec3f offset) {
  float floatM[16] = {
      offset.x, 0.0f, 0.0f, 0.f,
      0.0f, offset.y, 0.0f, 0.f,
      0.0f, 0.0f, offset.z, 0.f,
      0.0f, 0.0f, 0.0f, 1.0f
    };
  optix::Matrix4x4 mm(floatM);

  return mm;
}

// Scale functions
optix::Transform scale(optix::GeometryInstance gi, vec3f& scale, optix::Context &g_context) {
  check_child(gi); // check if child is NULL

  // get scale matrix
  optix::Matrix4x4 matrix = optix::Matrix4x4::scale(make_float3(scale.x, scale.y, scale.z));

  // add GeometryInstance to GeometryGroup
  optix::GeometryGroup group = g_context->createGeometryGroup();
  group->setAcceleration(g_context->createAcceleration("Trbvh"));
  group->addChild(gi);

  // apply Transform to GeometryGroup
  optix::Transform transform = g_context->createTransform();
  transform->setChild(group);
  transform->setMatrix(true, matrix.getData(), matrix.inverse().getData());
    
  return transform;
}

optix::Transform scale(optix::GeometryGroup gg, vec3f& scale, optix::Context &g_context) {
  check_child(gg); // check if child is NULL

  // get scale matrix
  optix::Matrix4x4 matrix = optix::Matrix4x4::scale(make_float3(scale.x, scale.y, scale.z));

  // apply Transform to GeometryGroup
  optix::Transform transform = g_context->createTransform();
  transform->setChild(gg);
  transform->setMatrix(true, matrix.getData(), matrix.inverse().getData());
    
  return transform;
}

optix::Transform scale(optix::Transform tr, vec3f& scale, optix::Context &g_context) {
  check_child(tr); // check if child is NULL

  // get scale matrix
  optix::Matrix4x4 matrix = optix::Matrix4x4::scale(make_float3(scale.x, scale.y, scale.z));

  // apply Transform to Transform
  optix::Transform transform = g_context->createTransform();
  transform->setChild(tr);
  transform->setMatrix(true, matrix.getData(), matrix.inverse().getData());
    
  return transform;
}

// rotat Y functions
optix::Matrix4x4 rotateMatrixY(float angle) {
  float floatM[16] = {
       cos(angle), 0.0f, -sin(angle), 0.0f,
             0.0f, 1.0f,        0.0f, 0.0f,
       sin(angle), 0.0f,  cos(angle), 0.0f,
             0.0f, 0.0f,        0.0f, 1.0f
    };
  optix::Matrix4x4 mm(floatM);

  return mm;
}

optix::Transform rotateY(optix::GeometryInstance gi, float angle, optix::Context &g_context) {
  check_child(gi); // check if child is NULL

  optix::Matrix4x4 matrix = rotateMatrixY(-angle * PI_F / 180.f);

  optix::GeometryGroup d_world = g_context->createGeometryGroup();
  d_world->setAcceleration(g_context->createAcceleration("Trbvh"));
  d_world->addChild(gi);

  optix::Transform translateTransform = g_context->createTransform();
  translateTransform->setChild(d_world);
  translateTransform->setMatrix(false, matrix.getData(), matrix.inverse().getData());
    
  return translateTransform;
}

optix::Transform rotateY(optix::GeometryGroup gi, float angle, optix::Context &g_context) {
  check_child(gi); // check if child is NULL

  optix::Matrix4x4 matrix = rotateMatrixY(-angle * PI_F / 180.f);

  optix::Transform transf = g_context->createTransform();
  transf->setChild(gi);
  transf->setMatrix(false, matrix.getData(), matrix.inverse().getData());
    
  return transf;
}

optix::Transform rotateY(optix::Transform gi, float angle, optix::Context &g_context) {
  check_child(gi); // check if child is NULL

  optix::Matrix4x4 matrix = rotateMatrixY(-angle * PI_F / 180.f);

  optix::Transform transf = g_context->createTransform();
  transf->setChild(gi);
  transf->setMatrix(false, matrix.getData(), matrix.inverse().getData());
    
  return transf;
}

#endif