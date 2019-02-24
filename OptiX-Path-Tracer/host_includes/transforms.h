#ifndef TRANSFORMSH
#define TRANSFORMSH

#include "../programs/vec.h"
#include "materials.h"

// debug functions to check if children are NULL
void check_child(GeometryInstance gi) {
  if (!gi) throw "Assigned GeometryInstance is NULL";
}

void check_child(GeometryGroup gg) {
  if (!gg) throw "Assigned GeometryGroup is NULL";
}

void check_child(Transform gi) {
  if (!gi) throw "Assigned Transform is NULL";
}

// functions to add transforms, groups and primitives to the scene graph
void addChild(GeometryInstance gi, Group &d_world, Context &g_context) {
  check_child(gi);  // check if child is NULL

  // add GeometryInstance to GeometryGroup
  GeometryGroup group = g_context->createGeometryGroup();
  group->setAcceleration(g_context->createAcceleration("Trbvh"));
  group->addChild(gi);

  // add GeometryGroup to Group
  d_world->addChild(group);
  d_world->getAcceleration()->markDirty();
}

void addChild(GeometryGroup gg, Group &d_world, Context &g_context) {
  check_child(gg);  // check if child is NULL

  // add GeometryGroup to Group
  d_world->addChild(gg);
  d_world->getAcceleration()->markDirty();
}

void addChild(Transform tr, Group &d_world, Context &g_context) {
  check_child(tr);  // check if child is NULL

  // add Transform to Group
  d_world->addChild(tr);
  d_world->getAcceleration()->markDirty();
}

// translate functions
Transform translate(GeometryInstance gi, float3 &pos, Context &g_context) {
  check_child(gi);  // check if child is NULL

  // get translate matrix
  Matrix4x4 matrix = Matrix4x4::translate(pos);

  // add GeometryInstance to GeometryGroup
  GeometryGroup group = g_context->createGeometryGroup();
  group->setAcceleration(g_context->createAcceleration("Trbvh"));
  group->addChild(gi);

  // apply Transform to GeometryGroup
  Transform transform = g_context->createTransform();
  transform->setChild(group);
  transform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

  return transform;
}

Transform translate(GeometryGroup gg, float3 &pos, Context &g_context) {
  check_child(gg);  // check if child is NULL

  // get translate matrix
  Matrix4x4 matrix = Matrix4x4::translate(pos);

  // apply Transform to GeometryGroup
  Transform transform = g_context->createTransform();
  transform->setChild(gg);
  transform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

  return transform;
}

Transform translate(Transform tr, float3 &pos, Context &g_context) {
  check_child(tr);  // check if child is NULL

  // get translate matrix
  Matrix4x4 matrix = Matrix4x4::translate(pos);

  // apply Transform to Transform
  Transform transform = g_context->createTransform();
  transform->setChild(tr);
  transform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

  return transform;
}

// rotate functions
Transform rotate(GeometryInstance gi, float angle, AXIS ax,
                 Context &g_context) {
  check_child(gi);  // check if child is NULL

  // check selected axis
  float3 axis;
  switch (ax) {
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
  Matrix4x4 matrix = Matrix4x4::rotate(angle * PI_F / 180.f, axis);

  // add GeometryInstance to GeometryGroup
  GeometryGroup group = g_context->createGeometryGroup();
  group->setAcceleration(g_context->createAcceleration("Trbvh"));
  group->addChild(gi);

  // apply Transform to GeometryGroup
  Transform transform = g_context->createTransform();
  transform->setChild(group);
  transform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

  return transform;
}

Transform rotate(GeometryGroup gg, float angle, AXIS ax, Context &g_context) {
  check_child(gg);  // check if child is NULL

  // check selected axis
  float3 axis;
  switch (ax) {
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
  Matrix4x4 matrix = Matrix4x4::rotate(angle * PI_F / 180.f, axis);

  // apply Transform to GeometryGroup
  Transform transform = g_context->createTransform();
  transform->setChild(gg);
  transform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

  return transform;
}

Transform rotate(Transform tr, float angle, AXIS ax, Context &g_context) {
  check_child(tr);  // check if child is NULL

  // check selected axis
  float3 axis;
  switch (ax) {
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
  Matrix4x4 matrix = Matrix4x4::rotate(angle * PI_F / 180.f, axis);

  // apply Transform to Transform
  Transform transform = g_context->createTransform();
  transform->setChild(tr);
  transform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

  return transform;
}

// Scale functions
Transform scale(GeometryInstance gi, float3 &scale, Context &g_context) {
  check_child(gi);  // check if child is NULL

  // get scale matrix
  Matrix4x4 matrix = Matrix4x4::scale(scale);

  // add GeometryInstance to GeometryGroup
  GeometryGroup group = g_context->createGeometryGroup();
  group->setAcceleration(g_context->createAcceleration("Trbvh"));
  group->addChild(gi);

  // apply Transform to GeometryGroup
  Transform transform = g_context->createTransform();
  transform->setChild(group);
  transform->setMatrix(true, matrix.getData(), matrix.inverse().getData());

  return transform;
}

Transform scale(GeometryGroup gg, float3 &scale, Context &g_context) {
  check_child(gg);  // check if child is NULL

  // get scale matrix
  Matrix4x4 matrix = Matrix4x4::scale(scale);

  // apply Transform to GeometryGroup
  Transform transform = g_context->createTransform();
  transform->setChild(gg);
  transform->setMatrix(true, matrix.getData(), matrix.inverse().getData());

  return transform;
}

Transform scale(Transform tr, float3 &scale, Context &g_context) {
  check_child(tr);  // check if child is NULL

  // get scale matrix
  Matrix4x4 matrix = Matrix4x4::scale(scale);

  // apply Transform to Transform
  Transform transform = g_context->createTransform();
  transform->setChild(tr);
  transform->setMatrix(true, matrix.getData(), matrix.inverse().getData());

  return transform;
}

#endif