#ifndef TRANSFORMSH
#define TRANSFORMSH

// transforms.hpp: Define Transform related types and functions

#include "host_common.hpp"
#include "materials.hpp"

//////////////////////////////////////////
// Transform Types & Auxiliar Functions //
//////////////////////////////////////////

// Throw exception if GeometryInstance object is NULL
void check_if_null(GeometryInstance gi) {
  if (!gi) throw "Assigned GeometryInstance is NULL";
}

// Throw exception if GeometryGroup object is NULL
void check_if_null(GeometryGroup gg) {
  if (!gg) throw "Assigned GeometryGroup is NULL";
}

// Throw exception if Transform object is NULL
void check_if_null(Transform gi) {
  if (!gi) throw "Assigned Transform is NULL";
}

// Types of transforms
typedef enum {
  Rotate_Transform,
  Translate_Transform,
  Scale_Transform
} Transform_Type;

// Parameters that define a transform operation
struct TransformParameter {
  TransformParameter(Transform_Type t, float a, AXIS ax, float3 s, float3 p)
      : type(t), angle(a), axis(ax), scale(s), pos(p) {}
  Transform_Type type;
  float angle;
  AXIS axis;
  float3 scale;
  float3 pos;
};

/////////////////////////
// Translate functions //
/////////////////////////

// Translate a GeometryInstance object in (x, y, z) coordinates
Transform translate(GeometryInstance gi, float3 &xyz, Context &g_context) {
  check_if_null(gi);

  // get translate matrix
  Matrix4x4 matrix = Matrix4x4::translate(xyz);

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

// Translate a GeometryGroup object in (x, y, z) coordinates
Transform translate(GeometryGroup gg, float3 &xyz, Context &g_context) {
  check_if_null(gg);

  // get translate matrix
  Matrix4x4 matrix = Matrix4x4::translate(xyz);

  // apply Transform to GeometryGroup
  Transform transform = g_context->createTransform();
  transform->setChild(gg);
  transform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

  return transform;
}

// Translate a Transform object in (x, y, z) coordinates
Transform translate(Transform tr, float3 &xyz, Context &g_context) {
  check_if_null(tr);

  // get translate matrix
  Matrix4x4 matrix = Matrix4x4::translate(xyz);

  // apply Transform to Transform
  Transform transform = g_context->createTransform();
  transform->setChild(tr);
  transform->setMatrix(false, matrix.getData(), matrix.inverse().getData());

  return transform;
}

//////////////////////
// Rotate functions //
//////////////////////

// Rotates a GeometryInstance object in (angle) degrees in the given axis
Transform rotate(GeometryInstance gi, float angle, AXIS ax,
                 Context &g_context) {
  check_if_null(gi);

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

// Rotates a GeometryGroup object in (angle) degrees in the given axis
Transform rotate(GeometryGroup gg, float angle, AXIS ax, Context &g_context) {
  check_if_null(gg);

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

// Rotates a Transform object in (angle) degrees in the given axis
Transform rotate(Transform tr, float angle, AXIS ax, Context &g_context) {
  check_if_null(tr);

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

/////////////////////
// Scale functions //
/////////////////////

// Scales a GeometryInstance object with (x, y, z) factors for each axis
Transform scale(GeometryInstance gi, float3 &xyz, Context &g_context) {
  check_if_null(gi);

  // get scale matrix
  Matrix4x4 matrix = Matrix4x4::scale(xyz);

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

// Scales a GeometryGroup object with (x, y, z) factors for each axis
Transform scale(GeometryGroup gg, float3 &xyz, Context &g_context) {
  check_if_null(gg);

  // get scale matrix
  Matrix4x4 matrix = Matrix4x4::scale(xyz);

  // apply Transform to GeometryGroup
  Transform transform = g_context->createTransform();
  transform->setChild(gg);
  transform->setMatrix(true, matrix.getData(), matrix.inverse().getData());

  return transform;
}

// Scales a Transform object with (x, y, z) factors for each axis
Transform scale(Transform tr, float3 &xyz, Context &g_context) {
  check_if_null(tr);

  // get scale matrix
  Matrix4x4 matrix = Matrix4x4::scale(xyz);

  // apply Transform to Transform
  Transform transform = g_context->createTransform();
  transform->setChild(tr);
  transform->setMatrix(true, matrix.getData(), matrix.inverse().getData());

  return transform;
}

///////////////////////////////
// Transform Apply functions //
///////////////////////////////

// Functions to apply Transforms: Apply and return a Transform, given a
// TransformParam from a list. Check if the list size if zero. If it's zero,
// just return the new transform, if it's not, call a recursion.

// Applies a Transform operation to a Transform
Transform applyTransform(Transform oldTransf,
                         std::vector<TransformParameter> &params,
                         Context &g_context) {
  TransformParameter param = params[params.size() - 1];
  params.pop_back();
  Transform newTransf;

  // check type of transform to be applied
  switch (param.type) {
    case Rotate_Transform:
      newTransf = rotate(oldTransf, param.angle, param.axis, g_context);
      break;

    case Translate_Transform:
      newTransf = translate(oldTransf, param.pos, g_context);
      break;

    case Scale_Transform:
      newTransf = scale(oldTransf, param.scale, g_context);
      break;

    default:
      throw "Invalid Transform operation";
  }

  // check if there are more transforms to be applied
  if (params.size() > 0)
    return applyTransform(newTransf, params, g_context);
  else
    return newTransf;
}

// Applies a Transform operation to a GeometryInstance
Transform applyTransform(GeometryInstance gi,
                         std::vector<TransformParameter> &params,
                         Context &g_context) {
  TransformParameter param = params[params.size() - 1];
  params.pop_back();
  Transform newTransf;

  // check type of transform to be applied
  switch (param.type) {
    case Rotate_Transform:
      newTransf = rotate(gi, param.angle, param.axis, g_context);
      break;

    case Translate_Transform:
      newTransf = translate(gi, param.pos, g_context);
      break;

    case Scale_Transform:
      newTransf = scale(gi, param.scale, g_context);
      break;

    default:
      throw "Invalid Transform operation";
  }

  // check if there are more transforms to be applied
  if (params.size() > 0)
    return applyTransform(newTransf, params, g_context);
  else
    return newTransf;
}

// Applies a Transform operation to a GeometryGroup
Transform applyTransform(GeometryGroup group,
                         std::vector<TransformParameter> &params,
                         Context &g_context) {
  TransformParameter param = params[params.size() - 1];
  params.pop_back();
  Transform newTransf;

  // check type of transform to be applied
  switch (param.type) {
    case Rotate_Transform:
      newTransf = rotate(group, param.angle, param.axis, g_context);
      break;

    case Translate_Transform:
      newTransf = translate(group, param.pos, g_context);
      break;

    case Scale_Transform:
      newTransf = scale(group, param.scale, g_context);
      break;

    default:
      throw "Invalid Transform operation";
  }

  // check if there are more transforms to be applied
  if (params.size() > 0)
    return applyTransform(newTransf, params, g_context);
  else
    return newTransf;
}

// Add a Transform child node to the scene graph
void addAndTransform(Transform tr, Group &d_world, Context &g_context,
                     std::vector<TransformParameter> params) {
  Transform transform;

  check_if_null(tr);

  // Add geometry to the scene graph and apply Transforms, if needed
  if (params.size() > 0)
    transform = applyTransform(tr, params, g_context);
  else
    transform = tr;

  // add Transform to Group object
  d_world->addChild(transform);
  d_world->getAcceleration()->markDirty();
}

// Add a GeometryGroup child node to the scene graph
void addAndTransform(GeometryGroup gg, Group &d_world, Context &g_context,
                     std::vector<TransformParameter> params) {
  check_if_null(gg);

  // Add geometry to the scene graph and apply Transforms, if needed
  if (params.size() == 0) {
    // add GeometryGroup to Group object
    d_world->addChild(gg);
    d_world->getAcceleration()->markDirty();
  } else {
    // Apply Transform and add to scene graph
    Transform t = applyTransform(gg, params, g_context);
    addAndTransform(t, d_world, g_context, params);
  }
}

// Add a GeometryInstance child node to the scene graph
void addAndTransform(GeometryInstance gi, Group &d_world, Context &g_context,
                     std::vector<TransformParameter> params) {
  check_if_null(gi);  // check if child is NULL

  // Add geometry to the scene graph and apply Transforms, if needed
  if (params.size() == 0) {
    // add GeometryInstance to GeometryGroup object
    GeometryGroup group = g_context->createGeometryGroup();
    group->setAcceleration(g_context->createAcceleration("Trbvh"));
    group->addChild(gi);

    // add GeometryGroup to Group
    d_world->addChild(group);
    d_world->getAcceleration()->markDirty();
  } else {
    // Apply Transform and add to scene graph
    Transform t = applyTransform(gi, params, g_context);
    addAndTransform(t, d_world, g_context, params);
  }
}

#endif