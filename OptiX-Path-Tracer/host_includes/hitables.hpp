#ifndef HITABLESH
#define HITABLESH

#include "host_common.hpp"
#include "materials.hpp"
#include "programs.hpp"
#include "transforms.hpp"

/*! The precompiled programs code (in ptx) that our cmake script
will precompile (to ptx) and link to the generated executable */
extern "C" const char Sphere_PTX[];
extern "C" const char Volume_Sphere_PTX[];
extern "C" const char Moving_Sphere_PTX[];
extern "C" const char AARect_PTX[];
extern "C" const char Box_PTX[];
extern "C" const char Volume_Box_PTX[];
extern "C" const char Triangle_PTX[];
extern "C" const char Cylinder_PTX[];

// Base geometry primitve class
class Hitable {
 public:
  std::vector<TransformParameter> transforms;  // vector of transforms

  Hitable(BRDF *material) : material(material) {}

  // Get GeometryInstance of Hitable element
  virtual GeometryInstance getGeometryInstance(Context &g_context) = 0;

  // Apply a rotation to the Hitable
  virtual void rotate(float angle, AXIS axis) {
    TransformParameter param(Rotate_Transform,   // Transform type
                             angle,              // Rotation Angle
                             axis,               // Rotation Axis
                             make_float3(0.f),   // Scale value
                             make_float3(0.f));  // Translation delta
    transforms.push_back(param);
  }

  // Apply a scale to the Hitable
  virtual void scale(float3 scale) {
    TransformParameter param(Scale_Transform,    // Transform type
                             0.f,                // Rotation Angle
                             X_AXIS,             // Rotation Axis
                             scale,              // Scale value
                             make_float3(0.f));  // Translation delta
    transforms.push_back(param);
  }

  // Apply a translation to the Hitable
  virtual void translate(float3 pos) {
    TransformParameter param(Translate_Transform,  // Transform type
                             0.f,                  // Rotation Angle
                             X_AXIS,               // Rotation Axis
                             make_float3(0.f),     // Scale value
                             pos);                 // Translation delta
    transforms.push_back(param);
  }

  // Add the Hitable to the scene graph
  virtual void addTo(Group &d_world, Context &g_context) {
    // reverse vector of transforms
    std::reverse(transforms.begin(), transforms.end());

    // create geometry instance
    GeometryInstance gi = getGeometryInstance(g_context);

    // apply transforms and add Hitable to the scene
    addAndTransform(gi, d_world, g_context, transforms);
  }

 protected:
  Geometry geometry;        // Geometry object
  BRDF *material;  // Host side material object

  // Creates GeometryInstance
  virtual GeometryInstance createGeometryInstance(Context &g_context) {
    GeometryInstance gi = g_context->createGeometryInstance();

    gi->setGeometry(geometry);
    gi->setMaterialCount(1);
    gi->setMaterial(0, material->assignTo(g_context));

    return gi;
  }
};

// Creates a Sphere
class Sphere : public Hitable {
 public:
  Sphere(const float3 &c, const float r, BRDF *material)
      : center(c), radius(r), Hitable(material) {}

  // Creates a GeometryInstance object of a sphere primitive
  virtual GeometryInstance getGeometryInstance(Context &g_context) override {
    GeometryInstance gi = g_context->createGeometryInstance();

    // set material
    gi->setMaterialCount(1);
    gi->setMaterial(0, material->assignTo(g_context));

    // Create Geometry parameters callable program
    Program prog = createProgram(Sphere_PTX, "Get_HitRecord", g_context);

    // Basic Parameters
    gi["center"]->setFloat(center.x, center.y, center.z);
    gi["radius"]->setFloat(radius);
    gi["Get_HitRecord"]->set(prog);

    // Create Geometry variable
    Geometry geometry = g_context->createGeometry();
    geometry->setPrimitiveCount(1);

    // Set intersection and bounding box programs
    Program bb = createProgram(Sphere_PTX, "get_bounds", g_context);
    geometry->setBoundingBoxProgram(bb);
    Program hit = createProgram(Sphere_PTX, "hit_sphere", g_context);
    geometry->setIntersectionProgram(hit);

    gi->setGeometry(geometry);

    return gi;
  }

 protected:
  const float3 center;  // center of the sphere
  const float radius;   // radius of the sphere
};

// FIXME: not working, adapt to OptiX's motion blur
class Moving_Sphere : public Hitable {
 public:
  Moving_Sphere(const float3 &c0, const float3 &c1, const float r,
                const float t0, const float t1, BRDF *material)
      : center0(c0),
        center1(c1),
        radius(r),
        time0(t0),
        time1(t1),
        Hitable(material) {}

  // Creates a GeometryInstance object of a motion blur sphere primitive
  virtual GeometryInstance getGeometryInstance(Context &g_context) override {
    // Create Geometry variable
    geometry = g_context->createGeometry();
    geometry->setPrimitiveCount(1);

    // Set bounding box program
    Program bb = createProgram(Moving_Sphere_PTX, "get_bounds", g_context);
    geometry->setBoundingBoxProgram(bb);

    // Set intersection program
    Program hit = createProgram(Moving_Sphere_PTX, "hit_sphere", g_context);
    geometry->setIntersectionProgram(hit);

    // Basic Parameters
    geometry["center0"]->setFloat(center0.x, center0.y, center0.z);
    geometry["center1"]->setFloat(center1.x, center1.y, center1.z);
    geometry["radius"]->setFloat(radius);
    geometry["time0"]->setFloat(time0);
    geometry["time1"]->setFloat(time1);
    geometry["index"]->setInt(0);

    // returns new GeometryInstance
    return createGeometryInstance(g_context);
  }

 protected:
  const float3 center0, center1;  // ending point of movement
  const float radius;             // radius of the sphere
  const float time0, time1;       // times of start and ending of movement
};

// Creates a sphere of volumetric material
class Volumetric_Sphere : public Hitable {
 public:
  Volumetric_Sphere(const float3 &c, const float r, const float d,
                    BRDF *material)
      : center(c), radius(r), density(d), Hitable(material) {}

  // Creates a GeometryInstance object of a volumetric sphere primitive
  virtual GeometryInstance getGeometryInstance(Context &g_context) override {
    // Create Geometry variable
    geometry = g_context->createGeometry();
    geometry->setPrimitiveCount(1);

    // Set bounding box program
    Program bound = createProgram(Volume_Sphere_PTX, "get_bounds", g_context);
    geometry->setBoundingBoxProgram(bound);

    // Set intersection program
    Program hit = createProgram(Volume_Sphere_PTX, "hit_sphere", g_context);
    geometry->setIntersectionProgram(hit);

    // Basic Parameters
    geometry["center"]->setFloat(center.x, center.y, center.z);
    geometry["radius"]->setFloat(radius);
    geometry["density"]->setFloat(density);
    geometry["index"]->setInt(0);

    // returns new GeometryInstance
    return createGeometryInstance(g_context);
  }

 protected:
  const float3 center;  // center of the sphere
  const float radius;   // radius of the sphere
  const float density;  // density of the sphere
};

/* Creates an axis aligned rectangle
The meaning of K, A0/A1 and B0/B1 change depending on the chosen axis. If ax is:
 - X_AXIS: a0 and a1 are points in the Y axis, b0 and b1 in the Z axix. k is a
 point in the X axis.
 - Y_AXIS: a0 and a1 are points in the X axis, b0 and b1 in the Z axis. k is a
 point in the Y axis.
 - Z_AXIS: a0 and a1 are points in the X axis, b0 and b1 in the Y axis. k is a
 point in the Z axis. */
class AARect : public Hitable {
 public:
  AARect(const float a0, const float a1, const float b0, const float b1,
         const float k, const bool flip, const AXIS ax, BRDF *material)
      : a0(a0),
        a1(a1),
        b0(b0),
        b1(b1),
        k(k),
        axis(ax),
        flip(flip),
        Hitable(material) {}

  // Creates a GeometryInstance object of a rectangle primitive
  virtual GeometryInstance getGeometryInstance(Context &g_context) override {
    GeometryInstance gi = g_context->createGeometryInstance();

    // set material
    gi->setMaterialCount(1);
    gi->setMaterial(0, material->assignTo(g_context));

    // Create a Geometry object and set programs
    Geometry geometry = g_context->createGeometry();
    geometry->setPrimitiveCount(1);

    // Create intersection and bounding box programs
    Program bound = createProgram(AARect_PTX, "Get_Bounds", g_context);
    geometry->setBoundingBoxProgram(bound);
    Program intersect = createProgram(AARect_PTX, "Hit_Rect", g_context);
    geometry->setIntersectionProgram(intersect);

    // Create Geometry parameters callable program
    Program prog = createProgram(AARect_PTX, "Get_HitRecord", g_context);

    // Basic Parameters
    gi["axis"]->setInt(int(axis));
    gi["a0"]->setFloat(a0);
    gi["a1"]->setFloat(a1);
    gi["b0"]->setFloat(b0);
    gi["b1"]->setFloat(b1);
    gi["k"]->setFloat(k);
    gi["flip"]->setInt(flip);
    gi["Get_HitRecord"]->set(prog);

    gi->setGeometry(geometry);

    return gi;
  }

 protected:
  const float a0, a1, b0, b1, k;  // rectangle coordinates
  const AXIS axis;                // axis to which rect is alligned to
  const bool flip;                // flip normal
};

// Creates a Box
class Box : public Hitable {
 public:
  Box(const float3 &p0, const float3 &p1, BRDF *material)
      : p0(p0), p1(p1), Hitable(material) {}

  // Creates a GeometryInstance object of a Box primitive
  virtual GeometryInstance getGeometryInstance(Context &g_context) override {
    GeometryInstance gi = g_context->createGeometryInstance();

    // set material
    gi->setMaterialCount(1);
    gi->setMaterial(0, material->assignTo(g_context));

    // Create a Geometry object and set programs
    Geometry geometry = g_context->createGeometry();
    geometry->setPrimitiveCount(1);

    // Set bounding box program
    Program bound = createProgram(Box_PTX, "Get_Bounds", g_context);
    geometry->setBoundingBoxProgram(bound);

    // Set intersection program
    Program intersect = createProgram(Box_PTX, "Intersect", g_context);
    geometry->setIntersectionProgram(intersect);

    // Create Geometry parameters callable program
    Program prog = createProgram(Box_PTX, "Get_HitRecord", g_context);

    // Basic parameters
    gi["boxmin"]->setFloat(p0.x, p0.y, p0.z);
    gi["boxmax"]->setFloat(p1.x, p1.y, p1.z);
    gi["Get_HitRecord"]->set(prog);

    gi->setGeometry(geometry);

    return gi;
  }

 protected:
  const float3 p0, p1;  // box is built by projecting two points
};

// Creates box with volumetric material
class Volumetric_Box : public Hitable {
 public:
  Volumetric_Box(const float3 &p0, const float3 &p1, const float d,
                 BRDF *material)
      : p0(p0), p1(p1), density(d), Hitable(material) {}

  // Creates a GeometryInstance object of a Volumetric Box primitive
  GeometryInstance getGeometryInstance(Context &g_context) {
    // Create Geometry variable
    geometry = g_context->createGeometry();
    geometry->setPrimitiveCount(1);

    // Set bounding box program
    Program bound = createProgram(Volume_Box_PTX, "get_bounds", g_context);
    geometry->setBoundingBoxProgram(bound);

    // Set intersection program
    Program intersect = createProgram(Volume_Box_PTX, "hit_volume", g_context);
    geometry->setIntersectionProgram(intersect);

    // Basic parameters
    geometry["boxmin"]->setFloat(p0.x, p0.y, p0.z);
    geometry["boxmax"]->setFloat(p1.x, p1.y, p1.z);
    geometry["density"]->setFloat(density);
    geometry["index"]->setInt(0);

    // returns new GeometryInstance
    return createGeometryInstance(g_context);
  }

 protected:
  const float3 p0, p1;  // box is built by projecting two points
  const float density;  // volumetric material density
};

// Creates triangle geometry primitive
class Triangle : public Hitable {
 public:
  Triangle(const float3 &a, const float2 &a_uv, const float3 &b,
           const float2 &b_uv, const float3 &c, const float2 &c_uv,
           BRDF *material)
      : a(a),
        b(b),
        c(c),
        a_uv(a_uv),
        b_uv(b_uv),
        c_uv(c_uv),
        Hitable(material) {}

  // Creates a GeometryInstance object of a Triangle primitive
  GeometryInstance getGeometryInstance(Context &g_context) {
    // Create Geometry variable
    geometry = g_context->createGeometry();
    geometry->setPrimitiveCount(1);

    // Set bounding box program
    Program bound = createProgram(Triangle_PTX, "get_bounds", g_context);
    geometry->setBoundingBoxProgram(bound);

    // Set intersection program
    Program intersect = createProgram(Triangle_PTX, "hit_triangle", g_context);
    geometry->setIntersectionProgram(intersect);

    // basic parameters
    geometry["a"]->setFloat(a.x, a.y, a.z);
    geometry["a_uv"]->setFloat(a_uv.x, a_uv.y);
    geometry["b"]->setFloat(b.x, b.y, b.z);
    geometry["b_uv"]->setFloat(b_uv.x, b_uv.y);
    geometry["c"]->setFloat(c.x, c.y, c.z);
    geometry["c_uv"]->setFloat(c_uv.x, c_uv.y);

    // precomputed variables
    const float3 e1(b - a);
    geometry["e1"]->setFloat(e1.x, e1.y, e1.z);

    const float3 e2(c - a);
    geometry["e2"]->setFloat(e2.x, e2.y, e2.z);

    // Geometric normal
    const float3 normal(unit_vector(cross(e1, e2)));
    geometry["normal"]->setFloat(normal.x, normal.y, normal.z);

    // primitive index
    geometry["index"]->setInt(0);

    // returns new GeometryInstance
    return createGeometryInstance(g_context);
  }

 protected:
  const float3 a, b, c;           // vertex coordinates
  const float2 a_uv, b_uv, c_uv;  // vertex texture coordinates
};

// Creates a Cylinder
class Cylinder : public Hitable {
 public:
  Cylinder(const float3 &oo, const float &ll, const float r,
           BRDF *material)
      : O(oo), length(ll), radius(r), Hitable(material) {}

  // Creates a GeometryInstance object of a sphere primitive
  virtual GeometryInstance getGeometryInstance(Context &g_context) override {
    GeometryInstance gi = g_context->createGeometryInstance();

    // set material
    gi->setMaterialCount(1);
    gi->setMaterial(0, material->assignTo(g_context));

    // Create a Geometry object and set programs
    Geometry geometry = g_context->createGeometry();
    geometry->setPrimitiveCount(1);

    // Set bounding box program
    Program bound = createProgram(Cylinder_PTX, "Get_Bounds", g_context);
    geometry->setBoundingBoxProgram(bound);

    // Set intersection program
    Program intersect = createProgram(Cylinder_PTX, "Intersect", g_context);
    geometry->setIntersectionProgram(intersect);

    // Create Geometry parameters callable program
    Program prog = createProgram(Cylinder_PTX, "Get_HitRecord", g_context);

    // Basic Parameters
    gi["O"]->setFloat(O.x, O.y, O.z);
    gi["L"]->setFloat(length);
    gi["R"]->setFloat(radius);
    gi["Get_HitRecord"]->set(prog);

    gi->setGeometry(geometry);

    return gi;
  }

 protected:
  const float3 O;      // origin of the cylinder
  const float length;  // length of the cylinder
  const float radius;  // radius of the cylinder
};

// List of Hitable objects
class Hitable_List {
 public:
  Hitable_List() {}

  // Appends a geometry to the list and returns its index
  int push(Hitable *hit) {
    int index = (int)hitList.size();

    hitList.push_back(hit);

    return index;
  }

  // Apply a rotation to the list
  void rotate(float angle, AXIS axis) {
    TransformParameter param(Rotate_Transform,   // Transform type
                             angle,              // Rotation Angle
                             axis,               // Rotation Axis
                             make_float3(0.f),   // Scale value
                             make_float3(0.f));  // Translation delta
    transforms.push_back(param);
  }

  // Apply a scale to the list
  void scale(float3 scale) {
    TransformParameter param(Scale_Transform,    // Transform type
                             0.f,                // Rotation Angle
                             X_AXIS,             // Rotation Axis
                             scale,              // Scale value
                             make_float3(0.f));  // Translation delta
    transforms.push_back(param);
  }

  // Apply a translation to the list
  void translate(float3 pos) {
    TransformParameter param(Translate_Transform,  // Transform type
                             0.f,                  // Rotation Angle
                             X_AXIS,               // Rotation Axis
                             make_float3(0.f),     // Scale value
                             pos);                 // Translation delta
    transforms.push_back(param);
  }

  // converts list to a GeometryGroup object
  GeometryGroup getGroup(Context &g_context) {
    GeometryGroup gg = g_context->createGeometryGroup();
    gg->setAcceleration(g_context->createAcceleration("Trbvh"));

    for (int i = 0; i < hitList.size(); i++)
      gg->addChild(hitList[i]->getGeometryInstance(g_context));

    return gg;
  }

  // adds and transforms Hitable_List as a group to the scene graph
  void addListTo(Group &d_world, Context &g_context) {
    GeometryGroup gg = getGroup(g_context);
    addAndTransform(gg, d_world, g_context, transforms);
  }

  // adds and transforms each list element to the scene graph individually
  void addElementsTo(Group &d_world, Context &g_context) {
    for (int i = 0; i < (int)hitList.size(); i++) {
      GeometryInstance gi = hitList[i]->getGeometryInstance(g_context);
      addAndTransform(gi, d_world, g_context, hitList[i]->transforms);
    }
  }

 protected:
  std::vector<Hitable *> hitList;
  std::vector<TransformParameter> transforms;
};

#endif