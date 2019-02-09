#ifndef HITABLESH
#define HITABLESH

#include "../programs/vec.h"
#include "materials.h"
#include "transforms.h"

#include "../lib/OBJ_Loader.h"

#include <map>

/*! The precompiled programs code (in ptx) that our cmake script
will precompile (to ptx) and link to the generated executable */
extern "C" const char sphere_programs[];
extern "C" const char volume_sphere_programs[];
extern "C" const char moving_sphere_programs[];
extern "C" const char aarect_programs[];
extern "C" const char box_programs[];
extern "C" const char volume_box_programs[];
extern "C" const char triangle_programs[];
extern "C" const char mesh_programs[];
extern "C" const char plane_programs[];

// Sphere constructor
GeometryInstance Sphere(const float3 &center, const float radius,
                        const Materials &material, Context &g_context) {
  // Create Geometry variable
  Geometry geometry = g_context->createGeometry();
  geometry->setPrimitiveCount(1);

  // Set bounding box program
  Program bound =
      g_context->createProgramFromPTXString(sphere_programs, "get_bounds");
  geometry->setBoundingBoxProgram(bound);

  // Set intersection program
  Program intersect =
      g_context->createProgramFromPTXString(sphere_programs, "hit_sphere");
  geometry->setIntersectionProgram(intersect);

  // Basic Parameters
  geometry["center"]->setFloat(center.x, center.y, center.z);
  geometry["radius"]->setFloat(radius);

  // Create GeometryInstance
  GeometryInstance gi = g_context->createGeometryInstance();
  gi->setGeometry(geometry);
  gi->setMaterialCount(1);

  // Assign material and texture programs
  Buffer texture_buffers =
      g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 1);
  callableProgramId<int(int)> *tex_data =
      static_cast<callableProgramId<int(int)> *>(texture_buffers->map());

  Program texture = material.assignTo(gi, g_context);
  tex_data[0] = callableProgramId<int(int)>(texture->getId());

  texture_buffers->unmap();
  gi["sample_texture"]->setBuffer(texture_buffers);

  return gi;
}

// Sphere of volumetric material
GeometryInstance Volume_Sphere(const float3 &center, const float radius,
                               const float density, const Materials &material,
                               Context &g_context) {
  // Create Geometry variable
  Geometry geometry = g_context->createGeometry();
  geometry->setPrimitiveCount(1);

  // Set bounding box program
  Program bound = g_context->createProgramFromPTXString(volume_sphere_programs,
                                                        "get_bounds");
  geometry->setBoundingBoxProgram(bound);

  // Set intersection program
  Program intersect = g_context->createProgramFromPTXString(
      volume_sphere_programs, "hit_sphere");
  geometry->setIntersectionProgram(intersect);

  // Basic Parameters
  geometry["center"]->setFloat(center.x, center.y, center.z);
  geometry["radius"]->setFloat(radius);
  geometry["density"]->setFloat(density);

  // Create GeometryInstance
  GeometryInstance gi = g_context->createGeometryInstance();
  gi->setGeometry(geometry);
  gi->setMaterialCount(1);

  // Assign material and texture programs
  Buffer texture_buffers =
      g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 1);
  callableProgramId<int(int)> *tex_data =
      static_cast<callableProgramId<int(int)> *>(texture_buffers->map());

  Program texture = material.assignTo(gi, g_context);
  tex_data[0] = callableProgramId<int(int)>(texture->getId());

  texture_buffers->unmap();
  gi["sample_texture"]->setBuffer(texture_buffers);

  return gi;
}

// Moving Sphere constructor
GeometryInstance Moving_Sphere(const float3 &center0, const float3 &center1,
                               const float t0, const float t1,
                               const float radius, const Materials &material,
                               Context &g_context) {
  // Create Geometry variable
  Geometry geometry = g_context->createGeometry();
  geometry->setPrimitiveCount(1);

  // Set bounding box program
  Program bound = g_context->createProgramFromPTXString(moving_sphere_programs,
                                                        "get_bounds");
  geometry->setBoundingBoxProgram(bound);

  // Set intersection program
  Program intersect = g_context->createProgramFromPTXString(
      moving_sphere_programs, "hit_sphere");
  geometry->setIntersectionProgram(intersect);

  // Basic Parameters
  geometry["center0"]->setFloat(center0.x, center0.y, center0.z);
  geometry["center1"]->setFloat(center1.x, center1.y, center1.z);
  geometry["radius"]->setFloat(radius);
  geometry["time0"]->setFloat(t0);
  geometry["time1"]->setFloat(t1);

  // Create GeometryInstance
  GeometryInstance gi = g_context->createGeometryInstance();
  gi->setGeometry(geometry);
  gi->setMaterialCount(1);

  // Assign material and texture programs
  Buffer texture_buffers =
      g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 1);
  callableProgramId<int(int)> *tex_data =
      static_cast<callableProgramId<int(int)> *>(texture_buffers->map());

  Program texture = material.assignTo(gi, g_context);
  tex_data[0] = callableProgramId<int(int)>(texture->getId());

  texture_buffers->unmap();
  gi["sample_texture"]->setBuffer(texture_buffers);

  return gi;
}

// Axis-alligned Rectangle constructor
GeometryInstance Rectangle(const float a0, const float a1, const float b0,
                           const float b1, const float k, const bool flip,
                           const AXIS ax, const Materials &material,
                           Context &g_context) {
  // Create Geometry variable
  Geometry geometry = g_context->createGeometry();
  geometry->setPrimitiveCount(1);

  // Set bounding box and intersection programs depending on axis
  Program bound, intersect;
  switch (ax) {
    case X_AXIS:
      bound = g_context->createProgramFromPTXString(aarect_programs,
                                                    "get_bounds_X");
      intersect =
          g_context->createProgramFromPTXString(aarect_programs, "hit_rect_X");
      break;
    case Y_AXIS:
      bound = g_context->createProgramFromPTXString(aarect_programs,
                                                    "get_bounds_Y");
      intersect =
          g_context->createProgramFromPTXString(aarect_programs, "hit_rect_Y");
      break;
    case Z_AXIS:
      bound = g_context->createProgramFromPTXString(aarect_programs,
                                                    "get_bounds_Z");
      intersect =
          g_context->createProgramFromPTXString(aarect_programs, "hit_rect_Z");
      break;
  }
  geometry->setBoundingBoxProgram(bound);
  geometry->setIntersectionProgram(intersect);

  // Basic Parameters
  geometry["a0"]->setFloat(a0);
  geometry["a1"]->setFloat(a1);
  geometry["b0"]->setFloat(b0);
  geometry["b1"]->setFloat(b1);
  geometry["k"]->setFloat(k);
  geometry["flip"]->setInt(flip);

  // Create GeometryInstance
  GeometryInstance gi = g_context->createGeometryInstance();
  gi->setGeometry(geometry);
  gi->setMaterialCount(1);

  // Assign material and texture programs
  Buffer texture_buffers =
      g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 1);
  callableProgramId<int(int)> *tex_data =
      static_cast<callableProgramId<int(int)> *>(texture_buffers->map());

  Program texture = material.assignTo(gi, g_context);
  tex_data[0] = callableProgramId<int(int)>(texture->getId());

  texture_buffers->unmap();
  gi["sample_texture"]->setBuffer(texture_buffers);

  return gi;
}

// Box made of rectangle primitives
GeometryGroup Box(const float3 &p0, const float3 &p1, Materials &material,
                  Context &g_context) {
  // Vector of primitives
  std::vector<GeometryInstance> d_list;

  d_list.push_back(Rectangle(p0.x, p1.x, p0.y, p1.y, p0.z, true, Z_AXIS,
                             material, g_context));
  d_list.push_back(Rectangle(p0.x, p1.x, p0.y, p1.y, p1.z, false, Z_AXIS,
                             material, g_context));

  d_list.push_back(Rectangle(p0.x, p1.x, p0.z, p1.z, p0.y, true, Y_AXIS,
                             material, g_context));
  d_list.push_back(Rectangle(p0.x, p1.x, p0.z, p1.z, p1.y, false, Y_AXIS,
                             material, g_context));

  d_list.push_back(Rectangle(p0.y, p1.y, p0.z, p1.z, p0.x, true, X_AXIS,
                             material, g_context));
  d_list.push_back(Rectangle(p0.y, p1.y, p0.z, p1.z, p1.x, false, X_AXIS,
                             material, g_context));

  GeometryGroup d_world = g_context->createGeometryGroup();
  d_world->setAcceleration(g_context->createAcceleration("Trbvh"));
  d_world->setChildCount((int)d_list.size());
  for (int i = 0; i < d_list.size(); i++) d_world->setChild(i, d_list[i]);

  return d_world;
}

// Box of volumetric material
GeometryInstance Volume_Box(const float3 &p0, const float3 &p1,
                            const float density, Materials &material,
                            Context &g_context) {
  // Create Geometry variable
  Geometry geometry = g_context->createGeometry();
  geometry->setPrimitiveCount(1);

  // Set bounding box program
  Program bound =
      g_context->createProgramFromPTXString(volume_box_programs, "get_bounds");
  geometry->setBoundingBoxProgram(bound);

  // Set intersection program
  Program intersect =
      g_context->createProgramFromPTXString(volume_box_programs, "hit_volume");
  geometry->setIntersectionProgram(intersect);

  // Basic parameters
  geometry["boxmin"]->setFloat(p0.x, p0.y, p0.z);
  geometry["boxmax"]->setFloat(p1.x, p1.y, p1.z);
  geometry["density"]->setFloat(density);

  // Create GeometryInstance
  GeometryInstance gi = g_context->createGeometryInstance();
  gi->setGeometry(geometry);
  gi->setMaterialCount(1);

  // Assign material and texture programs
  Buffer texture_buffers =
      g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 1);
  callableProgramId<int(int)> *tex_data =
      static_cast<callableProgramId<int(int)> *>(texture_buffers->map());

  Program texture = material.assignTo(gi, g_context);
  tex_data[0] = callableProgramId<int(int)>(texture->getId());

  texture_buffers->unmap();
  gi["sample_texture"]->setBuffer(texture_buffers);

  return gi;
}

// Triangle constructor
GeometryInstance Triangle(const float3 &a, const float2 &a_uv, const float3 &b,
                          const float2 &b_uv, const float3 &c,
                          const float2 &c_uv, const float scale,
                          const Materials &material, Context &g_context) {
  // Create Geometry variable
  Geometry geometry = g_context->createGeometry();
  geometry->setPrimitiveCount(1);

  // Set bounding box program
  Program bound =
      g_context->createProgramFromPTXString(triangle_programs, "get_bounds");
  geometry->setBoundingBoxProgram(bound);

  // Set intersection program
  Program intersect =
      g_context->createProgramFromPTXString(triangle_programs, "hit_triangle");
  geometry->setIntersectionProgram(intersect);

  // basic parameters
  geometry["a"]->setFloat(a.x, a.y, a.z);
  geometry["a_uv"]->setFloat(a_uv.x, a_uv.y);
  geometry["b"]->setFloat(b.x, b.y, b.z);
  geometry["b_uv"]->setFloat(b_uv.x, b_uv.y);
  geometry["c"]->setFloat(c.x, c.y, c.z);
  geometry["c_uv"]->setFloat(c_uv.x, c_uv.y);
  geometry["scale"]->setFloat(scale);

  // precomputed variables
  const float3 e1(b - a);
  geometry["e1"]->setFloat(e1.x, e1.y, e1.z);

  const float3 e2(c - a);
  geometry["e2"]->setFloat(e2.x, e2.y, e2.z);

  // Geometric normal
  const float3 normal(unit_vector(cross(e1, e2)));
  geometry["normal"]->setFloat(normal.x, normal.y, normal.z);

  // Create GeometryInstance
  GeometryInstance gi = g_context->createGeometryInstance();
  gi->setGeometry(geometry);
  gi->setMaterialCount(1);

  // Assign material and texture programs
  Buffer texture_buffers =
      g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 1);
  callableProgramId<int(int)> *tex_data =
      static_cast<callableProgramId<int(int)> *>(texture_buffers->map());

  Program texture = material.assignTo(gi, g_context);
  tex_data[0] = callableProgramId<int(int)>(texture->getId());

  texture_buffers->unmap();
  gi["sample_texture"]->setBuffer(texture_buffers);

  return gi;
}

// Triangle auxiliar struct definition
struct Triangle_Struct {
  Triangle_Struct(int &i,  // Primitive index
                  float3 &aa, float3 &bb,
                  float3 &cc,  // Vertex
                  float2 &aa_uv, float2 &bb_uv,
                  float2 &cc_uv,  // Texcoord
                  float3 &aa_n, float3 &bb_n,
                  float3 &cc_n,  // Vertex normals
                  int id)
      :  // Materials ID
        index(i),
        material_id(id),
        a(aa),
        b(bb),
        c(cc),
        a_uv(aa_uv),
        b_uv(bb_uv),
        c_uv(cc_uv),
        a_n(aa_n),
        b_n(bb_n),
        c_n(cc_n) {
    e1 = b - a;
    e2 = c - a;
  }

  int index, material_id;
  float3 a, b, c;
  float2 a_uv, b_uv, c_uv;
  float3 a_n, b_n, c_n;
  float3 e1, e2;
};

// Load and convert mesh
GeometryInstance Mesh(const std::string &fileName, Context &g_context,
                      Materials &custom_material,
                      bool use_custom_material = false,
                      const std::string assetsFolder = "assets/",
                      float scale = 1.f) {
  std::vector<Triangle_Struct> triangles;
  objl::Loader Loader;

  printf("Loading Model\n");
  bool loadout = Loader.LoadFile((char *)(assetsFolder + fileName).c_str());

  // If mesh wasn't loaded, output an error
  if (!loadout) {
    // Output Error
    printf(
        "Failed to Load File. May have failed to find it or it was not an .obj "
        "file.\n");
    return NULL;
  }
  printf("\nModel Loaded\n\n");

  // Load Materials
  std::map<std::string, int> material_map;
  std::vector<Materials *> material_list;
  if (use_custom_material)
    material_list.push_back(&custom_material);
  else {
    if (!Loader.LoadedMaterials.empty()) {
      for (int i = 0; i < Loader.LoadedMaterials.size(); i++) {
        material_map[Loader.LoadedMaterials[i].name] = i;

        Texture *tex;
        // Mesh has no texture but has a defined color.
        if (Loader.LoadedMaterials[i].map_Kd.empty())
          tex = new Constant_Texture(make_float3(
              Loader.LoadedMaterials[i].Kd.X, Loader.LoadedMaterials[i].Kd.Y,
              Loader.LoadedMaterials[i].Kd.Z));

        // Mesh has a defined texture.
        else {
          std::string imgPath = assetsFolder + Loader.LoadedMaterials[i].map_Kd;
          tex = new Image_Texture(imgPath);  // load diffuse map/texture
        }

        material_list.push_back(new Lambertian(tex));
      }
    }

    // Model has no assigned MTL file, assign random color
    else {
      material_list.push_back(
          new Lambertian(new Constant_Texture(make_float3(rnd()))));
    }
  }

  // Load Geometry
  for (int i = 0; i < Loader.LoadedMeshes.size(); i++) {
    // Copy one of the loaded meshes to be our current mesh
    objl::Mesh curMesh = Loader.LoadedMeshes[i];

    // Iterate over each face, going through every 3rd index
    float2 a_uv = make_float2(0.f);
    float2 b_uv = make_float2(0.f);
    float2 c_uv = make_float2(0.f);
    for (int j = 0; j < curMesh.Indices.size(); j += 3) {
      printf("Mesh %d / %d - %2.f%% Converted  \r", i + 1,
             (int)Loader.LoadedMeshes.size(),
             j * 100.f / curMesh.Indices.size());

      // Face vertex
      int ia = curMesh.Indices[j];
      float3 a = make_float3(scale * curMesh.Vertices[ia].Position.X,
                             scale * curMesh.Vertices[ia].Position.Y,
                             scale * curMesh.Vertices[ia].Position.Z);

      int ib = curMesh.Indices[j + 1];
      float3 b = make_float3(scale * curMesh.Vertices[ib].Position.X,
                             scale * curMesh.Vertices[ib].Position.Y,
                             scale * curMesh.Vertices[ib].Position.Z);

      int ic = curMesh.Indices[j + 2];
      float3 c = make_float3(scale * curMesh.Vertices[ic].Position.X,
                             scale * curMesh.Vertices[ic].Position.Y,
                             scale * curMesh.Vertices[ic].Position.Z);

      // Face UV coordinates
      a_uv = make_float2(curMesh.Vertices[ia].TextureCoordinate.X,
                         curMesh.Vertices[ia].TextureCoordinate.Y);
      b_uv = make_float2(curMesh.Vertices[ib].TextureCoordinate.X,
                         curMesh.Vertices[ib].TextureCoordinate.Y);
      c_uv = make_float2(curMesh.Vertices[ic].TextureCoordinate.X,
                         curMesh.Vertices[ic].TextureCoordinate.Y);

      // Face vertex normal
      float3 a_n = make_float3(curMesh.Vertices[ia].Normal.X,
                               curMesh.Vertices[ia].Normal.Y,
                               curMesh.Vertices[ia].Normal.Z);

      float3 b_n = make_float3(curMesh.Vertices[ib].Normal.X,
                               curMesh.Vertices[ib].Normal.Y,
                               curMesh.Vertices[ib].Normal.Z);

      float3 c_n = make_float3(curMesh.Vertices[ic].Normal.X,
                               curMesh.Vertices[ic].Normal.Y,
                               curMesh.Vertices[ic].Normal.Z);

      // Add triangle to vector
      int ind = (int)triangles.size();
      std::string name = curMesh.MeshMaterial.name;
      triangles.push_back(Triangle_Struct(ind, a, b, c, a_uv, b_uv, c_uv, a_n,
                                          b_n, c_n, material_map[name]));
    }
  }
  printf("\n");

  // create vertex_buffer
  int size = (int)triangles.size();
  Buffer vertex_buffer =
      g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 3 * size);
  float3 *vertex_map = static_cast<float3 *>(vertex_buffer->map());

  for (int i = 0; i < size; i++) {
    vertex_map[3 * i] =
        make_float3(triangles[i].a.x, triangles[i].a.y, triangles[i].a.z);
    vertex_map[3 * i + 1] =
        make_float3(triangles[i].b.x, triangles[i].b.y, triangles[i].b.z);
    vertex_map[3 * i + 2] =
        make_float3(triangles[i].c.x, triangles[i].c.y, triangles[i].c.z);
    printf("Vertex Buffer Assigned - %2.f%%  \r", i * 100.f / size);
  }
  printf("\n");
  vertex_buffer->unmap();

  // create e_buffer
  Buffer e_buffer =
      g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 2 * size);
  float3 *e_map = static_cast<float3 *>(e_buffer->map());

  for (int i = 0; i < size; i++) {
    e_map[2 * i] =
        make_float3(triangles[i].e1.x, triangles[i].e1.y, triangles[i].e1.z);
    e_map[2 * i + 1] =
        make_float3(triangles[i].e2.x, triangles[i].e2.y, triangles[i].e2.z);
    printf("E Buffer Assigned - %2.f%%  \r", i * 100.f / size);
  }
  printf("\n");
  e_buffer->unmap();

  // create normal_buffer
  Buffer normal_buffer =
      g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 3 * size);
  float3 *n_map = static_cast<float3 *>(normal_buffer->map());

  for (int i = 0; i < size; i++) {
    n_map[3 * i] =
        make_float3(triangles[i].a_n.x, triangles[i].a_n.y, triangles[i].a_n.z);
    n_map[3 * i + 1] =
        make_float3(triangles[i].b_n.x, triangles[i].b_n.y, triangles[i].b_n.z);
    n_map[3 * i + 2] =
        make_float3(triangles[i].c_n.x, triangles[i].c_n.y, triangles[i].c_n.z);
    printf("Normal Buffer Assigned - %2.f%%  \r", i * 100.f / size);
  }
  printf("\n");
  normal_buffer->unmap();

  // create texcoord_buffer
  Buffer texcoord_buffer =
      g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, 3 * size);
  float2 *tex_map = static_cast<float2 *>(texcoord_buffer->map());

  for (int i = 0; i < size; i++) {
    tex_map[3 * i] = make_float2(triangles[i].a_uv.x, triangles[i].a_uv.y);
    tex_map[3 * i + 1] = make_float2(triangles[i].b_uv.x, triangles[i].b_uv.y);
    tex_map[3 * i + 2] = make_float2(triangles[i].c_uv.x, triangles[i].c_uv.y);
    printf("Texture Buffer Assigned - %2.f%%  \r", i * 100.f / size);
  }
  printf("\n");
  texcoord_buffer->unmap();

  // create material_id_buffer
  Buffer material_id_buffer =
      g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, size);
  float *id_map = static_cast<float *>(material_id_buffer->map());

  for (int i = 0; i < size; i++) {
    id_map[i] = (float)triangles[i].material_id;
  }
  material_id_buffer->unmap();

  // Create Geometry variable
  Geometry geometry = g_context->createGeometry();
  geometry->setPrimitiveCount(size);

  // Set buffers
  geometry["vertex_buffer"]->setBuffer(vertex_buffer);
  geometry["e_buffer"]->setBuffer(e_buffer);
  geometry["normal_buffer"]->setBuffer(normal_buffer);
  geometry["texcoord_buffer"]->setBuffer(texcoord_buffer);
  geometry["material_id_buffer"]->setBuffer(material_id_buffer);

  // Set bounding box program
  Program bound =
      g_context->createProgramFromPTXString(mesh_programs, "mesh_bounds");
  geometry->setBoundingBoxProgram(bound);

  // Set intersection program
  Program intersect =
      g_context->createProgramFromPTXString(mesh_programs, "mesh_intersection");
  geometry->setIntersectionProgram(intersect);

  // Create GeometryInstance
  GeometryInstance gi = g_context->createGeometryInstance();
  gi->setGeometry(geometry);

  // using materials from the model
  if (!use_custom_material) {
    gi->setMaterialCount((int)material_list.size());

    Buffer texture_buffers = g_context->createBuffer(
        RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, material_list.size());
    callableProgramId<int(int)> *tex_data =
        static_cast<callableProgramId<int(int)> *>(texture_buffers->map());

    for (int i = 0; i < material_list.size(); i++) {
      printf("Materials Assigned - %2.f%%  \r", i * 100.f / size);
      Program texture = material_list[i]->assignTo(gi, g_context, i);
      tex_data[i] = callableProgramId<int(int)>(texture->getId());
    }
    printf("\n");

    texture_buffers->unmap();

    gi["sample_texture"]->setBuffer(texture_buffers);
    gi["single_mat"]->setInt(false);
  }

  // using a given material
  else {
    gi->setMaterialCount(1);

    Buffer texture_buffers =
        g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 1);
    callableProgramId<int(int)> *tex_data =
        static_cast<callableProgramId<int(int)> *>(texture_buffers->map());

    Program texture = custom_material.assignTo(gi, g_context, 0);
    tex_data[0] = callableProgramId<int(int)>(texture->getId());

    texture_buffers->unmap();

    gi["sample_texture"]->setBuffer(texture_buffers);
    gi["single_mat"]->setInt(true);
  }

  printf("\nFinished Loading Model\n\n");

  return gi;
}

// Plane constructor
GeometryInstance Plane(const float &center, const AXIS ax,
                       const bool invert_plane, const Materials &material,
                       Context &g_context) {
  // Create Geometry variable
  Geometry geometry = g_context->createGeometry();
  geometry->setPrimitiveCount(1);

  // Set bounding box and intersection programs
  Program bound =
      g_context->createProgramFromPTXString(plane_programs, "get_bounds");
  Program intersect =
      g_context->createProgramFromPTXString(plane_programs, "hit_plane");
  geometry->setBoundingBoxProgram(bound);
  geometry->setIntersectionProgram(intersect);

  // check if plane normal should be inverted
  int invert = invert_plane ? -1 : 1;

  // assign center and normal according to a given axis
  switch (ax) {
    case X_AXIS:
      geometry["center"]->setFloat(center, 0.f, 0.f);
      geometry["normal"]->setFloat(invert * 1.f, 0.f, 0.f);
      break;

    case Y_AXIS:
      geometry["center"]->setFloat(0.f, center, 0.f);
      geometry["normal"]->setFloat(0.f, invert * 1.f, 0.f);
      break;

    case Z_AXIS:
      geometry["center"]->setFloat(0.f, 0.f, center);
      geometry["normal"]->setFloat(0.f, 0.f, invert * 1.f);
      break;
  }

  // Create GeometryInstance
  GeometryInstance gi = g_context->createGeometryInstance();
  gi->setGeometry(geometry);
  gi->setMaterialCount(1);

  // Assign material and texture programs
  Buffer texture_buffers =
      g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 1);
  callableProgramId<int(int)> *tex_data =
      static_cast<callableProgramId<int(int)> *>(texture_buffers->map());

  Program texture = material.assignTo(gi, g_context);
  tex_data[0] = callableProgramId<int(int)>(texture->getId());

  texture_buffers->unmap();
  gi["sample_texture"]->setBuffer(texture_buffers);

  return gi;
}

#endif