#ifndef MESHH
#define MESHH

#include "hitables.hpp"

#include "../lib/tiny_obj_loader.h"

#include <map>

extern "C" const char Mesh_PTX[];

// Sources:
// - GeometryTriangles setup from the OptiX 6.0 SDK samples
// - File and material conversion from syoyo's tinyobj example:
// https://github.com/syoyo/tinyobjloader/tree/master/examples/viewer

// Parse and convert OBJ file
class Mesh {
  // - If no assets folder is given as parameter, model is in CWD.
  // - If no material is given as parameter, parse material from MTL file.
 public:
  Mesh(std::string fileName)
      : fileName(fileName), assetsFolder(""), givenMaterial(nullptr) {}

  Mesh(std::string fileName, std::string assetsFolder)
      : fileName(fileName),
        assetsFolder(assetsFolder),
        givenMaterial(nullptr) {}

  Mesh(std::string fileName, Host_Material *givenMaterial)
      : fileName(fileName), assetsFolder(""), givenMaterial(givenMaterial) {}

  Mesh(std::string fileName, std::string assetsFolder,
       Host_Material *givenMaterial)
      : fileName(fileName),
        assetsFolder(assetsFolder),
        givenMaterial(givenMaterial) {}

  // Get GeometryInstance of Mesh
  GeometryInstance getGeometryInstance(Context &g_context) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;

    // load obj & mtl files
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                                (assetsFolder + fileName).c_str(),
                                assetsFolder.c_str(), true);

    // Check if there was a warning while reading the file
    if (!warn.empty()) std::cout << "WARN: " << warn << std::endl;

    // Check if there was an error while reading the file
    if (!err.empty()) std::cerr << "ERR: " << err << std::endl;

    // If file wasn't read successfully, close
    if (!ret) {
      printf("Failed to load/parse .obj.");
      system("PAUSE");
      exit(0);
    }

    // Convert Materials
    Material device_material;  // material object to be passed to device
    std::map<std::string, int> material_map;  // [Name, index] map

    // if no material is given, convert MTL file
    if (givenMaterial == nullptr) {
      Texture_List textures;

      // for each material in the MTL file
      for (int m = 0; m < materials.size(); m++) {
        tinyobj::material_t *mp = &materials[m];

        // Create Texture from image file or color value
        Texture *tex;
        if (mp->diffuse_texname.length() > 0)
          tex = new Image_Texture(assetsFolder + mp->diffuse_texname);
        else
          tex = new Constant_Texture(mp->ambient[0],   // R
                                     mp->ambient[1],   // G
                                     mp->ambient[2]);  // B

        // Assign texture index to the Material's name
        int texId = textures.push(tex);
        material_map[mp->name] = texId;
      }

      // Create a vector of textures
      Host_Material *host_material;
      host_material = new Lambertian(new Vector_Texture(textures.texList));
      device_material = host_material->assignTo(g_context);
    }

    else  // a Material was given, assign it to the Mesh
      device_material = givenMaterial->assignTo(g_context);

    // Convert Geoemtry
    std::vector<int> mat_vector;             // material index vector
    std::vector<uint3> i_vector;             // face index vector
    std::vector<float2> t_vector;            // texcoord vector
    std::vector<float3> v_vector, n_vector;  // vertex and normal vector

    int index = 0;
    std::vector<tinyobj::shape_t>::const_iterator it;
    for (it = shapes.begin(); it < shapes.end(); ++it) {
      const tinyobj::shape_t &shape = *it;

      // for each face of the current mesh
      for (size_t f = 0; f < shape.mesh.indices.size() / 3; f++) {
        // Get the three indexes of the face (all faces are triangular)
        tinyobj::index_t idx0 = shape.mesh.indices[3 * f + 0];
        tinyobj::index_t idx1 = shape.mesh.indices[3 * f + 1];
        tinyobj::index_t idx2 = shape.mesh.indices[3 * f + 2];

        // get the vertex coordinates
        v_vector.push_back(make_Vertex3(attrib.vertices, idx0.vertex_index));
        v_vector.push_back(make_Vertex3(attrib.vertices, idx1.vertex_index));
        v_vector.push_back(make_Vertex3(attrib.vertices, idx2.vertex_index));

        // get the normal coordinates
        n_vector.push_back(make_Vertex3(attrib.normals, idx0.normal_index));
        n_vector.push_back(make_Vertex3(attrib.normals, idx1.normal_index));
        n_vector.push_back(make_Vertex3(attrib.normals, idx2.normal_index));

        // get the vertex tex coordinates
        t_vector.push_back(make_Vertex2(attrib.texcoords, idx0.texcoord_index));
        t_vector.push_back(make_Vertex2(attrib.texcoords, idx1.texcoord_index));
        t_vector.push_back(make_Vertex2(attrib.texcoords, idx2.texcoord_index));

        // set index vector
        i_vector.push_back(make_uint3(index++, index++, index++));

        // set material index vector
        if (givenMaterial == nullptr) {
          int m = material_map[materials[shape.mesh.material_ids[f]].name];
          mat_vector.push_back(m);
        } else
          mat_vector.push_back(0);  // uses the material given as parameter
      }
    }

    // Create a GeometryTriangles object
    optix::GeometryTriangles geom_tri = g_context->createGeometryTriangles();

    // create buffers
    Buffer v_buffer = createBuffer(v_vector, g_context);
    Buffer n_buffer = createBuffer(n_vector, g_context);
    Buffer t_buffer = createBuffer(t_vector, g_context);
    Buffer i_buffer = createBuffer(i_vector, g_context);
    Buffer m_buffer = createBuffer(mat_vector, g_context);

    // set GeometryTriangles parameters
    geom_tri->setPrimitiveCount((int)i_vector.size());
    geom_tri->setTriangleIndices(i_buffer, RT_FORMAT_UNSIGNED_INT3);
    geom_tri->setVertices((int)v_vector.size(), v_buffer, RT_FORMAT_FLOAT3);
    geom_tri->setBuildFlags(RTgeometrybuildflags(0));

    // Set buffers
    geom_tri["vertex_buffer"]->setBuffer(v_buffer);
    geom_tri["normal_buffer"]->setBuffer(n_buffer);
    geom_tri["texcoord_buffer"]->setBuffer(t_buffer);
    geom_tri["index_buffer"]->setBuffer(i_buffer);
    geom_tri["material_buffer"]->setBuffer(m_buffer);

    // Set attribute program
    Program att = createProgram(Mesh_PTX, "attributes", g_context);
    geom_tri->setAttributeProgram(att);

    // returns GeometryInstance
    return g_context->createGeometryInstance(geom_tri, device_material);
  }

  // Apply a rotation to the hitable
  void rotate(float angle, AXIS axis) {
    TransformParameter param(Rotate_Transform,   // Transform type
                             angle,              // Rotation Angle
                             axis,               // Rotation Axis
                             make_float3(0.f),   // Scale value
                             make_float3(0.f));  // Translation delta
    arr.push_back(param);
  }

  // Apply a scale to the hitable
  void scale(float3 scale) {
    TransformParameter param(Scale_Transform,    // Transform type
                             0.f,                // Rotation Angle
                             X_AXIS,             // Rotation Axis
                             scale,              // Scale value
                             make_float3(0.f));  // Translation delta
    arr.push_back(param);
  }

  // Apply a translation to the hitable
  void translate(float3 pos) {
    TransformParameter param(Translate_Transform,  // Transform type
                             0.f,                  // Rotation Angle
                             X_AXIS,               // Rotation Axis
                             make_float3(0.f),     // Scale value
                             pos);                 // Translation delta
    arr.push_back(param);
  }

  // Adds Hitable to the scene graph
  void addTo(Group &d_world, Context &g_context) {
    // reverse vector of transforms
    std::reverse(arr.begin(), arr.end());
    GeometryInstance gi = getGeometryInstance(g_context);
    addAndTransform(gi, d_world, g_context, arr);
  }

 private:
  // Make a float3 vertex out of a vector and an index
  float3 make_Vertex3(std::vector<tinyobj::real_t> &vertices, int index) {
    float x = vertices[3 * index + 0];
    float y = vertices[3 * index + 1];
    float z = vertices[3 * index + 2];
    return make_float3(x, y, z);
  }

  // Make a float2 vertex out of a vector and an index
  float2 make_Vertex2(std::vector<tinyobj::real_t> &vertices, int index) {
    float x = vertices[2 * index + 0];
    float y = vertices[2 * index + 1];
    return make_float2(x, y);
  }

  const Host_Material *givenMaterial;
  const std::string fileName, assetsFolder;
  std::vector<TransformParameter> arr;
};

// List of Mesh variables
class Mesh_List {
 public:
  Mesh_List() {}

  // Appends a mesh to the list and returns its index
  int push(Mesh *mesh) {
    int index = (int)list.size();

    list.push_back(mesh);

    return index;
  }

  // Apply a rotation to the hitable
  void rotate(float angle, AXIS axis) {
    TransformParameter param(Rotate_Transform,   // Transform type
                             angle,              // Rotation Angle
                             axis,               // Rotation Axis
                             make_float3(0.f),   // Scale value
                             make_float3(0.f));  // Translation delta
    arr.push_back(param);
  }

  // Apply a scale to the hitable
  void scale(float3 scale) {
    TransformParameter param(Scale_Transform,    // Transform type
                             0.f,                // Rotation Angle
                             X_AXIS,             // Rotation Axis
                             scale,              // Scale value
                             make_float3(0.f));  // Translation delta
    arr.push_back(param);
  }

  // Apply a translation to the hitable
  void translate(float3 pos) {
    TransformParameter param(Translate_Transform,  // Transform type
                             0.f,                  // Rotation Angle
                             X_AXIS,               // Rotation Axis
                             make_float3(0.f),     // Scale value
                             pos);                 // Translation delta
    arr.push_back(param);
  }

  // converts list to a GeometryGroup object
  GeometryGroup getGroup(Context &g_context) {
    GeometryGroup gg = g_context->createGeometryGroup();
    gg->setAcceleration(g_context->createAcceleration("Trbvh"));

    for (int i = 0; i < list.size(); i++)
      gg->addChild(list[i]->getGeometryInstance(g_context));

    return gg;
  }

  // FIXME: not working
  // adds and transforms Mesh_List as a whole to the scene graph
  void addListTo(Group &d_world, Context &g_context) {
    GeometryGroup gg = getGroup(g_context);
    addAndTransform(gg, d_world, g_context, arr);
  }
  // TODO: implement a function that applies the list transforms to the mesh
  // the mesh should still be added separately

  // adds and transforms each list element to the scene graph individually
  void addElementsTo(Group &d_world, Context &g_context) {
    for (int i = 0; i < (int)list.size(); i++)
      list[i]->addTo(d_world, g_context);
  }

 private:
  std::vector<Mesh *> list;
  std::vector<TransformParameter> arr;
};

#endif