#ifndef HITABLESH
#define HITABLESH

#include <optix.h>
#include <optixu/optixpp.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

#include "../programs/vec.h"
#include "materials.h"
#include "transforms.h"

#include "../lib/OBJ_Loader.h"

#include <map>

/*! the precompiled programs/raygen.cu code (in ptx) that our
  cmake magic will precompile (to ptx) and link to the generated
  executable (ie, we can simply declare and use this here as
  'extern'.  */
extern "C" const char embedded_sphere_programs[];
extern "C" const char embedded_moving_sphere_programs[];
extern "C" const char embedded_aarect_programs[];
extern "C" const char embedded_box_programs[];
extern "C" const char embedded_volume_sphere_programs[];
extern "C" const char embedded_volume_box_programs[];
extern "C" const char embedded_triangle_programs[];
extern "C" const char embedded_mesh_programs[];
extern "C" const char embedded_plane_programs[];

typedef enum{
  X_AXIS,
  Y_AXIS,
  Z_AXIS
} AXIS;

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

  optix::Buffer texture_buffers = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 1);
  optix::callableProgramId<int(int)>* tex_data = static_cast<optix::callableProgramId<int(int)>*>(texture_buffers->map());
    
  optix::Program texture = material.assignTo(gi, g_context);
  tex_data[0] = optix::callableProgramId<int(int)>(texture->getId());
  texture_buffers->unmap();
  gi["sample_texture"]->setBuffer(texture_buffers);
  
  return gi;
}

// Sphere constructor
optix::GeometryInstance createVolumeSphere(const vec3f &center, const float radius, const float density, const Material &material, optix::Context &g_context) {
  optix::Geometry geometry = g_context->createGeometry();
  
  geometry->setPrimitiveCount(1);
  geometry->setBoundingBoxProgram(g_context->createProgramFromPTXString(embedded_volume_sphere_programs, "get_bounds"));
  geometry->setIntersectionProgram(g_context->createProgramFromPTXString(embedded_volume_sphere_programs, "hit_sphere"));
  
  geometry["center"]->setFloat(center.x,center.y,center.z);
  geometry["radius"]->setFloat(radius);
  geometry["density"]->setFloat(density);

  optix::GeometryInstance gi = g_context->createGeometryInstance();
  gi->setGeometry(geometry);
  gi->setMaterialCount(1);

  optix::Buffer texture_buffers = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 1);
  optix::callableProgramId<int(int)>* tex_data = static_cast<optix::callableProgramId<int(int)>*>(texture_buffers->map());
    
  optix::Program texture = material.assignTo(gi, g_context);
  tex_data[0] = optix::callableProgramId<int(int)>(texture->getId());
  texture_buffers->unmap();
  gi["sample_texture"]->setBuffer(texture_buffers);
  
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

  optix::Buffer texture_buffers = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 1);
  optix::callableProgramId<int(int)>* tex_data = static_cast<optix::callableProgramId<int(int)>*>(texture_buffers->map());
    
  optix::Program texture = material.assignTo(gi, g_context);
  tex_data[0] = optix::callableProgramId<int(int)>(texture->getId());
  texture_buffers->unmap();
  gi["sample_texture"]->setBuffer(texture_buffers);
  
  return gi;
}

// Axis-alligned Rectangle constructors
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

  optix::Buffer texture_buffers = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 1);
  optix::callableProgramId<int(int)>* tex_data = static_cast<optix::callableProgramId<int(int)>*>(texture_buffers->map());
    
  optix::Program texture = material.assignTo(gi, g_context);
  tex_data[0] = optix::callableProgramId<int(int)>(texture->getId());
  texture_buffers->unmap();
  gi["sample_texture"]->setBuffer(texture_buffers);
  
  return gi;
}

optix::GeometryInstance createYRect(const float a0, 
                                    const float a1, 
                                    const float b0, 
                                    const float b1, 
                                    const float k, 
                                    const bool flip, 
                                    const Material &material, 
                                    optix::Context &g_context) {

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

  optix::Buffer texture_buffers = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 1);
  optix::callableProgramId<int(int)>* tex_data = static_cast<optix::callableProgramId<int(int)>*>(texture_buffers->map());
    
  optix::Program texture = material.assignTo(gi, g_context);
  tex_data[0] = optix::callableProgramId<int(int)>(texture->getId());
  texture_buffers->unmap();
  gi["sample_texture"]->setBuffer(texture_buffers);
  
  return gi;
}

optix::GeometryInstance createZRect(const float a0, 
                                    const float a1, 
                                    const float b0, 
                                    const float b1, 
                                    const float k, 
                                    const bool flip, 
                                    const Material &material, 
                                    optix::Context &g_context) {

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

  optix::Buffer texture_buffers = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 1);
  optix::callableProgramId<int(int)>* tex_data = static_cast<optix::callableProgramId<int(int)>*>(texture_buffers->map());
    
  optix::Program texture = material.assignTo(gi, g_context);
  tex_data[0] = optix::callableProgramId<int(int)>(texture->getId());
  texture_buffers->unmap();
  gi["sample_texture"]->setBuffer(texture_buffers);
  
  return gi;
}

// box made of rectangle primitives
optix::GeometryGroup createBox(const vec3f& p0, 
                               const vec3f& p1, 
                               Material &material, 
                               optix::Context &g_context) {

  std::vector<optix::GeometryInstance> d_list;

  d_list.push_back(createZRect(p0.x, p1.x, p0.y, p1.y, p0.z, true, material, g_context));//
  d_list.push_back(createZRect(p0.x, p1.x, p0.y, p1.y, p1.z, false, material, g_context));

  d_list.push_back(createYRect(p0.x, p1.x, p0.z, p1.z, p0.y, true, material, g_context));
  d_list.push_back(createYRect(p0.x, p1.x, p0.z, p1.z, p1.y, false, material, g_context));
  
  d_list.push_back(createXRect(p0.y, p1.y, p0.z, p1.z, p0.x, true, material, g_context));
  d_list.push_back(createXRect(p0.y, p1.y, p0.z, p1.z, p1.x, false, material, g_context));

  optix::GeometryGroup d_world = g_context->createGeometryGroup();
  d_world->setAcceleration(g_context->createAcceleration("Trbvh"));
  d_world->setChildCount((int)d_list.size());
  for (int i = 0; i < d_list.size(); i++)
    d_world->setChild(i, d_list[i]);

  return d_world;
}

optix::GeometryInstance createVolumeBox(const vec3f& p0, const vec3f& p1, const float density, Material &material, optix::Context &g_context) {
  optix::Geometry geometry = g_context->createGeometry();
  
  geometry->setPrimitiveCount(1);
  geometry->setBoundingBoxProgram(g_context->createProgramFromPTXString(embedded_volume_box_programs, "get_bounds"));
  geometry->setIntersectionProgram(g_context->createProgramFromPTXString(embedded_volume_box_programs, "hit_volume"));
  
  geometry["boxmin"]->setFloat(p0.x, p0.y, p0.z);
  geometry["boxmax"]->setFloat(p1.x, p1.y, p1.z);
  geometry["density"]->setFloat(density);

  optix::GeometryInstance gi = g_context->createGeometryInstance();
  gi->setGeometry(geometry);
  gi->setMaterialCount(1);

  optix::Buffer texture_buffers = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 1);
  optix::callableProgramId<int(int)>* tex_data = static_cast<optix::callableProgramId<int(int)>*>(texture_buffers->map());
    
  optix::Program texture = material.assignTo(gi, g_context);
  tex_data[0] = optix::callableProgramId<int(int)>(texture->getId());
  texture_buffers->unmap();
  gi["sample_texture"]->setBuffer(texture_buffers);
  
  return gi;
}

struct Triangle {
  Triangle(int &i, // index
           vec3f &aa, // vertex
           vec3f &bb, 
           vec3f &cc, 
           vec2f &aa_uv, // texcoord
           vec2f &bb_uv, 
           vec2f &cc_uv,
           int id) : 
           index(i), 
           a(aa), 
           b(bb), 
           c(cc), 
           a_uv(aa_uv), 
           b_uv(bb_uv), 
           c_uv(cc_uv),
           material_id(id) {
             e1 = b - a;
             e2 = c - a;
             normal = unit_vector(cross(e1, e2));
           }
  int index, material_id;
  vec3f a, b, c;
  vec2f a_uv, b_uv, c_uv;
  vec3f e1, e2, normal;
};

// Triangle constructor
optix::GeometryInstance createTriangle(const vec3f &a, 
                                       const vec2f &a_uv, 
                                       const vec3f &b, 
                                       const vec2f &b_uv, 
                                       const vec3f &c, 
                                       const vec2f &c_uv, 
                                       const float scale, 
                                       const Material &material, 
                                       optix::Context &g_context) {

  optix::Geometry geometry = g_context->createGeometry();
  
  geometry->setPrimitiveCount(1);
  geometry->setBoundingBoxProgram(g_context->createProgramFromPTXString(embedded_triangle_programs, "get_bounds"));
  geometry->setIntersectionProgram(g_context->createProgramFromPTXString(embedded_triangle_programs, "hit_triangle"));
  
  // basic parameters
  geometry["a"]->setFloat(a.x, a.y, a.z);
  geometry["a_uv"]->setFloat(a_uv.x, a_uv.y);
  geometry["b"]->setFloat(b.x, b.y, b.z);
  geometry["b_uv"]->setFloat(b_uv.x, b_uv.y);
  geometry["c"]->setFloat(c.x, c.y, c.z);
  geometry["c_uv"]->setFloat(c_uv.x, c_uv.y);
  geometry["scale"]->setFloat(scale);

  // precomputed variables
  const vec3f e1(b - a);
  geometry["e1"]->setFloat(e1.x, e1.y, e1.z);

  const vec3f e2(c - a);
  geometry["e2"]->setFloat(e2.x, e2.y, e2.z);

  const vec3f normal(unit_vector(cross(e1, e2)));
  geometry["normal"]->setFloat(normal.x, normal.y, normal.z);

  optix::GeometryInstance gi = g_context->createGeometryInstance();
  gi->setGeometry(geometry);
  gi->setMaterialCount(1);

  optix::Buffer texture_buffers = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 1);
  optix::callableProgramId<int(int)>* tex_data = static_cast<optix::callableProgramId<int(int)>*>(texture_buffers->map());
    
  optix::Program texture = material.assignTo(gi, g_context);
  tex_data[0] = optix::callableProgramId<int(int)>(texture->getId());
  texture_buffers->unmap();
  gi["sample_texture"]->setBuffer(texture_buffers);
  
  return gi;
}

optix::GeometryInstance Load_Mesh(const std::string &fileName, 
                          optix::Context &g_context, 
                          Material &custom_material,
                          bool use_custom_material = false,
                          const std::string assetsFolder="assets/", 
                          float scale = 1.f) {

  std::vector<Triangle> triangles;
	objl::Loader Loader;

  printf("Loading Model\n");
	bool loadout = Loader.LoadFile((char*)(assetsFolder + fileName).c_str());

  // If mesh wasn't loaded, output an error
	if (!loadout) {
		// Output Error
    printf("Failed to Load File. May have failed to find it or it was not an .obj file.\n");
		return NULL;
	}

  printf("\nModel Loaded\n\n");

  // Load Materials
  std::map<std::string, int> material_map;
  std::vector<Material*> material_list;
  if(use_custom_material)
    material_list.push_back(&custom_material);
  else{
    if(!Loader.LoadedMaterials.empty()){
      for(int i = 0; i < Loader.LoadedMaterials.size(); i++){
        material_map[Loader.LoadedMaterials[i].name] = i;

        Texture *tex;
        // Mesh has no texture but has a defined color.
        if(Loader.LoadedMaterials[i].map_Kd.empty())
          tex = new Constant_Texture(vec3f(Loader.LoadedMaterials[i].Kd.X, 
                                          Loader.LoadedMaterials[i].Kd.Y, 
                                          Loader.LoadedMaterials[i].Kd.Z));
        
        // Mesh has a defined texture.
        else{
          std::string imgPath = assetsFolder + Loader.LoadedMaterials[i].map_Kd;
          tex = new Image_Texture(imgPath); // load diffuse map/texture
        }

        material_list.push_back(new Lambertian(tex));
      }
    } 
    
    // Model has no assigned MTL file, assign random color
    else {
      material_list.push_back(new Lambertian(new Constant_Texture(vec3f(rnd()))));
    }
  }

  // Load Geometry
  for (int i = 0; i < Loader.LoadedMeshes.size(); i++) {
    // Copy one of the loaded meshes to be our current mesh
    objl::Mesh curMesh = Loader.LoadedMeshes[i];

    // Go through every 3rd index
    vec2f a_uv(0.f), b_uv(0.f), c_uv(0.f);
    for (int j = 0; j < curMesh.Indices.size(); j += 3) {
      printf("Mesh %d / %d - %2.f%% Converted  \r", i + 1, 
          (int)Loader.LoadedMeshes.size(), j * 100.f/curMesh.Indices.size());

      int ia = curMesh.Indices[j];
      vec3f a(scale * curMesh.Vertices[ia].Position.X,
              scale * curMesh.Vertices[ia].Position.Y,
              scale * curMesh.Vertices[ia].Position.Z);

      int ib = curMesh.Indices[j + 1];
      vec3f b(scale * curMesh.Vertices[ib].Position.X,
              scale * curMesh.Vertices[ib].Position.Y,
              scale * curMesh.Vertices[ib].Position.Z);

      int ic = curMesh.Indices[j + 2];
      vec3f c(scale * curMesh.Vertices[ic].Position.X,
              scale * curMesh.Vertices[ic].Position.Y,
              scale * curMesh.Vertices[ic].Position.Z);
      
      // get UV coordinates of the current triangle, if needed
      a_uv = vec2f(curMesh.Vertices[ia].TextureCoordinate.X,
                   curMesh.Vertices[ia].TextureCoordinate.Y);
      b_uv = vec2f(curMesh.Vertices[ib].TextureCoordinate.X,
                   curMesh.Vertices[ib].TextureCoordinate.Y);
      c_uv = vec2f(curMesh.Vertices[ic].TextureCoordinate.X,
                   curMesh.Vertices[ic].TextureCoordinate.Y);

      int ind = (int)triangles.size();
      std::string name = curMesh.MeshMaterial.name;
      triangles.push_back(Triangle(ind, a, b, c, a_uv, b_uv, c_uv, material_map[name]));
    }
  }

  // Triangles has all mesh data w/ materials and indexes
  printf("\n");
  printf("Meshes Converted\n");
  int size = (int)triangles.size();

  // create vertex_buffer
  optix::Buffer vertex_buffer = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 3 * size);
  float3 *vertex_map = static_cast<float3*>(vertex_buffer->map());

  for (int i = 0; i < size; i++) {
    printf("Vertex Buffer Assigned - %2.f%%  \r", i * 100.f / size);
    vertex_map[3 * i] = make_float3(triangles[i].a.x, triangles[i].a.y, triangles[i].a.z);
    vertex_map[3 * i + 1] = make_float3(triangles[i].b.x, triangles[i].b.y, triangles[i].b.z);
    vertex_map[3 * i + 2] = make_float3(triangles[i].c.x, triangles[i].c.y, triangles[i].c.z);
  }
  printf("\n");
  vertex_buffer->unmap();

  // create e_buffer
  optix::Buffer e_buffer = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 2 * size);
  float3 *e_map = static_cast<float3*>(e_buffer->map());

  for (int i = 0; i < size; i++) {
    printf("E Buffer Assigned - %2.f%%  \r", i * 100.f / size);
    e_map[2 * i] = make_float3(triangles[i].e1.x, triangles[i].e1.y, triangles[i].e1.z);
    e_map[2 * i + 1] = make_float3(triangles[i].e2.x, triangles[i].e2.y, triangles[i].e2.z);
  }
  printf("\n");
  e_buffer->unmap();

  // create normal_buffer
  optix::Buffer normal_buffer = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, size);
  float3 *n_map = static_cast<float3*>(normal_buffer->map());

  for (int i = 0; i < size; i++){
    printf("Normal Buffer Assigned - %2.f%%  \r", i * 100.f / size);
    n_map[i] = make_float3(triangles[i].normal.x, triangles[i].normal.y, triangles[i].normal.z);
  }
  printf("\n");
  normal_buffer->unmap();

  // create texcoord_buffer
  optix::Buffer texcoord_buffer = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, 3 * size);
  float2 *tex_map = static_cast<float2*>(texcoord_buffer->map());

  for (int i = 0; i < size; i++) {
    printf("Texture Buffer Assigned - %2.f%%  \r", i * 100.f / size);
    tex_map[3 * i] = make_float2(triangles[i].a_uv.x, triangles[i].a_uv.y);
    tex_map[3 * i + 1] = make_float2(triangles[i].b_uv.x, triangles[i].b_uv.y);
    tex_map[3 * i + 2] = make_float2(triangles[i].c_uv.x, triangles[i].c_uv.y);
  }
  printf("\n");
  texcoord_buffer->unmap();

  // create material_id_buffer
  optix::Buffer material_id_buffer = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, size);
  float *id_map = static_cast<float*>(material_id_buffer->map());
  
  for (int i = 0; i < size; i++) {
    id_map[i] = (float)triangles[i].material_id;
  }
  material_id_buffer->unmap();

  optix::Geometry geometry = g_context->createGeometry();
  geometry["vertex_buffer"]->setBuffer(vertex_buffer);
  geometry["e_buffer"]->setBuffer(e_buffer);
  geometry["normal_buffer"]->setBuffer(normal_buffer);
  geometry["texcoord_buffer"]->setBuffer(texcoord_buffer);
  geometry["material_id_buffer"]->setBuffer(material_id_buffer);
  geometry->setPrimitiveCount(size);
  geometry->setBoundingBoxProgram(g_context->createProgramFromPTXString(embedded_mesh_programs, "mesh_bounds"));
  geometry->setIntersectionProgram(g_context->createProgramFromPTXString(embedded_mesh_programs, "mesh_intersection"));

  optix::GeometryInstance gi = g_context->createGeometryInstance();
  gi->setGeometry(geometry);

  // use materials from the model
  if(!use_custom_material){
    gi->setMaterialCount((int)material_list.size());
    
    optix::Buffer texture_buffers = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, material_list.size());
    optix::callableProgramId<int(int)>* tex_data = static_cast<optix::callableProgramId<int(int)>*>(texture_buffers->map());

    for (int i = 0; i < material_list.size(); i++) {
      printf("Materials Assigned - %2.f%%  \r", i * 100.f / size);
      optix::Program texture = material_list[i]->assignTo(gi, g_context, i);
      tex_data[i] = optix::callableProgramId<int(int)>(texture->getId());
    }
    printf("\n");

    texture_buffers->unmap();
    
    gi["sample_texture"]->setBuffer(texture_buffers);
    gi["single_mat"]->setInt(false);
  }

  // used a given material
  else{
    gi->setMaterialCount(1);

    optix::Buffer texture_buffers = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 1);
    optix::callableProgramId<int(int)>* tex_data = static_cast<optix::callableProgramId<int(int)>*>(texture_buffers->map());

    optix::Program texture = custom_material.assignTo(gi, g_context, 0);
    tex_data[0] = optix::callableProgramId<int(int)>(texture->getId());
    
    texture_buffers->unmap();
    
    gi["sample_texture"]->setBuffer(texture_buffers);
    gi["single_mat"]->setInt(true);
  }

  printf("Finished Loading Model\n");

  return gi;

}

// Plane constructor
optix::GeometryInstance createPlane(const float &center, const AXIS ax, const bool invert_plane, const Material &material, optix::Context &g_context) {
  optix::Geometry geometry = g_context->createGeometry();
  
  geometry->setPrimitiveCount(1);
  geometry->setBoundingBoxProgram(g_context->createProgramFromPTXString(embedded_plane_programs, "get_bounds"));
  geometry->setIntersectionProgram(g_context->createProgramFromPTXString(embedded_plane_programs, "hit_plane"));

  int invert = invert_plane ? -1 : 1;
  
  switch(ax){
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

  optix::GeometryInstance gi = g_context->createGeometryInstance();
  gi->setGeometry(geometry);
  gi->setMaterialCount(1);

  optix::Buffer texture_buffers = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 1);
  optix::callableProgramId<int(int)>* tex_data = static_cast<optix::callableProgramId<int(int)>*>(texture_buffers->map());
    
  optix::Program texture = material.assignTo(gi, g_context);
  tex_data[0] = optix::callableProgramId<int(int)>(texture->getId());
  texture_buffers->unmap();
  gi["sample_texture"]->setBuffer(texture_buffers);
  
  return gi;
}

#endif