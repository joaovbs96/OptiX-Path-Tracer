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
  material.assignTo(gi, g_context);
  
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
  material.assignTo(gi, g_context);
  
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
  material.assignTo(gi, g_context);
  
  return gi;
}

// box made of rectangle primitives
optix::GeometryGroup createBox(const vec3f& p0, 
                               const vec3f& p1, 
                               Material &material, 
                               optix::Context &g_context){

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

optix::GeometryInstance createVolumeBox(const vec3f& p0, const vec3f& p1, const float density, Material &material, optix::Context &g_context){
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
  material.assignTo(gi, g_context);
  
  return gi;
}

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
  material.assignTo(gi, g_context);
  
  return gi;
}

optix::GeometryGroup Mesh(const std::string &fileName, 
                          optix::Context &g_context, 
                          const std::string assetsFolder="assets/", 
                          float scale = 1.f){

  std::vector<optix::GeometryInstance> triangles;
	objl::Loader Loader;

	std::cout << "Loading Model" << std::endl;
	bool loadout = Loader.LoadFile((char*)(assetsFolder + fileName).c_str());

	// Check if obj was loaded
	if (loadout) {
    std::cout << std::endl << "Model Loaded" << std::endl << std::endl;

		// Go through each loaded mesh
		for (int i = 0; i < Loader.LoadedMeshes.size(); i++) {
			// Copy one of the loaded meshes to be our current mesh
			std::cout << '\r' << "                         " << '\r'; // flush line
			std::cout << "Converting Mesh " << (i + 1) << "/" << Loader.LoadedMeshes.size();
			objl::Mesh curMesh = Loader.LoadedMeshes[i];

      Texture *tex;
      vec2f a_uv(0.f), b_uv(0.f), c_uv(0.f);
      bool get_uv = false;

			// Model has no assigned MTL file
      if(Loader.LoadedMaterials.empty())
        tex = new Constant_Texture(vec3f(rnd(), rnd(), rnd()));

      // Model has assigned MTL file, but mesh has no diffuse map
      if(curMesh.MeshMaterial.map_Kd.empty())
        tex = new Constant_Texture(vec3f(curMesh.MeshMaterial.Kd.X, 
                                         curMesh.MeshMaterial.Kd.Y, 
                                         curMesh.MeshMaterial.Kd.Z));

      // Model has assigned MTL file and mesh has diffuse map
      else{
        std::string imgPath = assetsFolder + curMesh.MeshMaterial.map_Kd;
        tex = new Image_Texture(imgPath); // load diffuse map/texture
        get_uv = true;
      }

      // Go through every 3rd index and print the
      // save that these indices represent
      Material *mat = new Lambertian(tex);
      for (int j = 0; j < curMesh.Indices.size(); j += 3) {
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
        if(get_uv){
          a_uv = vec2f(curMesh.Vertices[ia].TextureCoordinate.X,
                       curMesh.Vertices[ia].TextureCoordinate.Y);
          b_uv = vec2f(curMesh.Vertices[ib].TextureCoordinate.X,
                       curMesh.Vertices[ib].TextureCoordinate.Y);
          c_uv = vec2f(curMesh.Vertices[ic].TextureCoordinate.X,
                       curMesh.Vertices[ic].TextureCoordinate.Y);
        }

        triangles.push_back(createTriangle(a, a_uv, 
                                            b, b_uv, 
                                            c, c_uv,
                                            scale, *mat, g_context));
      }
		}
		std::cout << std::endl;
		std::cout << "Meshes Converted" << std::endl;
		
    optix::GeometryGroup group = g_context->createGeometryGroup();
    group->setAcceleration(g_context->createAcceleration("Trbvh"));
    group->setChildCount((int)triangles.size());
    
    for (int i = 0; i < triangles.size(); i++)
      group->setChild(i, triangles[i]);
    
    return group;
	}

	// If it wasn't, output an error
	else {
		// Output Error
		std::cout << "Failed to Load File. May have failed to find it or it was not an .obj file." << std::endl;
		return NULL;
	}

}

#endif