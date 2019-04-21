# ======================================================================== #
# Copyright 2018 Ingo Wald                                                 #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

# do some cmake magic to pre-compile the cuda file to ptx, and embed
# the resulting ptx code string into the final executable
cuda_compile_and_embed( exception_program programs/exception.cu )
cuda_compile_and_embed( raygen_program programs/raygen.cu )
cuda_compile_and_embed( sphere_programs programs/hitables/sphere.cu )
cuda_compile_and_embed( moving_sphere_programs programs/hitables/moving_sphere.cu )
cuda_compile_and_embed( miss_program programs/miss.cu )
cuda_compile_and_embed( metal_programs programs/materials/metal.cu )
cuda_compile_and_embed( dielectric_programs programs/materials/dielectric.cu )
cuda_compile_and_embed( lambertian_programs programs/materials/lambertian.cu )
cuda_compile_and_embed( constant_texture_programs programs/textures/constant_texture.cu )
cuda_compile_and_embed( aarect_programs programs/hitables/aarect.cu )
cuda_compile_and_embed( checker_texture_programs programs/textures/checkered_texture.cu )
cuda_compile_and_embed( noise_texture_programs programs/textures/noise_texture.cu )
cuda_compile_and_embed( image_texture_programs programs/textures/image_texture.cu )
cuda_compile_and_embed( light_programs programs/materials/diffuse_light.cu )
cuda_compile_and_embed( box_programs programs/hitables/box.cu )
cuda_compile_and_embed( isotropic_programs programs/materials/isotropic.cu )
cuda_compile_and_embed( volume_sphere_programs programs/hitables/volume_sphere.cu )
cuda_compile_and_embed( volume_box_programs programs/hitables/volume_box.cu )
cuda_compile_and_embed( cosine_pdf_programs programs/pdfs/cosine_pdf.cu )
cuda_compile_and_embed( rect_pdf_programs programs/pdfs/rect_pdf.cu )
cuda_compile_and_embed( mixture_pdf_programs programs/pdfs/mixture_pdf.cu )
cuda_compile_and_embed( sphere_pdf_programs programs/pdfs/sphere_pdf.cu )
cuda_compile_and_embed( buffer_pdf_programs programs/pdfs/buffer_pdf.cu )
cuda_compile_and_embed( triangle_programs programs/hitables/triangle.cu )
cuda_compile_and_embed( plane_programs programs/hitables/plane.cu )
cuda_compile_and_embed( hit_program programs/hit.cu )
cuda_compile_and_embed( vector_texture_programs programs/textures/vector.cu )
cuda_compile_and_embed( geom_tri_programs programs/hitables/geometry_triangles.cu )
cuda_compile_and_embed( normal_programs programs/materials/normal_shader.cu )
cuda_compile_and_embed( anisotropic_programs programs/materials/anisotropic_phong.cu )
cuda_compile_and_embed( oren_nayar_programs programs/materials/oren_nayar.cu )
cuda_compile_and_embed( torrance_sparrow_programs programs/materials/torrance_sparrow.cu )
cuda_compile_and_embed( cylinder_programs programs/hitables/cylinder.cu )
cuda_compile_and_embed( gradient_texture_programs programs/textures/gradient_texture.cu )
cuda_compile_and_embed( disney_programs programs/materials/disney.cu )

find_package(OpenGL REQUIRED) 

include_directories("lib/imgui/gl3w/")
include_directories("lib/imgui/glfw/include")

# OpenGL & imgui files
add_library(ImGuiLibs 
 lib/imgui/gl3w/GL/gl3w.c
 lib/imgui/imgui.cpp
 lib/imgui/imgui_impl_glfw.cpp 
 lib/imgui/imgui_impl_opengl3.cpp
 lib/imgui/imgui_widgets.cpp
 lib/imgui/imgui_demo.cpp 
 lib/imgui/imgui_draw.cpp 
 lib/imgui/imgui_stdlib.cpp
)

target_link_libraries(ImGuiLibs opengl32)
target_link_libraries(ImGuiLibs "${PROJECT_SOURCE_DIR}/OptiX-Path-Tracer/lib/imgui/glfw/lib-vc2010-64/glfw3.lib")

# this is doing the same using OptiX
add_executable(OptiX_Path_Tracer
  # C++ host code
  lib/HDRloader.cpp
  lib/tiny_obj_loader.cc
  main.cpp
  # embedded cuda kernels:
  ${exception_program}
  ${raygen_program}
  ${sphere_programs}
  ${moving_sphere_programs}
  ${lambertian_programs}
  ${miss_program}
  ${metal_programs}
  ${dielectric_programs}
  ${constant_texture_programs}
  ${aarect_programs}
  ${checker_texture_programs}
  ${noise_texture_programs}
  ${image_texture_programs}
  ${light_programs}
  ${box_programs}
  ${isotropic_programs}
  ${volume_sphere_programs}
  ${volume_box_programs}
  ${cosine_pdf_programs}
  ${rect_pdf_programs}
  ${mixture_pdf_programs}
  ${sphere_pdf_programs}
  ${buffer_pdf_programs}
  ${triangle_programs}
  ${plane_programs}
  ${hit_program}
  ${vector_texture_programs}
  ${geom_tri_programs}
  ${normal_programs}
  ${anisotropic_programs}
  ${oren_nayar_programs}
  ${torrance_sparrow_programs}
  ${cylinder_programs}
  ${gradient_texture_programs}
  ${disney_programs}
  )

target_link_libraries(OptiX_Path_Tracer ImGuiLibs)

target_link_libraries(OptiX_Path_Tracer ${optix_LIBRARY})