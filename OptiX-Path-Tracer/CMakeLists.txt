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
cuda_compile_and_embed( Exception_PTX programs/exception.cu )
cuda_compile_and_embed( Raygen_PTX programs/raygen.cu )
cuda_compile_and_embed( Sphere_PTX programs/hitables/sphere.cu )
cuda_compile_and_embed( Moving_Sphere_PTX programs/hitables/moving_sphere.cu )
cuda_compile_and_embed( Miss_PTX programs/miss.cu )
cuda_compile_and_embed( Metal_PTX programs/materials/metal.cu )
cuda_compile_and_embed( Dielectric_PTX programs/materials/dielectric.cu )
cuda_compile_and_embed( Lambertian_PTX programs/materials/lambertian.cu )
cuda_compile_and_embed( Color_PTX programs/textures/constant_texture.cu )
cuda_compile_and_embed( AARect_PTX programs/hitables/aarect.cu )
cuda_compile_and_embed( Checker_PTX programs/textures/checkered_texture.cu )
cuda_compile_and_embed( Noise_PTX programs/textures/noise_texture.cu )
cuda_compile_and_embed( Image_PTX programs/textures/image_texture.cu )
cuda_compile_and_embed( Light_PTX programs/materials/diffuse_light.cu )
cuda_compile_and_embed( Box_PTX programs/hitables/box.cu )
cuda_compile_and_embed( Isotropic_PTX programs/materials/isotropic.cu )
cuda_compile_and_embed( Volume_Sphere_PTX programs/hitables/volume_sphere.cu )
cuda_compile_and_embed( Volume_Box_PTX programs/hitables/volume_box.cu )
cuda_compile_and_embed( Rect_PDF_PTX programs/pdfs/rect_pdf.cu )
cuda_compile_and_embed( Sphere_PDF_PTX programs/pdfs/sphere_pdf.cu )
cuda_compile_and_embed( Triangle_PTX programs/hitables/triangle.cu )
cuda_compile_and_embed( Plane_PTX programs/hitables/plane.cu )
cuda_compile_and_embed( Hit_PTX programs/hit.cu )
cuda_compile_and_embed( Vector_Tex_PTX programs/textures/vector.cu )
cuda_compile_and_embed( Normal_PTX programs/materials/normal_shader.cu )
cuda_compile_and_embed( Ashikhmin_PTX programs/materials/ashikhmin_shirley.cu )
cuda_compile_and_embed( Oren_Nayar_PTX programs/materials/oren_nayar.cu )
cuda_compile_and_embed( Torrance_PTX programs/materials/torrance_sparrow.cu )
cuda_compile_and_embed( Cylinder_PTX programs/hitables/cylinder.cu )
cuda_compile_and_embed( Gradient_PTX programs/textures/gradient_texture.cu )

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
  
  # General Programs
  ${Exception_PTX}
  ${Raygen_PTX}
  ${Miss_PTX}
  ${Hit_PTX}

  # Surface Programs
  ${Sphere_PTX}
  ${Moving_Sphere_PTX}
  ${AARect_PTX}
  ${Box_PTX}
  ${Volume_Sphere_PTX}
  ${Volume_Box_PTX}
  ${Triangle_PTX}
  ${Cylinder_PTX}

  # Material Programs
  ${Lambertian_PTX}
  ${Metal_PTX}
  ${Dielectric_PTX}
  ${Light_PTX}
  ${Isotropic_PTX}
  ${Normal_PTX}
  ${Ashikhmin_PTX}
  ${Oren_Nayar_PTX}
  ${Torrance_PTX}
  
  # Texture Programs
  ${Color_PTX}
  ${Checker_PTX}
  ${Noise_PTX}
  ${Image_PTX}
  ${Gradient_PTX}
  ${Vector_Tex_PTX}
  
  #Sampling Programs
  ${Rect_PDF_PTX}
  ${Sphere_PDF_PTX}

  )

target_link_libraries(OptiX_Path_Tracer ImGuiLibs)

target_link_libraries(OptiX_Path_Tracer ${optix_LIBRARY})