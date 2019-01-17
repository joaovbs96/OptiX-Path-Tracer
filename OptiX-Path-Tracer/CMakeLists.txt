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
cuda_compile_and_embed( embedded_exception_program programs/exception.cu )
cuda_compile_and_embed( embedded_raygen_program programs/raygen.cu )
cuda_compile_and_embed( embedded_sphere_programs programs/hitables/sphere.cu )
cuda_compile_and_embed( embedded_moving_sphere_programs programs/hitables/moving_sphere.cu )
cuda_compile_and_embed( embedded_miss_program programs/miss.cu )
cuda_compile_and_embed( embedded_metal_programs programs/materials/metal.cu )
cuda_compile_and_embed( embedded_dielectric_programs programs/materials/dielectric.cu )
cuda_compile_and_embed( embedded_lambertian_programs programs/materials/lambertian.cu )
cuda_compile_and_embed( embedded_constant_texture_programs programs/textures/constant_texture.cu )
cuda_compile_and_embed( embedded_aarect_programs programs/hitables/aarect.cu )
cuda_compile_and_embed( embedded_checker_texture_programs programs/textures/checkered_texture.cu )
cuda_compile_and_embed( embedded_noise_texture_programs programs/textures/noise_texture.cu )
cuda_compile_and_embed( embedded_image_texture_programs programs/textures/image_texture.cu )
cuda_compile_and_embed( embedded_diffuse_light_programs programs/materials/diffuse_light.cu )
cuda_compile_and_embed( embedded_box_programs programs/hitables/box.cu )
cuda_compile_and_embed( embedded_isotropic_programs programs/materials/isotropic.cu )
cuda_compile_and_embed( embedded_volume_sphere_programs programs/hitables/volume_sphere.cu )
cuda_compile_and_embed( embedded_volume_box_programs programs/hitables/volume_box.cu )
cuda_compile_and_embed( embedded_cosine_pdf_programs programs/pdfs/cosine_pdf.cu )
cuda_compile_and_embed( embedded_rect_pdf_programs programs/pdfs/rect_pdf.cu )
cuda_compile_and_embed( embedded_mixture_pdf_programs programs/pdfs/mixture_pdf.cu )
cuda_compile_and_embed( embedded_sphere_pdf_programs programs/pdfs/sphere_pdf.cu )
cuda_compile_and_embed( embedded_buffer_pdf_programs programs/pdfs/buffer_pdf.cu )
cuda_compile_and_embed( embedded_triangle_programs programs/hitables/triangle.cu )

# this is doing the same using OptiX
add_executable(OptiX-Path-Tracer
  # C++ host code
  programs/vec.h
  programs/materials/material.h
  host_includes/hitables.h
  host_includes/materials.h
  host_includes/scenes.h
  host_includes/camera.h
  lib/stb_image.h
  lib/stb_image_write.h
  main.cpp
  # embedded cuda kernels:
  ${embedded_exception_program}
  ${embedded_raygen_program}
  ${embedded_sphere_programs}
  ${embedded_moving_sphere_programs}
  ${embedded_lambertian_programs}
  ${embedded_miss_program}
  ${embedded_metal_programs}
  ${embedded_dielectric_programs}
  ${embedded_constant_texture_programs}
  ${embedded_aarect_programs}
  ${embedded_checker_texture_programs}
  ${embedded_noise_texture_programs}
  ${embedded_image_texture_programs}
  ${embedded_diffuse_light_programs}
  ${embedded_box_programs}
  ${embedded_isotropic_programs}
  ${embedded_volume_sphere_programs}
  ${embedded_volume_box_programs}
  ${embedded_cosine_pdf_programs}
  ${embedded_rect_pdf_programs}
  ${embedded_mixture_pdf_programs}
  ${embedded_sphere_pdf_programs}
  ${embedded_buffer_pdf_programs}
  ${embedded_triangle_programs}
  )

target_link_libraries(OptiX-Path-Tracer
  ${optix_LIBRARY}
  )

