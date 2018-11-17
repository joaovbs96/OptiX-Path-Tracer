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
cuda_compile_and_embed(
  embedded_raygen_program programs/raygen.cu
  )
cuda_compile_and_embed(
  embedded_sphere_programs programs/sphere.cu
  )
cuda_compile_and_embed(
  embedded_miss_program programs/miss.cu
  )
cuda_compile_and_embed(
  embedded_metal_programs programs/metal.cu
  )
cuda_compile_and_embed(
  embedded_dielectric_programs programs/dielectric.cu
  )
cuda_compile_and_embed( 
  embedded_lambertian_programs programs/lambertian.cu
  )

# this is doing the same using OptiX
add_executable(finalChapter
  # C++ host code
  programs/vec.h
  programs/material.h
  finalChapter.cpp
  # embedded cuda kernels:
  ${embedded_raygen_program}
  ${embedded_sphere_programs}
  ${embedded_lambertian_programs}
  ${embedded_miss_program}
  ${embedded_metal_programs}
  ${embedded_dielectric_programs}
  )

target_link_libraries(finalChapter 
  ${optix_LIBRARY}
  )

