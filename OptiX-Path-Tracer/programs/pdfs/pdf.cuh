#pragma once

#include "../prd.cuh"
#include "../sampling.cuh"
#include "../vec.hpp"

// communication between hit functions and the value programs
struct PDFRecord {
  float distance;
  float3 normal;
};

// input structs for the PDF programs
struct PDFParams {
  __device__ PDFParams(PerRayData& prd)
      : origin(prd.origin),
        view_direction(prd.view_direction),
        normal(prd.shading_normal),
        geometric_normal(prd.geometric_normal),
        matParams(prd.matParams) {}

  const float3 view_direction;
  const float3 origin;
  const float3 normal;
  const float3 geometric_normal;
  float3 direction;
  float3 localDirection;
  BRDFParameters matParams;
};
