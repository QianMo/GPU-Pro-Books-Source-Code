#ifndef COMMON_TYPES_H_
#define COMMON_TYPES_H_

#include <glm/glm.hpp>
#include <cuda.h>

struct BoundingBox {
  glm::vec3 lower;
  glm::vec3 upper;
};

struct PointCloud {
  glm::vec3* positions;
  glm::vec4* colors;
  int size = 0;
};

struct VoxelGrid {
  glm::vec4* centers;
  glm::vec4* colors;
  int size = 0;
  float scale = 0.0f;
  BoundingBox bbox;
};

struct SVO {
  unsigned int* data;
  glm::vec3 center;
  float size;
};

struct negative {
  __host__ __device__ bool operator() (const int x) {
    return (x < 0);
  }
};

typedef long long int octkey;

#endif //COMMON_TYPES_H_
