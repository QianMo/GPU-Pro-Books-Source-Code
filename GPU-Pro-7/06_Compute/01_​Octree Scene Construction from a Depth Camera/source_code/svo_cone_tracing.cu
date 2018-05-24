
#include “common_types.h”

// C/STL Dependencies
#include <cmath>

// CUDA Dependencies
#include <cuda.h>
#include <vector_types.h>

// Thrust Dependencies
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/remove.h>

// GLM Dependency
#include <glm/gtc/matrix_transform.hpp>

//The maximum distance that we can see
__device__ const float MAX_RANGE = 10.0f;

//The starting distance from the origin to start the ray marching from
__device__ const float START_DIST = 0.002f;

//CUDA Kernel generates a ray for each pixel
__global__ void createRays(glm::vec2 resolution, float fov, glm::vec2 focal,
	   glm::vec3 x_dir, glm::vec3 y_dir, glm::vec3* rays) {

  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  //Don't do anything if the index is out of bounds
  if (idx >= (int)resolution.x * (int)resolution.y) {
    return;
  }

  //Compute the x/y coords of this thread
  int x = idx % (int)resolution.x;
  int y = idx / (int)resolution.x;

  //Calculate the perpendicular vector component magnitudes 
  //from the fov, resolution, and x/y values
  float fac = tan(fov * 3.14159f / 180.0f) / resolution.x;
  glm::vec2 mag;
  mag.x = ((float)x - resolution.x / 2.0f) / focal.x;
  mag.y = ((float)y - resolution.y / 2.0f) / focal.y;

  //Calculate the direction
  rays[idx] = START_DIST * glm::normalize( (mag.x * x_dir) 
      + (mag.y * y_dir) + (glm::cross(x_dir, -y_dir)));

}

//CUDA Kernel executes a single iteration of Voxel Cone Tracing
__global__ void coneTrace(uchar4* pos, int* ind, int numRays, 
	   glm::vec3 camera_origin, glm::vec3* rays, float pix_scale, 
	   unsigned int* octree, glm::vec3 oct_center, float oct_size) {

  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  //Don't do anything if the index is out of bounds
  if (idx >= numRays) {
    return;
  }

  int index = ind[idx];

  //Compute the target point
  glm::vec3 ray = rays[index];
  glm::vec3 target = camera_origin + ray;
  float ray_len = glm::length(ray);
  float pix_size = ray_len*pix_scale;
  int depth = ceil(log((float)(oct_size / pix_size)) / log(2.0f));

  //Descend into the octree and get the value
  int node_idx = 0;
  int child_idx = 0;
  float temp_oct_size = oct_size;
  bool is_occupied = true;
  for (int i = 0; i < depth; i++) {
    //Determine which octant the point lies in
    uint8_t x = target.x > oct_center.x ? 1 : 0;
    uint8_t y = target.y > oct_center.y ? 1 : 0;
    uint8_t z = target.z > oct_center.z ? 1 : 0;

    //Update the child number
    int child = (x + 2 * y + 4 * z);

    //Get the child number from the first three bits of the morton code
    node_idx = child_idx + child;

    if (!(octree[2 * node_idx] & 0x40000000)) {
      depth = i+1;
      break;
    }

    //The lowest 30 bits are the address of the child nodes
    child_idx = octree[2 * node_idx] & 0x3FFFFFFF;

    //Update the edge length
    temp_oct_size /= 2.0f;

    //Update the center
    oct_center.x += temp_oct_size * (x ? 1 : -1);
    oct_center.y += temp_oct_size * (y ? 1 : -1);
    oct_center.z += temp_oct_size * (z ? 1 : -1);
  }

  //Update the pixel value
  uchar4 value = pos[index];
  unsigned int oct_val = octree[2 * node_idx + 1];
  int alpha = max(0, (oct_val >> 24) - 127);
  if (is_occupied) {
    value.x += (uint8_t) (((float)alpha/127.0f)
        * (float)((oct_val & 0xFF)));
    value.y += (uint8_t) (((float)alpha/127.0f)
        * (float)((oct_val >> 8) & 0xFF));
    value.z += (uint8_t) (((float)alpha/127.0f)
        * (float)((oct_val >> 16) & 0xFF));

    //Flag the ray as finished if alpha is saturated
    if ((int)value.w + alpha < 127) {
      value.w += alpha;
    } else {
      value.w = 255;
      pos[index] = value;
      index = -1;
    }
  }

  if (index >= 0) {

    float new_dist = oct_size / pow(2.0f, (float)depth);

    //Update the ray length and flag if its gone past the max distance
    ray *= (ray_len + new_dist) / ray_len;
    rays[index] = ray;
    if (glm::length(ray) > MAX_RANGE) {
      //Scale up the color
      value.x *= 127.0f/(float)value.w;
      value.y *= 127.0f/(float)value.w;
      value.z *= 127.0f/(float)value.w;
      value.w = 255;
      pos[index] = value;
      index = -1;
    }

  }
  //Update the index
  ind[idx] = index;
}

//External function for executing VCT on an SVO scene
extern "C" void coneTraceSVO(const SVO& octree, const glm::vec2& resolution, 
       const float fov, const glm::vec2& focal, const glm::mat4& cameraPose, 
       uchar4* pos) {

  int numRays = (int)resolution.x * (int)resolution.y;

  glm::vec3 camera_origin = glm::vec3(glm::inverse(cameraPose)
      * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

  //Create rays
  glm::vec3* rays;
  cudaMalloc((void**)&rays, numRays * sizeof(glm::vec3));
  glm::vec3 x_dir = glm::vec3(glm::inverse(cameraPose) 
      * glm::vec4(-1.0f, 0.0f, 0.0f, 0.0f));
  glm::vec3 y_dir = glm::vec3(glm::inverse(cameraPose) 
      * glm::vec4(0.0f, -1.0f, 0.0f, 0.0f));
  createRays<<<ceil(numRays / 256.0f), 256>>>(resolution, fov, focal 
      x_dir, y_dir, rays);

  //Initialize distance and depth
  float pix_scale = tan(fov)/resolution.y;

  //Setup indices
  int* ind;
  cudaMalloc((void**)&ind, numRays*sizeof(glm::vec3));
  thrust::device_ptr<int> t_ind = thrust::device_pointer_cast<int>(ind);
  thrust::sequence(t_ind, t_ind+numRays, 0, 1);

  //Initialize the output
  cudaMemset(pos, 0, numRays*sizeof(uchar4));

  while (numRays > 0) {
    //Call the cone tracing kernel
    coneTrace<<<ceil(numRays / 256.0f), 256>>>(pos, ind, numRays, 
      camera_origin, rays, pix_scale, octree.data, 
      octree.center, octree.size);

    //Use thrust to remove rays that are saturated
    numRays = thrust::remove_if(t_ind, t_ind + numRays, negative()) - t_ind;
  }

  //Cleanup
  cudaFree(rays);
  cudaFree(ind);
}
