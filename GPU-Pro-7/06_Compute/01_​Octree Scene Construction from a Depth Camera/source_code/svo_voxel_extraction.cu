
#include "common_types.h"

// Thrust Dependencies
#include <thrust/device_ptr.h>
#include <thrust/remove.h>

__global__ void getOccupiedChildren(const unsigned int* octree, 
	   const octkey* parents, const int num_parents, octkey* children) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  //Don't do anything if its out of bounds
  if (index >= num_parents) {
    return;
  }

  //Get the key for the parent
  const octkey key = parents[index];
  octkey temp_key = key;

  //Flag whether the node has any children
  int has_children = true;

  //Get the pointer to the children
  int pointer = 0;
  while (temp_key != 1) {
    //Get the next child
    pointer += getFirstValueAndShiftDown(temp_key);
    has_children = octree[2 * pointer] & 0x40000000;
    pointer = octree[2 * pointer] & 0x3FFFFFFF;
  }

  //Loop through the children, and if they are occupied fill its key 
  //into the output
  for (int i = 0; i < 8; i++) {
    int child_val = -1;

    if (has_children) {
      unsigned int val2 = octree[2 * (pointer + i) + 1];
      if (((val2 >> 24) & 0xFF) > 127) {
        //Add the moved bits
        child_val = (key << 3) + i;
      }
    }

    children[8 * index + i] = child_val;
  }
}

__global__ void voxelGridFromKeys(unsigned int* octree, octkey* keys, 
	   int num_voxels, glm::vec3 center, float edge_length, 
	   glm::vec4* centers, glm::vec4* colors) {

  //Get the index for the thread
  int idx = blockIdx.x*blockDim.x + threadIdx.x;

  //Don't do anything if out of bounds
  if (idx >= num_voxels) {
    return;
  }

  octkey key = keys[idx];

  //Get the pointer to the voxel
  int node_idx = 0;
  int child_idx = 0;
  while (key != 1) {
    //Get the next child
    int pos = getFirstValueAndShiftDown(key);
    node_idx = child_idx + pos;
    child_idx = octree[2 * node_idx] & 0x3FFFFFFF;

    //Decode the value into xyz
    int x = pos & 0x1;
    int y = pos & 0x2;
    int z = pos & 0x4;

    //Half the edge length to use it for the update
    edge_length /= 2.0f;

    //Update the center
    center.x += edge_length * (x ? 1 : -1);
    center.y += edge_length * (y ? 1 : -1);
    center.z += edge_length * (z ? 1 : -1);
  }

  unsigned int val = octree[2 * node_idx + 1];

  //Fill in the voxel
  centers[idx] = glm::vec4(center.x, center.y, center.z, 1.0f);
  colors[idx].r = ((float)(val & 0xFF) / 255.0f);
  colors[idx].g = ((float)((val >> 8) & 0xFF) / 255.0f);
  colors[idx].b = ((float)((val >> 16) & 0xFF) / 255.0f);
  colors[idx].a = ((float)((val >> 24) & 0xFF) / 255.0f);

}

extern "C" void extractVoxelGridFromSVO(SVO &octree, 
       const int max_depth, float edge_length, VoxelGrid& grid) {

  //Loop through each pass until max_depth, and determine the number of 
  //nodes at the highest resolution, along with morton codes for them
  int num_voxels = 1;

  //Initialize a node list with empty key (only a leading 1) for the first 
  //set of children, and copy to GPU
  octkey initial_nodes[1] = {1};
  octkey* node_list;
  cudaMalloc((void**)&node_list, sizeof(octkey));
  cudaMemcpy(node_list, initial_nodes, sizeof(octkey), cudaMemcpyHostToDevice);

  for (int i = 0; i < max_depth; i++) {
    //Allocate space for this pass based on the number of keys (x8)
    octkey* new_nodes;
    cudaMalloc((void**)&new_nodes, 8*num_voxels*sizeof(octkey));

    //Run kernel on all of the keys (x8)
    getOccupiedChildren<<<ceil((float)num_voxels/256.0f), 256>>>(octree.data, 
        node_list, num_voxels, new_nodes);
    cudaDeviceSynchronize();

    //Thrust remove-if to get the set of keys for the next pass
    {
      thrust::device_ptr<octkey> t_nodes = 
          thrust::device_pointer_cast<octkey>(new_nodes);
      num_voxels = thrust::remove_if(t_nodes, t_nodes + 8 * num_voxels, 
          negative()) - t_nodes;
    }

    //Free up memory for the previous set of keys
    cudaFree(node_list);
    node_list = new_nodes;
  }

  //Allocate the voxel grid
  grid.size = num_voxels;
  cudaMalloc((void**)&grid.centers, num_voxels*sizeof(glm::vec4));
  cudaMalloc((void**)&grid.colors, num_voxels*sizeof(glm::vec4));

  //Extract the data into the grid
  voxelGridFromKeys<<<ceil((float)num_voxels / 256.0f), 256>>>(octree.data, 
      node_list, num_voxels, octree.center, edge_length, grid.centers, 
      grid.colors);
  cudaDeviceSynchronize();

  //Free up memory
  cudaFree(node_list);
}
