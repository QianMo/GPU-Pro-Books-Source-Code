
#include "common_types.h"

// Thrust Dependencies
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/remove.h>

__host__ void initOctree(unsigned int* &octree) {
  int oct[16];
  for (int i = 0; i < 16; i++) {
    oct[i] = 0;
  }
  cudaMalloc((void**)&octree, 16*sizeof(unsigned int));
  cudaMemcpy(octree, oct, 16*sizeof(unsigned int), cudaMemcpyHostToDevice);
}

__device__ octkey computeKey(const glm::vec3& point, glm::vec3 center, 
	   const int tree_depth, float edge_length) {

  //Check for invalid
  if (!isfinite(point.x) || !isfinite(point.z) || !isfinite(point.z)) {
    return 1;
  }

  //Initialize the output value with a leading 1 to specify the depth
  octkey key = 1;

  for (int i = 0; i < tree_depth; i++) {
    key = key << 3;

    //Determine which octant the point lies in
    uint8_t x = point.x > center.x ? 1 : 0;
    uint8_t y = point.y > center.y ? 1 : 0;
    uint8_t z = point.z > center.z ? 1 : 0;

    //Update the code
    key += (x + 2*y + 4*z);

    //Update the edge length
    edge_length /= 2.0f;

    //Update the center
    center.x += edge_length * (x ? 1 : -1);
    center.y += edge_length * (y ? 1 : -1);
    center.z += edge_length * (z ? 1 : -1);
  }

  return key;
}

__device__ int depthFromKey(octkey key) {
  const int bval[] =
  { 0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4 };

  int r = 0;
  if (key & 0x7FFF0000) { r += 16 / 1; key >>= 16 / 1; }
  if (key & 0x0000FF00) { r += 16 / 2; key >>= 16 / 2; }
  if (key & 0x000000F0) { r += 16 / 4; key >>= 16 / 4; }

  return (r + bval[key] - 1) / 3;
}

__device__ int getValueFromKey(const octkey key, const int depth) {
  return ((key >> 3*depth) & 0x7);
}

__device__ int getFirstValueAndShiftDown(octkey& key) {
  int depth = depthFromKey(key);
  int value = getValueFromKey(key, depth-1);
  key -= ((8 + value) << 3 * (depth - 1));
  key += (1 << 3 * (depth - 1));
  return value;
}

__global__ void computeKeys(const glm::vec3* voxels, const int numVoxels, 
	   const int max_depth, const glm::vec3 center, 
	   float edge_length, octkey* keys) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  //Don't do anything if its out of bounds
  if (index >= numVoxels) {
    return;
  }

  //Compute the full morton code
  keys[index] = computeKey(voxels[index], center, max_depth, edge_length);
}

__global__ void splitKeys(octkey* keys, const int numKeys, 
	   unsigned int* octree, octkey* left, octkey* right) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  //Don't do anything if its out of bounds
  if (index >= numKeys) {
    return;
  }

  octkey r_key = keys[index];
  octkey l_key = -1;
  octkey temp_key = 1;

  int node_idx = 0;

  //Determine the existing depth from the current octree data
  while (r_key >= 15) {
    //Get the child number from the first level of the key, and shift it down
    int value = getFirstValueAndShiftDown(r_key);
    temp_key = (temp_key << 3) + value;
    node_idx += value;

    //Check the flag
    if (!(octree[2 * node_idx] & 0x40000000)) {
      l_key = temp_key;
      break;
    }

    //The lowest 30 bits are the address of the child nodes
    node_idx = octree[2 * node_idx] & 0x3FFFFFFF;
  }

  //Fill in the results
  left[index] = l_key;
  right[index] = r_key;
}

__global__ void rightToLeftShift(octkey* left, octkey* right, 
	   const int numVoxels) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  //Don't do anything if its out of bounds
  if (index >= numVoxels) {
    return;
  }

  //Not valid if right is empty or if left is already invalid
  if (left[index] == -1 || right[index] == 1) {
    left[index] = -1;
    return;
  }

  //Get moving bits from right and remove them from it
  octkey r_key = right[index];
  int moved_bits = getFirstValueAndShiftDown(r_key);
  right[index] = r_key;

  //If this leaves left empty (only the leading 1) 
  //then right is no longer valid
  if (right[index] == 1) {
    left[index] = -1;
    return;
  }

  //Add the moved bits to the left key
  left[index] = (left[index] << 3) + moved_bits;
}

__host__ int prepassCheckResize(octkey* keys, const int numVoxels, 
	 const int max_depth, unsigned int* octree, octkey** &d_codes, 
	 int* &code_sizes) {

  int num_split_nodes = 0;

  //Allocate left and right morton code data
  octkey* d_left;
  cudaMalloc((void**)&d_left, numVoxels*sizeof(octkey));
  octkey* d_right;
  cudaMalloc((void**)&d_right, numVoxels*sizeof(octkey));
  octkey* temp_left;
  cudaMalloc((void**)&temp_left, numVoxels*sizeof(octkey));
  thrust::device_ptr<octkey> t_left = 
      thrust::device_pointer_cast<octkey>(temp_left);

  //Split the set of existing codes and new codes (right/left)
  splitKeys<<<ceil(float(numVoxels)/128.0f), 128>>>(keys, numVoxels, 
      octree, d_left, d_right);
  cudaDeviceSynchronize();

  //Allocate memory for output code data based on max_depth
  d_codes = (octkey**)malloc(max_depth*sizeof(octkey*));
  code_sizes = (int*)malloc(max_depth*sizeof(int));

  //Loop over passes
  for (int i = 0; i < max_depth; i++) {
    //Get a copy of left codes so we can modify
    cudaMemcpy(thrust::raw_pointer_cast(t_left), d_left, 
        numVoxels*sizeof(octkey), cudaMemcpyDeviceToDevice);

    //Get the valid codes
    int size = thrust::remove_if(t_left, t_left + numVoxels, negative()) 
        - t_left;

    //If there are no valid codes, we're done
    if (size == 0) {
      for (int j = i; j < max_depth; j++) {
        code_sizes[j] = 0;
      }
      break;
    }

    //Get the unique codes
    thrust::sort(t_left, t_left + size);
    size = thrust::unique(t_left, t_left + size) - t_left;

    //Allocate output and copy the data into it
    code_sizes[i] = size;
    cudaMalloc((void**)&(d_codes[i]), size*sizeof(octkey));
    cudaMemcpy(d_codes[i], thrust::raw_pointer_cast(t_left), 
        size*sizeof(octkey), cudaMemcpyDeviceToDevice);

    //Update the total number of nodes that are being split
    num_split_nodes += size;

    //Move one morton code from each right to left for the next depth
    rightToLeftShift<<<ceil((float)numVoxels/128.0f), 128>>>(d_left, 
        d_right, numVoxels);
  }

  //Cleanup
  cudaFree(d_left);
  cudaFree(d_right);
  cudaFree(temp_left);

  return num_split_nodes;
}

__global__ void splitNodes(const octkey* keys, int numKeys, 
	   unsigned int* octree, int num_nodes) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  //Don't do anything if its out of bounds
  if (index >= numKeys) {
    return;
  }

  //Get the key for this thread
  octkey key = keys[index];

  //Don't do anything if its an empty key
  if (key == 1) {
    return;
  }

  int node_idx = 0;
  int child_idx = 0;
  while (key != 1) {
    //Get the child number from the first three bits of the key
    node_idx = child_idx + getFirstValueAndShiftDown(key);

    //The lowest 30 bits are the address of the child nodes
    child_idx = octree[2 * node_idx] & 0x3FFFFFFF;
  }

  //Get a new node tile
  int newNode = num_nodes + 8 * index;

  //Point this node at the new tile, and flag it
  octree[2 * node_idx] = (1 << 30) +  (newNode & 0x3FFFFFFF);

  //Initialize new child nodes to 0's
  for (int off = 0; off < 8; off++) {
    octree[2 * (newNode + off)] = 0;
    octree[2 * (newNode + off) + 1] = 127 << 24;
  }
}

__host__ void expandTreeAtKeys(octkey** d_keys, int* numKeys, 
	 const int depth, unsigned int* octree, int& num_nodes) {

  for (size_t i = 0; i < depth; i++) {
    if (numKeys[i] == 0) {
      break;
    }

    splitNodes<<<ceil((float)numKeys[i]/128.0f), 128>>>(d_keys[i], 
        numKeys[i], octree, num_nodes);

    num_nodes += 8 * numKeys[i];
    cudaDeviceSynchronize();
  }
}

__global__ void fillNodes(const octkey* keys, int numKeys, 
	   const glm::vec4* values, unsigned int* octree) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  //Don't do anything if its out of bounds
  if (index >= numKeys) {
    return;
  }

  //Get the key for this thread
  octkey key = keys[index];

  //Check for invalid key
  if (key == 1) {
    return;
  }

  int node_idx = 0;
  int child_idx = 0;
  while (key != 1) {
    //Get the child number from the first three bits of the morton code
    node_idx = child_idx + getFirstValueAndShiftDown(key);

    //The lowest 30 bits are the address of the child nodes
    child_idx = octree[2 * node_idx] & 0x3FFFFFFF;
  }

  glm::vec4 new_value = values[index] * 256.0f;
  unsigned int current_value = octree[2 * node_idx + 1];

  int current_alpha = current_value >> 24;
  int current_r = current_value & 0xFF;
  int current_g = (current_value >> 8) & 0xFF;
  int current_b = (current_value >> 16) & 0xFF;

  //Implement a pseudo low-pass filter with laplace smoothing
  float f1 = 1 - ((float)current_alpha / 256.0f);
  float f2 = (float)current_alpha / 256.0f;
  new_value.r = new_value.r * f1 + current_r * f2;
  new_value.g = new_value.g * f1 + current_g * f2;
  new_value.b = new_value.b * f1 + current_b * f2;
  octree[2 * node_idx + 1] = ((int)new_value.r) + ((int)new_value.g << 8) 
      + ((int)new_value.b << 16) + (min(255, current_alpha + 2) << 24);
}

__global__ void averageChildren(octkey* keys, int numKeys, 
	   unsigned int* octree) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  //Don't do anything if its out of bounds
  if (index >= numKeys) {
    return;
  }

  //Get the key for this thread
  octkey key = keys[index];

  ///Get the depth of the key
  int depth = depthFromKey(key);

  //Remove the max depth level from the key
  key = key >> 3;

  //Fill in back into global memory this way
  keys[index] = key;

  int node_idx = 0;
  int child_idx = 0;
  while (key != 1) {
    //Get the child number from the first three bits of the morton code
    node_idx = child_idx + getFirstValueAndShiftDown(key);

    //The lowest 30 bits are the address of the child nodes
    child_idx = octree[2 * node_idx] & 0x3FFFFFFF;
  }

  //Loop through children values and average them
  glm::vec4 val = glm::vec4(0.0);
  int num_occ = 0;
  for (int i = 0; i < 8; i++) {
    int child_val = octree[2*(child_idx+i) + 1];
    if ((child_val >> 24) & 0xFF == 0) {
      //Don't count in the average if its not occupied
      continue;
    }
    val.r += (float) (child_val & 0xFF);
    val.g += (float) ((child_val >> 8) & 0xFF);
    val.b += (float) ((child_val >> 16) & 0xFF);
    //Assign the max albedo (avoids diluting it)
    val.a = max(val.a,(float) ((child_val >> 24) & 0xFF));
    num_occ++;
  }

  //Average the color values
  if (num_occ > 0) {
    val.r = val.r / (float)num_occ;
    val.g = val.g / (float)num_occ;
    val.b = val.b / (float)num_occ;
  }

  //Assign value of this node to the average
  octree[(2 * node_idx) + 1] = (int)val.r + ((int)val.g << 8) 
      + ((int)val.b << 16) + ((int)val.a << 24);
}

struct depth_is_zero {
  __device__ bool operator() (const int key) {
    return (depthFromKey(key) == 0);
  }
};


__host__ void mipmapNodes(octkey* keys, int numKeys, unsigned int* octree) {

  //Get a thrust pointer for the keys
  thrust::device_ptr<octkey> t_keys = 
      thrust::device_pointer_cast<octkey>(keys);

  //Check if any keys still have children
  while ((numKeys = thrust::remove_if(t_keys, t_keys + numKeys, 
      depth_is_zero()) - t_keys) > 0) {

    if (numKeys > 100000)
      numKeys = thrust::unique(t_keys, t_keys + numKeys) - t_keys;

    //Average the children at the given set of keys
    averageChildren<<<ceil((float)numKeys / 64.0f), 64>>>(keys, numKeys, 
        octree);
    cudaDeviceSynchronize();
  }

}

extern "C" void svoFromPointCloud(const PointCloud points, 
       const int max_depth, const float edge_length, SVO &octree) {

  //Initialize the octree with a base set of empty nodes if its empty
  if (octree.size == 0) {
    initOctree(octree.data);
    octree.size = 8;
  }
  //Allocate space for octree keys for each input
  octkey* d_keys;
  cudaMalloc((void**)&d_keys, size*sizeof(octkey));

  //Compute Keys
  computeKeys<<<ceil((float)size / 256.0f), 256>>>(points.positions, 
      points.size, max_depth, octree.center, edge_length, d_keys);
  cudaDeviceSynchronize();

  //Determine how many new nodes are needed in the octree, and the keys to 
  //the nodes that need split in each pass
  octkey** d_codes = NULL;
  int* code_sizes = NULL;
  int new_nodes = prepassCheckResize(d_keys, size, max_depth, octree.data, 
      d_codes, code_sizes);

  //Create a new octree with an updated size and copy over the old data. 
  //Free up the old copy when it is no longer needed
  unsigned int* new_octree;
  cudaMalloc((void**)&new_octree, 2 * (octree_size + 8 * new_nodes) 
      * sizeof(unsigned int));
  cudaMemcpy(new_octree, octree, 2 * octree_size * sizeof(unsigned int), 
      cudaMemcpyDeviceToDevice);
  cudaFree(octree.data);
  octree.data = new_octree;

  //Expand the tree now that the space has been allocated
  expandTreeAtKeys(d_codes, code_sizes, max_depth, octree.data, octree.size);
  //Free up the codes now that we no longer need them
  for (size_t i = 0; i < max_depth; i++) {
    if (code_sizes[i] > 0) {
      cudaFree(d_codes[i]);
    }
  }
  free(d_codes);
  free(code_sizes);

  //Write voxel values into the lowest level of the svo
  fillNodes<<<ceil((float)size / 256.0f), 256>>>(d_keys, points.size, 
      points.colors, octree.data);
  cudaDeviceSynchronize();

  //Mip-mapping (currently only without use of the brick pool)
  mipmapNodes(d_keys, size, octree.data);
  cudaDeviceSynchronize();

  //Free up the keys since they are no longer needed
  cudaFree(d_keys);
}
