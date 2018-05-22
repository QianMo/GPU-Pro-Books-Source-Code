#include "globalIllum.shi"
 
RWStructuredBuffer<VOXEL> fineGridBuffer: register(u0);
RWStructuredBuffer<VOXEL> coarseGridBuffer: register(u1);

[numthreads(8, 8, 8)]
void main(uint3 GroupID : SV_GroupID, uint3 DispatchThreadID : SV_DispatchThreadID,
		      uint3 GroupThreadID : SV_GroupThreadID, uint GroupIndex : SV_GroupIndex)
{
	int3 voxelPos = DispatchThreadID.xyz;
	int gridIndex = GetGridIndex(voxelPos);

	VOXEL gridElement;
	gridElement.colorMask = 0;	
	gridElement.normalMasks = uint4(0,0,0,0);
	gridElement.occlusion = 0;

	// clear fine and coarse resolution voxel-grid
  fineGridBuffer[gridIndex] = gridElement;
	coarseGridBuffer[gridIndex] = gridElement;
}
