/******************************************************/
/* Object-order Ray Tracing Demo (c) Tobias Zirr 2013 */
/******************************************************/

#define BE_VOXEL_REP_SETUP

#include <Pipelines/Tracing/VoxelRep.fx>
#include <Utility/Math.fx>
#include <Utility/Bits.fx>

Texture3D<uint> VoxelField : VoxelField;
RWTexture3D<uint> VoxelFieldUAV : VoxelFieldUAV;

static const uint MipGroupSize = 8;

[numthreads(MipGroupSize,MipGroupSize,MipGroupSize)]
void CSVoxelMip(uint3 dispatchIdx : SV_DispatchThreadID)
{
	uint3 gridIdx = dispatchIdx * 2;

	bool occupied = false;
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j)
			for (int k = 0; k < 2; ++k)
				occupied = occupied || VoxelField[gridIdx + uint3(k, j, i)];

	VoxelFieldUAV[dispatchIdx] = occupied;
}

technique11 VoxelMip
{
	pass
	{
		SetComputeShader( CompileShader(cs_5_0, CSVoxelMip()) );
	}
}
