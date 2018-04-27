shared cbuffer def
{
	float4x4 matWVP;

	float3 dims;
	float3 cellDims;

	float opacity;

	uint rotateShift;
	uint rotateMask;

	uint3 numCellsPerDim;
	uint numColors;
}

shared Buffer 			ColorBuf;
shared Buffer<float3>	Vertices;