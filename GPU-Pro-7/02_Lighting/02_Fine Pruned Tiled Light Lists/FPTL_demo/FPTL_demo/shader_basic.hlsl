
//--------------------------------------------------------------------------------------
// Constant Buffers
//--------------------------------------------------------------------------------------
#include "std_cbuffer.h"


//--------------------------------------------------------------------------------------
// shader input/output structure
//--------------------------------------------------------------------------------------
struct VS_INPUT
{
    float3 Position     : POSITION;
};

struct VS_OUTPUT
{
    float4 Position     : SV_POSITION;
};

VS_OUTPUT RenderSceneVS( VS_INPUT input)
{
	VS_OUTPUT Output;
	
	// Transform the position from object space to homogeneous projection space
	Output.Position = mul( float4(input.Position.xyz,1), g_mWorldViewProjection );

	return Output;    
}
