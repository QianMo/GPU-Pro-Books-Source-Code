#include "globals.shi"

cbuffer CUSTOM_UB: register(CUSTOM_UB_BP)
{
	struct
	{
    matrix gridViewProjMatrices[6];
		float4 gridCellSizes; 
	  float4 gridPositions[2];
		float4 snappedGridPositions[2];
	}customUB;
};

struct VS_OUTPUT
{
	float4 position: SV_POSITION;
	uint instanceID: INSTANCE_ID;
};

struct GS_OUTPUT
{
  float4 position: SV_POSITION;
  uint rtIndex : SV_RenderTargetArrayIndex; 
};

[maxvertexcount(4)]
void main(line VS_OUTPUT input[2],inout TriangleStream<GS_OUTPUT> outputStream)
{
  // generate a quad from input line (2 vertices)
	// ->generate 1 triangle-strip
	GS_OUTPUT output[4];

  // left/ lower vertex
	output[0].position =  float4(input[0].position.x,input[0].position.y,input[0].position.z,1.0f);   
		
	// right/ lower vertex
	output[1].position = float4(input[1].position.x,input[0].position.y,input[0].position.z,1.0f);   

	// left/ upper vertex
	output[2].position = float4(input[0].position.x,input[1].position.y,input[0].position.z,1.0f);  

  // right/ upper vertex
	output[3].position = float4(input[1].position.x,input[1].position.y,input[0].position.z,1.0f);  
	
	[unroll]
	for(int i=0;i<4;i++)
	{
    output[i].rtIndex = input[0].instanceID; // write 32 instances of quad into 32 slices of 2D texture array
    outputStream.Append(output[i]);
	}

	outputStream.RestartStrip();
}






