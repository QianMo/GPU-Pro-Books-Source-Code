struct VS_OUTPUT
{
	float4 position: SV_POSITION;
  float2 texCoords: TEXCOORD;
};

struct GS_OUTPUT
{
  float4 position: SV_POSITION;
  float2 texCoords: TEXCOORD;
};

[maxvertexcount(4)]
void main(line VS_OUTPUT input[2],inout TriangleStream<GS_OUTPUT> outputStream)
{
  // generate a quad from input line (2 vertices)
	// ->generate 1 triangle-strip
	GS_OUTPUT output[4];

  // left/ lower vertex
	output[0].position =  float4(input[0].position.x,input[0].position.y,input[0].position.z,1.0f);   
	output[0].texCoords = float2(input[0].texCoords.x,input[0].texCoords.y); 
	
	// right/ lower vertex
	output[1].position = float4(input[1].position.x,input[0].position.y,input[0].position.z,1.0f);   
	output[1].texCoords = float2(input[1].texCoords.x,input[0].texCoords.y); 

	// left/ upper vertex
	output[2].position = float4(input[0].position.x,input[1].position.y,input[0].position.z,1.0f);  
	output[2].texCoords = float2(input[0].texCoords.x,input[1].texCoords.y); 

  // right/ upper vertex
	output[3].position = float4(input[1].position.x,input[1].position.y,input[0].position.z,1.0f);  
	output[3].texCoords = float2(input[1].texCoords.x,input[1].texCoords.y); 

	[unroll]
	for(int i=0;i<4;i++)
    outputStream.Append(output[i]);
	
	outputStream.RestartStrip();
}






