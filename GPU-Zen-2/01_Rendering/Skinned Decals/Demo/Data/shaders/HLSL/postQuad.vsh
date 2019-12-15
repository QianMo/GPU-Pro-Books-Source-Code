struct VS_Output
{
	float4 position: SV_POSITION;
};

static const float2 positions[3] =
{
  float2(-1.0f, -1.0f),
  float2(3.0f, -1.0f),
  float2(-1.0f, 3.0f) 
};

VS_Output main(uint vertexID: SV_VertexID)
{
  VS_Output output;
	output.position = float4(positions[vertexID], 1.0f, 1.0f);
  return output;
}
