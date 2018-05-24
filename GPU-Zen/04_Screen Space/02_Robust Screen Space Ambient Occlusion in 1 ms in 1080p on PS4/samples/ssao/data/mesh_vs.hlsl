cbuffer ConstantBuffer: register(b0)
{
	matrix worldTransform;
	matrix viewProjTransform;
}


struct VS_INPUT
{
	float4 position: POSITION;
	float3 normal: NORMAL;
	float2 texCoord: TEXCOORD0;
};


struct VS_OUTPUT
{
	float4 position: SV_POSITION;
	float2 texCoord: TEXCOORD0;
	float3 normal: TEXCOORD1;
};


VS_OUTPUT VSMain(VS_INPUT input)
{
	VS_OUTPUT output = (VS_OUTPUT)0;
	
	// world-space
	float4 position_world = mul(worldTransform, input.position);
	float3 normal_world = mul(worldTransform, input.normal);
	
	output.position = mul(viewProjTransform, position_world);
	output.texCoord = input.texCoord;
	output.normal = normal_world;

	return output;
}