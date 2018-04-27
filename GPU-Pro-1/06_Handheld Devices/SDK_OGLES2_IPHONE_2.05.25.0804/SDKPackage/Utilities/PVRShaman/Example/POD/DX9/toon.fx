uniform float4x4 WVPMatrix 		: WORLDVIEWPROJECTION;
uniform float3x3 WorldViewIT	: WORLDVIEWINVERSETRANSPOSE;

struct VS_INPUT
{
	float3 Vertex		: POSITION;
	float3 Normal		: NORMAL;
	float2 UV			: TEXCOORD0;
	float3 Tangent		: TANGENT;
};

struct VS_OUTPUT
{
	float4 Position		  	: POSITION;
	float Intensity			: TEXCOORD0;
};

VS_OUTPUT VertexShader(in VS_INPUT In)
{
	VS_OUTPUT Out;
	
	Out.Position = mul(float4(In.Vertex, 1.0), WVPMatrix);
	
	float3 normal =  mul(float4(In.Normal, 1.0), WorldViewIT);
	Out.Intensity = abs( dot(float3(0.0, 0.0, 1.0), normalize(normal)) );
	
	return Out;
}

float4 PixelShader(in VS_OUTPUT In) : COLOR
{
	float4 color;
	
	if (In.Intensity > 0.95)
		color = float4(1.0,0.5,0.5,1.0);
	else if (In.Intensity > 0.5)
		color = float4(0.6,0.3,0.3,1.0);
	else if (In.Intensity > 0.25)
		color = float4(0.4,0.2,0.2,1.0);
	else
		color = float4(0.2,0.1,0.1,1.0);

	return color;	
};

	
technique ToonEffect
{
    pass P0
    {
        vertexShader = compile vs_2_0 VertexShader();
        pixelShader = compile ps_2_0 PixelShader();
    }
}
	