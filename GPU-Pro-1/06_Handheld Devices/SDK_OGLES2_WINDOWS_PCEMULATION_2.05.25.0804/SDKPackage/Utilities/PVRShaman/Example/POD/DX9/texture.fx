uniform float4x4 WVPMatrix		: WORLDVIEWPROJECTION;
uniform float3x3 WorldViewIT	: WORLDVIEWINVERSETRANSPOSE;
uniform float3 LightPos			: LIGHTPOSMODEL0;

texture tex0 : TEXTURE0 < string name = "maskmain.pvr"; >;
texture tex1 : TEXTURE1 < string name = "leaf.pvr"; >;

sampler2D sampler2d = sampler_state
{
	texture = (tex0);
	mipfilter = LINEAR;	
};

sampler2D envmap2d = sampler_state
{
	texture = (tex1);
	mipfilter = LINEAR;	
};

struct VS_INPUT
{
	float3 Vertex		: POSITION;
	float3 Normal		: NORMAL;
	float2 UVMain		: TEXCOORD0;
	float2 UVDetail		: TEXCOORD1;
};

struct VS_OUTPUT
{
	float4 Position				: POSITION;
	float DiffuseIntensity		: TEXCOORD0;
	float2 texCoordinateMain	: TEXCOORD1;
	float2 texCoordinateDetail	: TEXCOORD2;
	float2 varEnvMap			: TEXCOORD3;
};

VS_OUTPUT VertexShader(in VS_INPUT In)
{
	VS_OUTPUT Out;
	
	Out.Position = mul(float4(In.Vertex, 1.0), WVPMatrix);
	float3 transNormal = normalize(mul(In.Normal, WorldViewIT));
	
	float3 LightDirection = normalize(LightPos - In.Vertex);
	Out.DiffuseIntensity = 0.5 + dot(In.Normal, LightDirection) * 0.5;

	Out.texCoordinateMain = In.UVMain;
	Out.texCoordinateDetail = In.UVDetail;
	Out.varEnvMap = 0.5 + transNormal.xy * 0.5;

	return Out;
}

float4 PixelShader(in VS_OUTPUT In) : COLOR
{
	float3 envColour = 0.5 * tex2D(envmap2d, In.varEnvMap);
	float3 texMain = tex2D(sampler2d, In.texCoordinateMain);
	float3 texDetail = tex2D(sampler2d, In.texCoordinateDetail);
	
	float3 texColor = texMain * texDetail * (In.DiffuseIntensity + envColour);

	return float4(texColor, 1.0);
}

technique ReflectionEffect
{
    pass P0
    {
        vertexShader = compile vs_2_0 VertexShader();
        pixelShader = compile ps_2_0 PixelShader();
    }
}
