float4x4 WorldViewProjection 	: WORLDVIEWPROJECTION;
float4x4 WorldViewIT 			: WORLDVIEWINVERSETRANSPOSE;
float4 LightPosition 			: LIGHTPOSMODEL0;

Texture2D sampler2d : 	TEXTURE0 < string name = "maskmain.pvr"; >;
Texture2D envmap2d : 	TEXTURE1 < string name = "leaf.pvr"; >;

SamplerState samplerState
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Wrap;
    AddressV = Wrap;
};

struct VS_INPUT
{
	float3 Vertex		: POSITION;
	float3 Normal		: NORMAL;
	float2 UV			: TEXCOORD;
	float2 UVA			: TEXCOORDA;
	float3 Tangent		: TANGENT;
	float3 Binormal		: BINORMAL;
};

struct VS_OUTPUT
{
	float4 Position				: SV_POSITION;
	float DiffuseIntensity		: INTENSITY;
	float2 texCoordinateMain	: TEXCOORDMAIN;
	float2 texCoordinateDetail	: TEXCOORDDETAIL;
	float2 varEnvMap			: ENVIRONMENTMAP;
};

VS_OUTPUT VS(VS_INPUT In)
{
	VS_OUTPUT Out;
	Out.Position = mul(float4(In.Vertex, 1.0), WorldViewProjection);

	float4 transNormal = mul(float4(In.Normal, 1.0), WorldViewIT);
	
	float3 LightDir = normalize(LightPosition.xyz - In.Vertex);
	Out.DiffuseIntensity = 0.5 + dot(In.Normal, LightDir) * 0.5;

	Out.texCoordinateMain = In.UV;
	Out.texCoordinateDetail = In.UVA;
	Out.varEnvMap = 0.5 + transNormal.xy * 0.5;

	return Out;
}

float4 PS(VS_OUTPUT In) : SV_Target
{
	float3 envColour = 0.5 * envmap2d.Sample(samplerState, In.varEnvMap);
	float3 texMain = sampler2d.Sample(samplerState,In.texCoordinateMain);
	float3 texDetail = sampler2d.Sample(samplerState, In.texCoordinateDetail);
	
	float3 texColor = texMain * texDetail * (In.DiffuseIntensity + envColour);
	return float4(texColor, 1.0);
}

technique10 ReflectionEffect
{
    pass P0
    {
        SetVertexShader( CompileShader( vs_4_0, VS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, PS() ) );
    }
}
