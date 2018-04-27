float4x4 WVPMatrix		: WORLDVIEWPROJECTION;
float3x3 WorldViewIT	: WORLDVIEWINVERSETRANSPOSE;
float3 LightDirection	: LIGHTDIRWORLD1;
float3 EyePos			: EYEPOSWORLD;
	
struct VS_OUTPUT
{
	float4 Position			: POSITION;
	float diffuseIntensity	: TEXCOORD0;
	float specularIntensity	: TEXCOORD1;
};

struct VS_INPUT
{
	float4 vPos		: POSITION;
	float3 vNorm	: NORMAL;
};

const float4 cMaterial = float4(0.4,0.6,0.9,0.0);
const float3 cGrainDirection = float3(1.0, 2.0, 0.0);
const float3 cBaseColour = float3(0.9, 0.2, 0.2);

VS_OUTPUT VertexShader(in VS_INPUT In)
{
	VS_OUTPUT Out;

	Out.Position		= mul(In.vPos, WVPMatrix);
	float3 transNormal	= normalize(mul(In.vNorm, WorldViewIT));

	float3 normalXgrain = cross(transNormal, normalize(cGrainDirection));
	float3 tangent = normalize(cross(normalXgrain, transNormal));
	float LdotT = dot(tangent, normalize(LightDirection));
	float VdotT = dot(tangent, normalize(EyePos));

	float temp = sqrt(1.0 - LdotT * LdotT);
	float NdotL = temp;
	float VdotR = temp * sqrt(1.0 - VdotT * VdotT) - VdotT * LdotT;

	Out.diffuseIntensity = max(NdotL * cMaterial.x + cMaterial.y, 0.0);
	Out.specularIntensity = max(VdotR * VdotR * cMaterial.z + cMaterial.w, 0.0);

	return Out;
}

float4 PixelShader(in VS_OUTPUT In) : COLOR
{
	float3 finalColour = (cBaseColour * In.diffuseIntensity) + In.specularIntensity;
	return float4(finalColour, 1.0);
}

technique AnisotropicLighting
{
    pass P0
    {
        vertexShader = compile vs_2_0 VertexShader();
        pixelShader = compile ps_2_0 PixelShader();
    }
}
