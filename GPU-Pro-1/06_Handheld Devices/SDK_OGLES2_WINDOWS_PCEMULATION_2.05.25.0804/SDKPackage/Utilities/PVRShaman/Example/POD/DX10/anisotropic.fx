float4x4 WorldViewProjection	: WORLDVIEWPROJECTION;
float4x4 WorldViewIT			: WORLDVIEWINVERSETRANSPOSE;
float4 LightDirection			: LIGHTDIRWORLD1;
float4 EyePosWorld				: EYEPOSWORLD;

struct VS_INPUT
{
	float3 Vertex			: POSITION;
	float3 Normal			: NORMAL;
};

struct VS_OUTPUT
{
	float4 Position			: SV_POSITION;
	float  varDot			: LIGHTINTENSITY;
	float diffuseIntensity	: DIFFUSEINTENSITY;
	float specularIntensity	: SPECULARINTENSITY;
};

const float3 cBaseColour = float3(0.9, 0.2, 0.2);

VS_OUTPUT VertShader(VS_INPUT In)
{
	const float3 cGrainDirection = float3(1.0, 2.0, 0.0);
	const float4 cMaterial = float4(0.4,0.6,0.9,0.0);
	
	VS_OUTPUT Out;
	Out.Position = mul(float4(In.Vertex, 1.0), WorldViewProjection);
	
	float3 transNormal = mul(float4(In.Normal, 1.0), WorldViewIT).xyz;

	float3 normalXgrain = cross(transNormal, normalize(cGrainDirection));
	float3 tangent = normalize(cross(normalXgrain, transNormal));
	float LdotT = dot(tangent, normalize(LightDirection));
	float VdotT = dot(tangent, normalize(EyePosWorld));

	float temp = sqrt(1.0 - LdotT * LdotT);
	float NdotL = temp;
	float VdotR = temp * sqrt(1.0 - VdotT * VdotT) - VdotT * LdotT;

	Out.diffuseIntensity = max(NdotL * cMaterial.x + cMaterial.y, 0.0);
	Out.specularIntensity = max(VdotR * VdotR * cMaterial.z + cMaterial.w, 0.0);

	
	return Out;
}


float4 PixShader(VS_OUTPUT In) : SV_Target
{
	float3 finalColour = (cBaseColour * In.diffuseIntensity) + In.specularIntensity;
	return float4(finalColour, 1.0);
}



technique10 AnisotropicLighting
{
    pass P0
    {
        SetVertexShader( CompileShader( vs_4_0, VertShader() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, PixShader() ) );
    }
}

