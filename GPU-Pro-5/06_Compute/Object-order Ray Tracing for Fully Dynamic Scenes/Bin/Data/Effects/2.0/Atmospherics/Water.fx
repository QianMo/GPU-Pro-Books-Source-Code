#define BE_RENDERABLE_INCLUDE_WORLD
#define BE_RENDERABLE_INCLUDE_PROJ
#define BE_RENDERABLE_INCLUDE_ID

#include <Engine/Perspective.fx>
#include <Engine/Renderable.fx>
#include <Utility/Bits.fx>
#include <Utility/Math.fx>

Texture2D PlanarReflection : ReflectionTexture;
float4 DestinationResolution : Resolution;

cbuffer SetupConstants
{
	float4 GroundColor
	<
		String UIName = "Ground";
	> = float4(0.1f, 0.25f, 0.15f, 1.0f);

	float4 SurfaceColor
	<
		String UIName = "Surface";
	> = float4(0.2f, 0.25f, 0.35f, 1.0f);

	float2 TextureRepeat
	<
		String UIName = "Texture Repeat";
	> = float2(1.0f, 1.0f);

	float4 TextureSpeed
	<
		String UIName = "Texture Speed";
	> = float4(0.0f, 0.1f, 0.0f, 0.05f);

	float TextureStrength
	<
		String UIName = "Texture Strength";
	> = 0.1f;
}

Texture2D WavesTexture
<
	string UIName = "Waves";
>;

struct Pixel
{
	float4 Position		: SV_Position;
	float3 WorldPos		: TexCoord1;
};

Pixel VSMain(uint id : SV_VertexID)
{
	Pixel o;
	
	float4 objPos;
	objPos.xz = bitunzip2(id) / 32.0f * 2.0f - 1.0f;
	objPos.y = 0.0f;
	objPos.w = 1.0f;

	float4 worldPos = mul(objPos, World);
	o.Position = mul(worldPos, Perspective.ViewProj);
	o.WorldPos = worldPos.xyz;

	return o;
}

SamplerState ReflectionSampler
{
	Filter = MIN_MAG_MIP_LINEAR;
	AddressU = CLAMP;
	AddressV = CLAMP;
};

SamplerState LinearSampler
{
	Filter = MIN_MAG_MIP_LINEAR;
	AddressU = WRAP;
	AddressV = WRAP;
};

float Waves(float2 pos)
{
	float4 distT = (pos.x + 1337) * float4(0.01f, 0.027f, 0.071f, 0.33f) + Perspective.Time * float4(0.13f, -0.27f, 0.41f, -1.3f);
	float4 dist = sin(distT);

	float4 rollingT = (pos.y) * float4(0.01f, 0.027f, 0.051f, 0.094f) + Perspective.Time * float4(-0.03f, -0.07f, -0.08f, -0.13f);
	rollingT.x += dot(dist, float4(0.25f, 0.08, 0, 0));
	rollingT.y -= dot(dist, float4(0.08f, 0.15, 0.06, 0));
	rollingT.z += dot(dist, float4(0.01f, 0.05, 0.06, 0.02));
	rollingT.w -= dot(dist, float4(0.3f, 0.05, 0.06, 0.02));
	float4 rolling = 1 + cos(PI * 0.87f + frac(rollingT));
//	rolling.x *= abs( dot(dist, float4(1.0f, 0.05, 0.0, 0.0)) );
//	rolling.y *= abs( dot(dist, float4(1.0f, 0.2, 0.0, 0.0)) );
//	rolling.z *= abs( dot(dist, float4(1.0f, 0.0, 1.5, 0.0)) );
	
	float bump1 = WavesTexture.Sample(LinearSampler, pos * TextureRepeat + TextureSpeed.xy * Perspective.Time).x;
	float bump2 = WavesTexture.Sample(LinearSampler, pos * -TextureRepeat.yx + TextureSpeed.zw * Perspective.Time).x;

//	return rolling.z;
	return dot(rolling, float4(0.5f, 0.25f, 0.1f, 0.05f)) + TextureStrength * (bump1 + bump2 + bump1 * bump2); // 0.5f + 0.5f * dist.x; // rolling.x;
}

float4 PSMain(Pixel p) : SV_Target0
{
	float ddeps = 0.005f; //  * distance(p.WorldPos, Perspective.CamPos);
	float centerWave = Waves(p.WorldPos.xz);
	float rightWave = Waves(p.WorldPos.xz + float2(ddeps, 0.0f));
	float topWave = Waves(p.WorldPos.xz + float2(0.0f, ddeps));

	float2 speed = float2(0.1f, 0.3f);
	float wavelength = 1.0f;
	float4 wavedist = float4(5.0f, 2.0f, 1.0f, 1.0f);
	float4 normaldistX = float4(2.0f, 1.0f, 1.0f, 0.5f);
	float4 normaldistZ = float4(20.0f, 10.0f, 5.0f, 2.0f);
	float4 texDist = 0.015f * float4(0.2f, 0.2f, 0.1f, 0.05f);

	float4 times = Perspective.Time * float4(3.1f, 2.7f, 1.9f, 1.3f);
	float4 spaceX = (p.WorldPos.x + 2233) * float4(0.02f, 0.027f, 0.1f, 0.3f);
	float4 spaceZ = (p.WorldPos.z + 2233) * float4(0.5f, 1.3f, 2.3f, 3.1f);
	float4 sinesX = sin(wavelength * spaceX + speed.x * times);
	float4 sinesZ = sin(wavelength * spaceZ + dot(wavedist, sinesX) + speed.y * times);
//	float dist = dot(sin(times), 0.1f);
	
	float3 flatNormal = normalize(World[1].xyz);
	float3 normal = flatNormal;
	normal.x -= 30 * (rightWave - centerWave) / ddeps;
	normal.z -= 30 * (topWave - centerWave) / ddeps;
//	normal.x -= dot(sinesX, normaldistX);
///	normal.z -= dot(sinesZ, normaldistZ);
	normal = normalize(normal);

//	return float4(0.5f + 0.5f * normal, 1.0f);

	float3 camDir = normalize(p.WorldPos - Perspective.CamPos);
	float cosCam = abs( dot(normal, camDir) );
	float reflectivity = lerp(0.04f, 1.0f, pow(saturate(1.0f - cosCam), 5));

	float3 color = lerp(SurfaceColor.xyz, GroundColor.xyz, cosCam);

	float2 texSpeed = float2(2.0f, 1.7f);
	float4 texX = (p.WorldPos.x + 2233) * float4(1.7f, 2.7f, 4.7f, 8.3f);
	float4 texZ = (p.WorldPos.z + 2233) * float4(1.5f, 2.3f, 4.3f, 8.1f);
	float4 texSinesX = sin(wavelength * texX + texSpeed.x * times);
	float4 texSinesZ = sin(wavelength * texZ + dot(float4(5.0f, 2.0f, 1.0f, 1.0f), sinesX) + texSpeed.y * times);

	float3 normalDelta = mul(normal - flatNormal, (float3x3) Perspective.View);
	
	float2 texCoord = p.Position.xy * DestinationResolution.zw;
	texCoord += normalDelta.xy * 0.05f;
//	texCoord.x += dot(texSinesX + texSinesZ + sinesX + sinesZ, texDist);
//	texCoord.y += dot(texSinesZ - texSinesX + sinesX - sinesZ, texDist);
	color = lerp(color, PlanarReflection.Sample(ReflectionSampler, texCoord).xyz, reflectivity);
	
	return float4(color, 1.0f);
}

technique11 Default <
	string PipelineStage = "DefaultPipelineStage";
	string RenderQueue = "DefaultRenderQueue";
	bool EnableProcessing = true;
>
{
	pass
	{
		SetVertexShader( CompileShader(vs_4_0, VSMain()) );
		SetPixelShader( CompileShader(ps_4_0, PSMain()) );
	}
}

uint4 PSObjectIDs(float4 p : SV_Position) : SV_Target0
{
	return ObjectID;
}

technique11 ObjectIDs <
	string PipelineStage = "ObjectIDPipelineStage";
	string RenderQueue = "DefaultRenderQueue";
>
{
	pass
	{
		SetVertexShader( CompileShader(vs_4_0, VSMain()) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader(ps_4_0, PSObjectIDs()) );
	}
}