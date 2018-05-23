#include <Engine/Perspective.fx>
#include <Engine/DirectionalLight.fx>
#include <Pipelines/LPR/Scene.fx>
#include <Pipelines/LPR/Geometry.fx>

#include <Utility/Math.fx>

#include "Processing/BilateralAverage.fx"

tbuffer LightData
{
	DirectionalLightLayout DirectionalLight[16];
}
tbuffer ShadowData
{
	DirectionalShadowLayout DirectionalShadow[16];
}

Texture2DArray DirectionalLightShadowMaps : ShadowMaps;

// TODO: Check if available?
Texture2D AmbientTexture : AmbientTarget
<
	string TargetType = "Permanent";
	string Format = "R8G8B8A8U";
>;

Texture2D NoiseTexture
<
	string UIName = "Noise";
	string UIFile = "rotationalMatrix32.png";
>;

Texture2D ShadowTexture : ShadowTarget
<
	string TargetType = "Temporary";
	string Format = "R8G8B8A8U";
>;

float4 DestinationResolution : DestinationResolution;

struct Pixel
{
	float4 Position						: SV_Position;
	float2 TexCoord						: TexCoord0;
	float3 CamDir						: TexCoord1;
	nointerpolation uint LightIdx		: TexCoord2;
};

Pixel VSMain(uint v : SV_VertexID, uint i : SV_InstanceID)
{
	Pixel o;
	
	o.Position.x = (v & 1) ? 1.0f : -1.0f;
	o.Position.y = (v < 2) ? 1.0f : -1.0f;
	o.Position.zw = float2(0.0f, 1.0f);

	o.TexCoord = 0.5f + float2(0.5f, -0.5f) * o.Position.xy;
	
	o.CamDir = o.Position.xyw * float3(Perspective.ProjInv[0][0], Perspective.ProjInv[1][1], 1.0f);
	o.CamDir = mul(o.CamDir, (float3x3) Perspective.ViewInv);

	o.LightIdx = i;

	return o;
}

SamplerState DefaultSampler
{
	Filter = MIN_MAG_MIP_POINT;
};

SamplerState NoiseSampler
{
	AddressU = WRAP;
	AddressV = WRAP;
};

SamplerState ShadowSampler
{
	Filter = MIN_MAG_MIP_LINEAR;
};

static const int PoissonKernelSize = 8;

static const float2 PoissonKernel[PoissonKernelSize] = {
  float2(-0.07280093f, 0.38396510f),
  float2(-0.88017300f, 0.39960410f),
  float2(-0.15059570f,-0.69103160f),
  float2( 0.36638260f,-0.58297690f),
  float2( 0.80362130f, 0.18440510f),
  float2( 0.32283370f, 0.35518560f),
  float2(-0.70078040f,-0.12033080f),
  float2( 0.14763900f, 0.74279390f)
};
/*
static const float2 PoissonKernel[PoissonKernelSize] = {
  0.125f * float2( 1.0f, 0.0f),
  1.0f * float2( 0.7071f, 0.7071f ),
  0.25f * float2( 0.0f, 1.0f ),
  0.875f * float2( -0.7071f, 0.7071f ),
  0.375f * float2( -1.0f, 0.0f), 0.625f * 
  0.5f * float2( -0.7071f, -0.7071f ),
  0.625f * float2( 0.0f, -1.0f ),
  0.75f * float2( 0.7071f, -0.7071f ),
};
*/
#ifdef NONONO
static const float2 PoissonKernel[PoissonKernelSize] = {
  /* 0.15f */ float2( 1.0f, 0.0f),
  /* 1.0f */ float2( 0.7071f, 0.7071f ),
  /* 0.3f */ float2( 0.0f, 1.0f ),
  /* 0.9f */ float2( -0.7071f, 0.7071f ),
  /* 0.5f */ float2( -1.0f, 0.0f), 0.625f * 
  /* 0.8f */ float2( -0.7071f, -0.7071f ),
  /* 0.7f */ float2( 0.0f, -1.0f ),
  /* 0.7f */ float2( 0.7071f, -0.7071f ),
};
#endif

static const float2 SignedPoissonKernel[PoissonKernelSize] = {
  float2(-0.07280093f, 0.38396510f),
  float2(-0.88017300f, 0.39960410f),
  float2(-0.15059570f, 0.69103160f),
  float2( 0.36638260f, 0.58297690f),
  float2( 0.80362130f, 0.18440510f),
  float2( 0.32283370f, 0.35518560f),
  float2(-0.70078040f, 0.12033080f),
  float2( 0.14763900f, 0.74279390f)
};

float4 PSShadow(Pixel p) : SV_Target0
{
	float4 eyeGeometry = SceneGeometryTexture[p.Position.xy];
	float eyeDepth = ExtractDepth(eyeGeometry);
	
	int splitIndex = (int) dot(eyeDepth >= DirectionalShadow[p.LightIdx].ShadowSplitPlanes, 1.0f);

	clip(3 - splitIndex);

	float3 world = Perspective.CamPos.xyz + p.CamDir * eyeDepth;
	float3 worldNormal = ExtractNormal(eyeGeometry);

	float4 shadowCoord = mul(float4(world + 0.1f * worldNormal, 1.0f), DirectionalShadow[p.LightIdx].ShadowSplits[splitIndex].Proj);
	float2 shadowMapCoord = 0.5f + float2(0.5f, -0.5f) * shadowCoord.xy;

	clip( float4(shadowMapCoord.xy, 1.0f - shadowMapCoord.xy) );

	float shadowRange = DirectionalShadow[p.LightIdx].ShadowSplits[splitIndex].NearFar.y - DirectionalShadow[p.LightIdx].ShadowSplits[splitIndex].NearFar.x;

	float shadowDepth = DirectionalLightShadowMaps.SampleLevel(ShadowSampler, float3(shadowMapCoord, splitIndex), 0.0f).r;
	float shadowDeltaDepth = (shadowCoord.z - shadowDepth * shadowCoord.w) * shadowRange;
	clip(0.5f + shadowDeltaDepth);

	float3 shadowMapCoordZ = float3(shadowMapCoord, shadowCoord.z);
	
	float3 shadowMapCoordDDX = ddx(shadowMapCoordZ), shadowMapCoordDDY = ddy(shadowMapCoordZ);
	float2 shadowMapCoordDepthCorrection = float2(
		shadowMapCoordDDX.y * shadowMapCoordDDY.z - shadowMapCoordDDY.y * shadowMapCoordDDX.z,
		shadowMapCoordDDY.x * shadowMapCoordDDX.z - shadowMapCoordDDX.x * shadowMapCoordDDY.z )
		/ (shadowMapCoordDDY.x * shadowMapCoordDDX.y - shadowMapCoordDDX.x * shadowMapCoordDDY.y);

	float scaledRadius = 0.05f + 0.2f * clamp(0.2f * shadowDeltaDepth, 0.0f, 5.0f);
	float2 scaledOffset = scaledRadius * DirectionalShadow[p.LightIdx].ShadowSplits[splitIndex].PixelScale; // lerp( 2.0f, 0.25f, 1.0f / (1.0f + 0.02f * eyeDepth) )

	float shadowMultiplier = 50.0f;

	float testVisibility = saturate( 1.0f - shadowMultiplier * shadowDeltaDepth );
	float visibility = 0.0f;

	float4 noise = NoiseTexture.SampleLevel(NoiseSampler, p.Position.xy / 32.0f, 0) * 2.0f - 1.0f;

	float depthCorrectionAcc = 0.0f;

	for (int i = 0; i < 8; ++i)
	{
		float2 sampleOffset = scaledOffset * (PoissonKernel[i].x * noise.xy + PoissonKernel[i].y * noise.zw);

		float sampleDepthCorrection = min(0.0f, dot(sampleOffset, shadowMapCoordDepthCorrection));

		depthCorrectionAcc -= sampleDepthCorrection;

		float sampleDepth = DirectionalLightShadowMaps.SampleLevel(ShadowSampler, float3(shadowMapCoord + sampleOffset, splitIndex), 0.0f).r;
		float sampleDeltaDepth = (shadowCoord.z - sampleDepth * shadowCoord.w) * shadowRange + sampleDepthCorrection;
		visibility += saturate( 1.0f - shadowMultiplier * sampleDeltaDepth );
	}

	visibility = saturate( 2.5 * visibility / 8 );

	return visibility;
}

#ifdef SMOOTH_MINMAX

float smoothminmax(float a, float b, float dir)
{
	float m = 0.5f * (a + b);
	float d = 0.5f * abs(a - b);
	float dsq = d * d;
	float f = dsq * rcp(0.002f + dsq);

	return m + dir * d * f; // * smoothstep(0.0f, 0.2f, d);
}

float smoothmin(float a, float b) { return smoothminmax(a, b, -1.0f); }
float smoothmax(float a, float b) { return smoothminmax(a, b, 1.0f); }

#else

float smoothmin(float a, float b) { return min(a, b); }
float smoothmax(float a, float b) { return max(a, b); }

#endif

float tanSqFromCos(float cosAngle)
{
	float cosAngleSq = cosAngle * cosAngle;
	return saturate(1.0f - cosAngleSq) * rcp(cosAngleSq);
}

float tanFromCos(float cosAngle)
{
	return sqrt(tanSqFromCos(cosAngle));
}

float approxTanFor0to1(float zeroToOne)
{
	return (zeroToOne * (PI / 2.0f)) * rcp(1.0f - 0.88f * zeroToOne * zeroToOne);
}

float Schlick(float cosAngle)
{
	return pow(saturate(1.0f - cosAngle), 5);
}

float OrenNayar(float3 camDir, float3 lightDir, float3 normal,
				float roughness)
{
	float variance = sq(roughness * (PI / 2.0f));
	float A = 1.0f - 0.5f * variance / (variance + 0.33f);
	float B = 0.45f * variance / (variance + 0.09f);

	float cosCam = dot(normal, camDir);
	float cosLgt = dot(normal, lightDir);

	float3 camDirP = normalize(camDir - normal * 0.9999f * cosCam);
	float3 lightDirP = normalize(lightDir - normal * 0.9999f * cosLgt);
	
	float cosDelta = dot(camDirP, lightDirP);
	if (isnan(cosDelta)) cosDelta = 1.0f;

	float sinAlpha = pyt1( smoothmin(cosCam, cosLgt) );
	float cosBeta = smoothmax(cosCam, cosLgt);
	float tanBeta = pyt1(cosBeta) / (cosBeta);
	
	float r = saturate(cosLgt);
#ifdef DIFFUSE_NORM
	r *= (1.0f / PI);
#endif
	r *= A + B * saturate(0.5f + 0.5f * cosDelta) * sinAlpha * tanBeta; // max(0, cosDelta) // saturate(0.5f + 0.5f * cosDelta)
	return r;
}

float KelemenKarlos(float3 camDir, float3 lightDir, float3 normal, 
					float roughness)
{
	float variance = sq( max(approxTanFor0to1(roughness), 0.001f) );

	float cosLgt = dot(normal, lightDir);

	float3 halfVec = camDir + lightDir;
	halfVec += step(lengthsq(halfVec), 0.01f) * normal;
	float3 halfDir = normalize(halfVec);
	float cosHalf = dot(halfDir, normal);
	float tanHalfSq = tanSqFromCos(cosHalf);

	float ward = rcp(variance * PI * pow(cosHalf, 3)) * exp(-tanHalfSq * rcp(variance));

	return saturate(cosLgt) * ward / lengthsq(halfVec);
}

float4 PSMain(Pixel p, uniform bool bShadowed = true) : SV_Target0
{
	float4 eyeGeometry = SceneGeometryTexture[p.Position.xy];
	GBufferDiffuse gbDiffuse = ExtractDiffuse( SceneDiffuseTexture[p.Position.xy] );
	GBufferSpecular gbSpecular = ExtractSpecular( SceneSpecularTexture[p.Position.xy] );
	
	float3 diffuseColor = gbDiffuse.Color;
	float3 specularColor = gbSpecular.Color * (0.1f + 0.9f * gbSpecular.FresnelM); // / 5.0f; // * gbSpecular.Color;

//	return diffuseColor;

	float3 worldNormal = normalize( ExtractNormal(eyeGeometry) );
	float3 camDir = normalize(p.CamDir);

	float3 intensity = 1.0f;

	if (bShadowed)
		intensity = ShadowTexture.SampleLevel(DefaultSampler, p.TexCoord, 0).xyz;

	float3 alternateLightDir = DirectionalLight[p.LightIdx].Dir;
	float3 alternateIntensity = 0.0f;
	float alternateRoughness = gbDiffuse.Roughness;
	if (dot(worldNormal, alternateLightDir) > 0.0f)
	{
		alternateLightDir = -alternateLightDir;
		alternateIntensity = 1.0f;
		alternateRoughness += 3.0f * saturate(0.15f - alternateRoughness);
	}

	float3 halfVec = camDir + alternateLightDir;
	float3 halfDir = normalize(halfVec);

	float orenNayar = OrenNayar(-camDir, -DirectionalLight[p.LightIdx].Dir, worldNormal, gbDiffuse.Roughness);
	float kelemen = KelemenKarlos(-camDir, -alternateLightDir, worldNormal, alternateRoughness);

	float cosAngle = dot(worldNormal, -DirectionalLight[p.LightIdx].Dir);
	float cosAlternateAngle = dot(worldNormal, -alternateLightDir);
	float cosCamAngle = dot(worldNormal, -camDir);

	float reflectanceFresnel = Schlick(dot(camDir, halfDir));
	float3 specular = lerp(specularColor.xyz, 1.0f, reflectanceFresnel);
	float3 alternateSpecular = kelemen * specular * rcp(1.0f + dot(specular, 27.0f)) * alternateIntensity * (1.0f - intensity) * gbSpecular.Shininess;
	specular *= kelemen * intensity * (1.0f - alternateIntensity);

//	return float4(alternateSpecular, 1.0f);
//	return float4(kelemen * specular * rcp(1.0f + dot(specular, 27.0f)), 1.0f);
//	return float4(specular * rcp(0.001f + dot(specular, 27.0f)) * kelemen * DirectionalLight[p.LightIdx].SkyColor.xyz * gbSpecular.Shininess, 1.0f);

	float3 diffuseFresnelCoeff = 21.0f * rcp(20.0f * (1.0f - 0.97f * specularColor.xyz)) // (0.97f - 0.03f * specularColor.xyz)
		* (1.0f - Schlick(cosAlternateAngle)) * (1.0f - Schlick(cosCamAngle));

	// Angle fallof
	float negIntensity = saturate(0.5f - 0.35f * cosAngle); // * AmbientTexture.SampleLevel(DefaultSampler, p.TexCoord, 0).a;
//	float rimIntensity = (1 - abs(cosAngle)) * pow(1 - dot(-camDir, worldNormal), 8) * saturate( dot(camDir, -DirectionalLight[p.LightIdx].Dir) );
	float3 posIntensity = orenNayar * intensity;

	float3 ambient = diffuseColor.xyz * negIntensity * DirectionalLight[p.LightIdx].Color.w;

	float3 diffuse = diffuseColor.xyz * posIntensity;
	diffuse *= (1.0f - gbSpecular.Metalness);

	float3 radiance = lerp(diffuse, diffuse * diffuseFresnelCoeff + specular, gbSpecular.Shininess);
	radiance *=  DirectionalLight[p.LightIdx].Color.xyz;
	radiance += (ambient + alternateSpecular) * DirectionalLight[p.LightIdx].SkyColor.xyz;

//	radiance = diffuseFresnelCoeff;

	// Specular
/*	float3 fresnelLightDir = (cosAngle >= 0.0f) ? DirectionalLight[p.LightIdx].Dir : -DirectionalLight[p.LightIdx].Dir;
	float3 halfway = -normalize(camDir + fresnelLightDir);
	float fresnel = pow(1.0f - dot(halfway, -camDir), 5);
	
	float reflectCoeff = lerp(gbSpecular.FresnelR, 1.0f, fresnel);
	float3 specular = intensity * kelemen;

	float metalCoeff = lerp(gbSpecular.FresnelM, 1.0f, fresnel);
	float3 metalSpecular = lerp(specularColor.xyz, 1.0f, metalCoeff) * specular;

	specular = lerp(specular, metalSpecular, gbSpecular.Metalness);

	float3 radiance = lerp(diffuse, specular, reflectCoeff * gbSpecular.Shininess);
*/

//	float3 radiance = lerp(diffuse, specular, reflectCoeff * gbSpecular.Shininess);
//	float3 radiance = lerp(diffuse, (diffuse + specular * reflectCoeff), gbSpecular.Shininess);

//	float specExp = 1024.0f * (1.0f - gbDiffuse.Roughness) + 0.00001f;
//	float3 specular = specularColor.xyz * pow( saturate( dot(worldNormal, halfway) ) , specExp ); // * (specExp + 1.0f) * 0.5f;
	
	return float4( radiance, 0.0f );
}

technique11 Shadowed <
	bool EnableProcessing = true;
	string PipelineStage = "LightingPipelineStage";
	string RenderQueue = "DefaultRenderQueue";
>
{
	pass <
		string Color0 = "ShadowTarget";
		float4 ClearColor0 = float4(1.0f, 1.0f, 1.0f, 1.0f);
		bool bClearColor0 = true;

		string LightType = "DirectionalLight";
		bool Shadowed = true;
	>
	{
		SetBlendState( NULL, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xffffffff );

		SetVertexShader( CompileShader(vs_4_0, VSMain()) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader(ps_4_0, PSShadow()) );
	}
	
	pass <
		string Color0 = "ShadowTarget";
		
		string LightType = "DirectionalLight";
		bool Shadowed = true;
	>
	{
		SetVertexShader( CompileShader(vs_4_0, VSMain()) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader(ps_4_0, PSBilateralAverage(
			float2(1.0f, 0.0f), 3, 4,
			SceneGeometryTexture, DefaultSampler,
			ShadowTexture, ShadowSampler, DestinationResolution.zw )) );
	}

	pass <
		string Color0 = "ShadowTarget";
		
		string LightType = "DirectionalLight";
		bool Shadowed = true;
	>
	{
		SetVertexShader( CompileShader(vs_4_0, VSMain()) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader(ps_4_0, PSBilateralAverage(
			float2(0.0f, 1.0f), 3, 4,
			SceneGeometryTexture, DefaultSampler,
			ShadowTexture, ShadowSampler, DestinationResolution.zw )) );
	}
	
	pass <
		bool RevertTargets = true;
		
		string LightType = "DirectionalLight";
		bool Shadowed = true;

		bool RevertBlending = true;
	>
	{
		SetVertexShader( CompileShader(vs_4_0, VSMain()) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader(ps_4_0, PSMain(true)) );
	}
}

technique11 Default <
	string PipelineStage = "LightingPipelineStage";
	string RenderQueue = "DefaultRenderQueue";
>
{
	pass < string LightType = "DirectionalLight"; bool Shadowed = false; >
	{
		SetVertexShader( CompileShader(vs_4_0, VSMain()) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader(ps_4_0, PSMain(false)) );
	}
}
