#define BE_RENDERABLE_INCLUDE_WORLD

#include <Engine/Perspective.fx>
#include <Engine/Renderable.fx>
#include <Engine/PointLight.fx>
#include <Pipelines/LPR/Scene.fx>
#include <Pipelines/LPR/Geometry.fx>

cbuffer Light0
{
    PointLightLayout PointLight;
}

//Texture2D PointLightShadowMap : ShadowMaps0;

struct Vertex
{
    float4 Position	: Position;
};

struct Pixel
{
    float4 Position		: SV_Position;
    float4 ScreenSpace	: TexCoord0;
    float3 CamDir		: TexCoord1;
};

Pixel VSMain(Vertex v)
{
    Pixel o;
    
    float4 worldPosition = mul(v.Position, World);

    o.Position = mul(worldPosition, Perspective.ViewProj);
    o.ScreenSpace = float2(0.5f, 0.0f).xxyy * o.Position.w + float3(0.5f, -0.5f, 1.0f).xyzz * o.Position;
    o.Position.z = 0.0f;
//	o.Position.w = abs(o.Position.w);

    o.CamDir = worldPosition.xyz - Perspective.CamPos.xyz;

    return o;
}

SamplerState DefaultSampler
{
    Filter = MIN_MAG_MIP_POINT;
};

SamplerState ShadowSampler
{
    Filter = MIN_MAG_MIP_LINEAR;
};

float4 PSMain(Pixel p, uniform bool bShadowed = true) : SV_Target0
{
    float2 texCoord = p.ScreenSpace.xy / p.ScreenSpace.w;

    float4 eyeGeometry = SceneGeometryTexture[p.Position.xy];
    GBufferDiffuse gbDiffuse = ExtractDiffuse( SceneDiffuseTexture[p.Position.xy] );
    GBufferSpecular gbSpecular = ExtractSpecular( SceneSpecularTexture[p.Position.xy] );
    
    float3 diffuseColor = gbDiffuse.Color;
    float3 specularColor = gbSpecular.Color;

    float3 world = Perspective.CamPos.xyz + p.CamDir * eyeGeometry.x / dot(p.CamDir, Perspective.CamDir.xyz);

    float3 toWorld = world - PointLight.Pos.xyz;
    float3 toWorldDir = normalize(toWorld);

    // Distance falloff
    float distSq = dot(toWorld, toWorld);
    float intensity = 1.0f / (PointLight.AttenuationOffset + PointLight.Attenuation * distSq);
    intensity *= saturate(10.0f - 10.0f * distSq / (PointLight.Range * PointLight.Range));
    
    // Angle fallof
    intensity *= saturate( dot(eyeGeometry.yzw, -toWorldDir) );

/*	// Shadow
    if (bShadowed)
    {
        float4 shadowCoord = mul(float4(world + 0.1f * eyeGeometry.yzw, 1.0f), SpotLight.ShadowProj);
        float2 shadowMapCoord = 0.5f + float2(0.5f, -0.5f) * shadowCoord.xy / shadowCoord.w;

        float2 ditherVec = frac(p.Position.xy * 0.5f) > 0.26f;
        ditherVec.y += ditherVec.x;
        ditherVec.y *= step(ditherVec.y, 1.1f);
        
        float2 shadowMapOffsetCoord = shadowMapCoord.xy - (ditherVec - 0.5f) * SpotLight.ShadowPixel;
        float4 shadowMapPixelOffsets = float2(0.5f, -0.5f).xxyy * SpotLight.ShadowPixel.xyxy;

        float4 shadowDepths = float4(
                SpotLightShadowMap.SampleLevel(ShadowSampler, shadowMapOffsetCoord + shadowMapPixelOffsets.zw, 0.0f).r,
                SpotLightShadowMap.SampleLevel(ShadowSampler, shadowMapOffsetCoord + shadowMapPixelOffsets.xw, 0.0f).r,
                SpotLightShadowMap.SampleLevel(ShadowSampler, shadowMapOffsetCoord + shadowMapPixelOffsets.zy, 0.0f).r,
                SpotLightShadowMap.SampleLevel(ShadowSampler, shadowMapOffsetCoord + shadowMapPixelOffsets.xy, 0.0f).r
            );
        float4 shadowResults = saturate( 1.0f - (shadowCoord.z - shadowDepths * shadowCoord.w) * SpotLight.Range );
        
        intensity *= saturate( dot(shadowResults, 0.25f) );
    }
*/
    return float4(diffuseColor.xyz * PointLight.Color.xyz * intensity, 0.0f);
}

technique11 Default <
    string PipelineStage = "LightingPipelineStage";
    string RenderQueue = "DefaultRenderQueue";
>
{
    pass < string LightType = "PointLight"; bool Shadowed = false; >
    {
        SetVertexShader( CompileShader(vs_4_0, VSMain()) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader(ps_4_0, PSMain(false)) );
    }

    pass < string LightType = "PointLight"; bool Shadowed = true; >
    {
        SetVertexShader( CompileShader(vs_4_0, VSMain()) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader(ps_4_0, PSMain(true)) );
    }
}
