#include "Common.fxh"

cbuffer cbFrameParams 
{
    matrix g_mWorldViewProj;
    matrix g_mWorld;
    float4 g_CameraPos;
}

cbuffer cbLightParams
{
    // WARNING: these parameters are duplicated in AtmEffects11.fx
    float4 g_vDirectionOnSun = {0.f, 0.769666f, 0.638446f, 1.f}; ///< Direction on sun
    float4 g_vSunColorAndIntensityAtGround = {0.640682f, 0.591593f, 0.489432f, 100.f}; ///< Sun color
    float4 g_vAmbientLight = {0.191534f, 0.127689f, 0.25f, 0.f}; ///< Ambient light
}
cbuffer cbMaterial
{
	float4 g_f4MaterialDiffuseColor;
}

struct VS_RenderSceneInput
{
    float3 f3Position   : POSITION;  
    float3 f3Normal     : NORMAL;     
    float2 f2TexCoord   : TEXCOORD;
};

struct PS_RenderSceneInput
{
    float4 f4Position   : SV_Position;
    float2 f2TexCoord   : TEXCOORD0;
    float3 f3NormalWS   : NORMAL;
    float fDistToCamera : DISTANCE_TO_CAMERA;
};

Texture2D g_txDiffuse;

PS_RenderSceneInput RenderSDKMeshVS(VS_RenderSceneInput I)
{
    PS_RenderSceneInput O;
    float3 f3NormalWorldSpace;
    
    // Transform the position from object space to homogeneous projection space
    O.f4Position = mul( float4( I.f3Position, 1.0f ), g_mWorldViewProj );
    
    // Transform the normal from object space to world space    
    O.f3NormalWS = normalize( mul( I.f3Normal, (float3x3)g_mWorld ) );
    
    // Pass through texture coords
    O.f2TexCoord = I.f2TexCoord; 
    
    O.fDistToCamera = length( mul( float4(I.f3Position.xyz,1), g_mWorld ) - g_CameraPos.xyz );

    return O;
}


//--------------------------------------------------------------------------------------
// This shader outputs the pixel's color by passing through the lit 
// diffuse material color & modulating with the diffuse texture
//--------------------------------------------------------------------------------------
float4 PS_RenderSceneTextured( PS_RenderSceneInput I ) : SV_Target
{
	float3 f4Diffuse = g_txDiffuse.Sample( samLinearClamp, I.f2TexCoord ).rgb;
    // Calc diffuse color    
    float4 f4ShadedColor;
    f4ShadedColor.rgb = f4Diffuse * (g_vSunColorAndIntensityAtGround.rgb * max( 0, dot( I.f3NormalWS, g_vDirectionOnSun.xyz ) ) + g_vAmbientLight.rgb);
    f4ShadedColor.a = 1.0f;
    f4ShadedColor.rgb = ApplyFog(f4ShadedColor.rgb, I.fDistToCamera);
    return f4ShadedColor;
}


//--------------------------------------------------------------------------------------
// This shader outputs the pixel's color by passing through the lit 
// diffuse material color
//--------------------------------------------------------------------------------------
float4 PS_RenderScene( PS_RenderSceneInput I ) : SV_Target
{
    // Calc diffuse color    
    float4 f4ShadedColor;
    f4ShadedColor.rgb = g_f4MaterialDiffuseColor.rgb * (g_vSunColorAndIntensityAtGround.rgb * max( 0, dot( I.f3NormalWS, g_vDirectionOnSun.xyz ) ) + g_vAmbientLight.rgb);
    f4ShadedColor.a = 1.0f;
    return f4ShadedColor;
}



RasterizerState RS_Wireframe_NoCull
{
    FILLMODE = Wireframe;
    CullMode = None;
    //AntialiasedLineEnable = true;
};

RasterizerState RS_SolidFill//Set by the app; can be biased or not
{
    FILLMODE = Solid;
    CullMode = Back;
    //FrontCounterClockwise = true;
};

technique11 RenderSDKMesh_FL10
{
    pass PRender
    {
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( RS_SolidFill );
        SetDepthStencilState( DSS_EnableDepthTest, 0 );

        SetVertexShader( CompileShader(vs_4_0, RenderSDKMeshVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader(ps_4_0, PS_RenderScene() ) );
    }

    pass PRenderTextured
    {
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( RS_SolidFill );
        SetDepthStencilState( DSS_EnableDepthTest, 0 );

        SetVertexShader( CompileShader(vs_4_0, RenderSDKMeshVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader(ps_4_0, PS_RenderSceneTextured() ) );
    }
}
