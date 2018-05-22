
#include "Common.fxh"

#ifndef NUM_PARTICLES_IN_TRAIL
#	define NUM_PARTICLES_IN_TRAIL 64
#endif

cbuffer cbFrameParams 
{
	float3 g_vMeteoritePos;
	float3 g_vMeteoriteTrailDir;
	matrix g_mViewInvMatrix;
    matrix g_mWorldViewProj;
    float4 g_CameraPos;
}

cbuffer cbImmutable
{
	float g_ParticleScale = 1;
}
cbuffer cbTrailParticles
{
	float2 g_MinMaxParticleSize = float2(0.5, 2.0);
	float4 g_Particles[NUM_PARTICLES_IN_TRAIL];
};

cbuffer cbLightParams
{
    // WARNING: these parameters are duplicated in AtmEffects11.fx
    //float4 g_vDirectionOnSun = {0.f, 0.769666f, 0.638446f, 1.f}; ///< Direction on sun
    float4 g_vSunColorAndIntensityAtGround = {0.640682f, 0.591593f, 0.489432f, 100.f}; ///< Sun color
    float4 g_vAmbientLight = {0.191534f, 0.127689f, 0.25f, 0.f}; ///< Ambient light
}

struct PS_RenderSceneInput
{
    float4 f4Position   : SV_Position;
    float3 f3TexCoord   : TEXCOORD0;
    float fDistToCamera : DISTANCE_TO_CAMERA;
};

Texture2D g_txSmokeModulation;
Texture3D g_txSmoke;

PS_RenderSceneInput RenderTrailVS(in uint VertexID : SV_VertexID, 
                                  in uint InstID : SV_InstanceID)
{
	uint QuadNum = InstID;
	
    float4 DstTextureMinMaxUV = float4(-1,1,1,-1);
    float4 SrcElevAreaMinMaxUV = float4(0,0,1,1);
	
	float4 QuadXYUV[4] = 
	{
		float4(-1, 1,   0,0),
		float4(-1,-1,   0,1),
		float4( 1, 1,   1,0),
		float4( 1,-1,   1,1)
	};
	
	float fRelativePos = saturate( g_Particles[QuadNum].w );
	
	float3 SmokeQuadCenter = g_Particles[QuadNum].xyz;
	
	float3 CurrQuadVert = SmokeQuadCenter + mul( float3(QuadXYUV[VertexID].xy * lerp(g_MinMaxParticleSize.x, g_MinMaxParticleSize.y, fRelativePos), 0) * g_ParticleScale, (float3x3)g_mViewInvMatrix );
	
    PS_RenderSceneInput O;
    O.f4Position = mul( float4( CurrQuadVert, 1.0f ), g_mWorldViewProj );
    O.f3TexCoord.xy = QuadXYUV[VertexID].zw;
    O.f3TexCoord.z = fRelativePos;
    O.fDistToCamera = length( CurrQuadVert - g_CameraPos.xyz );

    return O;
}


//--------------------------------------------------------------------------------------
// This shader outputs the pixel's color by passing through the lit 
// diffuse material color & modulating with the diffuse texture
//--------------------------------------------------------------------------------------
float4 RenderTrailPS( PS_RenderSceneInput I ) : SV_Target
{
	float4 Color = g_txSmoke.Sample( samLinearClamp, I.f3TexCoord );
	float3 SmokeModulation = g_txSmokeModulation.Sample( samLinearClamp, float2(I.f3TexCoord.z,0.5) ).rgb;
	//Color.rgb = dot( float3(0.2125f, 0.7154f, 0.0721f), Color.rgb );
	Color.rgb *= SmokeModulation *  max(0.3, 3 * (1 - I.f3TexCoord.z)*(1-I.f3TexCoord.z) );
	Color.rgb *= g_vSunColorAndIntensityAtGround.rgb;


    Color.rgb = ApplyFog(Color.rgb, I.fDistToCamera);

	return Color;
}

RasterizerState RS_SolidFill
{
    FILLMODE = Solid;
    CullMode = Back;
    FrontCounterClockwise = true;
};


BlendState AlphaBlending
{
    BlendEnable[0] = TRUE;
    RenderTargetWriteMask[0] = 0x0F;
    BlendOp = ADD;
    SrcBlend = SRC_ALPHA;
    DestBlend = INV_SRC_ALPHA;
    SrcBlendAlpha = ZERO;
    DestBlendAlpha = INV_SRC_ALPHA;
};

DepthStencilState DSS_EnableDepthTestDisableWrites
{
    DepthEnable = TRUE;
    DepthWriteMask = 0;
};

technique11 RenderMeteoriteTrail_FL10
{
    pass PRender
    {
        SetBlendState( AlphaBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( RS_SolidFill );
        SetDepthStencilState( DSS_EnableDepthTestDisableWrites, 0 );

        SetVertexShader( CompileShader(vs_4_0, RenderTrailVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader(ps_4_0, RenderTrailPS() ) );
    }
}
