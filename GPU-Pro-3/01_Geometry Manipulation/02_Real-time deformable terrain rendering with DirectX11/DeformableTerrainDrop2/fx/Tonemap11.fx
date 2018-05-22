/* ************************************************************************* *\
                  INTEL CORPORATION PROPRIETARY INFORMATION
     This software is supplied under the terms of a license agreement or 
     nondisclosure agreement with Intel Corporation and may not be copied 
     or disclosed except in accordance with the terms of that agreement. 
          Copyright (C) 2009 Intel Corporation. All Rights Reserved.
\* ************************************************************************* */

/// @file Postprocess.fx
///       The file contains all techniques necessary to implement sample

static const float3 LUMINANCE_VECTOR  = float3(0.2125f, 0.7154f, 0.0721f);
static const float  MIDDLE_GRAY = 1.0;//= 0.72f;
static const float  MINIMAL_LUMINANCE = 1.0;

SamplerState samLinearClamp
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Clamp;
    AddressV = Clamp;
};

SamplerState samPointClamp
{
    Filter = MIN_MAG_MIP_POINT;
    AddressU = Clamp;
    AddressV = Clamp;
};

struct GenerateQuadVS_OUTPUT
{
    float4 m_ScreenPos_PS : SV_POSITION;
    float2 m_UV : TEXCOORD0; 
};

GenerateQuadVS_OUTPUT GenerateQuadVS( in uint VertexId : SV_VertexID )
{
    float4 DstTextureMinMaxUV = float4(-1,1,1,-1);
    float4 SrcElevAreaMinMaxUV = float4(0,0,1,1);
    
    GenerateQuadVS_OUTPUT Verts[4] = 
    {
        {float4(DstTextureMinMaxUV.xy, 0.5, 1.0), SrcElevAreaMinMaxUV.xy}, 
        {float4(DstTextureMinMaxUV.xw, 0.5, 1.0), SrcElevAreaMinMaxUV.xw},
        {float4(DstTextureMinMaxUV.zy, 0.5, 1.0), SrcElevAreaMinMaxUV.zy},
        {float4(DstTextureMinMaxUV.zw, 0.5, 1.0), SrcElevAreaMinMaxUV.zw}
    };

    return Verts[VertexId];
}

Texture2D<float3> g_tex2DHDRImage;
float CalculateLum_DownScale3x3PS ( GenerateQuadVS_OUTPUT In ) : SV_TARGET
{    
    float3 vColor;
    float fAverageLum = 0.0f;
        
    [unroll]
    for( int y = -1; y <= 1; y++ )
    {
        [unroll]
        for( int x = -1; x <= 1; x++ )
        {
            // Compute the sum of color values
            vColor = g_tex2DHDRImage.SampleLevel( samLinearClamp, In.m_UV, 0, int2(x,y) );
            fAverageLum += dot( vColor.rgb, LUMINANCE_VECTOR );
        }
    }
    
    fAverageLum /= 9.0;

    return fAverageLum;
}

Texture2D<float> g_tex2DLum;
float DownScale3x3PS ( GenerateQuadVS_OUTPUT In ) : SV_TARGET
{    
    float fAverageLum = 0.0f;

    [unroll]
    for( int y = -1; y <= 1; y++ )
    {
        [unroll]
        for( int x = -1; x <= 1; x++ )
        {
            // Compute the sum of color values
            fAverageLum += g_tex2DLum.SampleLevel( samLinearClamp, In.m_UV, 0, int2(x,y) );
        }
    }
    
    fAverageLum /= 9.0;

    return fAverageLum;
}


float4 ToneMapPS( GenerateQuadVS_OUTPUT In ) : SV_TARGET
{   
    float3 vColor = max(g_tex2DHDRImage.Sample( samPointClamp, In.m_UV ), 0);
//return vColor;

    float vLum = g_tex2DLum.Sample( samPointClamp, float2(0,0) );
    vLum = max(vLum, MINIMAL_LUMINANCE);
    //float3 vBloom = s2.Sample( samplerLinearClamp, input.m_ScreenPos_PSCoord );
    
    // Tone mapping
    vColor.rgb *= MIDDLE_GRAY / ( vLum + 0.001f );
    //vColor.rgb *= ( 1.0f + vColor/LUM_WHITE );
    //vColor.rgb /= ( 1.0f + vColor );
    
    //vColor.rgb += 0.6f * vBloom;
    
    return pow( float4(vColor, 0), 2.2);
}

RasterizerState RS_SolidFill_NoCull
{
    FILLMODE = Solid;
    CullMode = NONE;
};

DepthStencilState DSS_DisableDepthTest
{
    DepthEnable = FALSE;
    DepthWriteMask = ZERO;
};

// Blend state disabling blending
BlendState NoBlending
{
    BlendEnable[0] = FALSE;
    BlendEnable[1] = FALSE;
    BlendEnable[2] = FALSE;
};


technique10 CalculateLum_DownScale3x3Tech_l10
{
    pass P0
    {
        SetDepthStencilState( DSS_DisableDepthTest, 0 );
        SetRasterizerState( RS_SolidFill_NoCull );
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );

        SetVertexShader( CompileShader( vs_4_0, GenerateQuadVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, CalculateLum_DownScale3x3PS() ) );
    }
}

technique11 DownScale3x3Tech_fl10
{
    pass P0
    {
        SetDepthStencilState( DSS_DisableDepthTest, 0 );
        SetRasterizerState( RS_SolidFill_NoCull );
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );

        SetVertexShader( CompileShader( vs_4_0, GenerateQuadVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, DownScale3x3PS() ) );
    }
}

technique11 ToneMapTech_fl10
{
    pass P0
    {
        SetDepthStencilState( DSS_DisableDepthTest, 0 );
        SetRasterizerState( RS_SolidFill_NoCull );
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );

        SetVertexShader( CompileShader( vs_4_0, GenerateQuadVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, ToneMapPS() ) );
    }
}
