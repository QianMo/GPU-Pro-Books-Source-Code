#include "Common.fxh"

cbuffer cbModifiedAreaPlacement
{
    float4 g_vModifiedAreaPlacement = float4(-1,1,1,-1);
    float4 g_vModificationMinMaxUV = float4(0,0,1,1);
    uint g_TexArrayInd;
    uint g_uiReconstructionPrecision = 1;
}

Texture2D<float> g_tex2Displacement;
#define DISPLACEMENT_SAMPLING_SCALE 32767.f

Texture2D<float> g_tex2DOrigHeightMap;

SamplerState samPointBorder0
{
    Filter = MIN_MAG_MIP_POINT;
    AddressU = BORDER;
    AddressV = BORDER;
    BorderColor = float4(0,0,0,0);
};

SamplerState samLinearBorder0
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = BORDER;
    AddressV = BORDER;
    BorderColor = float4(0,0,0,0);
};

#ifndef USE_TEXTURE_ARRAY
#   define USE_TEXTURE_ARRAY 0
#endif

struct GenerateQuadVS_OUTPUT
{
    float4 m_ScreenPos_PS : SV_POSITION;
    float2 m_ModifiedRegionUV : TEXCOORD0;
    uint m_uiInstID : INST_ID; // Unused if texture arrays are not used
};

GenerateQuadVS_OUTPUT GenerateQuadVS( in uint VertexId : SV_VertexID, 
                                      in uint InstID : SV_InstanceID)
{
    float4 DstTextureMinMaxUV = g_vModifiedAreaPlacement;
    float4 DisplacementMinMaxUV = g_vModificationMinMaxUV;
    
    GenerateQuadVS_OUTPUT Verts[4] = 
    {
        {float4(DstTextureMinMaxUV.xy, 0.5, 1.0), DisplacementMinMaxUV.xy, InstID}, 
        {float4(DstTextureMinMaxUV.xw, 0.5, 1.0), DisplacementMinMaxUV.xw, InstID},
        {float4(DstTextureMinMaxUV.zy, 0.5, 1.0), DisplacementMinMaxUV.zy, InstID},
        {float4(DstTextureMinMaxUV.zw, 0.5, 1.0), DisplacementMinMaxUV.zw, InstID}
    };

    return Verts[VertexId];
}

#if USE_TEXTURE_ARRAY

struct SPassThroughGS_Output
{
    GenerateQuadVS_OUTPUT VSOutput;
    uint RenderTargetIndex : SV_RenderTargetArrayIndex;
    float m_fTexArrInd : TEX_ARRAY_IND_FLOAT;
};

[maxvertexcount(3)]
void PassThroughGS(triangle GenerateQuadVS_OUTPUT In[3], 
                   inout TriangleStream<SPassThroughGS_Output> triStream )
{
    uint InstID =In[0].m_uiInstID;
    for(int i=0; i<3; i++)
    {
        SPassThroughGS_Output Out;
        Out.VSOutput = In[i];
        Out.RenderTargetIndex = g_TexArrayInd;
        Out.m_fTexArrInd = (float)g_TexArrayInd;
        triStream.Append( Out );
    }
}

#endif



float ModifyElevMapPS(
#if USE_TEXTURE_ARRAY
                      SPassThroughGS_Output In
#else
                      GenerateQuadVS_OUTPUT In
#endif
                      ) : SV_TARGET
{
#if USE_TEXTURE_ARRAY
    float2 HeightMapUV = In.VSOutput.m_ScreenPos_PS.xy;
    float2 DisplacementUV = In.VSOutput.m_ModifiedRegionUV;
#else
    float2 HeightMapUV = In.m_ScreenPos_PS.xy;
    float2 DisplacementUV = In.m_ModifiedRegionUV;
#endif
    
    // Load original height
    int Height = g_tex2DOrigHeightMap.Load( int3(HeightMapUV, 0) ) * HEIGHT_MAP_SAMPLING_SCALE;
    // Load displacement
    int Displacement = g_tex2Displacement.SampleLevel(samLinearBorder0, DisplacementUV, 0)  * DISPLACEMENT_SAMPLING_SCALE;
    Height += Displacement;
    Height = clamp(Height, 0, HEIGHT_MAP_SAMPLING_SCALE);
    // Height map must store only dequantized values
    Height = DequantizeValue( QuantizeValue( Height, g_uiReconstructionPrecision), g_uiReconstructionPrecision );
    return float(Height) / HEIGHT_MAP_SAMPLING_SCALE;
}

technique11 ModifyElevMap_FL10
{
    pass
    {
        SetDepthStencilState( DSS_DisableDepthTest, 0 );
        SetRasterizerState( RS_SolidFill_NoCull );
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );

        SetVertexShader( CompileShader( vs_4_0, GenerateQuadVS() ) );
#if USE_TEXTURE_ARRAY
        SetGeometryShader( CompileShader( gs_4_0, PassThroughGS() ) );
#else
        SetGeometryShader( NULL );
#endif
        SetPixelShader( CompileShader( ps_4_0, ModifyElevMapPS() ) );
    }
}
