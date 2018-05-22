#include "Common.fxh"

#ifndef USE_TEXTURE_ARRAY
#   define USE_TEXTURE_ARRAY 1
#endif

#ifndef IS_PRECISE_BILINEAR_FILTERING
#   define IS_PRECISE_BILINEAR_FILTERING 0
#endif

cbuffer cbDecompressionParams
{
    uint g_uiReconstrPrecision;
    float4 g_OffspringsDstArea; // Destination rectangle on the decompressed height map in projection space [-1,1]x[-1,1]
    float4 g_ParentPatchSrcAreaUV; // Cooridnates of the corresponding parent patch height map region
    float4 g_ParentPatchUVClampArea; // valid parent patch coordinates area
    float4 g_SrcRfnmtLabelsIJRange; // Indices of the refinement data
}

cbuffer cbExtrapolationParams
{
    float4 g_DecodedOffspringsAreaUV;
#if USE_TEXTURE_ARRAY
    float4 g_SrcOffspringsAreaUV[4];
#else
    float4 g_SrcOffspringsAreaUV;
#endif
}

#if USE_TEXTURE_ARRAY
    Texture2DArray<float> g_tex2DElevationMapArr;
    uint g_uiParentPatchArrInd = 0;
    uint g_uiChildPatchArrInd[4];
#else
    Texture2D<float> g_tex2DParentHeightMap;
#endif

Texture2D<int> g_tex2DResiduals;
Texture2D<float> g_tex2DOffspringsHM;

struct DecompressOffspringsVS_OUTPUT
{
    float4 m_ScreenPos_PS : SV_POSITION;
    float2 m_ParentPatchUV : TEXCOORD0; 
    float2 m_ResidualIJ : IJ_INDICES;
};

DecompressOffspringsVS_OUTPUT DecompressOffspringsVS( in uint VertexId : SV_VertexID )
{
    DecompressOffspringsVS_OUTPUT Verts[4] = 
    {
        {float4(g_OffspringsDstArea.xy, 0.5, 1.0), g_ParentPatchSrcAreaUV.xy, g_SrcRfnmtLabelsIJRange.xy}, 
        {float4(g_OffspringsDstArea.xw, 0.5, 1.0), g_ParentPatchSrcAreaUV.xw, g_SrcRfnmtLabelsIJRange.xw},
        {float4(g_OffspringsDstArea.zy, 0.5, 1.0), g_ParentPatchSrcAreaUV.zy, g_SrcRfnmtLabelsIJRange.zy},
        {float4(g_OffspringsDstArea.zw, 0.5, 1.0), g_ParentPatchSrcAreaUV.zw, g_SrcRfnmtLabelsIJRange.zw}
    };

    return Verts[VertexId];
}

float DecompressOffspringsPS(DecompressOffspringsVS_OUTPUT In) : SV_TARGET
{
    // Filter parent patch height map and get the predicted value
    
    // Very important subtle point: bilinear texture filtering is done differently on NVidia and ATI hardware:
    // * ATI does precise filtering and returns exact weighted floating result
    // * NVidia internally performes filtering in integers and the result is always in form f/N where 
    //   f and N are integers; N is the normalization constant (N=65535 for 16-bit UNORM).
    //
    //  Suppose the following example:
    //     _____ _____
    //    |     |     |
    //    | 0/N * 1/N |    * - sampling point
    //    |_____|_____|
    //
    // On ATI GPU we will get honest 0.5/N, while on NVidia GPU we will get 1/N !
    // Since we need honest results, we need to perform bilinear filtering manually on NVidia

#if IS_PRECISE_BILINEAR_FILTERING

    // Clamp parent patch height map UV coordinates to the decoded area
    float2 ParentPatchUV = clamp( In.m_ParentPatchUV, g_ParentPatchUVClampArea.xy, g_ParentPatchUVClampArea.zw );

    // If bilinear texture filtering is done precisely by the GPU, we can simply use HW-supported sampler
#   if USE_TEXTURE_ARRAY
        int PredictedElev = g_tex2DElevationMapArr.SampleLevel(samLinearClamp, float3(ParentPatchUV, g_uiParentPatchArrInd), 0) * HEIGHT_MAP_SAMPLING_SCALE;
#   else
        int PredictedElev = g_tex2DParentHeightMap.SampleLevel(samLinearClamp, ParentPatchUV, 0) * HEIGHT_MAP_SAMPLING_SCALE;
#   endif

#else

    // If bilinear filtering is performed inexactly, we need to do it manually
    float2 ElevMapSize;
#   if USE_TEXTURE_ARRAY
        float Elems;
        g_tex2DElevationMapArr.GetDimensions(ElevMapSize.x, ElevMapSize.y, Elems);
#   else
        g_tex2DParentHeightMap.GetDimensions(ElevMapSize.x, ElevMapSize.y);
#   endif

    //        I0  -1       0         1         2 
    //           ----> <-------> <-------> <-------> 
    //            |   X    |    X    |    X    |    ....
    //         U  0       1/W       2/W       3/W         W = ElevMapSize.x
    //              0.5/W     1.5/W     2.5/W
    float2 IJ = In.m_ParentPatchUV*ElevMapSize - float2(0.5, 0.5);
    float2 IJ0 = floor(IJ);
    float2 BilinearWeights = IJ - IJ0;
    float2 SampleUV0 = (IJ0 + float2(0.5,0.5)) / ElevMapSize;
    
    float H[2][2];
    for(int i=0; i<2; i++)
        for(int j=0; j<2; j++)
        {
            float2 SampleUV = SampleUV0 + float2( i / ElevMapSize.x, j / ElevMapSize.y );

            // Clamp sample UV coordinates to the decoded area
            SampleUV = clamp( SampleUV, g_ParentPatchUVClampArea.xy, g_ParentPatchUVClampArea.zw );

#   if USE_TEXTURE_ARRAY
            H[i][j] = g_tex2DElevationMapArr.SampleLevel(samPointClamp, float3(SampleUV, g_uiParentPatchArrInd), 0) * HEIGHT_MAP_SAMPLING_SCALE;
#   else
            H[i][j] = g_tex2DParentHeightMap.SampleLevel(samPointClamp, SampleUV, 0) * HEIGHT_MAP_SAMPLING_SCALE;
#   endif
        }

    int PredictedElev = lerp( lerp(H[0][0], H[1][0], BilinearWeights.x),
                              lerp(H[0][1], H[1][1], BilinearWeights.x),
                              BilinearWeights.y );
#endif
    // Load residual
    int iQuantizedResidual = g_tex2DResiduals.Load( int3(In.m_ResidualIJ.xy, 0) );
    
    // Quantize predicted height
    int iQuantizedPredictedElev = QuantizeValue( PredictedElev, g_uiReconstrPrecision );
    
    // Add refinement label and dequantize
    int ReconstructedChildElev = DequantizeValue( iQuantizedPredictedElev + iQuantizedResidual, g_uiReconstrPrecision );

    // Scale to [0,1]. Note that 0.f corresponds to 0u and 1.f corresponds to 65535u
    return (float)ReconstructedChildElev / HEIGHT_MAP_SAMPLING_SCALE;
}

technique11 DecompressHeightMap_FL10
{
    pass
    {
        SetDepthStencilState( DSS_DisableDepthTest, 0 );
        SetRasterizerState( RS_SolidFill_NoCull );
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );

        SetVertexShader( CompileShader( vs_4_0, DecompressOffspringsVS() ) );
        SetHullShader(NULL);
        SetDomainShader(NULL);
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, DecompressOffspringsPS() ) );
    }
}



struct ExtrapolateChildHeightMapVS_OUTPUT
{
    float4 m_ScreenPos_PS : SV_POSITION;
    float2 m_UV : TEXCOORD0;
#if USE_TEXTURE_ARRAY
    uint m_ChildID : InstanceID;
#endif
};

ExtrapolateChildHeightMapVS_OUTPUT ExtrapolateChildHeightMapVS( in uint VertexId : SV_VertexID
#if USE_TEXTURE_ARRAY
                                                              , in uint InstID : SV_InstanceID
#endif
                                                              )
{
    // Render the whole texture
    float4 DstArea = float4(-1,1,1,-1);
    ExtrapolateChildHeightMapVS_OUTPUT Verts[4] = 
    {
#if USE_TEXTURE_ARRAY
        {float4(DstArea.xy, 0.5, 1.0), g_SrcOffspringsAreaUV[InstID].xy, InstID}, 
        {float4(DstArea.xw, 0.5, 1.0), g_SrcOffspringsAreaUV[InstID].xw, InstID},
        {float4(DstArea.zy, 0.5, 1.0), g_SrcOffspringsAreaUV[InstID].zy, InstID},
        {float4(DstArea.zw, 0.5, 1.0), g_SrcOffspringsAreaUV[InstID].zw, InstID}
#else
        {float4(DstArea.xy, 0.5, 1.0), g_SrcOffspringsAreaUV.xy}, 
        {float4(DstArea.xw, 0.5, 1.0), g_SrcOffspringsAreaUV.xw},
        {float4(DstArea.zy, 0.5, 1.0), g_SrcOffspringsAreaUV.zy},
        {float4(DstArea.zw, 0.5, 1.0), g_SrcOffspringsAreaUV.zw}
#endif
    };

    return Verts[VertexId];
}


float ExtrapolateLinear(float fVal0, float fVal1, float fStep)
{            
    //         fVal0 fVal1
    //   *     *     *
    //   |<--->|
    //    fStep
    return fVal0 - fStep * (fVal1-fVal0);
}

#if USE_TEXTURE_ARRAY
struct SSlectRenderTargetGS_Output
{
    ExtrapolateChildHeightMapVS_OUTPUT VSOut;
    uint RenderTargetIndex : SV_RenderTargetArrayIndex;
};

[maxvertexcount(3)]
void SlectRenderTargetGS(triangle ExtrapolateChildHeightMapVS_OUTPUT In[3], inout TriangleStream<SSlectRenderTargetGS_Output> triStream)
{
    for(int i=0; i<3; i++)
    {
        SSlectRenderTargetGS_Output Out;
        Out.VSOut.m_ScreenPos_PS = In[i].m_ScreenPos_PS;
        Out.VSOut.m_UV = In[i].m_UV;
        Out.VSOut.m_ChildID = In[0].m_ChildID;
        Out.RenderTargetIndex = g_uiChildPatchArrInd[ In[0].m_ChildID ];
        triStream.Append( Out );
    }
}
#endif



float ExtrapolateChildHeightMapPS(ExtrapolateChildHeightMapVS_OUTPUT In) : SV_TARGET
{
    // Clamp texture coordinate to the decoded region
    float2 Sample1UV = clamp(In.m_UV, g_DecodedOffspringsAreaUV.xy, g_DecodedOffspringsAreaUV.zw);
    float2 Sample2UV = Sample1UV;
    float2 Step = 0;
    
    float2 OffspringsHMTexelSize;
    g_tex2DOffspringsHM.GetDimensions(OffspringsHMTexelSize.x, OffspringsHMTexelSize.y);
    OffspringsHMTexelSize = 1.f/OffspringsHMTexelSize;

    // Extrapolate smaples outside the decoded region similiar to the way it is done on the CPU
    if( g_DecodedOffspringsAreaUV.y <= In.m_UV.y && In.m_UV.y <= g_DecodedOffspringsAreaUV.w )
    {
        if( In.m_UV.x < g_DecodedOffspringsAreaUV.x )
        {
            Sample2UV.x += OffspringsHMTexelSize.x;
            Step.x = (g_DecodedOffspringsAreaUV.x - In.m_UV.x) / OffspringsHMTexelSize.x;
        }
        else if( In.m_UV.x > g_DecodedOffspringsAreaUV.z )
        {
            Sample2UV.x -= OffspringsHMTexelSize.x;
            Step.x = (In.m_UV.x - g_DecodedOffspringsAreaUV.z) / OffspringsHMTexelSize.x;
        }
    }
    if( g_DecodedOffspringsAreaUV.x <= In.m_UV.x && In.m_UV.x <= g_DecodedOffspringsAreaUV.z )
    {
        if( In.m_UV.y < g_DecodedOffspringsAreaUV.y )
        {
            Sample2UV.y += OffspringsHMTexelSize.y;
            Step.y = (g_DecodedOffspringsAreaUV.y - In.m_UV.y) / OffspringsHMTexelSize.y;
        }
        else if( In.m_UV.y > g_DecodedOffspringsAreaUV.w )
        {
            Sample2UV.y -= OffspringsHMTexelSize.y;
            Step.y = (In.m_UV.y - g_DecodedOffspringsAreaUV.w) / OffspringsHMTexelSize.y;
        }
    }

    float Height1 = g_tex2DOffspringsHM.SampleLevel( samPointClamp, Sample1UV, 0 );
    float Height2 = g_tex2DOffspringsHM.SampleLevel( samPointClamp, Sample2UV, 0 );
    
    return ExtrapolateLinear(Height1, Height2, length(Step) );
}

technique11 ExtrapolateChildHeightMap_FL10
{
    pass
    {
        SetDepthStencilState( DSS_DisableDepthTest, 0 );
        SetRasterizerState( RS_SolidFill_NoCull );
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );

        SetVertexShader( CompileShader( vs_4_0, ExtrapolateChildHeightMapVS() ) );
        SetHullShader(NULL);
        SetDomainShader(NULL);
#if USE_TEXTURE_ARRAY
        SetGeometryShader( CompileShader( gs_4_0, SlectRenderTargetGS() ) );
#else
        SetGeometryShader( NULL );
#endif
        SetPixelShader( CompileShader( ps_4_0, ExtrapolateChildHeightMapPS() ) );
    }
}
