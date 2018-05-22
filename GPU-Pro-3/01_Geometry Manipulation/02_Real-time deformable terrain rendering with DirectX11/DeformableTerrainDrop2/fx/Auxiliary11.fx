#include "Common.fxh"

#ifndef NUM_MATERIALS
#   define NUM_MATERIALS 12
#endif

cbuffer cbImmutable
{
    float4 g_BoundaryExtensions = float4(2,2,2,2);
    float g_fElevationScale = 1;
    float4 g_GlobalMinMaxElevation;

    float4 g_HeightRanges[NUM_MATERIALS];
    float4 g_SlopeRanges[NUM_MATERIALS];
}

cbuffer cbPatchParams
{
    float g_fPatchSampleSpacingInterval;
    int g_iTessBlockSize;
}

#ifndef USE_TEXTURE_ARRAY
#   define USE_TEXTURE_ARRAY 1
#endif

#if USE_TEXTURE_ARRAY
    Texture2DArray<float> g_tex2DElevationMapArr;
    Texture2DArray<float2> g_tex2DNormalMapArr; // Normal map stores only x,y components. z component is calculated as sqrt(1 - x^2 - y^2)
    Texture2DArray<float> g_tex2DMaterialIdxMapArr;
    Texture2DArray<float> g_tex2DTessBlockErrorBoundsArr;

#ifndef MAX_TEXTURES_TO_UPDATE
#   define MAX_TEXTURES_TO_UPDATE 128
#endif
    cbuffer cbTexArrIndices
    {
        uint g_uiTexToUpdateIndices[MAX_TEXTURES_TO_UPDATE];// This array contains indices of textures to update during one draw call
    }
    cbuffer cbPatchSampleSpacingIntervals
    {
        float g_PatchSampleSpacingIntervalsArr[MAX_TEXTURES_TO_UPDATE];
    }
#else
    Texture2D<float> g_tex2DElevationMap;
    Texture2D<float2> g_tex2DNormalMap; // Normal map stores only x,y components. z component is calculated as sqrt(1 - x^2 - y^2)
    Texture2D<float> g_tex2DMaterialIdxMap;
    Texture2D<float> g_tex2DTessBlockErrorBounds;
#endif

Texture2D g_tex2DSlopeNoise;

SamplerState samLinearWrap
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = WRAP;
    AddressV = WRAP;
};

struct GenerateQuadVS_OUTPUT
{
    float4 m_ScreenPos_PS : SV_POSITION;
    float2 m_ElevationMapUV : TEXCOORD0;
    uint m_uiInstID : INST_ID; // Unused if texture arrays are not used
};

GenerateQuadVS_OUTPUT GenerateQuadVS( in uint VertexId : SV_VertexID, 
                                      in uint InstID : SV_InstanceID)
{
    float4 DstTextureMinMaxUV = float4(-1,1,1,-1);
    float4 SrcElevAreaMinMaxUV = float4(0,0,1,1);
    
    GenerateQuadVS_OUTPUT Verts[4] = 
    {
        {float4(DstTextureMinMaxUV.xy, 0.5, 1.0), SrcElevAreaMinMaxUV.xy, InstID}, 
        {float4(DstTextureMinMaxUV.xw, 0.5, 1.0), SrcElevAreaMinMaxUV.xw, InstID},
        {float4(DstTextureMinMaxUV.zy, 0.5, 1.0), SrcElevAreaMinMaxUV.zy, InstID},
        {float4(DstTextureMinMaxUV.zw, 0.5, 1.0), SrcElevAreaMinMaxUV.zw, InstID}
    };

    return Verts[VertexId];
}


GenerateQuadVS_OUTPUT CalculateTessBlockErrorsVS( in uint VertexId : SV_VertexID, 
                                                  in uint InstID : SV_InstanceID)
{
    float2 SrcElevDataTextureSize;
#if USE_TEXTURE_ARRAY
    float Elems;
    g_tex2DElevationMapArr.GetDimensions(SrcElevDataTextureSize.x, SrcElevDataTextureSize.y, Elems);
#else
    g_tex2DElevationMap.GetDimensions(SrcElevDataTextureSize.x, SrcElevDataTextureSize.y);
#endif

    float4 DstTextureMinMaxUV = float4(-1,1,1,-1);
    float4 SrcElevAreaMinMaxUV = float4(0,0,1,1);
    
    // Tessellation blocks do not cover boundary extensions, thus
    // it is neccessary to narrow the source height map UV range:
    SrcElevAreaMinMaxUV.xy += g_BoundaryExtensions.xy / SrcElevDataTextureSize;
    SrcElevAreaMinMaxUV.zw -= g_BoundaryExtensions.zw / SrcElevDataTextureSize;
    // During rasterization, height map UV will be interpolated to the center of each
    // tessellation block (x). However, we need the center of the height map texel (_):
    //    ___ ___ ___ ___     
    //   |   |   |   |   |
    //   |___|___|___|___|
    //   |   |   | _ |   |
    //   |___|___x___|___|
    //   |   |   |   |   |
    //   |___|___|___|___|
    //   |   |   |   |   |
    //   |___|___|___|___|
    // Thus it is necessary to add the following offset:
    SrcElevAreaMinMaxUV.xyzw += 0.5 / SrcElevDataTextureSize.xyxy;

    GenerateQuadVS_OUTPUT Verts[4] = 
    {
        {float4(DstTextureMinMaxUV.xy, 0.5, 1.0), SrcElevAreaMinMaxUV.xy, InstID}, 
        {float4(DstTextureMinMaxUV.xw, 0.5, 1.0), SrcElevAreaMinMaxUV.xw, InstID},
        {float4(DstTextureMinMaxUV.zy, 0.5, 1.0), SrcElevAreaMinMaxUV.zy, InstID},
        {float4(DstTextureMinMaxUV.zw, 0.5, 1.0), SrcElevAreaMinMaxUV.zw, InstID}
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
        Out.RenderTargetIndex = g_uiTexToUpdateIndices[ InstID ];
        Out.m_fTexArrInd = (float)g_uiTexToUpdateIndices[ InstID ];
        triStream.Append( Out );
    }
}

[maxvertexcount(3)]
void RenderCoarseNormalMapMIP_GS(triangle GenerateQuadVS_OUTPUT In[3], 
                                 inout TriangleStream<SPassThroughGS_Output> triStream )
{
    uint InstID = In[0].m_uiInstID;
    for(int i=0; i<3; i++)
    {
        SPassThroughGS_Output Out;
        Out.VSOutput = In[i];
        Out.RenderTargetIndex = InstID; // When coarser normal map MIPs are generated, we render to the
                                        // temporary texture array being CDX11PatchRender::MAX_TEXTURES_TO_UPDATE in size.
                                        // This array is populated serially: the first normal map is rendered to
                                        // array slice 0, the second one - to array slice 1 and so on.
                                        // These slices are then copied to appropriate normal map texture array element
        // Source height map is read from appropriate texture array element
        Out.m_fTexArrInd = (float)g_uiTexToUpdateIndices[ InstID ];
        triStream.Append( Out );
    }
}
#endif

float3 ComputeNormal(float2 ElevationMapUV,
                     float fPatchSampleSpacingInterval
#if USE_TEXTURE_ARRAY
                     , float fElevDataTexArrayIndex
#endif
                     )
{
#if USE_TEXTURE_ARRAY
    float2 ElevTexelSize;
    float Elems;
    g_tex2DElevationMapArr.GetDimensions( ElevTexelSize.x, ElevTexelSize.y, Elems );
    ElevTexelSize = 1/ElevTexelSize;
    //Here should be SampleLevel(..., Offset), but this causes crash on NVidia!
    #   define GET_ELEV(Offset) g_tex2DElevationMapArr.SampleLevel(samPointClamp, float3(ElevationMapUV.xy + float2(Offset.xy)*ElevTexelSize, fElevDataTexArrayIndex), 0 )
//#   define GET_ELEV(Offset) g_tex2DElevationMapArr.SampleLevel(samPointClamp, float3(ElevationMapUV.xy, g_fElevDataTexArrayIndex), 0, Offset)
#else
#   define GET_ELEV(Offset) g_tex2DElevationMap.SampleLevel(samPointClamp, ElevationMapUV.xy, 0, Offset)
#endif

#if 1
    float Height00 = GET_ELEV( int2( -1, -1) );
    float Height10 = GET_ELEV( int2(  0, -1) );
    float Height20 = GET_ELEV( int2( +1, -1) );

    float Height01 = GET_ELEV( int2( -1, 0) );
    //float Height11 = GET_ELEV( int2(  0, 0) );
    float Height21 = GET_ELEV( int2( +1, 0) );

    float Height02 = GET_ELEV( int2( -1, +1) );
    float Height12 = GET_ELEV( int2(  0, +1) );
    float Height22 = GET_ELEV( int2( +1, +1) );

    float3 Grad;
    Grad.x = (Height00+Height01+Height02) - (Height20+Height21+Height22);
    Grad.y = (Height00+Height10+Height20) - (Height02+Height12+Height22);
    Grad.z = fPatchSampleSpacingInterval * 6.f;
    //Grad.x = (3*Height00+10*Height01+3*Height02) - (3*Height20+10*Height21+3*Height22);
    //Grad.y = (3*Height00+10*Height10+3*Height20) - (3*Height02+10*Height12+3*Height22);
    //Grad.z = fPatchSampleSpacingInterval * 32.f;
#else
    float Height1 = GET_ELEV( int2( 1, 0) );
    float Height2 = GET_ELEV( int2(-1, 0) );
    float Height3 = GET_ELEV( int2( 0, 1) );
    float Height4 = GET_ELEV( int2( 0,-1) );
       
    float3 Grad;
    Grad.x = Height2 - Height1;
    Grad.y = Height4 - Height3;
    Grad.z = fPatchSampleSpacingInterval * 2.f;
#endif
    Grad.xy *= HEIGHT_MAP_SAMPLING_SCALE * g_fElevationScale;
    float3 Normal = normalize( Grad );

    return Normal;
}

float4 CalculateTessBlockErrors(float2 ElevationMapUV
#if USE_TEXTURE_ARRAY
                               , float fElevDataTexArrayIndex
                               , uint InstID
#endif
                                )
{
#define MAX_TESS_BLOCK_ERROR 1.7e+34f
    float TessBlockErrors[4] = {MAX_TESS_BLOCK_ERROR, MAX_TESS_BLOCK_ERROR, MAX_TESS_BLOCK_ERROR, MAX_TESS_BLOCK_ERROR};

    float2 SrcElevDataTextureSize;
#if USE_TEXTURE_ARRAY
    float Elems;
    g_tex2DElevationMapArr.GetDimensions(SrcElevDataTextureSize.x, SrcElevDataTextureSize.y, Elems);
#else
    g_tex2DElevationMap.GetDimensions(SrcElevDataTextureSize.x, SrcElevDataTextureSize.y);
#endif
    float2 ElevDataTexelSize = 1.f / SrcElevDataTextureSize;

#ifdef GET_ELEV
#   undef GET_ELEV
#endif

#if USE_TEXTURE_ARRAY
#   define GET_ELEV(Col, Row) g_tex2DElevationMapArr.Sample(samPointClamp, float3(ElevationMapUV.xy + float2(Col, Row) * ElevDataTexelSize, fElevDataTexArrayIndex))*HEIGHT_MAP_SAMPLING_SCALE
#else
#   define GET_ELEV(Col, Row) g_tex2DElevationMap.SampleLevel(samPointClamp, ElevationMapUV.xy + float2(Col, Row) * ElevDataTexelSize, 0)*HEIGHT_MAP_SAMPLING_SCALE
#endif

    for(int iCoarseLevel = 1; iCoarseLevel <= 4; iCoarseLevel++)
    {
        float fTessBlockCurrLevelError = 0.f;
        int iStep = 1 << iCoarseLevel;
        // Minimum tessellation factor for the tessellation block is 2, which 
        // corresponds to the step g_iTessBlockSize/2. There is no point in 
        // considering larger steps
        //
        //g_iTessBlockSize
        //  |<----->|
        //   _______ 
        //  |\  |  /|
        //  |__\|/__|
        //  |  /|\  |
        //  |/__|__\|
        if( iStep > g_iTessBlockSize/2 )
            break;

        // Tessellation block covers height map samples in the range [-g_iTessBlockSize/2, g_iTessBlockSize/2]
        for(int iRow = -(g_iTessBlockSize>>1); iRow <= (g_iTessBlockSize>>1); iRow++)
            for(int iCol = -(g_iTessBlockSize>>1); iCol <= (g_iTessBlockSize>>1); iCol++)
            {
                int iCol0 = iCol & (-iStep);
                int iRow0 = iRow & (-iStep);
                int iCol1 = iCol0 + iStep;
                int iRow1 = iRow0 + iStep;

                float fHorzWeight = ((float)iCol - (float)iCol0) / (float)iStep;
                float fVertWeight = ((float)iRow - (float)iRow0) / (float)iStep;

                float fElev00 = GET_ELEV(iCol0, iRow0);
                float fElev10 = GET_ELEV(iCol1, iRow0);
                float fElev01 = GET_ELEV(iCol0, iRow1);
                float fElev11 = GET_ELEV(iCol1, iRow1);
                float fInterpolatedElev = lerp( lerp(fElev00, fElev10, fHorzWeight ),
                                                lerp(fElev01, fElev11, fHorzWeight ),
                                                fVertWeight );

                float fCurrElev = GET_ELEV(iCol, iRow);
                float fCurrElevError = abs(fCurrElev - fInterpolatedElev);
                fTessBlockCurrLevelError = max(fTessBlockCurrLevelError, fCurrElevError);
            }

        TessBlockErrors[iCoarseLevel-1] = fTessBlockCurrLevelError;
    }

    // Load tessellation block error with respect to finer levels
    // TessBlockErrorBounds texture[array] has the same dimension as the tess block error texture
    // being rendered. Thus we can use ElevationMapUV to load the data. The only thing is that we 
    // need to remove offset by 0.5*ElevDataTexelSize
#if USE_TEXTURE_ARRAY
    float TessBlockErrorBound = g_tex2DTessBlockErrorBoundsArr.Sample(samPointClamp, float3(ElevationMapUV.xy - 0.5*ElevDataTexelSize, InstID)) * HEIGHT_MAP_SAMPLING_SCALE;
#else
    float TessBlockErrorBound = g_tex2DTessBlockErrorBounds.SampleLevel(samPointClamp, ElevationMapUV.xy  - 0.5*ElevDataTexelSize, 0) * HEIGHT_MAP_SAMPLING_SCALE;
#endif
    return float4(TessBlockErrors[0], TessBlockErrors[1], TessBlockErrors[2], TessBlockErrors[3]) + TessBlockErrorBound;
}

#if USE_TEXTURE_ARRAY

float2 GenerateNormalMapPS(SPassThroughGS_Output In) : SV_TARGET
{
    float3 Normal = ComputeNormal( In.VSOutput.m_ElevationMapUV.xy, g_PatchSampleSpacingIntervalsArr[In.VSOutput.m_uiInstID], (float)In.m_fTexArrInd );
    // Only xy components are stored. z component is calculated in the shader
    return Normal.xy;
}

float4 CalculateTessBlockErrorsPS(SPassThroughGS_Output In) : SV_TARGET
{
    return CalculateTessBlockErrors( In.VSOutput.m_ElevationMapUV.xy, (float)In.m_fTexArrInd, In.VSOutput.m_uiInstID );
}

#else

float2 GenerateNormalMapPS(GenerateQuadVS_OUTPUT In) : SV_TARGET
{
    float3 Normal = ComputeNormal( In.m_ElevationMapUV.xy, g_fPatchSampleSpacingInterval );
    // Only xy components are stored. z component is calculated in the shader
    return Normal.xy;
}
float4 CalculateTessBlockErrorsPS(GenerateQuadVS_OUTPUT In) : SV_TARGET
{
    return CalculateTessBlockErrors( In.m_ElevationMapUV.xy );
}

#endif

float g_fFinerNormalMapMIPLevel;
float2 GenerateCoarseNormalMapMIP(
#if USE_TEXTURE_ARRAY
                                  SPassThroughGS_Output In
#else
                                  GenerateQuadVS_OUTPUT In
#endif
                                  ) : SV_TARGET
{
    float3 Normal;
    // Just filter finer MIP level
#if USE_TEXTURE_ARRAY
    Normal.xy = g_tex2DNormalMapArr.SampleLevel(samLinearClamp, float3(In.VSOutput.m_ElevationMapUV.xy, (float)In.m_fTexArrInd), g_fFinerNormalMapMIPLevel);
#else
    Normal.xy = g_tex2DNormalMap.SampleLevel(samLinearClamp, In.m_ElevationMapUV.xy, g_fFinerNormalMapMIPLevel);
#endif
    Normal.z = sqrt( 1 - dot(Normal.xy, Normal.xy) );
    Normal = normalize(Normal);
    // Only xy components are stored. z component is calculated in the shader
    return Normal.xy;
}

float HatFunc(float fMin, float fMax, float fTrans, float fVal)
{
    float fLeftSide  =  1.f / fTrans * fVal + (1 - fMin/fTrans);
    float fRightSide = -1.f / fTrans * fVal + (1 + fMax/fTrans);
    float fRes = saturate( min( fLeftSide, fRightSide ) );
    return fRes;
}


float CalculateMaterialIdx(
#if USE_TEXTURE_ARRAY
                                  SPassThroughGS_Output In
#else
                                  GenerateQuadVS_OUTPUT In
#endif
                                  ) : SV_TARGET
{
    float3 Normal;
    float Height;
    float2 ElevMapUV;
#if USE_TEXTURE_ARRAY
    ElevMapUV = In.VSOutput.m_ElevationMapUV.xy;
    Normal.xy = g_tex2DNormalMapArr.SampleLevel(samPointClamp, float3(ElevMapUV, (float)In.m_fTexArrInd), 0);
    Height = g_tex2DElevationMapArr.Sample(samPointClamp, float3(ElevMapUV, (float)In.m_fTexArrInd) );
#else
    ElevMapUV = In.m_ElevationMapUV.xy;
    Normal.xy = g_tex2DNormalMap.SampleLevel(samPointClamp, ElevMapUV, 0);
    Height = g_tex2DElevationMap.SampleLevel(samPointClamp, ElevMapUV, 0);
#endif
    Height = (Height * HEIGHT_MAP_SAMPLING_SCALE - g_GlobalMinMaxElevation.x) / (g_GlobalMinMaxElevation.y - g_GlobalMinMaxElevation.x);
    Normal.z = sqrt( 1 - dot(Normal.xy, Normal.xy) );
    float Slope = 1/Normal.z;
    
    //float fSlopeNoise = g_tex2DSlopeNoise.Sample(samLinearWrap, ElevMapUV*2)-0.5;
    //Slope /= 1 + 0.2*fSlopeNoise;
    //Slope = max(Slope,1);

    uint MatID = 0;
    float fMaxCoverage = 0.f;
    

    for(uint iMat = 0; iMat < NUM_MATERIALS; iMat++)
    {
        float fHeightCov = HatFunc( g_HeightRanges[iMat].x, g_HeightRanges[iMat].y, g_HeightRanges[iMat].z, Height);
        float fSlopeCov = HatFunc( g_SlopeRanges[iMat].x, g_SlopeRanges[iMat].y, g_SlopeRanges[iMat].z, Slope);

        float fMatCov = fSlopeCov * fHeightCov;
        fMaxCoverage *= (1 - fMatCov);
        if( fMatCov > fMaxCoverage )
        {
            fMaxCoverage = fMatCov;
            MatID = iMat;
        }
    }

    return (MatID+0.5)/256;
}

float2 CalculateCoarseMaterialIdxMapMIP(
#if USE_TEXTURE_ARRAY
                                  SPassThroughGS_Output In
#else
                                  GenerateQuadVS_OUTPUT In
#endif
                                  ) : SV_TARGET
{
    float MaterialIdx;
    // Just filter finer MIP level
#if USE_TEXTURE_ARRAY
    MaterialIdx = g_tex2DMaterialIdxMapArr.SampleLevel(samPointClamp, float3(In.VSOutput.m_ElevationMapUV.xy, (float)In.m_fTexArrInd), g_fFinerNormalMapMIPLevel);
#else
    MaterialIdx = g_tex2DMaterialIdxMap.SampleLevel(samPointClamp, In.m_ElevationMapUV.xy, g_fFinerNormalMapMIPLevel);
#endif
    return MaterialIdx;
}


technique11 RenderNormalMap_FeatureLevel10
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
        SetPixelShader( CompileShader( ps_4_0, GenerateNormalMapPS() ) );
    }
}

technique11 RenderTessBlockErrors_FL10
{
    pass
    {
        SetDepthStencilState( DSS_DisableDepthTest, 0 );
        SetRasterizerState( RS_SolidFill_NoCull );
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );

        SetVertexShader( CompileShader( vs_4_0, CalculateTessBlockErrorsVS() ) );
#if USE_TEXTURE_ARRAY
        SetGeometryShader( CompileShader( gs_4_0, PassThroughGS() ) );
#else
        SetGeometryShader( NULL );
#endif
        SetPixelShader( CompileShader( ps_4_0, CalculateTessBlockErrorsPS() ) );
    }
}

technique11 GenerateCoarseNormalMapMIP_FL10
{
    pass
    {
        SetDepthStencilState( DSS_DisableDepthTest, 0 );
        SetRasterizerState( RS_SolidFill_NoCull );
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );

        SetVertexShader( CompileShader( vs_4_0, GenerateQuadVS() ) );
#if USE_TEXTURE_ARRAY
        SetGeometryShader( CompileShader( gs_4_0, RenderCoarseNormalMapMIP_GS() ) );
#else
        SetGeometryShader( NULL );
#endif
        SetPixelShader( CompileShader( ps_4_0, GenerateCoarseNormalMapMIP() ) );
    }
}


technique11 GenerateMaterialMap_FL10
{
    pass PGenerateMaterialIndices
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
        SetPixelShader( CompileShader( ps_4_0, CalculateMaterialIdx() ) );
    }

    pass PCalculateCoarseMIPLevel
    {
        SetDepthStencilState( DSS_DisableDepthTest, 0 );
        SetRasterizerState( RS_SolidFill_NoCull );
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );

        SetVertexShader( CompileShader( vs_4_0, GenerateQuadVS() ) );
#if USE_TEXTURE_ARRAY
        SetGeometryShader( CompileShader( gs_4_0, RenderCoarseNormalMapMIP_GS() ) );
#else
        SetGeometryShader( NULL );
#endif
        SetPixelShader( CompileShader( ps_4_0, CalculateCoarseMaterialIdxMapMIP() ) );
    }
}
