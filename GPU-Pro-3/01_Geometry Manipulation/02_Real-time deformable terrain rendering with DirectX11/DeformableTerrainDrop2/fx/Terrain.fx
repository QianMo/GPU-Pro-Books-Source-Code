#include "Common.fxh"

#ifndef PATCH_SIZE
#   define PATCH_SIZE 64
#endif

#ifndef USE_TEXTURE_ARRAY
#   define USE_TEXTURE_ARRAY 1
#endif

#ifndef ENABLE_HEIGHT_MAP_MORPH
#   define ENABLE_HEIGHT_MAP_MORPH 1
#endif

#ifndef ENABLE_NORMAL_MAP_MORPH
#   define ENABLE_NORMAL_MAP_MORPH 1
#endif

#ifndef ENABLE_DIFFUSE_TEX_MORPH
#   define ENABLE_DIFFUSE_TEX_MORPH 1
#endif

// Texturing modes
#define TM_HEIGHT_BASED 0             // Simple height-based texturing mode using 1D look-up table
#define TM_INDEX_MASK 1
#define TM_INDEX_MASK_WITH_NM 2

#ifndef TEXTURING_MODE
#   define TEXTURING_MODE TM_INDEX_MASK
#endif

#define POS_XYZ_SWIZZLE xzy

#ifndef SHADOWS
#   define SHADOWS 1
#endif

#if SHADOWS

#include "hulton.fxh"

#ifndef MAX_CASCADES
#   define MAX_CASCADES 8
#endif

#ifndef NUM_SHADOW_CASCADES
#   define NUM_SHADOW_CASCADES 3
#endif

#ifndef INTERVAL_SELECTION
#   define INTERVAL_SELECTION 0
#endif

#ifndef INTEGER_OFFSET_FILTERING
#   define INTEGER_OFFSET_FILTERING 1
#endif

Texture2D g_txShadow;

#ifndef FILTER_SIZE
#   define FILTER_SIZE 3
#endif

cbuffer ShadowMapImmutable
{
	float4 g_CascadeColors[MAX_CASCADES] = {
		float4(0,1,0,1),
		float4(0,0,1,1),
		float4(1,1,0,1),
		float4(0,1,1,1),
		float4(1,0,1,1),
		float4(0.3, 1, 0.7,1),
		float4(0.7, 0.3,1,1),
		float4(1, 0.7, 0.3, 1),
	};
}

cbuffer ShadowMapParams
{
	matrix g_mWorldToLightProj[MAX_CASCADES];  // Transform from view space to light projection space
	float4 g_cascadeZes[MAX_CASCADES/4];
	float4 g_cascadeScales[MAX_CASCADES];
	float4 g_cascadeBiases[MAX_CASCADES];
	float2 g_shadowTexelSize = float2(1.f/1024.f, 1.f/1024.f);
	bool g_isVisualizeCascades = false;
	float g_Apperture = 3.5; // screen space filter aperture
}

SamplerComparisonState g_samShadowCmp
{
    Filter = COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
	AddressU = CLAMP;//for spot light - BORDER
    AddressV = CLAMP;//for spot light - BORDER
    AddressW = CLAMP;
    ComparisonFunc = LESS;
    BorderColor = float4(1,0,0,0);
};

#endif // SHADOWS

cbuffer cbImmutable
{
    float4 g_GlobalMinMaxElevation;
    float2 g_ViewPortSize = float2(1280,1024);
    bool g_bFullResHWTessellatedTriang = false;
    float g_fElevationScale = 1.f;
}

cbuffer cbFrameParams 
{
    float g_fScrSpaceErrorThreshold = 1.f;
    matrix g_mWorldViewProj;
    float4 g_CameraPos;
};


cbuffer cbLightParams
{
    // WARNING: these parameters are duplicated in AtmEffects11.fx
    float4 g_vDirectionOnSun = {0.f, 0.769666f, 0.638446f, 1.f}; ///< Direction on sun
    float4 g_vSunColorAndIntensityAtGround = {0.640682f, 0.591593f, 0.489432f, 100.f}; ///< Sun color
    float4 g_vAmbientLight = {0.191534f, 0.127689f, 0.25f, 0.f}; ///< Ambient light
}

cbuffer cbPatchParams
{
    float g_PatchXYScale;
    float4 g_PatchLBCornerXY;
    float g_fFlangeWidth = 50.f;
    float g_fMorphCoeff = 0.f;
    int2 g_PatchOrderInSiblQuad;
}


SamplerState samMaterialIdx
{
    Filter = MIN_MAG_MIP_POINT;
    AddressU = Clamp;
    AddressV = Clamp;
    AddressW = Clamp;
    MipLODBias = -1;
};


SamplerState samLinearWrap
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Wrap;
    AddressV = Wrap;
};

SamplerState samAnisotropicWrap
{
    Filter = ANISOTROPIC;
    AddressU = Wrap;
    AddressV = Wrap;
    MaxAnisotropy = 8;
};


#if USE_TEXTURE_ARRAY

    Texture2DArray<float> g_tex2DElevationMapArr;
    Texture2DArray<float2> g_tex2DNormalMapArr; // Normal map stores only x,y components. z component is calculated as sqrt(1 - x^2 - y^2)
    Texture2DArray<float4> g_tex2DTessBlockErrorsArr;
    
    int g_iPatchTexArrayIndex = 0;
    int g_iParentPatchTexArrayIndex = 0;

#   if TEXTURING_MODE >= TM_INDEX_MASK
        Texture2DArray<float> g_tex2DMtrlIdxArr;
#   endif

#else

    Texture2D<float> g_tex2DElevationMap;
    Texture2D<float2> g_tex2DNormalMap; // Normal map stores only x,y components. z component is calculated as sqrt(1 - x^2 - y^2)
    Texture2D<float4> g_tex2DTessBlockErrors;

    Texture2D<float> g_tex2DParentElevMap;
    Texture2D<float2> g_tex2DParentNormalMap; // Normal map stores only x,y components. z component is calculated as sqrt(1 - x^2 - y^2)

#   if TEXTURING_MODE >= TM_INDEX_MASK
        Texture2D<float> g_tex2DMtrlIdx;
        Texture2D<float> g_tex2DParentMtrlIdx;
#   endif

#endif

Texture2D<float3> g_tex2DElevationColor;
Texture2D g_tex2DNoiseMap;

#ifndef ELEV_DATA_EXTENSION
#   define ELEV_DATA_EXTENSION 2
#endif

#if USE_TEXTURE_ARRAY
#   define LOAD_DATA_FROM_TEX_OR_TEX_ARR(Texture, Position) \
        Texture##Arr.Load( int4(Position, iPatchTexArrayIndex, 0) )

#   define SAMPLE_LEVEL_FROM_TEX_OR_TEX_ARR(Texture, Sampler, UV, Level, Offset ) \
        Texture##Arr.SampleLevel( Sampler, float3(UV, fPatchTexArrayIndex), Level, Offset )

#   define SAMPLE_TEX_OR_TEX_ARR(Texture, Sampler, UV ) \
        Texture##Arr.Sample( Sampler, float3(UV, fPatchTexArrayIndex) )

#   define SAMPLE_TEX_OR_TEX_ARR_WITH_OFFSET(Texture, Sampler, UV, Offset ) \
        Texture##Arr.Sample( Sampler, float3(UV, fPatchTexArrayIndex), Offset )

#else
#   define LOAD_DATA_FROM_TEX_OR_TEX_ARR(Texture, Position) \
        Texture.Load( int3(Position, 0) )

#   define SAMPLE_LEVEL_FROM_TEX_OR_TEX_ARR(Texture, Sampler, UV, Level, Offset ) \
        Texture.SampleLevel( Sampler, UV, Level, Offset )

#   define SAMPLE_TEX_OR_TEX_ARR(Texture, Sampler, UV ) \
        Texture.Sample( Sampler, UV )

#   define SAMPLE_TEX_OR_TEX_ARR_WITH_OFFSET(Texture, Sampler, UV, Offset ) \
        Texture.Sample( Sampler, UV, Offset )

#endif

#if TM_INDEX_MASK == TEXTURING_MODE
    Texture2DArray g_tex2DSrfMtrlArr;
    cbuffer cbTilingParams
    {
        float g_fTilingScale = 100.f;
    }
#endif


float3 DomainToWorld(float2 in_DomainUV,
                     float2 ElevDataTexUV,
                     float PatchXYScale,
                     float fZShift,
                     out float3 fNormal
#if USE_TEXTURE_ARRAY
                   , float fPatchTexArrayIndex
#endif
                       )
{
    float3 VertexCoords;
    float fHeight;
    
    fHeight = SAMPLE_LEVEL_FROM_TEX_OR_TEX_ARR( g_tex2DElevationMap, samLinearClamp, ElevDataTexUV.xy, 0, int2(0,0) );
    fHeight *= HEIGHT_MAP_SAMPLING_SCALE * g_fElevationScale;
    fHeight -= fZShift;

    fNormal = float3(0,0,1);

    // Note that 0 corresponds directly to left (bottom) patch boundary, while
    // 1 corresponds directly to right (top) boundary:
    //     
    //      PATCH_SIZE+1 samples
    //     |<----------....-->|
    //     0  1  2           PATCH_SIZE
    //     *  *  *  *  ....   *
    // U:  0  |               1   
    //   (1/PATCH_SIZE)
    VertexCoords.xy = in_DomainUV * (float)PATCH_SIZE * PatchXYScale;
    VertexCoords.z = fHeight;

    return VertexCoords;
}

struct RenderPatchVS_Output
{
    float4 Pos_PS           : SV_Position; // vertex position in projection space

#if ENABLE_HEIGHT_MAP_MORPH
    float4 HeightMapUV  : HEIGHT_MAP_UV;
#else 
    float2 HeightMapUV  : HEIGHT_MAP_UV;
#endif
            
#if ENABLE_NORMAL_MAP_MORPH
    float4 NormalMapUV  : NORMAL_MAP_UV;
#else 
    float2 NormalMapUV  : NORMAL_MAP_UV;
#endif

#if TEXTURING_MODE > TM_HEIGHT_BASED
#    if ENABLE_DIFFUSE_TEX_MORPH
        float4 DiffuseTexUV : DIFFUSE_TEX_UV;
#    else 
        float2 DiffuseTexUV : DIFFUSE_TEX_UV;
#    endif
#endif
    
    float fMorphCoeff : MORPH_COEFF;

#if TEXTURING_MODE > TM_HEIGHT_BASED
    float2 TileTexUV        : TileTextureUV;
#endif

#if SHADOWS
    float4 vPosLight        : LIGHT_SPACE_POSITION;
#endif

#if USE_TEXTURE_ARRAY
    float fPatchTexArrayIndex : PatchTexArrayIndex;
    float fParentPatchTexArrayIndex : ParentPatchTexArrayIndex;
#endif

#if TEXTURING_MODE == TM_INDEX_MASK_WITH_NM
    float3 Tangent : TANGENT;
    float3 BiTangent : BITANGENT;
    float3 Normal : Normal;
#endif

    float fDistToCamera : DISTANCE_TO_CAMERA;
};


#if SHADOWS

float3 GetCascadeIndex(float3 vPosLight, 
                       float cameraZ, 
                       out float depth, 
                       out float3 dTex_dX, 
                       out float3 dTex_dY)
{
	int iCascade=0;
	float3 ppPos = vPosLight;
	depth = ppPos.z;
	dTex_dX = ddx(ppPos);
	dTex_dY = ddy(ppPos);

#if 1!=NUM_SHADOW_CASCADES
	float2 ppPosLS = (ppPos.xy)/g_cascadeScales[0].xy+g_cascadeBiases[0].xy;
	float3 dTex_dX0 = dTex_dX/g_cascadeScales[0].xyz;
	float3 dTex_dY0 = dTex_dY/g_cascadeScales[0].xyz;
#   if INTERVAL_SELECTION
	    [unroll]for(int i=0; i<(NUM_SHADOW_CASCADES+3)/4; ++i)
        {
		    float4 v = float4(g_cascadeZes[i].x<cameraZ, g_cascadeZes[i].y<cameraZ, g_cascadeZes[i].z<cameraZ, g_cascadeZes[i].w<cameraZ);
		    iCascade += dot(float4(1,1,1,1), v);
	    }
	    ppPos.xy = (ppPosLS-g_cascadeBiases[iCascade].xy)*g_cascadeScales[iCascade].xy;
	    dTex_dX = dTex_dX0*g_cascadeScales[iCascade];
	    dTex_dY = dTex_dY0*g_cascadeScales[iCascade];
#   else
	    int iFitsLevel = -2;
	    for(;iCascade<NUM_SHADOW_CASCADES;++iCascade)
	    {
		    ppPos.xy = (ppPosLS-g_cascadeBiases[iCascade].xy)*g_cascadeScales[iCascade].xy;
		    float2 d = 0;

		    dTex_dX = dTex_dX0*g_cascadeScales[iCascade].xyz;
		    dTex_dY = dTex_dY0*g_cascadeScales[iCascade].xyz;

#           if !INTEGER_OFFSET_FILTERING
	            d = float2(g_shadowTexelSize.x, g_shadowTexelSize.y+g_Apperture*length(float2(dTex_dX.y, dTex_dY.y)));
#           else
	            d = g_shadowTexelSize*float2(1,NUM_SHADOW_CASCADES)*(1+FILTER_SIZE);
#           endif

		    if( (abs(ppPos.x)<1 && abs(ppPos.y)<1) )
            {
			    if( iCascade == NUM_SHADOW_CASCADES-1 || (abs(ppPos.x)<1-d.x && abs(ppPos.y)<1-d.y) )
                {
				    break;
			    }
			    iFitsLevel=iCascade;
	        }
	    }
	    iCascade = min(iCascade, NUM_SHADOW_CASCADES);
#   endif //INTERVAL_SELECTION
#endif
	depth = depth/g_cascadeScales[0].z*g_cascadeScales[iCascade].z;
	float3 texShadow = float3(0.5 * ppPos.xy  + float2( 0.5, 0.5 ), iCascade);
	texShadow.y = 1.0f - texShadow.y;    

	dTex_dX.xy*=0.5*float2(1,-1);
	dTex_dY.xy*=0.5*float2(1,-1);

	dTex_dX.y/=NUM_SHADOW_CASCADES;
	dTex_dY.y/=NUM_SHADOW_CASCADES;
	texShadow.y/=NUM_SHADOW_CASCADES;
	texShadow.y += ((float)iCascade)/NUM_SHADOW_CASCADES;

	return texShadow;
}

float2 ComputeReceiverPlaneDepthBias(float3 texCoordDX, float3 texCoordDY)
{    
    float2 biasUV;
    biasUV.x = texCoordDY.y * texCoordDX.z - texCoordDX.y * texCoordDY.z;
    biasUV.y = texCoordDX.x * texCoordDY.z - texCoordDY.x * texCoordDX.z;

    float Det = (texCoordDX.x * texCoordDY.y) - (texCoordDX.y * texCoordDY.x);
	biasUV /= Det;//sign(Det) * max( abs(Det), 1e-6 );
    return biasUV;
}

#if INTEGER_OFFSET_FILTERING

float pcf_IsNotInShadow(float fragDepth, float3 tex, Texture2D texShadowMap, float3 dTex_dX, float3 dTex_dY)
{
	float lit = 0.0f;
	float2 receiverPlaneDepthBias = ComputeReceiverPlaneDepthBias(dTex_dX, dTex_dY);
	float fractionalSamplingError = dot(g_shadowTexelSize.xy, abs(receiverPlaneDepthBias));
	fragDepth -= 1e-6 + fractionalSamplingError;
	float2 receiverPlaneDepthBiasTexel = g_shadowTexelSize.xy*receiverPlaneDepthBias;
    [unroll]for( int i = -FILTER_SIZE; i <= FILTER_SIZE; ++i )
        [unroll]for( int j = -FILTER_SIZE; j <= FILTER_SIZE; ++j )
        {   
			int2 offset= int2(i,j);
            lit += texShadowMap.SampleCmpLevelZero( g_samShadowCmp, tex.xy, fragDepth+dot(offset, receiverPlaneDepthBiasTexel), offset);
        }
	float w = (2*FILTER_SIZE +1)*(2*FILTER_SIZE +1);
    return lit / w;
}

#else //INTEGER_OFFSET_FILTERING

float pcf_IsNotInShadow(float fragDepth, float3 tex, Texture2D texShadowMap, float3 dTex_dX, float3 dTex_dY)
{
	float lit = 0.0f;
	int iCount = (2*FILTER_SIZE +1)*(2*FILTER_SIZE +1);
	float2 receiverPlaneDepthBias = ComputeReceiverPlaneDepthBias(dTex_dX, dTex_dY);
	float fractionalSamplingError = dot(g_shadowTexelSize.xy, abs(receiverPlaneDepthBias));
    //float g_fMaxFractionalSamplingError = 0.001;
    //float g_fMinFractionalSamplingError = 0.0001;
    //fractionalSamplingError = max( min(fractionalSamplingError, g_fMaxFractionalSamplingError), g_fMinFractionalSamplingError );
	fragDepth -= 1e-6 + fractionalSamplingError;
	for(int i=0; i<iCount; ++i){
		float2 delta = Hulton[i];
		float2 vX = float2(dTex_dX.x, dTex_dY.x);
		float2 vY = float2(dTex_dX.y, dTex_dY.y);
		delta = float2(dot(delta,vX), dot(delta,vY))*g_Apperture;
		float fragDepthCurr = fragDepth+dot(delta, receiverPlaneDepthBias);
		lit += texShadowMap.SampleCmpLevelZero( g_samShadowCmp, tex.xy+delta, fragDepthCurr);
	}
	
	float w = (2*FILTER_SIZE +1)*(2*FILTER_SIZE +1);
	return lit/w;
}
#endif// INTEGER_OFFSET_FILTERING

#endif // SHADOWS


#if TEXTURING_MODE >= TM_INDEX_MASK
float3 GetTerrainSurfaceColor(in float2 MaterialIdxUV,
                              in float2 TileTexUV,
#if USE_TEXTURE_ARRAY
                              in Texture2DArray<float> tex2DMtrlIdxArr,
                              in float fPatchTexArrayIndex
#else
                              in Texture2D<float> tex2DMtrlIdx
#endif
                              )
{
    float2 MtrlIdxTexSize;
#   if USE_TEXTURE_ARRAY
        float Elems;
        tex2DMtrlIdxArr.GetDimensions(MtrlIdxTexSize.x, MtrlIdxTexSize.y, Elems);
#   else
        tex2DMtrlIdx.GetDimensions(MtrlIdxTexSize.x, MtrlIdxTexSize.y);
#   endif

    float2 TexUVSclaed = MaterialIdxUV.xy*MtrlIdxTexSize.xy - float2(0.5, 0.5);
    float2 TexIJ = floor(TexUVSclaed);
    float2 BilinearWeights = TexUVSclaed - TexIJ;
    TexIJ = (TexIJ + float2(0.5, 0.5)) / MtrlIdxTexSize.xy;
    
    float2 Noise = g_tex2DNoiseMap.Sample(samLinearWrap, MaterialIdxUV.xy * 50).xy;
    //BilinearWeights *= saturate(Noise*3);
    BilinearWeights = saturate(BilinearWeights + 0.5*(Noise - 0.5));

    const int FilterSize = 2;
    float3 Colors[FilterSize][FilterSize];
    int i0 = (FilterSize-1)/2;
    int j0 = (FilterSize-1)/2;
    [unroll]for(int i=-i0; i<=(FilterSize/2); i++)
        [unroll]for(int j=-j0; j<=(FilterSize/2); j++)
        {
            float MtrIdx = SAMPLE_TEX_OR_TEX_ARR_WITH_OFFSET( tex2DMtrlIdx, samMaterialIdx, TexIJ.xy, int2(i,j) )*255;
            Colors[i0+i][j0+j] = g_tex2DSrfMtrlArr.Sample(samAnisotropicWrap, float3(TileTexUV.xy, MtrIdx) ).rgb;
        }

    float3 SurfaceColor = lerp( lerp(Colors[0][0], Colors[1][0], BilinearWeights.x),
                                lerp(Colors[0][1], Colors[1][1], BilinearWeights.x),
                                BilinearWeights.y );
    return SurfaceColor;
}
#endif


float3 RenderPatchPS(RenderPatchVS_Output In) : SV_Target
{
    float3 SurfaceColor;
    
#if USE_TEXTURE_ARRAY
    float fPatchTexArrayIndex = In.fPatchTexArrayIndex;
#endif
    
#if TEXTURING_MODE >= TM_INDEX_MASK
    SurfaceColor = 
        GetTerrainSurfaceColor(In.DiffuseTexUV.xy, 
                               In.TileTexUV.xy,
#   if USE_TEXTURE_ARRAY
                               g_tex2DMtrlIdxArr,
                               fPatchTexArrayIndex
#   else
                               g_tex2DMtrlIdx
#   endif
                           );
#   if ENABLE_DIFFUSE_TEX_MORPH
        float3 ParentPatchSurfaceColor = 
            GetTerrainSurfaceColor(In.DiffuseTexUV.zw, 
                                   In.TileTexUV.xy,
#       if USE_TEXTURE_ARRAY
                                   g_tex2DMtrlIdxArr,
                                   In.fParentPatchTexArrayIndex
#       else
                                   g_tex2DParentMtrlIdx
#       endif
                                   );
        SurfaceColor = lerp(SurfaceColor, ParentPatchSurfaceColor, In.fMorphCoeff);
#   endif

#else
    // It is more accurate to calculate average elevation in the pixel shader rather than in the vertex shader
    float Elev = SAMPLE_TEX_OR_TEX_ARR( g_tex2DElevationMap, samLinearClamp, In.HeightMapUV.xy );
#   if ENABLE_HEIGHT_MAP_MORPH
#       if USE_TEXTURE_ARRAY
            float ParentElev = g_tex2DElevationMapArr.Sample(samLinearClamp, float3(In.HeightMapUV.zw, In.fParentPatchTexArrayIndex) );
#       else
            float ParentElev = g_tex2DParentElevMap.Sample(samLinearClamp, In.HeightMapUV.zw );
#       endif
        Elev = lerp(Elev, ParentElev, In.fMorphCoeff);
#   endif
    float NormalizedElev = (Elev * HEIGHT_MAP_SAMPLING_SCALE - g_GlobalMinMaxElevation.x) / (g_GlobalMinMaxElevation.y - g_GlobalMinMaxElevation.x);
    SurfaceColor.rgb = g_tex2DElevationColor.Sample( samLinearClamp, float2(NormalizedElev, 0.5) );
#endif

    float3 Normal; 
    Normal.xy = SAMPLE_TEX_OR_TEX_ARR( g_tex2DNormalMap, samLinearClamp, In.NormalMapUV.xy);

#if ENABLE_NORMAL_MAP_MORPH
#   if USE_TEXTURE_ARRAY
        float2 ParentNormalXY = g_tex2DNormalMapArr.Sample( samLinearClamp, float3(In.NormalMapUV.zw, In.fParentPatchTexArrayIndex));
#   else
        float2 ParentNormalXY = g_tex2DParentNormalMap.Sample(samLinearClamp, In.NormalMapUV.zw);
#   endif
    Normal.xy = lerp(Normal.xy, ParentNormalXY.xy, In.fMorphCoeff);
#endif
    Normal.z = sqrt(1 - dot(Normal.xy,Normal.xy));

#if TEXTURING_MODE == TM_INDEX_MASK_WITH_NM
    Normal = In.Tangent * TangentSpaceNormal.x + In.BiTangent * TangentSpaceNormal.y + Normal * TangentSpaceNormal.z;
#endif

    Normal = Normal.POS_XYZ_SWIZZLE;

    Normal = normalize( Normal );

    float DiffuseIllumination = max(0, dot(Normal.xyz, g_vDirectionOnSun.xyz));

    float3 lightColor = g_vSunColorAndIntensityAtGround.rgb;

#if SHADOWS
	float3 dTex_dX, dTex_dY;
	float depth;
	float3 texShadow = GetCascadeIndex(In.vPosLight.xyz/In.vPosLight.w, In.Pos_PS.z, depth, dTex_dX, dTex_dY);
    //texShadow.z=0;
    //return texShadow;
	//return In.vPosLight.xyz/In.vPosLight.w;
	//return depth;
	float3 CascadeColor=(g_isVisualizeCascades)?g_CascadeColors[texShadow.z].xyz:float3(1,1,1);
	float inShadow = pcf_IsNotInShadow(depth, texShadow, g_txShadow, dTex_dX, dTex_dY);
    //return inShadow;
    DiffuseIllumination *= max(inShadow, 0.1);
	lightColor *= CascadeColor;
    //return g_CascadeColors[texShadow.z].xyz;
#endif
    SurfaceColor.rgb *= (DiffuseIllumination*lightColor + g_vAmbientLight.rgb);
    SurfaceColor.rgb = ApplyFog(SurfaceColor.rgb, In.fDistToCamera);

    return SurfaceColor;
}

float3 RenderWireframePatchPS(RenderPatchVS_Output In) : SV_Target
{
    return ApplyFog(float3(0,0,0), In.fDistToCamera);
}







// It is incredible, but WITH tessellated skirts, the terrain is rendered almost 10% FASTER!!!
#ifndef RENDER_TESSELLATED_SKIRTS
#   define RENDER_TESSELLATED_SKIRTS 1
#endif

struct RENDER_TESSELLATED_PATCH_VS_OUTPUT
{
    uint uiPatchID : PATCH_ID;
    float PatchXYScale : PATCH_XY_SCALE;
    float4 PatchLBCornerXY : PATCH_LB_CORNER_XY;
    float fPatchFlangeWidth : PATCH_FLANGE_WIDTH;
    float fMorphCoeff : MORPH_COEFF;
    int2 PatchOrderInSiblQuad : PATCH_ORDER_IN_SIBL_QUAD;
#if USE_TEXTURE_ARRAY
    int iPatchTexArrayIndex : PATCH_TEX_ARRAY_INDEX_INT;
    int iParentPatchTexArrayIndex : PARENT_PATCH_TEX_ARRAY_INDEX_INT;
#endif
};

RENDER_TESSELLATED_PATCH_VS_OUTPUT RenderTessellatedPatchVS(uint PatchID : SV_VertexID)
{
    RENDER_TESSELLATED_PATCH_VS_OUTPUT Out;

    Out.uiPatchID = PatchID;
    Out.PatchXYScale    = g_PatchXYScale;
    Out.PatchLBCornerXY = g_PatchLBCornerXY;
    Out.fPatchFlangeWidth = g_fFlangeWidth;
    Out.fMorphCoeff = g_fMorphCoeff;
    Out.PatchOrderInSiblQuad = g_PatchOrderInSiblQuad;
#if USE_TEXTURE_ARRAY
    Out.iPatchTexArrayIndex  = g_iPatchTexArrayIndex;
    Out.iParentPatchTexArrayIndex = g_iParentPatchTexArrayIndex;
#endif
    return Out;
};

#if USE_TEXTURE_ARRAY
RENDER_TESSELLATED_PATCH_VS_OUTPUT RenderTessellatedPatchVS_Instanced(uint PatchID : SV_VertexID,
                                                                      float PatchXYScale : PATCH_XY_SCALE,
                                                                      float4 PatchLBCornerXY : PATCH_LB_CORNER_XY,
                                                                      float fPatchFlangeWidth : PATCH_FLANGE_WIDTH,
                                                                      int   iPatchTexArrayIndex : PATCH_TEX_ARRAY_INDEX,
                                                                      int   iParentPatchTexArrayIndex : PARENT_PATCH_TEX_ARRAY_INDEX,
                                                                      float fMorphCoeff :     PATCH_MORPH_COEFF,
                                                                      int2 PatchOrderInSiblQuad : PATCH_ORDER_IN_SIBL_QUAD)
{
    RENDER_TESSELLATED_PATCH_VS_OUTPUT Out;
    Out.uiPatchID = PatchID;
    Out.PatchXYScale    = PatchXYScale;
    Out.PatchLBCornerXY = PatchLBCornerXY;
    Out.fPatchFlangeWidth = fPatchFlangeWidth;
    Out.iPatchTexArrayIndex  = iPatchTexArrayIndex;
    Out.iParentPatchTexArrayIndex  = iParentPatchTexArrayIndex;
    Out.fMorphCoeff = fMorphCoeff;
    Out.PatchOrderInSiblQuad = PatchOrderInSiblQuad;
    return Out;
};
#endif

struct HS_CONTROL_POINT_OUTPUT
{
	uint uiPatchID : PATCH_ID;
    float PatchXYScale : PATCH_XY_SCALE;
    float2 PatchLBCornerXY : PATCH_LB_CORNER_XY;
    float fPatchFlangeWidth : PATCH_FLANGE_WIDTH;
    int2 PatchOrderInSiblQuad : PATCH_ORDER_IN_SIBL_QUAD;
    float fMorphCoeff   : MORPH_COEFF;
#if USE_TEXTURE_ARRAY
    float fPatchTexArrayIndex : TEX_ARRAY_IND;
    float fParentPatchTexArrayIndex : PARENT_TEX_ARRAY_IND;
#endif
};

//--------------------------------------------------------------------------------------
// Hull shader
//--------------------------------------------------------------------------------------
#ifndef BLOCK_SIZE
#   define BLOCK_SIZE 16
#endif

#define NUM_BLOCKS_ALONG_PATCH_EDGE (PATCH_SIZE/BLOCK_SIZE)
#define MIN_EDGE_TESS_FACTOR 2

float GetScrSpaceError(float EdgeError, float3 EdgeCenterWorldSpacePos, float3 Normal)
{
    float3 EdgeCenterShiftedPoint1_WS = EdgeCenterWorldSpacePos - Normal * EdgeError/2.f;
    float3 EdgeCenterShiftedPoint2_WS = EdgeCenterWorldSpacePos + Normal * EdgeError/2.f;
    float4 EdgeCenterShiftedPoint1_PS = mul( float4(EdgeCenterShiftedPoint1_WS.POS_XYZ_SWIZZLE, 1), g_mWorldViewProj );
    float4 EdgeCenterShiftedPoint2_PS = mul( float4(EdgeCenterShiftedPoint2_WS.POS_XYZ_SWIZZLE, 1), g_mWorldViewProj );
    EdgeCenterShiftedPoint1_PS /= EdgeCenterShiftedPoint1_PS.w;
    EdgeCenterShiftedPoint2_PS /= EdgeCenterShiftedPoint2_PS.w;
    
    EdgeCenterShiftedPoint1_PS.xy *= g_ViewPortSize/2;
    EdgeCenterShiftedPoint2_PS.xy *= g_ViewPortSize/2;

    float ScrSpaceError = length(EdgeCenterShiftedPoint1_PS.xy - EdgeCenterShiftedPoint2_PS.xy);
    
    return ScrSpaceError;
}

float CalculateEdgeTessFactor(float4 Errors, 
                              float3 EdgeCenterWorldSpacePos, 
                              float3 Normal)
{
    if( g_bFullResHWTessellatedTriang )
        return BLOCK_SIZE;

    float EdgeTessFactor = BLOCK_SIZE;
#define MAX_ERR         3.4e+38f
    float4 ScrSpaceErrors = float4(MAX_ERR, MAX_ERR, MAX_ERR, MAX_ERR);
#if (BLOCK_SIZE >= 4)
    // We can simplify an edge by a factor of 2
    ScrSpaceErrors.x = GetScrSpaceError(Errors.x, EdgeCenterWorldSpacePos.xyz, Normal);
#endif
#if (BLOCK_SIZE >= 8)
    // We can simplify an edge by a factor of 4
    ScrSpaceErrors.y = GetScrSpaceError(Errors.y, EdgeCenterWorldSpacePos.xyz, Normal);
#endif
#if (BLOCK_SIZE >= 16)
    // We can simplify an edge by a factor of 8
    ScrSpaceErrors.z = GetScrSpaceError(Errors.z, EdgeCenterWorldSpacePos.xyz, Normal);
#endif
#if (BLOCK_SIZE >= 32)
    // We can simplify an edge by a factor of 16
    ScrSpaceErrors.w = GetScrSpaceError(Errors.w, EdgeCenterWorldSpacePos.xyz, Normal);
#endif
    // Compare screen space errors with the threshold
    float4 Cmp = (ScrSpaceErrors.xyzw < g_fScrSpaceErrorThreshold.xxxx);
    // Calculate number of errors less than the threshold
    float SimplPower = dot( Cmp, float4(1,1,1,1) );
    // Compute simplification factor
    float SimplFactor = exp2( SimplPower );
    // Calculate edge tessellation factor
    EdgeTessFactor /= SimplFactor;
//#if (BLOCK_SIZE >= 4)
//    if( ScrSpaceErrors.x < g_fScrSpaceErrorThreshold )
//    {
//        EdgeTessFactor/=2; // Simplify by a factor of 2
//#       if (BLOCK_SIZE >= 8)
//        if( ScrSpaceErrors.y < g_fScrSpaceErrorThreshold )
//        {
//            EdgeTessFactor/=2; // Simplify by a factor of 4
//#           if (BLOCK_SIZE >= 16)
//            if( ScrSpaceErrors.z < g_fScrSpaceErrorThreshold )
//            {
//                EdgeTessFactor/=2; // Simplify by a factor of 8
//#               if (BLOCK_SIZE >= 32)
//                if( ScrSpaceErrors.w < g_fScrSpaceErrorThreshold )
//                {
//                    EdgeTessFactor/=2; // Simplify by a factor of 16
//                }
//#               endif
//            }
//#           endif
//        }
//#       endif
//    }
//#endif
    return max(EdgeTessFactor, MIN_EDGE_TESS_FACTOR);
}


// Output patch constant data.
struct HS_CONSTANT_DATA_OUTPUT
{
    float Edges[4]        : SV_TessFactor;
    float Inside[2]       : SV_InsideTessFactor;
};

HS_CONSTANT_DATA_OUTPUT ConstantHS( InputPatch<RENDER_TESSELLATED_PATCH_VS_OUTPUT, 1> p/*, uint PatchID : SV_PrimitiveID*/ )
{
#if USE_TEXTURE_ARRAY
    int iPatchTexArrayIndex = p[0].iPatchTexArrayIndex;
    float fPatchTexArrayIndex = (float)iPatchTexArrayIndex;
#endif
    float PatchXYScale = p[0].PatchXYScale;
    float2 PatchLBCornerXY = p[0].PatchLBCornerXY.xy;

    HS_CONSTANT_DATA_OUTPUT output = (HS_CONSTANT_DATA_OUTPUT)0;
    float4 vEdgeTessellationFactors;
        
#if RENDER_TESSELLATED_SKIRTS
    // Additional tessellation blocks around patch interior are used to form skirt
    //  
    //   |     ||     |     |
    //   |-1,1 || 0,1 | 1,1 |
    //   |_____||_____|_____|....
    //   |     ||     |     |
    //   |-1,0 || 0,0 | 1,0 |
    //   |_____||_____|_____|....
    //    =====  ===== ===== 
    //   |     ||     |     |
    //   |-1,-1|| 0,-1| 1,-1|
    //   |_____||_____|_____|....
    int iBlockHorzOrder = (p[0].uiPatchID % (NUM_BLOCKS_ALONG_PATCH_EDGE+2)) - 1;
    int iBlockVertOrder = (p[0].uiPatchID / (NUM_BLOCKS_ALONG_PATCH_EDGE+2)) - 1;
    // WARNING!!!!   p[0].uiPatchID is not the same as PatchID !!!!!
#else
    int iBlockHorzOrder = p[0].uiPatchID % NUM_BLOCKS_ALONG_PATCH_EDGE;
    int iBlockVertOrder = p[0].uiPatchID / NUM_BLOCKS_ALONG_PATCH_EDGE;
    // WARNING!!!!   p[0].uiPatchID is not the same as PatchID !!!!!
#endif

    // Load tessellation block errors. Errors for the additional flange blocks are zeroes
    float4 TessBlockErrors            = LOAD_DATA_FROM_TEX_OR_TEX_ARR( g_tex2DTessBlockErrors, int2(iBlockHorzOrder,   iBlockVertOrder)   ) * g_fElevationScale;
    float4 LeftNeighbTessBlockErrors  = LOAD_DATA_FROM_TEX_OR_TEX_ARR( g_tex2DTessBlockErrors, int2(iBlockHorzOrder-1, iBlockVertOrder)   ) * g_fElevationScale;
    float4 RightNeighbTessBlockErrors = LOAD_DATA_FROM_TEX_OR_TEX_ARR( g_tex2DTessBlockErrors, int2(iBlockHorzOrder+1, iBlockVertOrder)   ) * g_fElevationScale;
    float4 BottomNeighbTessBlockErrors= LOAD_DATA_FROM_TEX_OR_TEX_ARR( g_tex2DTessBlockErrors, int2(iBlockHorzOrder,   iBlockVertOrder-1) ) * g_fElevationScale;
    float4 TopNeighbTessBlockErrors   = LOAD_DATA_FROM_TEX_OR_TEX_ARR( g_tex2DTessBlockErrors, int2(iBlockHorzOrder,   iBlockVertOrder+1) ) * g_fElevationScale;


    // Compute UV coordinates range for the current tessellation block
    // Note that 0 corresponds directly to left (bottom) boundary, while
    // 1 corresponds directly to right (top) patch boundary:
    //     
    //      PATCH_SIZE+1 samples
    //     |<----------....-->|
    //     0  1  2           PATCH_SIZE
    //     *  *  *  *  ....   *
    // U:  0  |               1   
    //   (1/PATCH_SIZE)
    float4 QuadUVRange = float4(iBlockHorzOrder, iBlockVertOrder, iBlockHorzOrder+1, iBlockVertOrder+1) * ((float)BLOCK_SIZE/(float)PATCH_SIZE);
#if RENDER_TESSELLATED_SKIRTS
    // We need to clamp the range to fall into the patch interior:
    //     ______                    0 ______       0 ______ 
    //    |      |        |           |      |         
    //    |      |   =>   |           |      |   =>    
    //    |______|        |         -A|______|         
    //   -A      0        0
    // A = (float)BLOCK_SIZE/(float)PATCH_SIZE
    QuadUVRange = clamp(QuadUVRange, 0, 1);
#endif

    float2 ElevDataTexSize;
#if USE_TEXTURE_ARRAY
    float Elems;
    g_tex2DElevationMapArr.GetDimensions( ElevDataTexSize.x, ElevDataTexSize.y, Elems );
#else
    g_tex2DElevationMap.GetDimensions( ElevDataTexSize.x, ElevDataTexSize.y );
#endif
    float2 ElevDataTexelSize = 1.f / ElevDataTexSize;
    
    // Get elevation data texture UV coordinates range for this tessellation block

    //     
    //                      PATCH_SIZE+1 samples
    //                     |<----------....-->|
    //              -2 -1  0  1  2           PATCH_SIZE
    //               *  *  *  *  *  *  ....   *
    //                     |                  |
    //QuadUVRange          0                  1
    //ElevDataTexUVRange   X                  Y
    //                     |<---------------->|
    //                 PATCH_SIZE / ElevDataTexSize.xy
    // 
    // X = ((float)ELEV_DATA_EXTENSION + 0.5) * ElevDataTexelSize.xy
    // Y = ((float)ELEV_DATA_EXTENSION + PATCH_SIZE + 0.5) * ElevDataTexelSize.xy

    float4 ElevDataTexUVRange = QuadUVRange * PATCH_SIZE / ElevDataTexSize.xyxy;
    ElevDataTexUVRange += ((float)ELEV_DATA_EXTENSION + 0.5) * ElevDataTexelSize.xyxy;

#if RENDER_TESSELLATED_SKIRTS
//  QuadUVRange has already been clamped, so we do not need to clamp ElevDataTexUVRange as well
//    ElevDataTexUVRange = clamp(ElevDataTexUVRange, 
//                               ((float)ELEV_DATA_EXTENSION + 0.5) * ElevDataTexelSize.xyxy, 
//                               ((float)ELEV_DATA_EXTENSION + PATCH_SIZE + 0.5) * ElevDataTexelSize.xyxy);
#endif
    
    // Assign tessellation levels
    float3 LeftEdgeCenterPos, RightEdgeCenterPos, BottomEdgeCenterPos, TopEdgeCenterPos;

    float3 LeftNeighbNormal = float3(0,0,1);
    float3 RightNeighbNormal = float3(0,0,1);
    float3 BottomNeighbNormal = float3(0,0,1);
    float3 TopNeighbNormal = float3(0,0,1);

#if USE_TEXTURE_ARRAY
    LeftEdgeCenterPos  = DomainToWorld(float2(QuadUVRange.x, (QuadUVRange.y+QuadUVRange.w)/2), float2(ElevDataTexUVRange.x, (ElevDataTexUVRange.y+ElevDataTexUVRange.w)/2), PatchXYScale, 0.f, LeftNeighbNormal,   fPatchTexArrayIndex);
    RightEdgeCenterPos = DomainToWorld(float2(QuadUVRange.z, (QuadUVRange.y+QuadUVRange.w)/2), float2(ElevDataTexUVRange.z, (ElevDataTexUVRange.y+ElevDataTexUVRange.w)/2), PatchXYScale, 0.f, RightNeighbNormal,  fPatchTexArrayIndex);
    BottomEdgeCenterPos= DomainToWorld(float2((QuadUVRange.x+QuadUVRange.z)/2, QuadUVRange.y), float2((ElevDataTexUVRange.x+ElevDataTexUVRange.z)/2, ElevDataTexUVRange.y), PatchXYScale, 0.f, BottomNeighbNormal, fPatchTexArrayIndex);
    TopEdgeCenterPos   = DomainToWorld(float2((QuadUVRange.x+QuadUVRange.z)/2, QuadUVRange.w), float2((ElevDataTexUVRange.x+ElevDataTexUVRange.z)/2, ElevDataTexUVRange.w), PatchXYScale, 0.f, TopNeighbNormal,    fPatchTexArrayIndex);
#else
    LeftEdgeCenterPos  = DomainToWorld(float2(QuadUVRange.x, (QuadUVRange.y+QuadUVRange.w)/2), float2(ElevDataTexUVRange.x, (ElevDataTexUVRange.y+ElevDataTexUVRange.w)/2), PatchXYScale, 0.f, LeftNeighbNormal);
    RightEdgeCenterPos = DomainToWorld(float2(QuadUVRange.z, (QuadUVRange.y+QuadUVRange.w)/2), float2(ElevDataTexUVRange.z, (ElevDataTexUVRange.y+ElevDataTexUVRange.w)/2), PatchXYScale, 0.f, RightNeighbNormal);
    BottomEdgeCenterPos= DomainToWorld(float2((QuadUVRange.x+QuadUVRange.z)/2, QuadUVRange.y), float2((ElevDataTexUVRange.x+ElevDataTexUVRange.z)/2, ElevDataTexUVRange.y), PatchXYScale, 0.f, BottomNeighbNormal);
    TopEdgeCenterPos   = DomainToWorld(float2((QuadUVRange.x+QuadUVRange.z)/2, QuadUVRange.w), float2((ElevDataTexUVRange.x+ElevDataTexUVRange.z)/2, ElevDataTexUVRange.w), PatchXYScale, 0.f, TopNeighbNormal);
#endif

    LeftEdgeCenterPos.xy  += PatchLBCornerXY;
    RightEdgeCenterPos.xy += PatchLBCornerXY;
    BottomEdgeCenterPos.xy+= PatchLBCornerXY;
    TopEdgeCenterPos.xy   += PatchLBCornerXY;

    output.Edges[0] = CalculateEdgeTessFactor( max(TessBlockErrors, LeftNeighbTessBlockErrors),   LeftEdgeCenterPos,   LeftNeighbNormal );
    output.Edges[1] = CalculateEdgeTessFactor( max(TessBlockErrors, BottomNeighbTessBlockErrors), BottomEdgeCenterPos, BottomNeighbNormal );
    output.Edges[2] = CalculateEdgeTessFactor( max(TessBlockErrors, RightNeighbTessBlockErrors),  RightEdgeCenterPos,  RightNeighbNormal );
    output.Edges[3] = CalculateEdgeTessFactor( max(TessBlockErrors, TopNeighbTessBlockErrors),    TopEdgeCenterPos,    TopNeighbNormal  );
    // Calculate interior tessellation factors as the minimum af edge tess factors:
    output.Inside[1] = 
    output.Inside[0] = min(output.Edges[0], min(output.Edges[2], min(output.Edges[1], output.Edges[3])));

#if RENDER_TESSELLATED_SKIRTS    
    if( iBlockHorzOrder < 0 || iBlockHorzOrder >= NUM_BLOCKS_ALONG_PATCH_EDGE || 
        iBlockVertOrder < 0 || iBlockVertOrder >= NUM_BLOCKS_ALONG_PATCH_EDGE )
    {
        // Tessellation factors for the edges that are not shared with the interior tessellation blocks
        // will be minimal because all tess block errors loaded from the texture will be zeroes

        // Left and right border patches with neighbouring patches' tessellation factors:
        //        _____                           _____    
        //       |     ||                       ||     |
        //       |  0  ||                       ||  0  |
        //  _   _|_____||_____             _____||_____|_   _
        // |     |     ||     |           |     ||     |     |
        //    0  |  0  ||  R  |           |  L  ||  0  |  0
        // |_   _|_____||_____|           |_____||_____|_   _|
        //       |     ||                       ||     |
        //       |  0  ||                       ||  0  |
        //       |_____||                       ||_____|
        //        
        //   -2    -1      0                N-1     N     N+1      N = NUM_BLOCKS_ALONG_PATCH_EDGE 
                                             
        // output.Inside[1] == horz tess factor
        // output.Edges[0] == left edge tess factor
        // output.Edges[2] == right edge tess factor
        output.Inside[1] = output.Edges[0] = output.Edges[2] = max(output.Edges[0], output.Edges[2]);

        // output.Inside[2] == vert tess factor
        // output.Edges[1] == bottom edge tess factor
        // output.Edges[3] == top edge tess factor
        output.Inside[0] = output.Edges[1] = output.Edges[3] = max(output.Edges[1], output.Edges[3]);
    }
#endif
    //Process2DQuadTessFactorsAvg

    return output;
}


[domain("quad")]
[partitioning("fractional_even")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(1)]
[patchconstantfunc("ConstantHS")]
[maxtessfactor( (float)(BLOCK_SIZE+2.f) )]
HS_CONTROL_POINT_OUTPUT RenderTessellatedPatchHS(InputPatch<RENDER_TESSELLATED_PATCH_VS_OUTPUT, 1> inputPatch, uint uCPID : SV_OutputControlPointID )
{
	HS_CONTROL_POINT_OUTPUT	output = (HS_CONTROL_POINT_OUTPUT)0;
	
    // Copy inputs to outputs
    output.uiPatchID =	inputPatch[uCPID].uiPatchID;

    output.PatchXYScale      = inputPatch[uCPID].PatchXYScale;
    output.PatchLBCornerXY   = inputPatch[uCPID].PatchLBCornerXY.xy;
    output.fPatchFlangeWidth = inputPatch[uCPID].fPatchFlangeWidth;
    output.fMorphCoeff       = inputPatch[uCPID].fMorphCoeff;
    output.PatchOrderInSiblQuad = inputPatch[uCPID].PatchOrderInSiblQuad;
#if USE_TEXTURE_ARRAY
    output.fPatchTexArrayIndex = (float)inputPatch[uCPID].iPatchTexArrayIndex;
    output.fParentPatchTexArrayIndex = (float)inputPatch[uCPID].iParentPatchTexArrayIndex;
#endif

    return output;
}


//--------------------------------------------------------------------------------------
// Domain Shader
//--------------------------------------------------------------------------------------
[domain("quad")]
RenderPatchVS_Output RenderTessellatedPatchDS( HS_CONSTANT_DATA_OUTPUT input, float2 QuadUV : SV_DomainLocation, const OutputPatch<HS_CONTROL_POINT_OUTPUT, 1> QuadPatch )
{
#if USE_TEXTURE_ARRAY
    float fPatchTexArrayIndex = QuadPatch[0].fPatchTexArrayIndex;
    float fParentPatchTexArrayIndex = QuadPatch[0].fParentPatchTexArrayIndex;
#endif
    float PatchXYScale = QuadPatch[0].PatchXYScale;
    float2 PatchLBCornerXY = QuadPatch[0].PatchLBCornerXY.xy;
    int2 PatchOrderInSiblQuad =QuadPatch[0].PatchOrderInSiblQuad;
    float fMorphCoeff = QuadPatch[0].fMorphCoeff;

    RenderPatchVS_Output Out = (RenderPatchVS_Output)0;

#if RENDER_TESSELLATED_SKIRTS
    // Additional tessellation blocks around patch interior are used to form skirt
    //  
    //   |     ||     |     |
    //   |-1,1 || 0,1 | 1,1 |
    //   |_____||_____|_____|....
    //   |     ||     |     |
    //   |-1,0 || 0,0 | 1,0 |
    //   |_____||_____|_____|....
    //    =====  ===== ===== 
    //   |     ||     |     |
    //   |-1,-1|| 0,-1| 1,-1|
    //   |_____||_____|_____|....
    int iBlockHorzOrder = (QuadPatch[0].uiPatchID % (NUM_BLOCKS_ALONG_PATCH_EDGE+2)) - 1;
    int iBlockVertOrder = (QuadPatch[0].uiPatchID / (NUM_BLOCKS_ALONG_PATCH_EDGE+2)) - 1;
#else
    int iBlockHorzOrder = QuadPatch[0].uiPatchID % NUM_BLOCKS_ALONG_PATCH_EDGE;
    int iBlockVertOrder = QuadPatch[0].uiPatchID / NUM_BLOCKS_ALONG_PATCH_EDGE;
#endif

    // Scale domain UV coordinates. Note that 0 corresponds directly to quad's left (bottom)
    // boundary while 1 corresponds directly to quad's right (top) boundary
    //    1  _____ 
    //      |     |  
    //      |     |
    //    0 |_____|
    //     0      1
    QuadUV *= (float)BLOCK_SIZE/(float)PATCH_SIZE;
    // Shift coordinates according to block location
    QuadUV += float2(iBlockHorzOrder, iBlockVertOrder) * ((float)BLOCK_SIZE/(float)PATCH_SIZE);

    float fFlangeShift = 0;

#if RENDER_TESSELLATED_SKIRTS
    // Additional vertices with domain coordinates outside [0,1] are used to form flange
    if( QuadUV.x < 0 || QuadUV.x > 1 ||
        QuadUV.y < 0 || QuadUV.y > 1 )
        fFlangeShift = QuadPatch[0].fPatchFlangeWidth;
#endif

    // Clamp coordinates to patch interior
    QuadUV = clamp(QuadUV, 0, 1);

    float2 ElevDataTexSize;
#if USE_TEXTURE_ARRAY
    float Elems;
    g_tex2DElevationMapArr.GetDimensions( ElevDataTexSize.x, ElevDataTexSize.y, Elems );
#else
    g_tex2DElevationMap.GetDimensions( ElevDataTexSize.x, ElevDataTexSize.y );
#endif
    float2 ElevDataTexelSize = 1.f / ElevDataTexSize;

    // Get elevation data texture UV coordinates

    //     
    //                      PATCH_SIZE+1 samples
    //                     |<----------....-->|
    //              -2 -1  0  1  2           PATCH_SIZE
    //               *  *  *  *  *  *  ....   *
    //                     |                  |
    //QuadUV               0                  1
    //ElevDataTexUV        X                  Y
    //                     |<---------------->|
    //                 PATCH_SIZE / ElevDataTexSize.xy
    // 
    // X = ((float)ELEV_DATA_EXTENSION + 0.5) * ElevDataTexelSize.xy
    // Y = ((float)ELEV_DATA_EXTENSION + PATCH_SIZE + 0.5) * ElevDataTexelSize.xy
    
    float2 ElevDataTexUV = QuadUV * float2(PATCH_SIZE, PATCH_SIZE) / ElevDataTexSize;
    float2 HeightMapUVUnShifted = ElevDataTexUV + float2(ELEV_DATA_EXTENSION, ELEV_DATA_EXTENSION) * ElevDataTexelSize;
    ElevDataTexUV += float2((float)ELEV_DATA_EXTENSION + 0.5, (float)ELEV_DATA_EXTENSION + 0.5) * ElevDataTexelSize;
    
    Out.HeightMapUV.xy = ElevDataTexUV;

    // There is no need to clamp ElevDataTexUV as QuadUV falls into the range [0,1]
    // Clamp elevation data UV to the allowable range 
    //ElevDataTexUV = clamp( 0.f + float2( (float)ELEV_DATA_EXTENSION + 0.5, (float)ELEV_DATA_EXTENSION + 0.5) * ElevDataTexelSize,
    //                       1.f - float2( (float)(ELEV_DATA_EXTENSION-1) + 0.5, (float)(ELEV_DATA_EXTENSION-1) + 0.5) * ElevDataTexelSize,
    //                       ElevDataTexUV);

    float3 VertexPos_WS;
    float3 Normal;

    VertexPos_WS = DomainToWorld(QuadUV, ElevDataTexUV, PatchXYScale, fFlangeShift, Normal 
#if USE_TEXTURE_ARRAY
        , fPatchTexArrayIndex
#endif
        );

#if TEXTURING_MODE == TM_INDEX_MASK_WITH_NM
    float3 RNeighbPos = DomainToWorld(QuadUV + float2(1.f, 0.f/(float)PATCH_SIZE), ElevDataTexUV + float2(1.f/ElevDataTexSize.x,0), PatchXYScale, fFlangeShift, Normal 
                                        #if USE_TEXTURE_ARRAY
                                                , fPatchTexArrayIndex
                                        #endif
                                        );
    float3 TNeighbPos = DomainToWorld(QuadUV + float2(0.f, 1.f/(float)PATCH_SIZE), ElevDataTexUV + float2(0.f, 1.f/ElevDataTexSize.y), PatchXYScale, fFlangeShift, Normal 
                                        #if USE_TEXTURE_ARRAY
                                                , fPatchTexArrayIndex
                                        #endif
                                        );
    float3 Tangent = RNeighbPos - VertexPos_WS;
    float3 BiTangent = TNeighbPos - VertexPos_WS;

    Out.Tangent   = normalize(Tangent);
    Out.BiTangent = normalize(BiTangent);
    Out.Normal    = normalize(Normal);
    //Out.Normal = normalize( cross(Tangent, BiTangent) );
#endif

    VertexPos_WS.xy += PatchLBCornerXY.xy;

    Out.Pos_PS = mul( float4(VertexPos_WS.POS_XYZ_SWIZZLE,1.f), g_mWorldViewProj );
    Out.fDistToCamera = length( VertexPos_WS.POS_XYZ_SWIZZLE - g_CameraPos.xyz );

#if SHADOWS
    Out.vPosLight = mul( float4(VertexPos_WS.POS_XYZ_SWIZZLE,1), g_mWorldToLightProj[0] );
#endif
    
    // Calculate texture UV coordinates
    float2 NormalMapTexSize;
    float2 DiffuseTexSize;
#if USE_TEXTURE_ARRAY
    g_tex2DNormalMapArr.GetDimensions( NormalMapTexSize.x, NormalMapTexSize.y, Elems );
#   if TM_INDEX_MASK == TEXTURING_MODE
        g_tex2DMtrlIdxArr.GetDimensions( DiffuseTexSize.x, DiffuseTexSize.y, Elems );
#   endif
#else
    g_tex2DNormalMap.GetDimensions( NormalMapTexSize.x, NormalMapTexSize.y );
#   if TM_INDEX_MASK == TEXTURING_MODE
        g_tex2DMtrlIdx.GetDimensions( DiffuseTexSize.x, DiffuseTexSize.y );
#   endif
#endif
    // Normal map and diffuse texture size must be scales of height map sizes!
    // + float2(0.5,0.5) is necessary to offset the coordinates to the center of the appropriate neight/normal map texel
    Out.NormalMapUV.xy = HeightMapUVUnShifted + float2(0.5,0.5)/NormalMapTexSize.xy;

#if TEXTURING_MODE > TM_HEIGHT_BASED
    // Normal map and diffuse texture size must be scales of height map sizes!
    Out.DiffuseTexUV.xy = HeightMapUVUnShifted + float2(0.5,0.5) / DiffuseTexSize.xy;
#endif
    

#if ENABLE_HEIGHT_MAP_MORPH || ENABLE_NORMAL_MAP_MORPH || TEXTURING_MODE > TM_HEIGHT_BASED && ENABLE_DIFFUSE_TEX_MORPH
    float2 ParentElevDataTexSize = 0;
#   if USE_TEXTURE_ARRAY
        ParentElevDataTexSize = ElevDataTexSize;
#   else
        g_tex2DParentElevMap.GetDimensions( ParentElevDataTexSize.x, ParentElevDataTexSize.y );
#   endif
    float2 ParentHeightMapUVUnShifted = (QuadUV+float2(PatchOrderInSiblQuad.xy))*float2(PATCH_SIZE, PATCH_SIZE)/(2.f*ParentElevDataTexSize) + float2(ELEV_DATA_EXTENSION, ELEV_DATA_EXTENSION) / ParentElevDataTexSize;

#   if ENABLE_HEIGHT_MAP_MORPH 
        Out.HeightMapUV.zw = ParentHeightMapUVUnShifted + float2(0.5,0.5) / ParentElevDataTexSize.xy;
#   endif

#   if ENABLE_NORMAL_MAP_MORPH
        float2 ParentNormalMapTexSize;
#       if USE_TEXTURE_ARRAY
            ParentNormalMapTexSize = NormalMapTexSize;
#       else
            g_tex2DParentNormalMap.GetDimensions( ParentNormalMapTexSize.x, ParentNormalMapTexSize.y );
#       endif
            Out.NormalMapUV.zw = ParentHeightMapUVUnShifted + float2(0.5,0.5) / ParentNormalMapTexSize.xy;
#   endif

#   if TEXTURING_MODE > TM_HEIGHT_BASED && ENABLE_DIFFUSE_TEX_MORPH
        float2 ParentDiffuseTexSize;
#       if USE_TEXTURE_ARRAY
            ParentDiffuseTexSize = DiffuseTexSize;
#       else
            g_tex2DParentMtrlIdx.GetDimensions( ParentDiffuseTexSize.x, ParentDiffuseTexSize.y );
#       endif
        // Do we need to offset diffuse coords as well?
        Out.DiffuseTexUV.zw = ParentHeightMapUVUnShifted + float2(0.5,0.5) / ParentDiffuseTexSize.xy;
#   endif
#endif

#if TEXTURING_MODE > TM_HEIGHT_BASED
    Out.TileTexUV = VertexPos_WS.xy / g_fTilingScale;
#endif

#if USE_TEXTURE_ARRAY
    Out.fPatchTexArrayIndex = fPatchTexArrayIndex;
    Out.fParentPatchTexArrayIndex = fParentPatchTexArrayIndex;
#endif

    Out.fMorphCoeff = fMorphCoeff;

    return Out;
}


RasterizerState RS_Wireframe_NoCull
{
    FILLMODE = Wireframe;
    CullMode = None;
    //AntialiasedLineEnable = true;
};

RasterizerState RS_SolidFill;//Set by the app; can be biased or not
//{
//    FILLMODE = Solid;
//    CullMode = Back;
//    FrontCounterClockwise = true;
//};

technique11 RenderPatchTessellated_FL11
{
    pass PRenderSolid
    {
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( RS_SolidFill );
        SetDepthStencilState( DSS_EnableDepthTest, 0 );

        SetVertexShader( CompileShader(vs_5_0, RenderTessellatedPatchVS() ) );
        SetHullShader( CompileShader(hs_5_0, RenderTessellatedPatchHS() ) );
        SetDomainShader( CompileShader(ds_5_0, RenderTessellatedPatchDS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader(ps_5_0, RenderPatchPS() ) );
    }

    pass PRenderWireframe
    {
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState(RS_Wireframe_NoCull);
        SetDepthStencilState( DSS_EnableDepthTest, 0 );

        SetVertexShader( CompileShader(vs_5_0, RenderTessellatedPatchVS() ) );
        SetHullShader( CompileShader(hs_5_0, RenderTessellatedPatchHS() ) );
        SetDomainShader( CompileShader(ds_5_0, RenderTessellatedPatchDS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader(ps_5_0, RenderWireframePatchPS() ) );
    }

    pass PRenderZOnly
    {
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState(RS_SolidFill_NoCull);
        SetDepthStencilState( DSS_EnableDepthTest, 0 );

        SetVertexShader( CompileShader(vs_5_0, RenderTessellatedPatchVS() ) );
        SetHullShader( CompileShader(hs_5_0, RenderTessellatedPatchHS() ) );
        SetDomainShader( CompileShader(ds_5_0, RenderTessellatedPatchDS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( NULL );
    }
}

#if USE_TEXTURE_ARRAY
technique11 RenderPatchTessellatedInstanced_FL11
{
    pass PRenderSolid
    {
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( RS_SolidFill );
        SetDepthStencilState( DSS_EnableDepthTest, 0 );

        SetVertexShader( CompileShader(vs_5_0, RenderTessellatedPatchVS_Instanced() ) );
        SetHullShader( CompileShader(hs_5_0, RenderTessellatedPatchHS() ) );
        SetDomainShader( CompileShader(ds_5_0, RenderTessellatedPatchDS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader(ps_5_0, RenderPatchPS() ) );
    }

    pass PRenderWireframe
    {
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState(RS_Wireframe_NoCull);
        SetDepthStencilState( DSS_EnableDepthTest, 0 );

        SetVertexShader( CompileShader(vs_5_0, RenderTessellatedPatchVS_Instanced() ) );
        SetHullShader( CompileShader(hs_5_0, RenderTessellatedPatchHS() ) );
        SetDomainShader( CompileShader(ds_5_0, RenderTessellatedPatchDS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader(ps_5_0, RenderWireframePatchPS() ) );
    }

    pass PRenderZOnly
    {
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState(RS_SolidFill_NoCull);
        SetDepthStencilState( DSS_EnableDepthTest, 0 );

        SetVertexShader( CompileShader(vs_5_0, RenderTessellatedPatchVS_Instanced() ) );
        SetHullShader( CompileShader(hs_5_0, RenderTessellatedPatchHS() ) );
        SetDomainShader( CompileShader(ds_5_0, RenderTessellatedPatchDS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( NULL );
    }
}
#endif //USE_TEXTURE_ARRAY




float4 g_vTerrainMapPos_PS;
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
    DstTextureMinMaxUV.xy = g_vTerrainMapPos_PS.xy;
    DstTextureMinMaxUV.zw = DstTextureMinMaxUV.xy + g_vTerrainMapPos_PS.zw * float2(1,1);
    float2 ElevDataTexSize;
#if USE_TEXTURE_ARRAY
    float Elems;
    g_tex2DElevationMapArr.GetDimensions( ElevDataTexSize.x, ElevDataTexSize.y, Elems );
#else
    g_tex2DElevationMap.GetDimensions( ElevDataTexSize.x, ElevDataTexSize.y );
#endif

    float4 SrcElevAreaMinMaxUV = float4(ELEV_DATA_EXTENSION/ElevDataTexSize.x, 
										ELEV_DATA_EXTENSION/ElevDataTexSize.y,
										1-ELEV_DATA_EXTENSION/ElevDataTexSize.x,
										1-ELEV_DATA_EXTENSION/ElevDataTexSize.y);
    
    GenerateQuadVS_OUTPUT Verts[4] = 
    {
        {float4(DstTextureMinMaxUV.xy, 0.5, 1.0), SrcElevAreaMinMaxUV.xy, InstID}, 
        {float4(DstTextureMinMaxUV.xw, 0.5, 1.0), SrcElevAreaMinMaxUV.xw, InstID},
        {float4(DstTextureMinMaxUV.zy, 0.5, 1.0), SrcElevAreaMinMaxUV.zy, InstID},
        {float4(DstTextureMinMaxUV.zw, 0.5, 1.0), SrcElevAreaMinMaxUV.zw, InstID}
    };

    return Verts[VertexId];
}


float4 RenderHeigtMapPreviewPS(GenerateQuadVS_OUTPUT In) : SV_TARGET
{
#if USE_TEXTURE_ARRAY
    float fPatchTexArrayIndex = g_iPatchTexArrayIndex;
#endif

	float fHeight = SAMPLE_LEVEL_FROM_TEX_OR_TEX_ARR( g_tex2DElevationMap, samLinearClamp, In.m_ElevationMapUV.xy, 0, int2(0,0) ) * HEIGHT_MAP_SAMPLING_SCALE;
    fHeight = (fHeight-g_GlobalMinMaxElevation.x)/g_GlobalMinMaxElevation.y;
	return float4(fHeight.xxx, 0.8);
}

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

technique11 RenderHeightMapPreview_FeatureLevel10
{
    pass
    {
        SetDepthStencilState( DSS_DisableDepthTest, 0 );
        SetRasterizerState( RS_SolidFill_NoCull );
        SetBlendState( AlphaBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );

        SetVertexShader( CompileShader( vs_4_0, GenerateQuadVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, RenderHeigtMapPreviewPS() ) );
    }
}