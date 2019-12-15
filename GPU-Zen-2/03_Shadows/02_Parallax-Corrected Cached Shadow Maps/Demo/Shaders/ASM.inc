#include "ASMShaderData.inc"
#include "_Shaders/ASMLayerShaderData.inc"

static const float2 ASMDepthAtlasSizeOverDepthTileSize = float2( gs_ASMDepthAtlasTextureWidth, gs_ASMDepthAtlasTextureHeight ) / gs_ASMTileSize;
static const float2 ASMDEMTileSizeOverDEMAtlasSize = gs_ASMDEMTileSize / float2( gs_ASMDEMAtlasTextureWidth, gs_ASMDEMAtlasTextureHeight );
static const float2 ASMDEMTileCoord = 1.5 / float2( gs_ASMDEMAtlasTextureWidth, gs_ASMDEMAtlasTextureHeight );
static const float2 ASMDEMTileSize = ( gs_ASMDEMTileSize - 3.0 ) / float2( gs_ASMDEMAtlasTextureWidth, gs_ASMDEMAtlasTextureHeight );
static const float2 ASMDepthAtlasSize = float2( gs_ASMDepthAtlasTextureWidth, gs_ASMDepthAtlasTextureHeight );
static const float2 ASMOneOverDepthAtlasSize = 1.0 / float2( gs_ASMDepthAtlasTextureWidth, gs_ASMDepthAtlasTextureHeight );
static const float ASMHalfOverDepthTileSize = 0.5 / gs_ASMTileSize;

Texture2D<float4> LongRangeIndexTexture                : register(t9);
Texture2D<float4> LongRangePreRenderIndexTexture       : register(t10);
Texture2D<float> LongRangeDepthAtlasTexture            : register(t11);
Texture2D<float> LongRangeDepthExtentMapAtlasTexture   : register(t12);
Texture2D<float> LongRangePreRenderLODClampTexture     : register(t13);

SamplerComparisonState ShadowCmpSampler  : register(s10);
sampler ColorBlackTransparentBorderPoint : register(s11);
sampler ClampLinearPoint                 : register(s12);

float3 GetASMTileCoord( float3 indexCoord, float4 indirectionTextureData )
{
    return GetASMTileCoord( indexCoord, indirectionTextureData, ASMOneOverDepthAtlasSize );
}

float UnpackASMFadeInConstant( float w )
{
    return 2.0 * frac( abs( w ) );
}

float3 GetIndirectionTextureCoord( float4x4 indexTexMat, float3 worldPos )
{
    return mul( indexTexMat, float4( worldPos, 1.0 ) ).xyz;
}

float PCF9(
    float3 shadowMapCoord,
    float2 kernelSize )
{
    float2 tapOffset[9] =
    {
        float2( 0.00, 0.00),
        float2( 1.20, 0.00),
        float2(-1.20, 0.00),
        float2( 0.00, 1.20),
        float2( 0.00,-1.20),
        float2( 0.84, 0.84),
        float2(-0.84, 0.84),
        float2(-0.84,-0.84),
        float2( 0.84,-0.84),
    };

    float tapWeight[9] =
    {
        0.120892,
        0.110858,
        0.110858,
        0.110858,
        0.110858,
        0.111050,
        0.111050,
        0.111050,
        0.111050,
    };

    float shadowFactor = 0;
    [unroll] for( int i = 0; i < 9; ++i )
    {
        float2 sampleCoord = shadowMapCoord.xy + kernelSize * tapOffset[i];
        shadowFactor += tapWeight[i] * LongRangeDepthAtlasTexture.SampleCmpLevelZero( ShadowCmpSampler, sampleCoord, shadowMapCoord.z );
    }
    return shadowFactor;
}

float ASMBlockerDistance(
    Texture2D indirectionTexture,
    float3 worldPos,
    float4x4 indexTexMat,
    float3 blockerSearchDir,
    float mip )
{
    float3 indexCoord = GetIndirectionTextureCoord( indexTexMat, worldPos );
    float4 t = indirectionTexture.SampleLevel( ColorBlackTransparentBorderPoint, indexCoord.xy, mip );

    float3 tileCoord = GetASMTileCoord( indexCoord, t );

    float2 tileCoordBoundsMin = floor( tileCoord.xy * ASMDepthAtlasSizeOverDepthTileSize - ASMHalfOverDepthTileSize ) * ASMDEMTileSizeOverDEMAtlasSize + ASMDEMTileCoord;
    float2 tileCoordBoundsMax = tileCoordBoundsMin + ASMDEMTileSize;

    float numBlockers = 0;
    float blockerSum = 0;

    if( t.w != 0 )
    {
        float3 sampleCoord = tileCoord;
        sampleCoord.z += 0.5 * blockerSearchDir.z * 0.1;
        [loop] for( int i = 0; i < 7; ++i )
        {
            float demValue = LongRangeDepthExtentMapAtlasTexture.SampleLevel( ClampLinearPoint, clamp( sampleCoord.xy, tileCoordBoundsMin, tileCoordBoundsMax ), 0 );
            [flatten] if( demValue < sampleCoord.z )
            {
                blockerSum += demValue;
                numBlockers += 1.0;
            }
            sampleCoord += blockerSearchDir * 0.1;
        }
    }

    float blockerDepth = numBlockers > 0 ? blockerSum * rcp( numBlockers ) : 1.0 ;
    return saturate( tileCoord.z - blockerDepth ) * gs_ASMTileFarPlane;
}

struct ASMFrustumDesc
{
    float3 indexCoord;
    Texture2D indirectionTexture;
};

bool ClampedDepthDifference( float mip, ASMFrustumDesc frustumDesc, inout float depthDiff )
{
    float4 t = frustumDesc.indirectionTexture.SampleLevel( ColorBlackTransparentBorderPoint, frustumDesc.indexCoord.xy, mip );
    if( t.w > 0 )
    {
        float3 tileCoord = GetASMTileCoord( frustumDesc.indexCoord, t );
        float demDepth = LongRangeDepthExtentMapAtlasTexture.SampleLevel( ClampLinearPoint, tileCoord.xy, 3 );
        depthDiff = saturate( tileCoord.z - demDepth );
        return true;
    }
    return false;
}

void SampleASMShadowMap( float mip, float kernelSize, ASMFrustumDesc frustumDesc, float3 debugColor, inout float3 shadowFactor, inout float fadeInFactor )
{
    float4 t = frustumDesc.indirectionTexture.SampleLevel( ColorBlackTransparentBorderPoint, frustumDesc.indexCoord.xy, mip );

    [branch] if( t.w != 0 )
    {
#if SHADOWS_CROSSFADE
        fadeInFactor = UnpackASMFadeInConstant( t.w );
#else
        fadeInFactor = 0;
#endif

        float3 tileCoord = GetASMTileCoord( frustumDesc.indexCoord, t );

#if ASM_PCF9
        shadowFactor = PCF9( tileCoord, kernelSize * ASMOneOverDepthAtlasSize );
#else
        shadowFactor = LongRangeDepthAtlasTexture.SampleCmpLevelZero( ShadowCmpSampler, tileCoord.xy, tileCoord.z );
#endif

        shadowFactor *= debugColor;
    }
}

float3 GetASMShadowFactor( float mip, float kernelSize, ASMFrustumDesc frustumDesc, ASMFrustumDesc preRenderFrustumDesc )
{
    float3 shadowFactor = 1.0;
    float fadeInFactor = 1.0;

#if ASM_PRERENDER_AVAILABLE
    float lodClamp = LongRangePreRenderLODClampTexture.SampleLevel( ColorBlackTransparentBorderPoint, preRenderFrustumDesc.indexCoord.xy, 0 ) * gs_ASMMaxRefinement;
    SampleASMShadowMap( max( mip, lodClamp ), kernelSize, preRenderFrustumDesc, float3(1,0,0), shadowFactor, fadeInFactor );
#endif

    [branch] if( fadeInFactor > 0 )
    {
        float3 shadowFactor1 = 1.0;
        float fadeInFactor1 = 1.0;

        SampleASMShadowMap( mip, kernelSize, frustumDesc, float3(1,1,0), shadowFactor1, fadeInFactor1 );
        shadowFactor = lerp( shadowFactor, shadowFactor1, fadeInFactor );
    }

    return shadowFactor;
}

float3 SampleLongRangeShadows( float3 worldPos )
{
    float3 result = 1.0;

    {
        float demLOD = gs_ASMMaxRefinement;

        ASMFrustumDesc frustumDesc;
        frustumDesc.indirectionTexture = LongRangeIndexTexture;

        ASMFrustumDesc preRenderFrustumDesc;
        preRenderFrustumDesc.indirectionTexture = LongRangePreRenderIndexTexture;

#if SHADOWS_PARALLAX
        float blockerDistance = ASMBlockerDistance( frustumDesc.indirectionTexture, worldPos, LongRangeIndexTexMat, LongRangeBlockerSearchVector, demLOD );
        float preRenderBlockerDistance = ASMBlockerDistance( preRenderFrustumDesc.indirectionTexture, worldPos, LongRangePreRenderIndexTexMat, LongRangePreRenderBlockerSearchVector, demLOD );
#else
        float blockerDistance = 0;
        float preRenderBlockerDistance = 0;
#endif

        frustumDesc.indexCoord = GetIndirectionTextureCoord( LongRangeIndexTexMat, worldPos + blockerDistance * LongRangeReceiverWarpVector );
        preRenderFrustumDesc.indexCoord = GetIndirectionTextureCoord( LongRangePreRenderIndexTexMat, worldPos + preRenderBlockerDistance * LongRangePreRenderReceiverWarpVector );

        float d = 0;
#if ASM_MRF
        if( ClampedDepthDifference( demLOD, frustumDesc, d ) )
#endif
        {
            float penumbraSizeFactor = saturate( 40.0 * d.x - 0.05 );
            float kernelSize = saturate( d.x * 100 + 0.5 );

            float lod = penumbraSizeFactor * gs_ASMMaxRefinement;
            float mip = floor( lod );

            float3 shadowFactor = GetASMShadowFactor( mip.x, kernelSize.x, frustumDesc, preRenderFrustumDesc );

            if( penumbraSizeFactor > 0.0 && penumbraSizeFactor < 1.0 )
            {
                float3 shadowFactor1 = GetASMShadowFactor( mip.x + 1, kernelSize.x, frustumDesc, preRenderFrustumDesc );
                shadowFactor = lerp( shadowFactor, shadowFactor1, lod.x - mip.x );
            }

            result = shadowFactor;
        }
    }

    return result;
}
