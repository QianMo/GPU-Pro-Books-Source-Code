Texture2D<float> DepthTexture : register(t0);

sampler PointClamp : register(s0);

// --------------------------------------------------------------------------------------------------------------------

#ifdef COPY_DEPTH

#define ATLAS_QUADS_MISC 1
#define ATLAS_QUADS_TEXCOORD 1
#define AtlasQuadsVS mainVS
#include "AtlasQuads.inc"

float mainPS( AtlasQuads_VS2PS input ) : SV_Depth
{
    float tileDepth = DepthTexture.SampleLevel( PointClamp, input.uv, 0 ) + input.miscData.z;
    return max( 0.0, min( 1.0, tileDepth ) );
}

#endif

// --------------------------------------------------------------------------------------------------------------------

#ifdef COPY_DEM

#define ATLAS_QUADS_MISC 1
#define ATLAS_QUADS_TEXCOORD 1
#define AtlasQuadsVS mainVS
#include "AtlasQuads.inc"

float4 mainPS( AtlasQuads_VS2PS input ) : SV_Target
{
    return DepthTexture.SampleLevel( PointClamp, input.uv, 0 );
}

#endif

// --------------------------------------------------------------------------------------------------------------------

#ifdef COMPUTE_DEM

#define ATLAS_QUADS_MISC 1
#define ATLAS_QUADS_TEXCOORD 1
#define AtlasQuadsVS mainVS
#include "AtlasQuads.inc"

float4 mainPS( AtlasQuads_VS2PS input ) : SV_Target
{
    const float2 nbrSampleOffsets[8] =
    {
        float2( 1.0,  0.0), float2( 0.0,  1.0), float2( 1.0,  1.0), float2(-1.0, -1.0),
        float2( 0.0, -1.0), float2( 1.0, -1.0), float2(-1.0,  0.0), float2(-1.0,  1.0),
    };

    float minZ = DepthTexture.SampleLevel( PointClamp, input.uv, 0 );
    [unroll] for ( int i = 0; i < 8; ++i )
        minZ = min( minZ, DepthTexture.SampleLevel( PointClamp, input.uv + nbrSampleOffsets[i] * input.miscData.xy, 0 ) );

    return minZ + input.miscData.z;
}

#endif

// --------------------------------------------------------------------------------------------------------------------

#ifdef FILL_INDIRECTION_TEXTURE

#define ATLAS_QUADS_MISC 1
#define ATLAS_QUADS_TEXCOORD 0
#define AtlasQuadsVS mainVS
#include "AtlasQuads.inc"

float4 mainPS( AtlasQuads_VS2PS input ) : SV_Target
{
    return input.miscData;
}

#endif

// --------------------------------------------------------------------------------------------------------------------

#ifdef CLEAR_DEPTH

#define ATLAS_QUADS_MISC 0
#define ATLAS_QUADS_TEXCOORD 0
#define AtlasQuadsVS mainVS
#include "AtlasQuads.inc"

float mainPS( AtlasQuads_VS2PS input ) : SV_Depth
{
    return 1.0;
}

#endif
