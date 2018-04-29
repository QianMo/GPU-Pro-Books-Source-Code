//--------------------------------------------------------------------------------------
// File: PS_OIT_LinkedList.hlsl
// 
//--------------------------------------------------------------------------------------
#include "PS_OIT_Include.hlsl"

//--------------------------------------------------------------------------------------
// External defines
//--------------------------------------------------------------------------------------
// NUM_SAMPLES
#ifndef NUM_SAMPLES
#define NUM_SAMPLES 1
#endif

//--------------------------------------------------------------------------------------
// Defines
//--------------------------------------------------------------------------------------
#ifndef MAX_SORTED_FRAGMENTS
#define MAX_SORTED_FRAGMENTS 18
#endif

//--------------------------------------------------------------------------------------
// Structures
//--------------------------------------------------------------------------------------
struct PS_INPUT
{
    float3 vNormal : NORMAL;
    float2 vTex    : TEXCOORD;
    
    float4 vPos    : SV_POSITION;
    
    bool   bFrontFace : SV_ISFRONTFACE;
        
#if NUM_SAMPLES>1
    uint    uCoverage : SV_COVERAGE;
#endif
};

struct PS_SIMPLE_INPUT
{
    float2 vTex  : TEXCOORD;
    float4 vPos  : SV_POSITION;

#if NUM_SAMPLES>1
    uint uSample : SV_SAMPLEINDEX;
#endif
};

struct PS_SIMPLE_INPUT_WITH_RESOLVE
{
    float2 vTex  : TEXCOORD;
    float4 vPos  : SV_POSITION;
};

struct Fragment_And_Link_Buffer_STRUCT
{
    uint    uPixelColor;
    uint    uDepthAndCoverage;       // Coverage is only used in the MSAA case
    uint    uNext;
};

//--------------------------------------------------------------------------------------
// UAVs
//--------------------------------------------------------------------------------------
RWByteAddressBuffer StartOffsetBuffer                                     : register(u1);
RWStructuredBuffer<Fragment_And_Link_Buffer_STRUCT> FragmentAndLinkBuffer : register(u2);

//--------------------------------------------------------------------------------------
// Globals
//--------------------------------------------------------------------------------------
static uint2 SortedFragments[MAX_SORTED_FRAGMENTS+1];

//--------------------------------------------------------------------------------------
// Pixel Shader
// PS_StoreFragments
//--------------------------------------------------------------------------------------
[earlydepthstencil]
float4 PS_StoreFragments( PS_INPUT input) : SV_Target
{
    // Renormalize normal
    input.vNormal = normalize(input.vNormal);

    // Invert normal when dealing with back faces
    if (!input.bFrontFace) input.vNormal = -input.vNormal;

    // Lighting
    float fLightIntensity = saturate(saturate(dot(input.vNormal.xyz, g_vLightVector.xyz)) + 0.2);
    float4 vColor = float4(g_vMeshColor.xyz * fLightIntensity, g_vMeshColor.w);
    
    // Texturing
    float4 vTextureColor = g_txDiffuse.Sample( g_samLinear, input.vTex );
    vColor.xyz *= vTextureColor.xyz;
    
    // Retrieve current pixel count and increase counter
    uint uPixelCount = FragmentAndLinkBuffer.IncrementCounter();

    // Calculate position in tile
    int2 uTilePosition = floor(input.vPos.xy - g_vRectangleCoordinates.xy);
    
    // Exchange indices in StartOffsetTexture corresponding to pixel location 
    uint uStartOffsetLinearAddress = 4 *( uTilePosition.x + int(g_vTileSize.x) * uTilePosition.y );
    uint uOldStartOffset;
    StartOffsetBuffer.InterlockedExchange(uStartOffsetLinearAddress, uPixelCount, uOldStartOffset);
    
    // Append new element at the end of the Fragment and Link Buffer
    Fragment_And_Link_Buffer_STRUCT Element;
    Element.uPixelColor         = PackFloat4IntoUint(vColor);
#if NUM_SAMPLES>1
    Element.uDepthAndCoverage   = PackDepthAndCoverageIntoUint(input.vPos.z, input.uCoverage);
#else
    Element.uDepthAndCoverage   = PackDepthIntoUint(input.vPos.z);
#endif
    Element.uNext               = uOldStartOffset;
    FragmentAndLinkBuffer[uPixelCount] = Element;
    
    // This won't write anything into the RT because color writes are off    
    return float4(0,0,0,0);
}


//--------------------------------------------------------------------------------------
// SRVs
//--------------------------------------------------------------------------------------
Buffer<uint> StartOffsetBufferSRV : register(t0);
StructuredBuffer<Fragment_And_Link_Buffer_STRUCT> FragmentAndLinkBufferSRV : register(t1);
#if NUM_SAMPLES>1
Texture2DMS<float4, NUM_SAMPLES> BackgroundTexture : register(t3);
#else
Texture2D BackgroundTexture : register(t3);
#endif

//--------------------------------------------------------------------------------------
// Pixel Shader
// PS_RenderFragments
//--------------------------------------------------------------------------------------
[earlydepthstencil]
float4 PS_RenderFragments( PS_SIMPLE_INPUT input) : SV_Target
{
    // Calculate position in tile
    int2 uTilePosition = floor(input.vPos.xy - g_vRectangleCoordinates.xy);
    
    // Calculate start offset buffer address
    uint uStartOffsetLinearAddress = uint( uTilePosition.x + g_vTileSize.x * uTilePosition.y );
    
    // Fetch offset of first fragment for current pixel
    uint uOffset = StartOffsetBufferSRV.Load(uStartOffsetLinearAddress);

    // Fetch structure element at this offset
    int nNumFragments = 0;
    while (uOffset!=0xFFFFFFFF)
    {
        // Retrieve fragment at current offset
        Fragment_And_Link_Buffer_STRUCT Element = FragmentAndLinkBufferSRV[uOffset];
        
#if NUM_SAMPLES>1
        // Only include fragment in sorted list if coverage mask includes the sample currently being rendered
        uint uCoverage = UnpackCoverageIntoUint(Element.uDepthAndCoverage);
        if ( uCoverage & (1<<input.uSample) )
        {
#endif
        // Copy fragment color and depth into sorted list
        SortedFragments[nNumFragments] = uint2(Element.uPixelColor, Element.uDepthAndCoverage);
        
        // Sort fragments in front to back (increasing) order using insertion sorting
        // max(j-1,0) is used to cater for the case where nNumFragments=0 (cheaper than a branch)
        int j = nNumFragments;
        [loop]while ( (j>0) && (SortedFragments[max(j-1, 0)].y > SortedFragments[j].y) )
        {
            // Swap required
            int jminusone = max(j-1, 0);
            uint2 Tmp                  = SortedFragments[j];
            SortedFragments[j]         = SortedFragments[jminusone];
            SortedFragments[jminusone] = Tmp;
            j--;
        }
        
        // Increase number of fragment if under the limit
        nNumFragments = min(nNumFragments+1, MAX_SORTED_FRAGMENTS);
        
#if NUM_SAMPLES>1
        }
#endif
        
        // Retrieve next offset
        [flatten]uOffset = Element.uNext;
    }
        
    // Retrieve current color from background color
#if NUM_SAMPLES>1
    float4 vCurrentColor = BackgroundTexture.Load(int3(input.vPos.xy, 0), input.uSample);
#else
    float4 vCurrentColor = BackgroundTexture.Load(int3(input.vPos.xy, 0));
#endif

    // Render fragments using SRCALPHA-INVSRCALPHA blending
    for (int k=nNumFragments-1; k>=0; k--)
    {
        float4 vFragmentColor = UnpackUintIntoFloat4(SortedFragments[k].x);
        vCurrentColor.xyz     = lerp(vCurrentColor.xyz, vFragmentColor.xyz, vFragmentColor.w);
    }
    
#if 0
   // Use under-blending: produces the same result as traditional back-to-front alpha blending
    float4 vCurrentColor = float4(0,0,0,1);
    for (int k=0; k<nNumFragments; k++)
    {
        float4 vFragmentColor = UnpackUintIntoFloat4(SortedFragments[k].uPixelColor);
        vCurrentColor.xyz = vCurrentColor.w*vFragmentColor.w*vFragmentColor.xyz + vCurrentColor.xyz;
        vCurrentColor.w   =  (1.0 - vFragmentColor.w)*vCurrentColor.w;
    }
#endif

    // Return manually-blended color
    return vCurrentColor;
}


#if NUM_SAMPLES>1

//--------------------------------------------------------------------------------------
// Pixel Shader
// PS_RenderFragmentsWithResolve
//--------------------------------------------------------------------------------------
float4 PS_RenderFragmentsWithResolve( PS_SIMPLE_INPUT_WITH_RESOLVE input) : SV_Target
{
    // Calculate position in tile
    int2 uTilePosition = floor(input.vPos.xy - g_vRectangleCoordinates.xy);
    
    // Calculate start offset buffer address
    uint uStartOffsetLinearAddress = uint( uTilePosition.x + g_vTileSize.x * uTilePosition.y );
    //uint uStartOffsetLinearAddress = uint( floor(input.vPos.x) + g_vScreenDimensions.x * floor(input.vPos.y) );
    
    // Fetch offset of first fragment for current pixel
    uint uOffset = StartOffsetBufferSRV.Load(uStartOffsetLinearAddress);

    int nNumFragments = 0;
    Fragment_And_Link_Buffer_STRUCT Element;
    
    // Fetch structure element at this offset
    while (uOffset!=0xFFFFFFFF)
    {
        // Retrieve fragment at current offset
        Element = FragmentAndLinkBufferSRV[uOffset];
        
        // Copy fragment color and depth into sorted list
        // (float depth is directly cast into a uint - this is OK since 
        // depth comparisons will still work after casting)
        SortedFragments[nNumFragments] = uint2(Element.uPixelColor, Element.uDepthAndCoverage);
        
        // Sort fragments in front to back (increasing) order using insertion sorting
        // max(j-1,0) is used to cater for the case where nNumFragments=0
        int j = nNumFragments;
        [loop]while ( (j>0) && (SortedFragments[max(j-1, 0)].y > SortedFragments[j].y) )
        {
            // Swap required
            int jminusone = max(j-1, 0);
            uint2 Tmp                  = SortedFragments[j];
            SortedFragments[j]         = SortedFragments[jminusone];
            SortedFragments[jminusone] = Tmp;
            j--;
        }
        
        // Increase number of fragment
        nNumFragments = min(nNumFragments+1, MAX_SORTED_FRAGMENTS);
        
        // Retrieve next offset
        [flatten]uOffset = Element.uNext;
    }
    
    // Retrieve color of individual samples
    float3 vCurrentColorSample[NUM_SAMPLES];
    [unroll]for (uint uSample=0; uSample<NUM_SAMPLES; uSample++)
    {
        vCurrentColorSample[uSample] = BackgroundTexture.Load(int3(input.vPos.xy, 0), uSample);
    }
    
    // Render fragments using SRCALPHA-INVSRCALPHA blending
    for (int k=nNumFragments-1; k>=0; k--)
    {
        // Retrieve fragment color
        float4 vFragmentColor = UnpackUintIntoFloat4(SortedFragments[k].x);
        
        // Retrieve sample coverage
        uint uCoverage = UnpackCoverageIntoUint(SortedFragments[k].y);
        
        // Blend current color sample with fragment color if covered
        // If the sample is not covered the color will be unchanged
        [unroll]for (uint uSample=0; uSample<NUM_SAMPLES; uSample++)
        {
            float fIsSampleCovered = ( uCoverage & (1<<uSample) ) ? 1.0 : 0.0;
            vCurrentColorSample[uSample].xyz = lerp( vCurrentColorSample[uSample].xyz, vFragmentColor.xyz, vFragmentColor.w * fIsSampleCovered);
        }
    }
    
    // Resolve samples into a single color
    float4 vCurrentColor = float4(0,0,0,1);
    [unroll]for (uint uSample=0; uSample<NUM_SAMPLES; uSample++)
    {
        vCurrentColor.xyz += vCurrentColorSample[uSample];
    }
    vCurrentColor.xyz *= (1.0 / NUM_SAMPLES);

    // Return manually-blended color
    return vCurrentColor;
}
#endif