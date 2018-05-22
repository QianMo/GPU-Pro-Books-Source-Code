//--------------------------------------------------------------------------------------
// File: Reflections11.hlsl
//
// This file implements the hlsl part of a grid based reflection
// technique that uses scattering writes
//
// Reflection are generated via a grid of cells with SH coeffs encoding reflected colors
// for directions entering a grid cell
//
// Contributed by the AMD Developer Relations team
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

cbuffer cbConstants : register( b1 )
{
    float4x4 g_f4x4View;                      // View matrix
    float4x4 g_f4x4WorldViewProjection;       // World * View * Projection matrix
    float4x4 g_f4x4ViewLight;                 // Light View matrix
    float4x4 g_f4x4ProjLight;                 // Light Proj matrix
    float4x4 g_f4x4WorldViewProjLight;        // World * ViewLight * Projection Light matrix
    float4   g_vCamPos;                       // world space cam pos
    float4   g_vShadowMapDimensions;          // dimensions of the shadow map / LV
    float4   g_vBackBufferDimensions;         // dimensions of the backbuffer
    float4   g_vWorldSpaceLightDirection;     // WS direction of light
    float4   g_RTBbox;                        // right top back of bounding box
    float4   g_LBFbox;                        // left bottom front of bounding box
    float4   g_BBoxSize;                      // size of bounding box
    float4   g_GridCellSize;                  // size of a grid cell
    float4   g_InvGridCellSize;               // size of a grid cell
    float    g_fW;                            // W from the projection matrix
    float    g_fH;                            // H from the projection matrix
    float2   g_fPad;                          // pad
};

// constant buffer used to pass index/vertex buffer
// data for the current draw call to CS_RenderTris
cbuffer cbVtxFetchConstants : register( b0 )
{
    uint  g_IndexCount;
    uint  g_StartIndexLocation;
    uint  g_VertexStart;
    uint  g_VertexStride;
    uint  g_BaseVertexLocation;
    uint  g_texID;
    uint2 pad;
};

struct SPLAT
{
    float3 pos;
    float3 n;
    float3 col;
};

struct LINKEDSPLAT
{
    uint data0;
    uint data1;
    uint data2;
    uint data3;
    uint prev;
};

struct DIData
{
    uint vertexcountperinstance;
    uint instancecount;
    uint startvertexlocation;
    uint startinstancelocation;
};

uint offset( uint3 pos )
{
    return ( pos.x + pos.y * CELLS_XYZ + pos.z * CELLS_XYZ * CELLS_XYZ );
}

/*
#define SWIZZLE_SIZE_SHIFT 3

//uint swizzledOffset( uint3 pos )
uint offset( uint3 pos )
{
    uint3 bpos = ( pos >> SWIZZLE_SIZE_SHIFT );
    uint3 spos = pos & (uint(0x1<<SWIZZLE_SIZE_SHIFT)-1).xxx;
    
    uint soff = uint(0x1<<SWIZZLE_SIZE_SHIFT) * uint(0x1<<SWIZZLE_SIZE_SHIFT) * uint(0x1<<SWIZZLE_SIZE_SHIFT) * 
               ( bpos.x + bpos.y * ( CELLS_XYZ / uint(0x1<<SWIZZLE_SIZE_SHIFT) ) + 
                 bpos.z * ( CELLS_XYZ / uint(0x1<<SWIZZLE_SIZE_SHIFT) ) * ( CELLS_XYZ / uint(0x1<<SWIZZLE_SIZE_SHIFT) ) );
    
    soff += spos.x + spos.y * uint(0x1<<SWIZZLE_SIZE_SHIFT) + spos.z * uint(0x1<<SWIZZLE_SIZE_SHIFT) * uint(0x1<<SWIZZLE_SIZE_SHIFT);
    
    return soff; 
}
*/

//--------------------------------------------------------------------------------------
// helper function that packs a normal into a uint
//--------------------------------------------------------------------------------------
uint pack_normal( float3 norm )
{
   float3 n = 0.5f * ( (1.0f).xxx + norm );

   uint3 un = uint3( 1023.0f * n );

   return uint( un.x | ( un.y << 10 ) | ( un.z << 20 ) );
}

//--------------------------------------------------------------------------------------
// helper function that unpacks packs a normal from a uint
//--------------------------------------------------------------------------------------
float3 unpack_normal( uint norm )
{
   float3 res;
   
   res = float3( norm & 1023, norm&(1023<<10), norm&(1023<<20) ) * 
         float3( 2.0f/1023.f, 
                 2.0f/(1023.0f*1024.0f), 
                 2.0f/(1024.0f*1024.0f*1023.0f) );
                 
   res -= (1.0f).xxx;

   return res;
}

uint pack_color( float3 col )
{
   uint3 un = uint3( 256.0f * col );

   return uint( un.x | ( un.y << 10 ) | ( un.z << 20 ) );
}

//--------------------------------------------------------------------------------------
// helper function that unpacks packs a normal from a float2
//--------------------------------------------------------------------------------------
float3 unpack_color( uint col )
{
   float3 res;
   
   res = float3( col & 1023, col&(1023<<10), col&(1023<<20) ) * 
         float3( 1.0f/256.f, 
                 1.0f/(256.0f*1024.0f), 
                 1.0f/(1024.0f*1024.0f*256.0f) );
   return res;
}

// pack two 16 bit floats into a unit
uint pack( float2 d )
{
    return ( f32tof16( d.x ) + ( f32tof16( d.y ) << 16 ) );
}

// pack two 16 bit floats into a unit
uint pack( float d0, float d1 )
{
    return ( f32tof16( d0 ) + ( f32tof16( d1 ) << 16 ) );
}

// pack two 16 bit floats into a unit
uint pack( float d0 )
{
    return ( f32tof16( d0 ) );
}

float unpackf1( uint d )
{
    return f16tof32( d );
}

// unpack two floats from a uint
float2 unpack( uint d )
{
    return float2( f16tof32( d ), f16tof32( d >> 16 ) );
}

LINKEDSPLAT pack( SPLAT t )
{
    LINKEDSPLAT res;
    
    res.data0 = pack( t.pos.xy );
    res.data1 = pack( t.pos.z );
    res.data2 = pack_normal( t.n );
    res.data3 = pack_color( t.col );
    res.prev  = 0xffffffff;
    
    return res;
}

SPLAT unpack( LINKEDSPLAT lt )
{
    SPLAT res;
    float2 tmp;
    
    res.pos.xy    = unpack( lt.data0 );
    res.pos.z     = unpackf1( lt.data1 );
    res.n         = unpack_normal( lt.data2 );
    res.col       = unpack_color( lt.data3 );
    
    return res;
}

//======================================================================================
// Textures and Samplers
//======================================================================================

// current scene texture
Texture2D                           g_txCurrentObject               : register( t0 );

// contains all splats
StructuredBuffer<SPLAT>             SplatBufferSRV                  : register( t0  );

// index buffer SRV used by CS_RenderTris
Buffer<uint>                        g_bufIndices                    : register( t0  );

// vertex buffer SRV used by CS_RenderTris
Buffer<float4>                      g_bufVertices                   : register( t1  );

// GB
Texture2D<uint4>                    g_txPackedGB                    : register( t1 );

Texture2D                           g_txReflections                 : register( t2 );

// per per grid cell start offset of last array entry/ list entry 
Buffer<uint>                        StartOffsetBufferSRV            : register( t2  );

// contains all splats
StructuredBuffer<LINKEDSPLAT>       LinkedSplatBufferSRV            : register( t3  );

Texture2D                           g_txNReflections                : register( t3 );
Texture2DArray                      g_tx2dArray                     : register( t3 );

// shadow map
Texture2D<float>                    g_txShadowMap                   : register( t4 );


//--------------------------------------------------------------------------------------
// SRVs/UAVs
//--------------------------------------------------------------------------------------

// buffer 
RWStructuredBuffer<SPLAT>           SplatBuffer           : register( u0 );
RWStructuredBuffer<LINKEDSPLAT>     LinkedSplatsBuffer    : register( u0 );
RWByteAddressBuffer                 StartOffsetBuffer     : register( u1 );
RWByteAddressBuffer                 DrawIndirectBuffer    : register( u2 );

// Samplers
SamplerState                g_SamplePointBorder  : register( s0 );
SamplerState                g_SampleLinearClamp  : register( s1 );
SamplerState                g_SamplePointWrap    : register( s2 );
SamplerState                g_SampleLinearWrap   : register( s3 );
SamplerState                g_SamplePointClamp   : register( s4 );
SamplerComparisonState      g_SamplePointCmp     : register( s5 );

//======================================================================================
// Vertex & Pixel shader structures
//======================================================================================

struct VS_SIMPLE_INPUT
{
    float3 vPosition : POSITION;
    float2 vTex      : TEXCOORD;
};

struct VS_SIMPLE_OUTPUT
{
    float2 vTex      : TEXCOORD;
    float4 vPosition : SV_POSITION;
};

struct PS_SIMPLE_INPUT
{
    float2 vTex : TEXCOORD;
    float4 vPos : SV_POSITION;
};

struct VS_RenderSceneInput
{
    float3 f3Position : POSITION;  
    float3 f3Normal   : NORMAL;     
    float2 f2TexCoord : TEXTURE0;
};

struct PS_RenderSceneInput
{
    float4 f4Position           : SV_POSITION;
    float4 f4Diffuse            : COLOR0; 
    float2 f2TexCoord           : TEXTURE0;
    float3 f3WorldSpacePos      : TEXTURE1;
    float3 f3WorldSpaceNormal   : TEXTURE2;
};

struct PS_RenderOutput
{
    float4 f4Color : SV_Target0;
};

struct PS_SceneGBufferOut
{
    uint4 u4PackedData        : SV_Target0;
};

struct PS_ReflectionsOut
{
    float4 f4ReflectedLight : SV_Target0;
    float4 f4Normal         : SV_Target1;
};

struct PS_SimpleRenderSceneInput
{
    float4 f4Position           : SV_POSITION;
};

#define FS 5
#define FS2 ( FS/2 )

//======================================================================================
// This function implements a fast uniform shadow filter
//======================================================================================
float filter_shadow( float3 tc )
{
   float  s   = 0.0f;
   float2 stc = ( g_vShadowMapDimensions.xy * tc.xy ) + float2( 0.5, 0.5 );
   float2 tcs = floor( stc );
   float2 fc;
   int    row;
   int    col;

   fc.xy = stc - tcs;
   tc.xy = tcs * g_vShadowMapDimensions.zw;
   
   tc.z -= 0.005f;
   
   // loop over the rows
   for( row = -FS2; row <= FS2; row += 2 )
   {
       [unroll]for( col = -FS2; col <= FS2; col += 2 )
       {
            float4 v = g_txShadowMap.GatherCmpRed( g_SamplePointCmp, tc.xy, tc.z, int2( col, row ) ); 
            
            if( row == -FS2 ) // top row
            {
                if( col == -FS2 ) // left
                    s += dot( float4( 1.0-fc.x, 1.0, 1.0-fc.y, (1.0-fc.x)*(1.0-fc.y) ), v );
                else if( col == FS2 ) // right
                    s += dot( float4( 1.0f, fc.x, fc.x*(1.0-fc.y), 1.0-fc.y ), v );
                else // center
                    s += dot( float4( 1.0, 1.0, 1.0-fc.y, 1.0-fc.y ), v );
            }
            else if( row == FS2 )  // bottom row
            {
                if( col == -FS2 ) // left
                    s += dot( float4( (1.0-fc.x)*fc.y, fc.y, 1.0, (1.0-fc.x) ), v );
                else if( col == FS2 ) // right
                    s += dot( float4( fc.y, fc.x*fc.y, fc.x, 1.0 ), v );
                else // center
                    s += dot( float4(fc.yy,1.0,1.0), v );
            }
            else // center rows
            {
                if( col == -FS2 ) // left
                    s += dot( float4( (1.0-fc.x), 1.0, 1.0, (1.0-fc.x) ), v ); 
                else if( col == FS2 ) // right
                    s += dot( float4( 1.0, fc.x, fc.x, 1.0 ), v ); 
                else // center
                    s += dot( (1.0).xxxx, v ); 
            }
        }
   }
  
   return s*(1.0f/(FS*FS));
}

//--------------------------------------------------------------------------------------
// Pass through Vertex Shader for full screen passes
//--------------------------------------------------------------------------------------
VS_SIMPLE_OUTPUT VSPassThrough( VS_SIMPLE_INPUT I )
{
    VS_SIMPLE_OUTPUT output = (VS_SIMPLE_OUTPUT)0;
    
    output.vPosition = float4(I.vPosition.xyz, 1.0);
    output.vTex      = I.vTex;
    
    return output;
}

//======================================================================================
// This shader computes standard transform and lighting for rendering a
// to the LV (reflective shadow map)
//======================================================================================
PS_RenderSceneInput VS_RenderLightView( VS_RenderSceneInput I )
{
    PS_RenderSceneInput O;
    
    // Transform the position from object space to homogeneous projection light space
    O.f4Position = mul( float4( I.f3Position, 1.0f ), 
                        g_f4x4WorldViewProjLight );

    // Transform the normal from object space to world space    
    float3 f3NormalWorldSpace = normalize( I.f3Normal );
    
    O.f3WorldSpaceNormal = f3NormalWorldSpace;
    
    // Calc diffuse color
    float3 f3LightDir = -g_vWorldSpaceLightDirection.xyz; 
    O.f4Diffuse = float4( max( 0, dot( f3NormalWorldSpace, f3LightDir ) ).xxx, 1.0f);  
    
    // copy tex coord
    O.f2TexCoord = I.f2TexCoord;

    // copy world space position
    O.f3WorldSpacePos = I.f3Position;
    
    return O;    
}

//======================================================================================
// This shader computes standard transform and lighting
//======================================================================================
PS_RenderSceneInput VS_RenderScene( VS_RenderSceneInput I )
{
    PS_RenderSceneInput O;
    float3 f3NormalWorldSpace;
    float4 pos = mul( float4( I.f3Position, 1.0f ), 
                      g_f4x4WorldViewProjection );
    
    // Transform the position from object space to homogeneous projection space
    O.f4Position = pos;
    
    // Transform the normal from object space to world space    
    f3NormalWorldSpace = normalize( I.f3Normal );
    
    O.f3WorldSpaceNormal = f3NormalWorldSpace;
    
    // Calc diffuse color
    float3 f3LightDir = -g_vWorldSpaceLightDirection.xyz; 
    O.f4Diffuse = float4(max( 0, dot( f3NormalWorldSpace, f3LightDir ) ).xxx, 1.0f);  
    
    // copy tex coord
    O.f2TexCoord = I.f2TexCoord;

    // copy world space position
    O.f3WorldSpacePos = I.f3Position;

    return O;    
}

//======================================================================================
// This shader outputs the per pixel normal and the world space position and the textured
// and lit color - it writes the main gbuffer as seen from the eye - version used
// when only indirect light is used (no indirect shadows)
//======================================================================================
PS_SceneGBufferOut PS_RenderNormalDepthAndColor( PS_RenderSceneInput I )
{
    PS_SceneGBufferOut O;
    
    float4 f4Col   = I.f4Diffuse * g_txCurrentObject.Sample( g_SampleLinearWrap, I.f2TexCoord );
    
    O.u4PackedData.x = pack( I.f3WorldSpacePos.xy );
    O.u4PackedData.y = pack( I.f3WorldSpacePos.z, f4Col.x );
    O.u4PackedData.z = pack_normal( normalize( I.f3WorldSpaceNormal.xyz ) );
    O.u4PackedData.w = pack( f4Col.yz );

    return O;
}

/*
void PS_RenderSplats( PS_RenderSceneInput I )
{
    SPLAT s;
    uint newOffset = SplatBuffer.IncrementCounter();
    
    float4 f4Col   = I.f4Diffuse * g_txCurrentObject.Sample( g_SampleLinearWrap, I.f2TexCoord );

    s.pos = I.f3WorldSpacePos.xyz;
    s.n   = normalize(  I.f3WorldSpaceNormal.xyz );
    s.col = f4Col.xyz;
    
	SplatBuffer[ newOffset ] = s;
}
*/

void PS_RenderSplats( PS_RenderSceneInput I )
{
    SPLAT s;
    uint oldOffset;
    
    
    float4 f4Col   = I.f4Diffuse * g_txCurrentObject.Sample( g_SampleLinearWrap, I.f2TexCoord );
    float4 f4SMC   = mul( float4( I.f3WorldSpacePos.xyz, 1.0f ), g_f4x4WorldViewProjLight );
    float2 rtc     = float2( 0.0f,  1.0f ) +  float2( 0.5f, -0.5f ) * 
                    ( (1.0f).xx + ( f4SMC.xy / f4SMC.w ) );
    
    // compute a shadow term
    float  fShadow = filter_shadow( float3( rtc, f4SMC.z / f4SMC.w ) );

    s.pos = I.f3WorldSpacePos.xyz;
    s.n   = normalize(  I.f3WorldSpaceNormal.xyz );
    s.col = 1.3f * saturate( fShadow + 0.3f ) * f4Col.xyz;
    LINKEDSPLAT ls = pack( s );

	float3 f3GridP = max(  I.f3WorldSpacePos - g_LBFbox.xyz, (0.0f).xxx );

    // compute triangle bounding box
    float3 f3TLBF = max( f3GridP - 0.4f * g_GridCellSize.xyz, (0.0f).xxx );
    float3 f3TRTB = max( f3GridP + 0.4f * g_GridCellSize.xyz, (0.0f).xxx );
    
    // figure out the range of cells touching the bb of the triangles
    float3 f3Start = min( (float(CELLS_XYZ-1)).xxx, 
                     max( f3TLBF * g_InvGridCellSize.xyz, (0.0f).xxx ) );
    float3 f3Stop  = min( (float(CELLS_XYZ-1)).xxx, 
                     max( f3TRTB * g_InvGridCellSize.xyz, (0.0f).xxx ) );
    uint3  start = uint3( f3Start );
    uint3  stop  = uint3( f3Stop );
    
    // iterate over cells
    for( uint zi = start.z; zi <= stop.z; ++zi )
    {
        for( uint yi = start.y; yi <= stop.y; ++yi )
        {
            for( uint xi = start.x; xi <= stop.x; ++xi )
            {
                // alloc new offset
                uint newOffset = LinkedSplatsBuffer.IncrementCounter();
                uint oldOffset;
                
                // update grid offset buffer
                StartOffsetBuffer.InterlockedExchange( 4 * offset( uint3(xi,yi,zi) ), 
                                                           newOffset, oldOffset );
                
                ls.prev = oldOffset;
                
                // add splat to the grid
                LinkedSplatsBuffer[ newOffset ] = ls;
            }
        }
    }
}

float4 PS_Render( PS_RenderSceneInput I ) : SV_TARGET
{
    return I.f4Diffuse * g_txCurrentObject.Sample( g_SampleLinearWrap, I.f2TexCoord );
}

PS_SimpleRenderSceneInput VS_ComputeSplatCount( uint vid : SV_VertexID )
{
    PS_SimpleRenderSceneInput O;

    // Transform the position from object space to homogeneous projection space
    O.f4Position = float4(0,0,0,1);
    
    return O;    
}

float4 PS_Debug( PS_RenderSceneInput I ) : SV_TARGET
{
    return I.f4Diffuse;
    //return (1.0f).xxxx;
}

PS_RenderSceneInput VS_Debug( uint vid : SV_VertexID )
{
    PS_RenderSceneInput O;
    
    LINKEDSPLAT ls = LinkedSplatBufferSRV[ vid ];
    SPLAT        s = unpack( ls ); 

    float3 f3NormalWorldSpace;
    float4 pos = mul( float4( s.pos, 1.0f ), 
                      g_f4x4WorldViewProjection );
    
    // Transform the position from object space to homogeneous projection space
    O.f4Position = pos;
    
    // Transform the normal from object space to world space    
    f3NormalWorldSpace = normalize( s.n );
    
    O.f3WorldSpaceNormal = f3NormalWorldSpace;
    
    // Calc diffuse color
    float3 f3LightDir = -g_vWorldSpaceLightDirection.xyz; 
    O.f4Diffuse = float4( s.col, 1.0f);  
    
    // copy tex coord
    O.f2TexCoord = (0.0f).xx;

    // copy world space position
    O.f3WorldSpacePos = s.pos;

    return O;    
}

void PS_ComputeSplatCount( PS_SimpleRenderSceneInput I )
{
	DrawIndirectBuffer.Store( 0, LinkedSplatsBuffer.IncrementCounter() );
	DrawIndirectBuffer.Store( 4 , 1 );
	DrawIndirectBuffer.Store( 8, 0 );
	DrawIndirectBuffer.Store( 12, 0 );
}

float4
intersect_splats( float3 orig, float3 dir,
                  SPLAT s0, 
                  SPLAT s1, 
                  SPLAT s2, 
                  SPLAT s3,
                  float4 f4Mask )
{
    float4 w, denom, k, dw;

    // compute initial weights
    w.x = saturate( dot( normalize( orig - s0.pos ), s0.n ) );
    w.y = saturate( dot( normalize( orig - s1.pos ), s1.n ) );
    w.z = saturate( dot( normalize( orig - s2.pos ), s2.n ) );
    w.w = saturate( dot( normalize( orig - s3.pos ), s3.n ) );
    
    // compute closest distance to splat
    // ( ( orig + k * dir ) - s.pos ) * s.n = 0
    // s.n * orig + k * dir * s.n  - s.pos * s.n = 0
    // k = ( s.pos * s.n - orig * s.n ) /  ( dir * s.n )
    denom.x = dot( dir, s0.n );
    denom.y = dot( dir, s1.n );
    denom.z = dot( dir, s2.n );
    denom.w = dot( dir, s3.n );
    
    k.x = dot( ( s0.pos - orig ), s0.n );
    k.y = dot( ( s1.pos - orig ), s1.n );
    k.z = dot( ( s2.pos - orig ), s2.n );
    k.w = dot( ( s3.pos - orig ), s3.n );
    
    k /= denom;
    
    k *= ( denom != (0.0f).xxxx ? (1.0f).xxxx : (0.0f).xxxx );
    
    w *= ( k > ( 0.0f ).xxxx ) ? ( 1.0f ).xxxx : ( 0.0f ).xxxx;
    
    // change w to reflect distance from splat center
    float3 temp0 = orig + k.x * dir - s0.pos;
    dw.x = 0.001f + dot( temp0, temp0 );
    float3 temp1 = orig + k.y * dir - s1.pos;
    dw.y = 0.001f + dot( temp1, temp1 );
    float3 temp2 = orig + k.z * dir - s2.pos;
    dw.z = 0.001f + dot( temp2, temp2 );
    float3 temp3 = orig + k.w * dir - s3.pos;
    dw.w = 0.001f + dot( temp3, temp3 );
    
    // combine weights
    w *= ( dw < (0.08f).xxxx ? 1.0f : 0.0f ) * f4Mask;
 
    // compute result
    return float4( w.x * s0.col + w.y * s1.col + 
                   w.z * s2.col + w.w * s3.col,
                   dot( w, (1.0f).xxxx ) );
}

//======================================================================================
// This function walks the 3d grid to check for intersections of
// splats and the given ray
//======================================================================================
float3 traceReflectedRay( float3 f3OrgP, float3 f3D )
{
    float   fI;
    float3  f3Inc, f3P;
    float3  f3GridDst = floor( ( f3OrgP - g_LBFbox.xyz + f3D  ) * g_InvGridCellSize.xyz + (0.5f).xxx );
    float3  f3GridOrg = floor( ( f3OrgP - g_LBFbox.xyz ) * g_InvGridCellSize.xyz + (0.5f).xxx  );
    float3  f3GridD   = f3GridDst - f3GridOrg;
    float4  f4ResCol  = (0.0f).xxxx;

    // compute length for ray march
    float fLen   = ceil( max( max( abs( f3GridD.x ), abs( f3GridD.y ) ), abs( f3GridD.z ) ) );
                                   
    // setup march along the ray
    if( fLen > 0.0f )
        f3Inc = f3GridD / fLen;
    else
        f3Inc = (0.0f).xxx;

    // do the march
    for( fI  =  0.0f, f3P = f3GridOrg; 
         fI <=  fLen; 
         fI +=  1.0f, f3P += f3Inc )
    {
        int3    vPos             = int3( f3P );
        float3  f3Pos            = f3P  + (0.5f).xxx;

        // clip to avoid out of buffer accesses
        if( all( vPos >= (0).xxx ) && all( vPos < (CELLS_XYZ).xxx ) )
        {
            // Fetch offset of last fragment for current pixel
            uint uOffset = StartOffsetBufferSRV.Load(offset(uint3(vPos))).x;
            
            // Loop through each splat in the current grid cell
            while(uOffset != 0xffffffff)
            {
				float4 f4Mask = float4(1.0f,0.0f,0.0f,0.0f);
				
                // Retrieve 4 triangles at current offset
                LINKEDSPLAT s0 = LinkedSplatBufferSRV[ uOffset ];
                [flatten] if( s0.prev != 0xffffffff )
                {
					uOffset  = s0.prev;
					f4Mask.y = 1.0f;
                } 
                LINKEDSPLAT s1 = LinkedSplatBufferSRV[ uOffset ];
                [flatten] if( s1.prev != 0xffffffff )
                {
					uOffset  = s1.prev;
					f4Mask.z = 1.0f;
                } 
                LINKEDSPLAT s2 = LinkedSplatBufferSRV[ uOffset ];
                [flatten] if( s2.prev != 0xffffffff )
                {
					uOffset  = s2.prev;
					f4Mask.w = 1.0f;
                } 
                LINKEDSPLAT s3 = LinkedSplatBufferSRV[ uOffset ];
                uOffset = s3.prev;
                
                f4ResCol += intersect_splats( f3OrgP, f3D, 
                                              unpack(s0), unpack(s1), 
                                              unpack(s2), unpack(s3), f4Mask );
                                              
            }
        }
        
        if( f4ResCol.w > 0.0f )
        {
            f4ResCol.xyz /= f4ResCol.w;
            break;
        }
        
        f4ResCol  = (0.0f).xxxx;
    }

    return f4ResCol.xyz;
} 

static const float fBilinearWeights[16] =
{
    9.0f/16.0f,3.0f/16.0f,3.0f/16.0f,1.0f/16.0f,
    3.0f/16.0f,9.0f/16.0f,1.0f/16.0f,3.0f/16.0f,
    3.0f/16.0f,1.0f/16.0f,9.0f/16.0f,3.0f/16.0f,
    1.0f/16.0f,3.0f/16.0f,3.0f/16.0f,9.0f/16.0f
};

static const int2 viOffsets[4] =
{
    int2( 0,0 ), int2( 1,0 ), int2( 0,1 ), int2( 1,1 )
};

//======================================================================================
// This shader implements a bi-lateral upsampling pass for the indirect light buffer
//======================================================================================
PS_RenderOutput PS_UpsampleReflections( PS_SIMPLE_INPUT I )
{
    PS_RenderOutput O;
    
    float   fTotalWeight = 0.0f;
    float4  f4IL = (0.0f).xxxx;
    int3    adr  = int3( int2( I.vPos.xy ), 0 );
    int     si   = int( ( ( adr.y & 0x1 ) ? 2 : 1 ) + ( adr.x & 0x1 ) );
    
    uint4  u4Data      = g_txPackedGB.Load( adr );
    float2 tmp0        = unpack( u4Data.y );
    float3 f3Val1      = float3( unpack( u4Data.x ), tmp0.x );
    float3 f3Val2      = unpack_normal( u4Data.z );
    float  CamZ        = mul( float4( f3Val1, 1.0f ), g_f4x4View ).z;
    float4 vNormDHiRes = float4( f3Val2, CamZ );
    
    for( int i = 0; i < 4; ++i )
    {
        int3   i3Lp          = int3( ( ( adr.xy >> 1 ) + viOffsets[i] ), 0 );
        uint4  u4Data        = g_txPackedGB.Load( i3Lp << 1 );
        float2 tmp0          = unpack( u4Data.y );
        float3 f3Val1        = float3( unpack( u4Data.x ), tmp0.x ); 
        float3 f3Val2        = unpack_normal( u4Data.z ); 
        float  CamZ          = mul( float4( f3Val1, 1.0f ), g_f4x4View ).z;
        float4 vNormDCoarse  = float4( f3Val2, CamZ );
        float4 vShadeCoarse  = g_txReflections.Load( i3Lp );
        float  fDepthWeight  = 1.0f / ( 0.000001f + abs( vNormDHiRes.w - vNormDCoarse.w ) );
        float  fNormalWeight = pow( saturate( dot( vNormDCoarse.xyz, vNormDHiRes.xyz ) ), 32 );
        float  fWeight       = fDepthWeight * fNormalWeight * fBilinearWeights[si+i*4];
        
        f4IL         += vShadeCoarse.xyzw * fWeight;
        fTotalWeight += fWeight;
    }
    
    [flatten]if( fTotalWeight < 0.0001f )
        O.f4Color = g_txReflections.Load( int3( ( adr.xy >> 1 ) + viOffsets[0],0 ) );
    else
        O.f4Color = float4( f4IL / fTotalWeight );
        
    return O;
}

PS_ReflectionsOut PS_RenderReflections( PS_SIMPLE_INPUT I )
{
    PS_ReflectionsOut O;

    int3   tc      = int3( int2( I.vPos.xy ) << 1, 0 );
    uint4  u4Data  = g_txPackedGB.Load( tc );
    float3 f3CCol;
    float3 f3CPos  = float3( unpack( u4Data.x ), unpack( u4Data.y ).x );
    float3 f3CN    = unpack_normal( u4Data.z );
    float4 f4SMC   = mul( float4( f3CPos.xyz, 1.0f ), g_f4x4WorldViewProjLight );
    float2 rtc     = float2( 0.0f,  1.0f ) +  float2( 0.5f, -0.5f ) * 
                     ( (1.0f).xx + ( f4SMC.xy / f4SMC.w ) );
    
    // compute a shadow term
    float  fShadow = filter_shadow( float3( rtc, f4SMC.z / f4SMC.w ) );

    f3CCol.x  = unpack( u4Data.y ).y;
    f3CCol.yz = unpack( u4Data.w );
    
    float3 f3Ref = (0.0f).xxx;

    // cancel invalid normals
    f3CN *= ( dot( f3CN, f3CN ) > 2.0f ) ? 0.0f : 1.0f;
    
    float fFac = dot( f3CN, f3CN ) != 0.0f ? 1.0f : 0.0f; 
    
    if( fFac > 0.0f )
    {
        // compute color and reflection
        float3 refDir = reflect( normalize(f3CPos - g_vCamPos.xyz ), f3CN );
        f3Ref = traceReflectedRay( f3CPos, 10.0f * refDir );
    }

    // ouput color taking account direct light, shadow and indirect light
    O.f4ReflectedLight =  float4( f3Ref, fShadow );
    O.f4Normal         =  float4( f3CN, mul( float4( f3CPos, 1.0f ), g_f4x4View ).z );

    return O;
}


#define IL_BW   7
#define IL_BW2 (IL_BW/2)

float4 bilateral_blur_horz( PS_SIMPLE_INPUT I, Texture2D image, 
                            Texture2D norm_d, uniform int half_w, float2 inv_tex )
{
   float4 f4CC  = image.SampleLevel( g_SamplePointClamp, I.vTex, 0 );
   float4 f4CN  = norm_d.SampleLevel( g_SamplePointClamp, I.vTex, 0 );
   float3 s     = f4CC.xyz;
   float  sum_w = 1.0f;
   float  fCol;
   int    col;

   [unroll]for( col = -half_w, fCol = -half_w; col <= half_w; fCol += 1.0f, ++col )
   {
         if( col != 0 )
         {
            float4 f4C = image.SampleLevel( g_SamplePointClamp, I.vTex + inv_tex * float2( fCol, 0 ), 0 );
            float4 f4N = norm_d.SampleLevel( g_SamplePointClamp, I.vTex + inv_tex * float2( fCol, 0 ), 0 );

            float  w = pow( saturate( dot( f4CN.xyz, f4N.xyz ) ), 16 ) * rcp( 0.1f + abs( f4N.w - f4CN.w ) );
         
            s     += w * f4C.xyz;
            sum_w += w;
         }
   }    

   return float4( s/sum_w, f4CC.w );
}

PS_RenderOutput PS_BilateralBlurHorz( PS_SIMPLE_INPUT I )
{
   PS_RenderOutput O;
   
   O.f4Color =  bilateral_blur_horz( I, g_txReflections, g_txNReflections, 
                                     IL_BW2, g_vBackBufferDimensions.zw * 2.0f );

   return O;
}

float4 bilateral_blur_vert( PS_SIMPLE_INPUT I, Texture2D image, Texture2D norm_d, 
                            uniform int half_w, float2 inv_tex )
{
   float4 f4CC  = image.SampleLevel( g_SamplePointClamp, I.vTex, 0 );
   float4 f4CN  = norm_d.SampleLevel( g_SamplePointClamp, I.vTex, 0 );
   float3 s     = f4CC.xyz;
   float  sum_w = 1.0f;
   float fRow;
   int row;

   [unroll]for( fRow = -half_w, row = -half_w; row <= half_w; fRow += 1.0f, ++row )
   {
         if( row != 0 )
         {
            float4 f4C = image.SampleLevel( g_SamplePointClamp, I.vTex + inv_tex * float2( 0, fRow ), 0 );
            float4 f4N = norm_d.SampleLevel( g_SamplePointClamp, I.vTex + inv_tex * float2( 0, fRow ), 0 );

            float  w = pow( saturate( dot( f4CN.xyz, f4N.xyz ) ), 16 ) * rcp( 0.1f + abs( f4N.w - f4CN.w ) );
         
            s     += w * f4C.xyz;
            sum_w += w;
         }
   }    

   return float4( s/sum_w, f4CC.w );
}

PS_RenderOutput PS_BilateralBlurVert( PS_SIMPLE_INPUT I )
{
   PS_RenderOutput O;
   
   O.f4Color =  bilateral_blur_vert( I, g_txReflections, g_txNReflections, 
                                     IL_BW2, g_vBackBufferDimensions.zw * 2.0f  );

   return O;
}


//======================================================================================
// This shader outputs the pixel's color using the lit 
// diffuse material color, a shadow term and it computes an indirect light term
// that is also added to the light taking indirect shadow into account
//======================================================================================
PS_RenderOutput PS_RenderScene( PS_SIMPLE_INPUT I )
{
    PS_RenderOutput O;

    float3 f3CCol;
    uint4  u4Data  = g_txPackedGB.Load( int3( int2( I.vPos.xy ), 0 ) );
    float3 f3CPos  = float3( unpack( u4Data.x ), unpack( u4Data.y ).x );

    f3CCol.x  = unpack( u4Data.y ).y;
    f3CCol.yz = unpack( u4Data.w );
    
    float4 f4Ref = g_txReflections.Load( int3( int2( I.vPos.xy ), 0 ) );
   
    // ouput color taking account direct light, shadow and indirect light
    O.f4Color =  1.3f * float4( 0.7f * saturate( f4Ref.w + 0.3f ) * f3CCol + 0.7f * f4Ref.xyz, 0 );

    return O;
}

//======================================================================================
// This shader outputs the pixel's color using the lit 
// diffuse material color, a shadow term and it computes an indirect light term
// that is also added to the light taking indirect shadow into account
//======================================================================================
PS_RenderOutput PS_RenderScene_NoReflections( PS_SIMPLE_INPUT I )
{
    PS_RenderOutput O;

    float3 f3CCol;
    uint4  u4Data  = g_txPackedGB.Load( int3( int2( I.vPos.xy ), 0 ) );
    float3 f3CPos  = float3( unpack( u4Data.x ), unpack( u4Data.y ).x );
    float4 f4SMC   = mul( float4( f3CPos, 1.0f ), g_f4x4WorldViewProjLight );
    float3 LSp     = f4SMC.xyz / f4SMC.w;

    // transform from RT space to texture space.
    float2 ShadowTexC = float2( 0.0f,  1.0f ) + 
                        float2( 0.5f, -0.5f ) * float2( LSp.xy + ( 1.0f ).xx );

    // compute a shadow term
    float  fShadow = filter_shadow( float3( ShadowTexC, LSp.z ) );

    f3CCol.x  = unpack( u4Data.y ).y;
    f3CCol.yz = unpack( u4Data.w );

    // ouput color taking account direct light, shadow and indirect light
    O.f4Color = 1.3f * float4( saturate( fShadow + 0.3f ) * f3CCol, 0 );

    return O;
}

//====================================================================================
// EOF
//====================================================================================
