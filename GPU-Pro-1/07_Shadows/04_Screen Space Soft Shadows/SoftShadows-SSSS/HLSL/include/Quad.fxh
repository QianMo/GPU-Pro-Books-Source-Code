/*********************************************************************NVMH3****
*******************************************************************************
$Revision: #4 $

Copyright NVIDIA Corporation 2008
TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
*AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS
BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY
LOSS) ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF
NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

Comments:
    Header file with lots of useful macros, types, and functions for use
	with textures and render-to-texture-buffer effects in DirectX.
	Such textures are written to a "Quad" -- that is, a two-triangle
	quadrilateral that covers the viewport.

Example Macro Usages: /////////////////////////////////////////////////

  Texture-declaration Macros:

      // simple 2D wrap-mode texture
    FILE_TEXTURE_2D(SurfTexture,SurfSampler,"myfile.dds")
      // with user-specified addr mode
    FILE_TEXTURE_2D_MODAL(SpotTexture,SpotSampler,"myfile.dds",CLAMP)

  RenderTarget Texture-declaration Macros:

      // declare screen-sized render targets
    DECLARE_QUAD_TEX(ObjTexture,ObjSampler,"A8R8G8B8")
    DECLARE_QUAD_DEPTH_BUFFER(DepthTexture,"D24S8")
      // scaled versions of the above:
    DECLARE_SIZED_QUAD_TEX(GlowTexture,GlowSampler,"A8R8G8B8",0.5)
    DECLARE_SIZED_QUAD_DEPTH_BUFFER(DepthTexture,"D24S8",0.5)
      // fixed-size variation:
    DECLARE_SIZED_TEX(BlahMap,BlahSampler,"R32F",128,1)
      // for shadows etc:
    DECLARE_SQUARE_QUAD_TEX(ShadTexture,ShadObjSampler,"A16R16G16B16F",512)
    DECLARE_SQUARE_QUAD_DEPTH_BUFFER(ShadDepth,"D24S8",512)

  Data types used in shaders:

    QUAD_REAL & variations -- "half" but you can force QUAD_REAL
      to be "float" by defining the symbol QUAD_FLOAT

  Compile-Time Flags (assign these macros before including "Quad.fxh"):

    TWEAKABLE_TEXEL_OFFSET	// shows in parameter panel
    NO_TEXEL_OFFSET	// disables DirectX half-texel offset

Structure defined by theis File: ///////////////////////////////////////

    QuadVertexOutput -- used by shaders, a simple connection
	    for "Draw=buffer" passes, passing data between VS and PS

Shader Functions for "Draw=buffer" passes Defined by the File: /////////

    QuadVertexOutput ScreenQuadVS():
	    standard vertex shader for screen-aligned quads - uses global scope
    QuadVertexOutput ScreenQuadVS2(texelOffsets):
	    standard vertex shader for screen-aligned quads - no global scope
    QUAD_REAL4 TexQuadPS(QuadVertexOutput IN,uniform sampler2D InputSampler)
	    pass this pixel shader a sampler -- will draw it to the screen
    QUAD_REAL4 TexQuadBiasPS(QuadVertexOutput IN,
			uniform sampler2D InputSampler,
			QUAD_REAL TBias)
	    Same as above, but uses tex2Dbias()

Utility Functions for Texture-Based Lookup Tables:

    QUAD_REAL scale_lookup(QUAD_REAL Value,const QUAD_REAL TableSize)
    QUAD_REAL2 scale_lookup(QUAD_REAL2 Value,const QUAD_REAL TableSize)
    QUAD_REAL3 scale_lookup(QUAD_REAL3 Value,const QUAD_REAL TableSize)

Other Utility Functions:

    QUAD_REAL4 premultiply(QUAD_REAL4 C)
    QUAD_REAL4 unpremultiply(QUAD_REAL4 C) // uses macro value NV_ALPHA_EPSILON

Global Variables: ///////////////////////////////////////////////////////

    // cautions: unlike many common global variables, these identifiers
    //    do not begin with "g"
    QUAD_REAL QuadTexOffset // reconciles difference between DirectX
	  // pixel and texel centers
    QUAD_REAL2 QuadScreenSize // contains the dimensions
          // of the render window




To learn more about shading, shaders, and to bounce ideas off other shader
    authors and users, visit the NVIDIA Shader Library Forums at:

    http://developer.nvidia.com/forums/

*******************************************************************************
******************************************************************************/



#ifndef _H_QUAD_
#define _H_QUAD_

// This optional flag deterines if RenderTargets are visible in the properties
//    pane so that they can be connected to shared surfaces
//
// #define SHARED_BG_IMAGE

#ifdef SHARED_BG_IMAGE
#define TARGETWIDGET ""
#else /* ! SHARED_BG_IMAGE */
#define TARGETWIDGET "None"
#endif /* ! SHARED_BG_IMAGE */

// Numeric types we are likely to encounter....
// Redefine these before including "Quad.fxh" if you want
//	to use a type other than "half" for these data or just
//	define the symbol QUAD_FLOAT to use "floats"
#ifndef QUAD_REAL
#ifdef QUAD_FLOAT
#define QUAD_REAL float
#define QUAD_REAL2 float2
#define QUAD_REAL3 float3
#define QUAD_REAL4 float4
#define QUAD_REAL3x3 float3x3
#define QUAD_REAL4x3 float4x3
#define QUAD_REAL3x4 float3x4
#define QUAD_REAL4x4 float4x4
#else /* ! QUAD_FLOAT */
#define QUAD_REAL half
#define QUAD_REAL2 half2
#define QUAD_REAL3 half3
#define QUAD_REAL4 half4
#define QUAD_REAL3x3 half3x3
#define QUAD_REAL4x3 half4x3
#define QUAD_REAL3x4 half3x4
#define QUAD_REAL4x4 half4x4
#endif /* ! QUAD_FLOAT */
#endif /* ! QUAD_REAL */

#ifndef NVIDIA_GREEN
#define NVIDIA_GREEN (QUAD_REAL3(0.4627,.7255,0))
#endif /* NVIDIA_GREEN */

///////////////////////////////////////////////////////////////////////
/// Texture-Declaration Macros ////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

//
// Modal 2D File Textures
//
// example usage: FILE_TEXTURE_2D_MODAL(GlowMap,GlowSampler,"myfile.dds",CLAMP)
//
#if DIRECT3D_VERSION < 0xa00
#define FILE_TEXTURE_2D_MODAL(TexName,SampName,Filename,AddrMode) texture TexName < \
	string ResourceName = (Filename); \
    string ResourceType = "2D"; \
>; \
sampler2D SampName = sampler_state { \
    texture = <TexName>; \
    AddressU = AddrMode; AddressV = AddrMode; \
    MagFilter = Linear; MipFilter = Linear; MinFilter = Linear; };
#else /* DIRECT3D_VERSION */
#define FILE_TEXTURE_2D_MODAL(TexName,SampName,Filename,AddrMode) texture TexName < \
	string ResourceName = (Filename); \
    string ResourceType = "2D"; \
>; \
sampler2D SampName = sampler_state { \
    texture = <TexName>; \
    AddressU = AddrMode; AddressV = AddrMode; \
    Filter=MIN_MAG_MIP_LINEAR; };
#endif /* DIRECT3D_VERSION */

//
// Simple 2D File Textures
//
// example usage: FILE_TEXTURE_2D(GlowMap,GlowSampler,"myfile.dds")
//
#define FILE_TEXTURE_2D(TextureName,SamplerName,Diskfile) FILE_TEXTURE_2D_MODAL(TextureName,SamplerName,(Diskfile),WRAP)

//
// Use this variation of DECLARE_QUAD_TEX() if you want a *scaled* render target
//
// example usage: DECLARE_SIZED_QUAD_TEX(GlowMap,GlowSampler,"A8R8G8B8",1.0)
#if DIRECT3D_VERSION < 0xa00
#define DECLARE_SIZED_QUAD_TEX(TexName,SampName,PixFmt,Multiple) texture TexName : RENDERCOLORTARGET < \
    float2 ViewPortRatio = {Multiple,Multiple}; \
    int MipLevels = 1; \
    string Format = PixFmt ; \
    string UIWidget = (TARGETWIDGET); \
>; \
sampler2D SampName = sampler_state { \
    texture = <TexName>; \
    AddressU = Clamp; \
    AddressV = Clamp; \
    MagFilter = Linear; \
    MinFilter = Linear; \
    MipFilter = Point; };
#else /* DIRECT3D_VERSION */
#define DECLARE_SIZED_QUAD_TEX(TexName,SampName,PixFmt,Multiple) texture TexName : RENDERCOLORTARGET < \
    float2 ViewPortRatio = {Multiple,Multiple}; \
    int MipLevels = 1; \
    string Format = PixFmt ; \
    string UIWidget = (TARGETWIDGET); \
>; \
sampler2D SampName = sampler_state { \
    texture = <TexName>; \
    AddressU = Clamp; \
    AddressV = Clamp; \
    Filter=MIN_MAG_LINEAR_MIP_POINT; };
#endif /* DIRECT3D_VERSION */

//
// Use this macro to easily declare typical color render targets
//
// example usage: DECLARE_QUAD_TEX(ObjMap,ObjSampler,"A8R8G8B8")
#define DECLARE_QUAD_TEX(TextureName,SamplerName,PixelFormat) DECLARE_SIZED_QUAD_TEX(TextureName,SamplerName,(PixelFormat),1.0)

//
// Use this macro to easily declare variable-sized depth render targets
//
// example usage: DECLARE_SIZED_QUAD_DEPTH_BUFFER(DepthMap,"D24S8",0.5)
#define DECLARE_SIZED_QUAD_DEPTH_BUFFER(TextureName,PixelFormat,Multiple) texture TextureName : RENDERDEPTHSTENCILTARGET < \
    float2 ViewPortRatio = {Multiple,Multiple}; \
    string Format = (PixelFormat); \
    string UIWidget = (TARGETWIDGET); \
>; 

//
// Use this macro to easily declare typical depth render targets
//
// example usage: DECLARE_QUAD_DEPTH_BUFFER(DepthMap,"D24S8")
#define DECLARE_QUAD_DEPTH_BUFFER(TexName,PixFmt) DECLARE_SIZED_QUAD_DEPTH_BUFFER(TexName,PixFmt,1.0)

//
// declare exact-sized arbitrary texture
//
// example usage: DECLARE_SIZED_TEX(BlahMap,BlahSampler,"R32F",128,1)
#if DIRECT3D_VERSION < 0xa00
#define DECLARE_SIZED_TEX(Tex,Samp,Fmt,Wd,Ht) texture Tex : RENDERCOLORTARGET < \
    float2 Dimensions = { Wd, Ht }; \
    string Format = Fmt ; \
    string UIWidget = (TARGETWIDGET);\
    int miplevels=1;\
>; \
sampler2D Samp = sampler_state { \
    texture = <Tex>; \
    AddressU = Clamp; \
    AddressV = Clamp; \
    MagFilter = Linear; \
    MinFilter = Linear; \
    MipFilter = Point; };
#else /* DIRECT3D_VERSION >= 0xa00 */
#define DECLARE_SIZED_TEX(Tex,Samp,Fmt,Wd,Ht) texture Tex : RENDERCOLORTARGET < \
    float2 Dimensions = { Wd, Ht }; \
    string Format = Fmt ; \
    string UIWidget = (TARGETWIDGET);\
    int miplevels=1;\
>; \
sampler2D Samp = sampler_state { \
    texture = <Tex>; \
    AddressU = Clamp; \
    AddressV = Clamp; \
    Filter=MIN_MAG_LINEAR_MIP_POINT; };
#endif /* DIRECT3D_VERSION >= 0xa00 */

//
// declare exact-sized square texture, as for shadow maps
//
// example usage: DECLARE_SQUARE_QUAD_TEX(ShadMap,ShadObjSampler,"A16R16G16B16F",512)
#define DECLARE_SQUARE_QUAD_TEX(TexName,SampName,PixFmt,Size) DECLARE_SIZED_TEX(TexName,SampName,(PixFmt),Size,Size)

//
// likewise for shadow depth targets
//
// example usage: DECLARE_SQUARE_QUAD_DEPTH_BUFFER(ShadDepth,"D24S8",512)
#define DECLARE_SQUARE_QUAD_DEPTH_BUFFER(TextureName,PixelFormat,Size) texture TextureName : RENDERDEPTHSTENCILTARGET < \
    float2 Dimensions = { Size, Size }; \
    string Format = (PixelFormat) ; \
    string UIWidget = (TARGETWIDGET); \
>; 

////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////// Utility Functions ////////
////////////////////////////////////////////////////////////////////////////

//
// Scale inputs for use with texture-based lookup tables. A value ranging from
// zero to one needs a slight scaling and offset to be sure to point at the
// centers of the first and last pixels of that lookup texture. Pass the integer
// size of the table in TableSize. For now we'll assume that all tables are 1D,
// square, or cube-shaped -- all axes of equal size
//
// Cost of this operation for pixel shaders: two const-register
//   entries and a MAD (one cycle)
//
QUAD_REAL scale_lookup(QUAD_REAL Value,const QUAD_REAL TableSize)
{
    QUAD_REAL scale = ((TableSize - 1.0)/TableSize);
    QUAD_REAL texShift = (0.5 / TableSize);
    return (scale*Value + texShift);
}

QUAD_REAL2 scale_lookup(QUAD_REAL2 Value,const QUAD_REAL TableSize)
{
    QUAD_REAL scale = ((TableSize - 1.0)/TableSize);
    QUAD_REAL texShift = (0.5 / TableSize);
    return (scale.xx*Value + texShift.xx);
}

QUAD_REAL3 scale_lookup(QUAD_REAL3 Value,const QUAD_REAL TableSize)
{
    QUAD_REAL scale = ((TableSize - 1.0)/TableSize);
    QUAD_REAL texShift = (0.5 / TableSize);
    return (scale.xxx*Value + texShift.xxx);
}

// pre-multiply and un-pre-mutliply functions. The precision
//	of thse operations is often limited to 8-bit so don't
//	always count on them!
// The macro value of NV_ALPHA_EPSILON, if defined, is used to
//	avoid IEEE "NaN" values that may occur when erroneously
//	dividing by a zero alpha (thanks to Pete Warden @ Apple
//	Computer for the suggestion in GPU GEMS II)

// multiply color by alpha to turn an un-premultipied
//	pixel value into a premultiplied one
QUAD_REAL4 premultiply(QUAD_REAL4 C)
{
    return QUAD_REAL4((C.w*C.xyz),C.w);
}

#define NV_ALPHA_EPSILON 0.0001

// given a premultiplied pixel color, try to undo the premultiplication.
// beware of precision errors
QUAD_REAL4 unpremultiply(QUAD_REAL4 C)
{
#ifdef NV_ALPHA_EPSILON
    QUAD_REAL a = C.w + NV_ALPHA_EPSILON;
    return QUAD_REAL4((C.xyz / a),C.w);
#else /* ! NV_ALPHA_EPSILON */
    return QUAD_REAL4((C.xyz / C.w),C.w);
#endif /* ! NV_ALPHA_EPSILON */
}

/////////////////////////////////////////////////////////////////////////
// Structure Declaration ////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

struct QuadVertexOutput {
    QUAD_REAL4 Position	: POSITION;
    QUAD_REAL2 UV	: TEXCOORD0;
};

/////////////////////////////////////////////////////////////////////////
// Hidden tweakables declared by this .fxh file /////////////////////////
/////////////////////////////////////////////////////////////////////////

QUAD_REAL2 QuadScreenSize : VIEWPORTPIXELSIZE <
    string UIName="Screen Size";
    string UIWidget="None";
>;

// DirectX has a half-texel offset for RTT
#ifndef NO_TEXEL_OFFSET

#ifdef TWEAKABLE_TEXEL_OFFSET
QUAD_REAL QuadTexOffset <
    string UIName="Texel Alignment Offset";
> = 0.5;
#else /* !TWEAKABLE_TEXEL_OFFSET */
QUAD_REAL QuadTexOffset <
    string UIName="Texel Alignment Offset";
    string UIWidget="None";
> = 0.5;
#endif /* !TWEAKABLE_TEXEL_OFFSET */

static QUAD_REAL2 QuadTexelOffsets =
	QUAD_REAL2(QuadTexOffset/(QuadScreenSize.x),
		   QuadTexOffset/(QuadScreenSize.y));

#endif /* NO_TEXEL_OFFSET */

////////////////////////////////////////////////////////////
////////////////////////////////// vertex shaders //////////
////////////////////////////////////////////////////////////

QuadVertexOutput ScreenQuadVS(
    QUAD_REAL3 Position : POSITION, 
    QUAD_REAL3 TexCoord : TEXCOORD0
) {
    QuadVertexOutput OUT;
    OUT.Position = QUAD_REAL4(Position, 1);
#ifdef NO_TEXEL_OFFSET
    OUT.UV = TexCoord.xy;
#else /* NO_TEXEL_OFFSET */
    OUT.UV = QUAD_REAL2(TexCoord.xy+QuadTexelOffsets); 
#endif /* NO_TEXEL_OFFSET */
    return OUT;
}

QuadVertexOutput ScreenQuadVS2(
    QUAD_REAL3 Position : POSITION, 
    QUAD_REAL3 TexCoord : TEXCOORD0,
    uniform QUAD_REAL2 TexelOffsets
) {
    QuadVertexOutput OUT;
    OUT.Position = QUAD_REAL4(Position, 1);
    OUT.UV = QUAD_REAL2(TexCoord.xy+TexelOffsets); 
    return OUT;
}

//////////////////////////////////////////////////////
////////////////////////////////// pixel shaders /////
//////////////////////////////////////////////////////

//
// Draw textures into screen-aligned quad
//
QUAD_REAL4 TexQuadPS(QuadVertexOutput IN,
    uniform sampler2D InputSampler) : COLOR
{   
    QUAD_REAL4 texCol = tex2D(InputSampler, IN.UV);
    return texCol;
}  

QUAD_REAL4 TexQuadBiasPS(QuadVertexOutput IN,
    uniform sampler2D InputSampler,QUAD_REAL TBias) : COLOR
{   
    QUAD_REAL4 texCol = tex2Dbias(InputSampler, QUAD_REAL4(IN.UV,0,TBias));
    return texCol;
}  

#endif /* _QUAD_FXH */

////////////// eof ///
