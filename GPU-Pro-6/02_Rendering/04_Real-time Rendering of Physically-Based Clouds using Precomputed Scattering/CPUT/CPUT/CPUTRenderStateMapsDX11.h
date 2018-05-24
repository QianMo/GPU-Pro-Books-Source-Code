//--------------------------------------------------------------------------------------
// Copyright 2013 Intel Corporation
// All Rights Reserved
//
// Permission is granted to use, copy, distribute and prepare derivative works of this
// software for any purpose and without fee, provided, that the above copyright notice
// and this statement appear in all copies.  Intel makes no representations about the
// suitability of this software for any purpose.  THIS SOFTWARE IS PROVIDED "AS IS."
// INTEL SPECIFICALLY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, AND ALL LIABILITY,
// INCLUDING CONSEQUENTIAL AND OTHER INDIRECT DAMAGES, FOR THE USE OF THIS SOFTWARE,
// INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PROPRIETARY RIGHTS, AND INCLUDING THE
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  Intel does not
// assume any responsibility for any errors which may appear in this software nor any
// responsibility to update it.
//--------------------------------------------------------------------------------------

#ifndef _CPUTRENDERSTATEMAPSDX11_H
#define _CPUTRENDERSTATEMAPSDX11_H

// TODO: Should this be in a cpp file instead of an h file?
//       We just put it here to get it out of the way
//       in the implementation file (CPUTRenderStateBlockDX11.cpp)

//-----------------------------------------------------------------------------
static const CPUTRenderStateMapEntry pBlendDescMap[] =
{
    { _L("alphatocoverageenable"),  ePARAM_TYPE_BOOL,  (UINT)offsetof(D3D11_BLEND_DESC, AlphaToCoverageEnable)},
    { _L("independentblendenable"), ePARAM_TYPE_BOOL,  (UINT)offsetof(D3D11_BLEND_DESC, IndependentBlendEnable)},
    { _L("blendfactor1"),           ePARAM_TYPE_FLOAT, (UINT)offsetof(CPUTRenderStateDX11, BlendFactor[0] )},
    { _L("blendfactor2"),           ePARAM_TYPE_FLOAT, (UINT)offsetof(CPUTRenderStateDX11, BlendFactor[1] )},
    { _L("blendfactor3"),           ePARAM_TYPE_FLOAT, (UINT)offsetof(CPUTRenderStateDX11, BlendFactor[2] )},
    { _L("blendfactor4"),           ePARAM_TYPE_FLOAT, (UINT)offsetof(CPUTRenderStateDX11, BlendFactor[3] )},
    { _L("samplemask"),             ePARAM_TYPE_UINT,  (UINT)offsetof(CPUTRenderStateDX11, SampleMask )},
    { _L(""), ePARAM_TYPE_TYPELESS, 0}
};

//-----------------------------------------------------------------------------
static const CPUTRenderStateMapEntry pRenderTargetBlendDescMap[] =
{
    { _L("blendenable"),           ePARAM_TYPE_BOOL,           (UINT)offsetof(D3D11_RENDER_TARGET_BLEND_DESC, BlendEnable )},
    { _L("srcblend"),              ePARAM_TYPE_D3D11_BLEND,    (UINT)offsetof(D3D11_RENDER_TARGET_BLEND_DESC, SrcBlend )},
    { _L("destblend"),             ePARAM_TYPE_D3D11_BLEND,    (UINT)offsetof(D3D11_RENDER_TARGET_BLEND_DESC, DestBlend )},
    { _L("blendop"),               ePARAM_TYPE_D3D11_BLEND_OP, (UINT)offsetof(D3D11_RENDER_TARGET_BLEND_DESC, BlendOp )},
    { _L("srcblendalpha"),         ePARAM_TYPE_D3D11_BLEND,    (UINT)offsetof(D3D11_RENDER_TARGET_BLEND_DESC, SrcBlendAlpha )},
    { _L("destblendalpha"),        ePARAM_TYPE_D3D11_BLEND,    (UINT)offsetof(D3D11_RENDER_TARGET_BLEND_DESC, DestBlendAlpha )},
    { _L("blendopalpha"),          ePARAM_TYPE_D3D11_BLEND_OP, (UINT)offsetof(D3D11_RENDER_TARGET_BLEND_DESC, BlendOpAlpha )},
    { _L("rendertargetwritemask"), ePARAM_TYPE_UCHAR,          (UINT)offsetof(D3D11_RENDER_TARGET_BLEND_DESC, RenderTargetWriteMask )},
    { _L(""), ePARAM_TYPE_TYPELESS, 0}
};

//-----------------------------------------------------------------------------
static const CPUTRenderStateMapEntry pDepthStencilDescMap[] =
{
    { _L("depthenable"),                 ePARAM_TYPE_BOOL,                  (UINT)offsetof(D3D11_DEPTH_STENCIL_DESC, DepthEnable)},
    { _L("depthwritemask"),              ePARAM_TYPE_DEPTH_WRITE_MASK,      (UINT)offsetof(D3D11_DEPTH_STENCIL_DESC, DepthWriteMask)},
    { _L("depthfunc"),                   ePARAM_TYPE_D3D11_COMPARISON_FUNC, (UINT)offsetof(D3D11_DEPTH_STENCIL_DESC, DepthFunc)},
    { _L("StencilEnable"),               ePARAM_TYPE_BOOL,                  (UINT)offsetof(D3D11_DEPTH_STENCIL_DESC, StencilEnable)},
    { _L("StencilReadMask"),             ePARAM_TYPE_UCHAR,                 (UINT)offsetof(D3D11_DEPTH_STENCIL_DESC, StencilWriteMask)},
    { _L("StencilWriteMask"),            ePARAM_TYPE_UCHAR,                 (UINT)offsetof(D3D11_DEPTH_STENCIL_DESC, DepthWriteMask)},
    { _L("FrontFaceStencilFailOp"),      ePARAM_TYPE_D3D11_STENCIL_OP,      (UINT)offsetof(D3D11_DEPTH_STENCIL_DESC, FrontFace.StencilFailOp )},
    { _L("FrontFaceStencilDepthFailOp"), ePARAM_TYPE_D3D11_STENCIL_OP,      (UINT)offsetof(D3D11_DEPTH_STENCIL_DESC, FrontFace.StencilDepthFailOp)},
    { _L("FrontFaceStencilPassOp"),      ePARAM_TYPE_D3D11_STENCIL_OP,      (UINT)offsetof(D3D11_DEPTH_STENCIL_DESC, FrontFace.StencilPassOp)},
    { _L("FrontFaceStencilFunc"),        ePARAM_TYPE_D3D11_COMPARISON_FUNC, (UINT)offsetof(D3D11_DEPTH_STENCIL_DESC, FrontFace.StencilFunc)},
    { _L("BackFaceStencilFailOp"),       ePARAM_TYPE_D3D11_STENCIL_OP,      (UINT)offsetof(D3D11_DEPTH_STENCIL_DESC, BackFace.StencilDepthFailOp)},
    { _L("BackFaceStencilDepthFailOp"),  ePARAM_TYPE_D3D11_STENCIL_OP,      (UINT)offsetof(D3D11_DEPTH_STENCIL_DESC, BackFace.StencilDepthFailOp)},
    { _L("BackFaceStencilPassOp"),       ePARAM_TYPE_D3D11_STENCIL_OP,      (UINT)offsetof(D3D11_DEPTH_STENCIL_DESC, BackFace.StencilPassOp)},
    { _L("BackFaceStencilFunc"),         ePARAM_TYPE_D3D11_COMPARISON_FUNC, (UINT)offsetof(D3D11_DEPTH_STENCIL_DESC, BackFace.StencilFunc)},
    { _L(""), ePARAM_TYPE_TYPELESS, 0}
};

//-----------------------------------------------------------------------------
static const CPUTRenderStateMapEntry pRasterizerDescMap[] =
{
    { _L("FillMode"),              ePARAM_TYPE_D3D11_FILL_MODE, (UINT)offsetof(D3D11_RASTERIZER_DESC, FillMode)},
    { _L("CullMode"),              ePARAM_TYPE_D3D11_CULL_MODE, (UINT)offsetof(D3D11_RASTERIZER_DESC, CullMode)},
    { _L("FrontCounterClockwise"), ePARAM_TYPE_BOOL,            (UINT)offsetof(D3D11_RASTERIZER_DESC, FrontCounterClockwise)},
    { _L("DepthBias"),             ePARAM_TYPE_INT,             (UINT)offsetof(D3D11_RASTERIZER_DESC, DepthBias)},
    { _L("DepthBiasClamp"),        ePARAM_TYPE_FLOAT,           (UINT)offsetof(D3D11_RASTERIZER_DESC, DepthBiasClamp)},
    { _L("SlopeScaledDepthBias"),  ePARAM_TYPE_FLOAT,           (UINT)offsetof(D3D11_RASTERIZER_DESC, SlopeScaledDepthBias)},
    { _L("DepthClipEnable"),       ePARAM_TYPE_BOOL,            (UINT)offsetof(D3D11_RASTERIZER_DESC, DepthClipEnable)},
    { _L("ScissorEnable"),         ePARAM_TYPE_BOOL,            (UINT)offsetof(D3D11_RASTERIZER_DESC, ScissorEnable)},
    { _L("MultisampleEnable"),     ePARAM_TYPE_BOOL,            (UINT)offsetof(D3D11_RASTERIZER_DESC, MultisampleEnable)},
    { _L("AntialiasedLineEnable"), ePARAM_TYPE_BOOL,            (UINT)offsetof(D3D11_RASTERIZER_DESC, AntialiasedLineEnable)},
    { _L(""), ePARAM_TYPE_TYPELESS, 0}
};

//-----------------------------------------------------------------------------
static const CPUTRenderStateMapEntry pSamplerDescMap[] =
{
    { _L("Filter"),         ePARAM_TYPE_D3D11_FILTER,               (UINT)offsetof(D3D11_SAMPLER_DESC, Filter)},
    { _L("AddressU"),       ePARAM_TYPE_D3D11_TEXTURE_ADDRESS_MODE, (UINT)offsetof(D3D11_SAMPLER_DESC, AddressU)},
    { _L("AddressV"),       ePARAM_TYPE_D3D11_TEXTURE_ADDRESS_MODE, (UINT)offsetof(D3D11_SAMPLER_DESC, AddressV)},
    { _L("AddressW"),       ePARAM_TYPE_D3D11_TEXTURE_ADDRESS_MODE, (UINT)offsetof(D3D11_SAMPLER_DESC, AddressW)},
    { _L("MipLODBias"),     ePARAM_TYPE_FLOAT,                      (UINT)offsetof(D3D11_SAMPLER_DESC, MipLODBias)},
    { _L("MaxAnisotropy"),  ePARAM_TYPE_UINT,                       (UINT)offsetof(D3D11_SAMPLER_DESC, MaxAnisotropy)},
    { _L("ComparisonFunc"), ePARAM_TYPE_D3D11_COMPARISON_FUNC,      (UINT)offsetof(D3D11_SAMPLER_DESC, ComparisonFunc)},
    { _L("BorderColor0"),   ePARAM_TYPE_FLOAT,                      (UINT)offsetof(D3D11_SAMPLER_DESC, BorderColor[0])},
    { _L("BorderColor1"),   ePARAM_TYPE_FLOAT,                      (UINT)offsetof(D3D11_SAMPLER_DESC, BorderColor[1])},
    { _L("BorderColor2"),   ePARAM_TYPE_FLOAT,                      (UINT)offsetof(D3D11_SAMPLER_DESC, BorderColor[2])},
    { _L("BorderColor3"),   ePARAM_TYPE_FLOAT,                      (UINT)offsetof(D3D11_SAMPLER_DESC, BorderColor[3])},
    { _L("MinLOD"),         ePARAM_TYPE_FLOAT,                      (UINT)offsetof(D3D11_SAMPLER_DESC, MinLOD)},
    { _L("MaxLOD"),         ePARAM_TYPE_FLOAT,                      (UINT)offsetof(D3D11_SAMPLER_DESC, MaxLOD)},
    { _L(""), ePARAM_TYPE_TYPELESS, 0}
};


//-----------------------------------------------------------------------------
static const StringToIntMapEntry pBlendMap[] = {
    { _L("d3d11_blend_zero"),             D3D11_BLEND_ZERO },
    { _L("d3d11_blend_one"),              D3D11_BLEND_ONE },
    { _L("d3d11_blend_src_color"),        D3D11_BLEND_SRC_COLOR },
    { _L("d3d11_blend_inv_src_color"),    D3D11_BLEND_INV_SRC_COLOR },
    { _L("d3d11_blend_src_alpha"),        D3D11_BLEND_SRC_ALPHA },
    { _L("d3d11_blend_inv_src_alpha"),    D3D11_BLEND_INV_SRC_ALPHA },
    { _L("d3d11_blend_dest_alpha"),       D3D11_BLEND_DEST_ALPHA },
    { _L("d3d11_blend_inv_dest_alpha"),   D3D11_BLEND_INV_DEST_ALPHA },
    { _L("d3d11_blend_dest_color"),       D3D11_BLEND_DEST_COLOR },
    { _L("d3d11_blend_inv_dest_color"),   D3D11_BLEND_INV_DEST_COLOR },
    { _L("d3d11_blend_src_alpha_sat"),    D3D11_BLEND_SRC_ALPHA_SAT },
    { _L("d3d11_blend_blend_factor"),     D3D11_BLEND_BLEND_FACTOR },
    { _L("d3d11_blend_inv_blend_factor"), D3D11_BLEND_INV_BLEND_FACTOR },
    { _L("d3d11_blend_src1_color"),       D3D11_BLEND_SRC1_COLOR },
    { _L("d3d11_blend_inv_src1_color"),   D3D11_BLEND_INV_SRC1_COLOR },
    { _L("d3d11_blend_src1_alpha"),       D3D11_BLEND_SRC1_ALPHA },
    { _L("d3d11_blend_inv_src1_alpha"),   D3D11_BLEND_INV_SRC1_ALPHA },
    { _L(""), -1 }
};

//-----------------------------------------------------------------------------
static const StringToIntMapEntry pBlendOpMap[] = {
    { _L("d3d11_blend_op_add"),          D3D11_BLEND_OP_ADD },
    { _L("d3d11_blend_op_subtract"),     D3D11_BLEND_OP_SUBTRACT },
    { _L("d3d11_blend_op_rev_subtract"), D3D11_BLEND_OP_REV_SUBTRACT },
    { _L("d3d11_blend_op_min"),          D3D11_BLEND_OP_MIN },
    { _L("d3d11_blend_op_max"),          D3D11_BLEND_OP_MAX },
    { _L(""), -1 }
};

//-----------------------------------------------------------------------------
static const StringToIntMapEntry pDepthWriteMaskMap[] = {
    { _L("D3D11_DEPTH_WRITE_MASK_ZERO"), D3D11_DEPTH_WRITE_MASK_ZERO },
    { _L("D3D11_DEPTH_WRITE_MASK_ALL"),  D3D11_DEPTH_WRITE_MASK_ALL },
    { _L(""), -1 }
};

//-----------------------------------------------------------------------------
static const StringToIntMapEntry pComparisonMap[] = {
    { _L("D3D11_COMPARISON_NEVER"),         D3D11_COMPARISON_NEVER },
    { _L("D3D11_COMPARISON_LESS"),          D3D11_COMPARISON_LESS },
    { _L("D3D11_COMPARISON_EQUAL"),         D3D11_COMPARISON_EQUAL },
    { _L("D3D11_COMPARISON_LESS_EQUAL"),    D3D11_COMPARISON_LESS_EQUAL },
    { _L("D3D11_COMPARISON_GREATER"),       D3D11_COMPARISON_GREATER },
    { _L("D3D11_COMPARISON_NOT_EQUAL"),     D3D11_COMPARISON_NOT_EQUAL},
    { _L("D3D11_COMPARISON_GREATER_EQUAL"), D3D11_COMPARISON_GREATER_EQUAL},
    { _L("D3D11_COMPARISON_ALWAYS"),        D3D11_COMPARISON_ALWAYS},
    { _L(""), -1 }
};

//-----------------------------------------------------------------------------
static const StringToIntMapEntry pStencilOpMap[] = {
    { _L("D3D11_STENCIL_OP_KEEP"),     D3D11_STENCIL_OP_KEEP },
    { _L("D3D11_STENCIL_OP_ZERO"),     D3D11_STENCIL_OP_ZERO },
    { _L("D3D11_STENCIL_OP_REPLACE"),  D3D11_STENCIL_OP_REPLACE },
    { _L("D3D11_STENCIL_OP_INCR_SAT"), D3D11_STENCIL_OP_INCR_SAT },
    { _L("D3D11_STENCIL_OP_DECR_SAT"), D3D11_STENCIL_OP_DECR_SAT },
    { _L("D3D11_STENCIL_OP_INVERT"),   D3D11_STENCIL_OP_INVERT },
    { _L("D3D11_STENCIL_OP_INCR"),     D3D11_STENCIL_OP_INCR },
    { _L("D3D11_STENCIL_OP_DECR"),     D3D11_STENCIL_OP_DECR },
    { _L(""), -1 }
};

//-----------------------------------------------------------------------------
static const StringToIntMapEntry pFillModeMap[] = {
    { _L("D3D11_FILL_WIREFRAME"),      D3D11_FILL_WIREFRAME },
    { _L("D3D11_FILL_SOLID"),          D3D11_FILL_SOLID },
    { _L(""), -1 }
};

//-----------------------------------------------------------------------------
static const StringToIntMapEntry pCullModeMap[] = {
    { _L("D3D11_CULL_NONE"),           D3D11_CULL_NONE },
    { _L("D3D11_CULL_FRONT"),          D3D11_CULL_FRONT },
    { _L("D3D11_CULL_BACK"),           D3D11_CULL_BACK },
    { _L(""), -1 }
};

//-----------------------------------------------------------------------------
static const StringToIntMapEntry pFilterMap[] = {
    { _L("D3D11_FILTER_MIN_MAG_MIP_POINT"),                          D3D11_FILTER_MIN_MAG_MIP_POINT },
    { _L("D3D11_FILTER_MIN_MAG_POINT_MIP_LINEAR"),                   D3D11_FILTER_MIN_MAG_POINT_MIP_LINEAR },
    { _L("D3D11_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT"),             D3D11_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT },
    { _L("D3D11_FILTER_MIN_POINT_MAG_MIP_LINEAR"),                   D3D11_FILTER_MIN_POINT_MAG_MIP_LINEAR },
    { _L("D3D11_FILTER_MIN_LINEAR_MAG_MIP_POINT"),                   D3D11_FILTER_MIN_LINEAR_MAG_MIP_POINT },
    { _L("D3D11_FILTER_MIN_LINEAR_MAG_POINT_MIP_LINEAR"),            D3D11_FILTER_MIN_LINEAR_MAG_POINT_MIP_LINEAR },
    { _L("D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT"),                   D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT },
    { _L("D3D11_FILTER_MIN_MAG_MIP_LINEAR"),                         D3D11_FILTER_MIN_MAG_MIP_LINEAR },
    { _L("D3D11_FILTER_ANISOTROPIC"),                                D3D11_FILTER_ANISOTROPIC },
    { _L("D3D11_FILTER_COMPARISON_MIN_MAG_MIP_POINT"),               D3D11_FILTER_COMPARISON_MIN_MAG_MIP_POINT },
    { _L("D3D11_FILTER_COMPARISON_MIN_MAG_POINT_MIP_LINEAR"),        D3D11_FILTER_COMPARISON_MIN_MAG_POINT_MIP_LINEAR },
    { _L("D3D11_FILTER_COMPARISON_MIN_POINT_MAG_LINEAR_MIP_POINT"),  D3D11_FILTER_COMPARISON_MIN_POINT_MAG_LINEAR_MIP_POINT },
    { _L("D3D11_FILTER_COMPARISON_MIN_POINT_MAG_MIP_LINEAR"),        D3D11_FILTER_COMPARISON_MIN_POINT_MAG_MIP_LINEAR },
    { _L("D3D11_FILTER_COMPARISON_MIN_LINEAR_MAG_MIP_POINT"),        D3D11_FILTER_COMPARISON_MIN_LINEAR_MAG_MIP_POINT },
    { _L("D3D11_FILTER_COMPARISON_MIN_LINEAR_MAG_POINT_MIP_LINEAR"), D3D11_FILTER_COMPARISON_MIN_LINEAR_MAG_POINT_MIP_LINEAR },
    { _L("D3D11_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT"),        D3D11_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT },
    { _L("D3D11_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR"),              D3D11_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR },
    { _L("D3D11_FILTER_COMPARISON_ANISOTROPIC"),                     D3D11_FILTER_COMPARISON_ANISOTROPIC },
    // { _L("D3D11_FILTER_TEXT_1BIT"),                                  D3D11_FILTER_TEXT_1BIT }, // DX docs list this, but not in actual structure
    { _L(""), -1 }
};

//-----------------------------------------------------------------------------
static const StringToIntMapEntry pTextureAddressMap[] = {
    { _L("D3D11_TEXTURE_ADDRESS_WRAP"),        D3D11_TEXTURE_ADDRESS_WRAP },
    { _L("D3D11_TEXTURE_ADDRESS_MIRROR"),      D3D11_TEXTURE_ADDRESS_MIRROR },
    { _L("D3D11_TEXTURE_ADDRESS_CLAMP"),       D3D11_TEXTURE_ADDRESS_CLAMP },
    { _L("D3D11_TEXTURE_ADDRESS_BORDER"),      D3D11_TEXTURE_ADDRESS_BORDER },
    { _L("D3D11_TEXTURE_ADDRESS_MIRROR_ONCE"), D3D11_TEXTURE_ADDRESS_MIRROR_ONCE },
    { _L(""), -1 }
};

#endif //_CPUTRENDERSTATEMAPSDX11_H
