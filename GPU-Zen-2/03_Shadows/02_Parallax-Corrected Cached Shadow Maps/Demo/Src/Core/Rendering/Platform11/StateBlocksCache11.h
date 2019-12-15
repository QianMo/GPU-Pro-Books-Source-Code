#ifndef __STATE_BLOCKS_CACHE11_H
#define __STATE_BLOCKS_CACHE11_H

#include <d3d11.h>
#include "../../Util/Cache.h"
#include "Objects11.h"

struct SamplerDesc11 : public D3D11_SAMPLER_DESC
{
  SamplerDesc11(D3D11_FILTER _Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR,
                D3D11_TEXTURE_ADDRESS_MODE _AddressU = D3D11_TEXTURE_ADDRESS_CLAMP,
                D3D11_TEXTURE_ADDRESS_MODE _AddressV = D3D11_TEXTURE_ADDRESS_CLAMP,
                D3D11_TEXTURE_ADDRESS_MODE _AddressW = D3D11_TEXTURE_ADDRESS_CLAMP,
                D3D11_COMPARISON_FUNC _ComparisonFunc = D3D11_COMPARISON_NEVER,
                const Vec4& _BorderColor = Vec4::Zero(),
                float _MipLODBias = 0,
                unsigned _MaxAnisotropy = 16,
                float _MinLOD = -FLT_MAX, 
                float _MaxLOD = FLT_MAX)
  {
    memset(this, 0, sizeof(*this));
    Filter = _Filter;
    AddressU = _AddressU;
    AddressV = _AddressV;
    AddressW =_AddressW;
    ComparisonFunc = _ComparisonFunc;
    _BorderColor.Store(BorderColor);
    MipLODBias = _MipLODBias;
    MaxAnisotropy = _MaxAnisotropy;
    MinLOD = _MinLOD;
    MaxLOD = _MaxLOD;
  }

  typedef D3D11_SAMPLER_DESC Description;
  typedef SamplerState11 CacheEntry;
  typedef ID3D11Device* UserParam;

  static finline void Allocate(const UserParam& Device11, const Description& d, CacheEntry& e)
  {
    ID3D11SamplerState* pState = NULL;
    Device11->CreateSamplerState(&d, &pState);
    e.Init(pState);
    SAFE_RELEASE(pState);
  }
  static finline void Free(const UserParam&, CacheEntry& e)
  {
    e.Clear();
  }
};

struct RasterizerDesc11 : public D3D11_RASTERIZER_DESC
{
  RasterizerDesc11(D3D11_FILL_MODE _FillMode = D3D11_FILL_SOLID,
                   D3D11_CULL_MODE _CullMode = D3D11_CULL_BACK,
                   bool _FrontCounterClockwise = false,
                   int _DepthBias = 0,
                   float _DepthBiasClamp = 0.0f,
                   float _SlopeScaledDepthBias = 0.0f,
                   bool _DepthClipEnable = true,
                   bool _ScissorEnable = false,
                   bool _MultisampleEnable = false,
                   bool _AntialiasedLineEnable = false)
  {
    memset(this, 0, sizeof(*this));
    FillMode = _FillMode;
    CullMode = _CullMode;
    FrontCounterClockwise = _FrontCounterClockwise;
    DepthBias = _DepthBias;
    DepthBiasClamp = _DepthBiasClamp;
    SlopeScaledDepthBias = _SlopeScaledDepthBias;
    DepthClipEnable = _DepthClipEnable;
    ScissorEnable = _ScissorEnable;
    MultisampleEnable = _MultisampleEnable;
    AntialiasedLineEnable = _AntialiasedLineEnable;
  }

  typedef D3D11_RASTERIZER_DESC Description;
  typedef RasterizerState11 CacheEntry;
  typedef ID3D11Device* UserParam;

  static finline void Allocate(const UserParam& Device11, const Description& d, CacheEntry& e)
  {
    ID3D11RasterizerState* pState = NULL;
    Device11->CreateRasterizerState(&d, &pState);
    e.Init(pState);
    SAFE_RELEASE(pState);
  }
  static finline void Free(const UserParam&, CacheEntry& e)
  {
    e.Clear();
  }
};

struct BlendDesc11 : public D3D11_BLEND_DESC
{
  BlendDesc11(bool _AlphaToCoverageEnable = false,
              bool _BlendEnable = false,
              D3D11_BLEND _SrcBlend = D3D11_BLEND_ONE,
              D3D11_BLEND _DestBlend = D3D11_BLEND_ZERO,
              D3D11_BLEND_OP _BlendOp = D3D11_BLEND_OP_ADD,
              D3D11_BLEND _SrcBlendAlpha = D3D11_BLEND_ONE,
              D3D11_BLEND _DestBlendAlpha = D3D11_BLEND_ZERO,
              D3D11_BLEND_OP _BlendOpAlpha = D3D11_BLEND_OP_ADD,
              UINT8 _RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL)
  {
    memset(this, 0, sizeof(*this));
    AlphaToCoverageEnable = _AlphaToCoverageEnable;
    IndependentBlendEnable = FALSE;
    RenderTarget[0].BlendEnable = _BlendEnable;
    RenderTarget[0].SrcBlend = _SrcBlend;
    RenderTarget[0].DestBlend = _DestBlend;
    RenderTarget[0].BlendOp = _BlendOp;
    RenderTarget[0].SrcBlendAlpha = _SrcBlendAlpha;
    RenderTarget[0].DestBlendAlpha = _DestBlendAlpha;
    RenderTarget[0].BlendOpAlpha = _BlendOpAlpha;
    RenderTarget[0].RenderTargetWriteMask = _RenderTargetWriteMask;
  }

  typedef D3D11_BLEND_DESC Description;
  typedef BlendState11 CacheEntry;
  typedef ID3D11Device* UserParam;

  static finline void Allocate(const UserParam& Device11, const Description& d, CacheEntry& e)
  {
    ID3D11BlendState* pState = NULL;
    Device11->CreateBlendState(&d, &pState);
    e.Init(pState);
    SAFE_RELEASE(pState);
  }
  static finline void Free(const UserParam&, CacheEntry& e)
  {
    e.Clear();
  }
};

struct DepthStencilDesc11 : public D3D11_DEPTH_STENCIL_DESC
{
  DepthStencilDesc11(bool _DepthEnable = true,
                     D3D11_DEPTH_WRITE_MASK _DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL,
                     D3D11_COMPARISON_FUNC _DepthFunc = D3D11_COMPARISON_LESS,
                     bool _StencilEnable = false,
                     unsigned char _StencilReadMask = D3D11_DEFAULT_STENCIL_READ_MASK,
                     unsigned char _StencilWriteMask = D3D11_DEFAULT_STENCIL_WRITE_MASK,
                     D3D11_STENCIL_OP _StencilFailOp = D3D11_STENCIL_OP_KEEP,
                     D3D11_STENCIL_OP _StencilDepthFailOp = D3D11_STENCIL_OP_KEEP,
                     D3D11_STENCIL_OP _StencilPassOp = D3D11_STENCIL_OP_KEEP,
                     D3D11_COMPARISON_FUNC _StencilFunc = D3D11_COMPARISON_ALWAYS)
  {
    memset(this, 0, sizeof(*this));
    DepthEnable = _DepthEnable;
    DepthWriteMask = _DepthWriteMask;
    DepthFunc = _DepthFunc;
    StencilEnable = _StencilEnable;
    StencilReadMask = _StencilReadMask;
    StencilWriteMask = _StencilWriteMask;
    FrontFace.StencilFailOp =  _StencilFailOp;
    FrontFace.StencilDepthFailOp = _StencilDepthFailOp;
    FrontFace.StencilPassOp = _StencilPassOp;
    FrontFace.StencilFunc = _StencilFunc;
    BackFace = FrontFace;
  }

  typedef D3D11_DEPTH_STENCIL_DESC Description;
  typedef DepthStencilState11 CacheEntry;
  typedef ID3D11Device* UserParam;

  static finline void Allocate(const UserParam& Device11, const Description& d, CacheEntry& e)
  {
    ID3D11DepthStencilState* pState = NULL;
    Device11->CreateDepthStencilState(&d, &pState);
    e.Init(pState);
    SAFE_RELEASE(pState);
  }
  static finline void Free(const UserParam&, CacheEntry& e)
  {
    e.Clear();
  }
};

typedef ConcurrentCache<SamplerDesc11> SamplerCache11;
typedef ConcurrentCache<RasterizerDesc11> RasterizerCache11;
typedef ConcurrentCache<BlendDesc11> BlendCache11;
typedef ConcurrentCache<DepthStencilDesc11> DepthStencilCache11;

#endif //#ifndef __STATE_BLOCKS_CACHE11_H
