/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_D3DX_EFFECTS_11
#define BE_GRAPHICS_D3DX_EFFECTS_11

#include "beGraphics.h"
#include <D3DX11Effect.h>
#include "beD3D11.h"

namespace beGraphics
{

namespace DX11
{

// DirectX 11 types redefined.
namespace API
{

/// D3DX 11 effect.
typedef ID3DX11Effect Effect;

/// D3DX 11 effect group.
typedef ID3DX11EffectGroup EffectGroup;
/// D3DX 11 effect technique.
typedef ID3DX11EffectTechnique EffectTechnique;
/// D3DX 11 effect pass.
typedef ID3DX11EffectPass EffectPass;

/// D3DX 11 effect technique desc.
typedef D3DX11_TECHNIQUE_DESC EffectTechniqueDesc;
/// D3DX 11 effect pass desc.
typedef D3DX11_PASS_DESC EffectPassDesc;

/// D3DX 11 effect variable.
typedef ID3DX11EffectVariable EffectVariable;
/// D3DX 11 effect variable type.
typedef ID3DX11EffectType EffectType;

/// D3DX 11 effect variable.
typedef ID3DX11EffectScalarVariable EffectScalar;
/// D3DX 11 effect variable.
typedef ID3DX11EffectVectorVariable EffectVector;
/// D3DX 11 effect variable.
typedef ID3DX11EffectMatrixVariable EffectMatrix;

/// D3DX 11 effect variable.
typedef ID3DX11EffectStringVariable EffectString;

/// D3DX 11 effect variable.
typedef ID3DX11EffectShaderResourceVariable EffectShaderResource;
/// D3DX 11 effect variable.
typedef ID3DX11EffectConstantBuffer EffectConstantBuffer;
/// D3DX 11 effect variable.
typedef ID3DX11EffectRenderTargetViewVariable EffectRenderTarget;
/// D3DX 11 effect variable.
typedef ID3DX11EffectDepthStencilViewVariable EffectDepthStencil;
/// D3DX 11 effect variable.
typedef ID3DX11EffectUnorderedAccessViewVariable EffectUnorderedAccessView;

/// D3DX 11 effect variable.
typedef ID3DX11EffectSamplerVariable EffectSamplerState;
/// D3DX 11 effect variable.
typedef ID3DX11EffectRasterizerVariable EffectRasterizerState;
/// D3DX 11 effect variable.
typedef ID3DX11EffectDepthStencilVariable EffectDepthStencilState;
/// D3DX 11 effect variable.
typedef ID3DX11EffectBlendVariable EffectBlendState;

/// D3DX 11 effect variable.
typedef ID3DX11EffectInterfaceVariable EffectInterface;
/// D3DX 11 effect variable.
typedef ID3DX11EffectClassInstanceVariable EffectClassInstance;

/// D3DX 11 effect variable desc.
typedef D3DX11_EFFECT_VARIABLE_DESC EffectVariableDesc;
/// D3DX 11 effect type desc.
typedef D3DX11_EFFECT_TYPE_DESC EffectTypeDesc;

} // namespace

//using namespace API;

} // namespace

} // namespace

#endif