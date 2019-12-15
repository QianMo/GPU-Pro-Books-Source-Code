#ifndef __RENDER_CONTEXT11_H
#define __RENDER_CONTEXT11_H

#include "StateBlocksCache11.h"
#include "Objects11.h"

enum RenderState
{
  FirstRasterizerRenderState = 0,
  RS_FILL_MODE = FirstRasterizerRenderState,
  RS_CULL_MODE,
  RS_FRONT_COUNTER_CLOCKWISE,
  RS_DEPTH_BIAS,
  RS_DEPTH_BIAS_CLAMP,
  RS_SLOPE_SCALED_DEPTH_BIAS,
  RS_DEPTH_CLIP_ENABLE,
  RS_SCISSOR_ENABLE,
  RS_MULTISAMPLE_ENABLE,
  RS_ANTIALIASED_LINE_ENABLE,
  LastRasterizerRenderState,

  FirstBlendRenderState = LastRasterizerRenderState,
  RS_ALPHA_TO_COVERAGE_ENABLE = FirstBlendRenderState,
  RS_BLEND_ENABLE,
  RS_SRC_BLEND,
  RS_DEST_BLEND,
  RS_BLEND_OP,
  RS_SRC_BLEND_ALPHA,
  RS_DEST_BLEND_ALPHA,
  RS_BLEND_OP_ALPHA,
  RS_RENDER_TARGET_WRITE_MASK,
  LastBlendRenderState,

  FirstDepthStencilRenderState = LastBlendRenderState,
  RS_DEPTH_ENABLE = FirstDepthStencilRenderState,
  RS_DEPTH_WRITE_MASK,
  RS_DEPTH_FUNC,
  RS_STENCIL_ENABLE,
  RS_STENCIL_READ_MASK,
  RS_STENCIL_WRITE_MASK,
  RS_STENCIL_FAIL_OP,
  RS_STENCIL_DEPTH_FAIL_OP,
  RS_STENCIL_PASS_OP,
  RS_STENCIL_FUNC,
  RS_BACKFACE_STENCIL_FAIL_OP,
  RS_BACKFACE_STENCIL_DEPTH_FAIL_OP,
  RS_BACKFACE_STENCIL_PASS_OP,
  RS_BACKFACE_STENCIL_FUNC,
  LastDepthStencilRenderState,

  RS_STENCIL_REF,
  RS_STENCIL_FRONT_AND_BACK_FAIL_OP,
  RS_STENCIL_FRONT_AND_BACK_DEPTH_FAIL_OP,
  RS_STENCIL_FRONT_AND_BACK_PASS_OP,
  RS_STENCIL_FRONT_AND_BACK_FUNC,
};

template<class T, unsigned c_FirstState, unsigned c_LastState> class StateBlock11
{
public:
  StateBlock11()
  {
    Reset();
  }
  finline void Set(const T& a)
  {
    m_Desc = a;
    m_BitMask = c_AllStatesMask;
  }
  finline const T& Get() const
  {
    _ASSERT(m_BitMask==c_AllStatesMask);
    return m_Desc;
  }
  finline bool operator== (const StateBlock11<T,c_FirstState,c_LastState>& a) const
  {
    return m_BitMask==a.m_BitMask && !memcmp(&m_Desc, &a.m_Desc, sizeof(T));
  }

protected:
  T m_Desc;
  T m_Mask;
  static const unsigned c_AllStatesMask = (1<<(c_LastState - c_FirstState)) - 1;
  unsigned m_BitMask;

  template<unsigned N, class T, class V> finline void Set(T& d, T& m, V v)
  {
    d = (T)v; memset(&m, 0, sizeof(m));
    m_BitMask |= (1<<(N - c_FirstState));
  }
  template<unsigned N, class T> finline void Unset(T& d, T& m)
  {
    memset(&d,  0, sizeof(d));
    memset(&m, ~0, sizeof(m));
    m_BitMask &= ~(1<<(N - c_FirstState));
  }
  finline void Reset()
  {
    memset(&m_Desc,  0, sizeof(m_Desc));
    memset(&m_Mask, ~0, sizeof(m_Mask));
    m_BitMask = 0;
  }
  finline void ApplyTo(StateBlock11<T, c_FirstState, c_LastState>& Dst) const
  {
    if(m_BitMask==c_AllStatesMask)
    {
      Dst.m_Desc = m_Desc;
      Dst.m_BitMask = c_AllStatesMask;
    }
    else if(m_BitMask!=0)
    {
      static_assert(!(sizeof(T)%sizeof(unsigned)), "size of the parameter must be multiple of 4");
      const unsigned* __restrict SrcDesc = (unsigned*)&m_Desc;
      const unsigned* __restrict SrcMask = (unsigned*)&m_Mask;
      unsigned* __restrict DstDesc = (unsigned*)&Dst.m_Desc;
      unsigned* __restrict DstMask = (unsigned*)&Dst.m_Mask;
      const int n = sizeof(T)/sizeof(unsigned);
      for(int i=0; i<n; ++i)
      {
        DstDesc[i] = (SrcDesc[i] | (DstDesc[i] & SrcMask[i]));
        DstMask[i] &= SrcMask[i];
      }
      Dst.m_BitMask |= m_BitMask;
    }
  }
};

typedef StateBlock11<RasterizerDesc11, FirstRasterizerRenderState, LastRasterizerRenderState> RasterizerStateBlock11;
typedef StateBlock11<BlendDesc11, FirstBlendRenderState, LastBlendRenderState> BlendStateBlock11;
typedef StateBlock11<DepthStencilDesc11, FirstDepthStencilRenderState, LastDepthStencilRenderState> DepthStencilStateBlock11;
typedef std::pair<const DepthStencil11*, const ShaderResource11*> DepthStencilPair;

#define RCDATA_MEMBERS(DECL) \
  DECL(D3D11_VIEWPORT, Viewport) \
  DECL(const PixelShader11*, PixelShader) \
  DECL(const GeometryShader11*, GeometryShader) \
  DECL(const VertexShader11*, VertexShader) \
  DECL(const ComputeShader11*, ComputeShader) \
  DECL(const HullShader11*, HullShader) \
  DECL(const DomainShader11*, DomainShader) \
  DECL(const InputLayout11*, InputLayout) \
  DECL(DepthStencilPair, DepthStencil) \
  DECL(const Buffer11*, IndexBuffer) \
  DECL(const RasterizerState11*, RasterizerState) \
  DECL(const BlendState11*, BlendState) \
  DECL(const DepthStencilState11*, DepthStencilState) \
  DECL(size_t, IndexBufferOffset) \
  DECL(D3D11_PRIMITIVE_TOPOLOGY, PrimitiveTopology) \
  DECL(unsigned char, StencilRef)

#define RCDATA_STRUCT(t, n) t n;
struct RenderContextDataBlock { RCDATA_MEMBERS(RCDATA_STRUCT) };

#define RCDATA_ENUM(t, n) RCData_##n,
enum { RCDATA_MEMBERS(RCDATA_ENUM) RCData_MembersCnt };

#define RCDATA_IS_SET(a, b) finline bool Is##b##Set() const { return (m_BitMask&(1<<RCData_##b))!=0; }
#define RCDATA_GET(a, b) finline a Get##b() const { _ASSERT(Is##b##Set()); return m_Desc.##b; }
#define RCDATA_SET(a, b) finline void Set##b(a v) { Set<RCData_##b>(m_Desc.##b, m_Mask.##b, v); } finline void Bind##b(a v) { Set<RCData_##b>(m_Desc.##b, m_Mask.##b, v); }
#define RCDATA_UNSET(a, b) finline void Unset##b() { Unset<RCData_##b>(m_Desc.##b, m_Mask.##b); } finline void Unbind##b() { Unset<RCData_##b>(m_Desc.##b, m_Mask.##b); }

class RCData : public StateBlock11<RenderContextDataBlock, 0, RCData_MembersCnt>
{
public:
  RCDATA_MEMBERS(RCDATA_IS_SET)
  RCDATA_MEMBERS(RCDATA_GET)
  RCDATA_MEMBERS(RCDATA_SET)
  RCDATA_MEMBERS(RCDATA_UNSET)
};

template<class T, unsigned _N> class RenderContextArray
{
public:
  static const unsigned N = _N;
  static const unsigned M = (_N + sizeof(unsigned)*8 - 1)/(sizeof(unsigned)*8);

  RenderContextArray()
  {
    memset(m_BitMask, 0, sizeof(m_BitMask));
  }
  finline void Set(unsigned i, const T& a)
  {
    _ASSERT(i<N);
    m_Objects[i] = a;
    m_BitMask[i>>5] |= (1<<(i&31));
  }
  finline void Unset(unsigned i)
  {
//    memset(&m_Objects[i], 0, sizeof(T));
    m_BitMask[i>>5] &= ~(1<<(i&31));
  }
  finline bool IsSet(unsigned i) const
  {
    _ASSERT(i<N);
    return (m_BitMask[i>>5]&(1<<(i&31)))!=0;
  }
  finline const T& Get(unsigned i) const
  {
    _ASSERT(IsSet(i));
    return m_Objects[i];
  }
  finline bool IsSet() const
  {
    unsigned m = m_BitMask[0];
    for(unsigned i=1; i<M; ++i)
      m |= m_BitMask[i];
    return m!=0;
  }
  finline bool operator== (const RenderContextArray<T,_N>& a) const
  {
    for(unsigned i=0; i<N; ++i)
      if(IsSet(i) && m_Objects[i]!=a.m_Objects[i])
        return false;
    return true;
  }

protected:
  T m_Objects[N];
  unsigned m_BitMask[M];

  finline void Reset()
  {
    memset(m_BitMask, 0, sizeof(m_BitMask));
  }
  finline void ApplyTo(RenderContextArray<T, N>& Dst) const
  {
    if(IsSet())
    {
      for(unsigned i=0; i<N; ++i)
        if(IsSet(i)) Dst.m_Objects[i] = m_Objects[i];
      Dst.m_BitMask[0] |= m_BitMask[0];
      for(unsigned i=1; i<M; ++i)
        Dst.m_BitMask[i] |= m_BitMask[i];
    }
  }
};

class PSResources11 : public RenderContextArray<const ShaderResource11*, 16> { };
class GSResources11 : public RenderContextArray<const ShaderResource11*, 8> { };
class VSResources11 : public RenderContextArray<const ShaderResource11*, 8> { };
class CSResources11 : public RenderContextArray<const ShaderResource11*, 12> { };
class HSResources11 : public RenderContextArray<const ShaderResource11*, 1> { };
class DSResources11 : public RenderContextArray<const ShaderResource11*, 3> { };
class RenderTargets11 : public RenderContextArray<std::pair<const RenderTarget11*, const ShaderResource11*>, 4> { };
class UnorderedAccessResources11 : public RenderContextArray<std::pair<const UnorderedAccessResource11*, const ShaderResource11*>, 4> { };
class VertexBuffers11 : public RenderContextArray<std::pair<const Buffer11*, size_t>, 4> { };
class PSSamplers : public RenderContextArray<const SamplerState11*, PSResources11::N> { };
class GSSamplers : public RenderContextArray<const SamplerState11*, GSResources11::N> { };
class VSSamplers : public RenderContextArray<const SamplerState11*, VSResources11::N> { };
class CSSamplers : public RenderContextArray<const SamplerState11*, CSResources11::N> { };
class HSSamplers : public RenderContextArray<const SamplerState11*, HSResources11::N> { };
class DSSamplers : public RenderContextArray<const SamplerState11*, DSResources11::N> { };

#define RENDER_STATE_I(b, s, f) template<> finline void SetRenderState<s>(int v)    { b##Block11::Set<s>(b##Block11::m_Desc.##f, b##Block11::m_Mask.##f, v); RCData::Unset##b(); }
#define RENDER_STATE_F(b, s, f) template<> finline void SetRenderStateF<s>(float v) { b##Block11::Set<s>(b##Block11::m_Desc.##f, b##Block11::m_Mask.##f, v); RCData::Unset##b(); }
#define RENDER_STATE_B(b, s, f) template<> finline void SetRenderStateB<s>(bool v)  { b##Block11::Set<s>(b##Block11::m_Desc.##f, b##Block11::m_Mask.##f, v); RCData::Unset##b(); }
#define UNBIND_RESOURCE(a) finline void Unbind(const a##11* p##a) { if(Is##a##Set() && Get##a()==p##a) Unbind##a(); }

class RenderContext11 : public RasterizerStateBlock11, public BlendStateBlock11, public DepthStencilStateBlock11,
                        public PSResources11, public GSResources11, public VSResources11, public CSResources11,
                        public RenderTargets11, public UnorderedAccessResources11, public VertexBuffers11,
                        public PSSamplers, public GSSamplers, public VSSamplers, public CSSamplers,
                        public HSResources11, public DSResources11, public HSSamplers, public DSSamplers,
                        public RCData
{
public:
  template<RenderState> finline void SetRenderState(int)    { static_assert(false, "not implemented"); }
  template<RenderState> finline void SetRenderStateF(float) { static_assert(false, "not implemented"); }
  template<RenderState> finline void SetRenderStateB(bool)  { static_assert(false, "not implemented"); }

  RENDER_STATE_I(RasterizerState, RS_FILL_MODE, FillMode);
  RENDER_STATE_I(RasterizerState, RS_CULL_MODE, CullMode);
  RENDER_STATE_B(RasterizerState, RS_FRONT_COUNTER_CLOCKWISE, FrontCounterClockwise);
  RENDER_STATE_I(RasterizerState, RS_DEPTH_BIAS, DepthBias);
  RENDER_STATE_F(RasterizerState, RS_DEPTH_BIAS_CLAMP, DepthBiasClamp);
  RENDER_STATE_F(RasterizerState, RS_SLOPE_SCALED_DEPTH_BIAS, SlopeScaledDepthBias);
  RENDER_STATE_B(RasterizerState, RS_DEPTH_CLIP_ENABLE, DepthClipEnable);
  RENDER_STATE_B(RasterizerState, RS_SCISSOR_ENABLE, ScissorEnable);
  RENDER_STATE_B(RasterizerState, RS_MULTISAMPLE_ENABLE, MultisampleEnable);
  RENDER_STATE_B(RasterizerState, RS_ANTIALIASED_LINE_ENABLE, AntialiasedLineEnable);

  RENDER_STATE_B(BlendState, RS_ALPHA_TO_COVERAGE_ENABLE, AlphaToCoverageEnable);
  RENDER_STATE_B(BlendState, RS_BLEND_ENABLE, RenderTarget[0].BlendEnable);
  RENDER_STATE_I(BlendState, RS_SRC_BLEND, RenderTarget[0].SrcBlend);
  RENDER_STATE_I(BlendState, RS_DEST_BLEND, RenderTarget[0].DestBlend);
  RENDER_STATE_I(BlendState, RS_BLEND_OP, RenderTarget[0].BlendOp);
  RENDER_STATE_I(BlendState, RS_SRC_BLEND_ALPHA, RenderTarget[0].SrcBlendAlpha);
  RENDER_STATE_I(BlendState, RS_DEST_BLEND_ALPHA, RenderTarget[0].DestBlendAlpha);
  RENDER_STATE_I(BlendState, RS_BLEND_OP_ALPHA, RenderTarget[0].BlendOpAlpha);
  RENDER_STATE_I(BlendState, RS_RENDER_TARGET_WRITE_MASK, RenderTarget[0].RenderTargetWriteMask);

  RENDER_STATE_B(DepthStencilState, RS_DEPTH_ENABLE, DepthEnable);
  RENDER_STATE_I(DepthStencilState, RS_DEPTH_WRITE_MASK, DepthWriteMask);
  RENDER_STATE_I(DepthStencilState, RS_DEPTH_FUNC, DepthFunc);
  RENDER_STATE_B(DepthStencilState, RS_STENCIL_ENABLE, StencilEnable);
  RENDER_STATE_I(DepthStencilState, RS_STENCIL_READ_MASK, StencilReadMask);
  RENDER_STATE_I(DepthStencilState, RS_STENCIL_WRITE_MASK, StencilWriteMask);
  RENDER_STATE_I(DepthStencilState, RS_STENCIL_FAIL_OP, FrontFace.StencilFailOp);
  RENDER_STATE_I(DepthStencilState, RS_STENCIL_DEPTH_FAIL_OP, FrontFace.StencilDepthFailOp);
  RENDER_STATE_I(DepthStencilState, RS_STENCIL_PASS_OP, FrontFace.StencilPassOp);
  RENDER_STATE_I(DepthStencilState, RS_STENCIL_FUNC, FrontFace.StencilFunc);
  RENDER_STATE_I(DepthStencilState, RS_BACKFACE_STENCIL_FAIL_OP, BackFace.StencilFailOp);
  RENDER_STATE_I(DepthStencilState, RS_BACKFACE_STENCIL_DEPTH_FAIL_OP, BackFace.StencilDepthFailOp);
  RENDER_STATE_I(DepthStencilState, RS_BACKFACE_STENCIL_PASS_OP, BackFace.StencilPassOp);
  RENDER_STATE_I(DepthStencilState, RS_BACKFACE_STENCIL_FUNC, BackFace.StencilFunc);

  template<> finline void SetRenderState<RS_STENCIL_REF>(int i) { SetStencilRef((unsigned char)i); }
  template<> finline void SetRenderState<RS_STENCIL_FRONT_AND_BACK_FAIL_OP>(int i) { SetRenderState<RS_STENCIL_FAIL_OP>(i); SetRenderState<RS_BACKFACE_STENCIL_FAIL_OP>(i); }
  template<> finline void SetRenderState<RS_STENCIL_FRONT_AND_BACK_DEPTH_FAIL_OP>(int i) { SetRenderState<RS_STENCIL_DEPTH_FAIL_OP>(i); SetRenderState<RS_BACKFACE_STENCIL_DEPTH_FAIL_OP>(i); }
  template<> finline void SetRenderState<RS_STENCIL_FRONT_AND_BACK_PASS_OP>(int i) { SetRenderState<RS_STENCIL_PASS_OP>(i); SetRenderState<RS_BACKFACE_STENCIL_PASS_OP>(i); }
  template<> finline void SetRenderState<RS_STENCIL_FRONT_AND_BACK_FUNC>(int i) { SetRenderState<RS_STENCIL_FUNC>(i); SetRenderState<RS_BACKFACE_STENCIL_FUNC>(i); }

  finline void BindPS(unsigned Slot, const ShaderResource11* pRes) { _ASSERT(pRes!=NULL); PSResources11::Set(Slot, pRes); }
  finline void BindGS(unsigned Slot, const ShaderResource11* pRes) { _ASSERT(pRes!=NULL); GSResources11::Set(Slot, pRes); }
  finline void BindVS(unsigned Slot, const ShaderResource11* pRes) { _ASSERT(pRes!=NULL); VSResources11::Set(Slot, pRes); }
  finline void BindCS(unsigned Slot, const ShaderResource11* pRes) { _ASSERT(pRes!=NULL); CSResources11::Set(Slot, pRes); }
  finline void BindHS(unsigned Slot, const ShaderResource11* pRes) { _ASSERT(pRes!=NULL); HSResources11::Set(Slot, pRes); }
  finline void BindDS(unsigned Slot, const ShaderResource11* pRes) { _ASSERT(pRes!=NULL); DSResources11::Set(Slot, pRes); }
  finline void UnbindPS(unsigned Slot) { PSResources11::Unset(Slot); }
  finline void UnbindGS(unsigned Slot) { GSResources11::Unset(Slot); }
  finline void UnbindVS(unsigned Slot) { VSResources11::Unset(Slot); }
  finline void UnbindCS(unsigned Slot) { CSResources11::Unset(Slot); }
  finline void UnbindHS(unsigned Slot) { HSResources11::Unset(Slot); }
  finline void UnbindDS(unsigned Slot) { DSResources11::Unset(Slot); }
  finline void BindRT(unsigned Slot, const RenderTarget11* pRT, const ShaderResource11* pRes) { _ASSERT(pRT!=NULL); RenderTargets11::Set(Slot, std::pair<const RenderTarget11*, const ShaderResource11*>(pRT, pRes)); }
  finline void UnbindRT(unsigned Slot) { RenderTargets11::Unset(Slot); }
  finline void BindUA(unsigned Slot, const UnorderedAccessResource11* pUA, const ShaderResource11* pRes) { _ASSERT(pRes!=NULL); UnorderedAccessResources11::Set(Slot, std::pair<const UnorderedAccessResource11*, const ShaderResource11*>(pUA, pRes)); }
  finline void UnbindUA(unsigned Slot) { UnorderedAccessResources11::Unset(Slot); }
  finline void BindVertexBuffer(unsigned Slot, const Buffer11* pRes, size_t Offset) { _ASSERT(pRes!=NULL); VertexBuffers11::Set(Slot, std::pair<const Buffer11*, size_t>(pRes, Offset)); }
  finline void UnbindVertexBuffer(unsigned Slot) { VertexBuffers11::Unset(Slot); }
  finline void BindDepthStencil(const DepthStencil11* pDS, const ShaderResource11* pRes) { SetDepthStencil(std::pair<const DepthStencil11*, const ShaderResource11*>(pDS, pRes)); }

  finline void BindIndexBuffer(const Buffer11* pRes, size_t Offset) { _ASSERT(pRes!=NULL); SetIndexBuffer(pRes); SetIndexBufferOffset(Offset); }
  finline void UnbindIndexBuffer() { UnsetIndexBuffer(); UnsetIndexBufferOffset(); }
  finline void BindPS(const PixelShader11* pShader)    { _ASSERT(pShader!=NULL); SetPixelShader(pShader); }
  finline void BindGS(const GeometryShader11* pShader) { _ASSERT(pShader!=NULL); SetGeometryShader(pShader); }
  finline void BindVS(const VertexShader11* pShader)   { _ASSERT(pShader!=NULL); SetVertexShader(pShader); }
  finline void BindCS(const ComputeShader11* pShader)  { _ASSERT(pShader!=NULL); SetComputeShader(pShader); }
  finline void BindHS(const HullShader11* pShader)     { _ASSERT(pShader!=NULL); SetHullShader(pShader); }
  finline void BindDS(const DomainShader11* pShader)   { _ASSERT(pShader!=NULL); SetDomainShader(pShader); }

  finline void SetSamplerPS(unsigned Slot, const SamplerState11* pState) { _ASSERT(pState!=NULL); PSSamplers::Set(Slot, pState); }
  finline void SetSamplerGS(unsigned Slot, const SamplerState11* pState) { _ASSERT(pState!=NULL); GSSamplers::Set(Slot, pState); }
  finline void SetSamplerVS(unsigned Slot, const SamplerState11* pState) { _ASSERT(pState!=NULL); VSSamplers::Set(Slot, pState); }
  finline void SetSamplerCS(unsigned Slot, const SamplerState11* pState) { _ASSERT(pState!=NULL); CSSamplers::Set(Slot, pState); }
  finline void SetSamplerHS(unsigned Slot, const SamplerState11* pState) { _ASSERT(pState!=NULL); HSSamplers::Set(Slot, pState); }
  finline void SetSamplerDS(unsigned Slot, const SamplerState11* pState) { _ASSERT(pState!=NULL); DSSamplers::Set(Slot, pState); }
  finline void UnsetSamplerPS(unsigned Slot) { PSSamplers::Unset(Slot); }
  finline void UnsetSamplerGS(unsigned Slot) { GSSamplers::Unset(Slot); }
  finline void UnsetSamplerVS(unsigned Slot) { VSSamplers::Unset(Slot); }
  finline void UnsetSamplerCS(unsigned Slot) { CSSamplers::Unset(Slot); }
  finline void UnsetSamplerHS(unsigned Slot) { HSSamplers::Unset(Slot); }
  finline void UnsetSamplerDS(unsigned Slot) { DSSamplers::Unset(Slot); }

  void ApplyTo(RenderContext11& c) const
  {
    RasterizerStateBlock11::ApplyTo(c);
    BlendStateBlock11::ApplyTo(c);
    DepthStencilStateBlock11::ApplyTo(c);
    PSResources11::ApplyTo(c);
    GSResources11::ApplyTo(static_cast<GSResources11&>(c));
    VSResources11::ApplyTo(static_cast<VSResources11&>(c));
    CSResources11::ApplyTo(c);
    HSResources11::ApplyTo(static_cast<HSResources11&>(c));
    DSResources11::ApplyTo(static_cast<DSResources11&>(c));
    RenderTargets11::ApplyTo(c);
    UnorderedAccessResources11::ApplyTo(c);
    VertexBuffers11::ApplyTo(c);
    RCData::ApplyTo(c);
    PSSamplers::ApplyTo(c);
    GSSamplers::ApplyTo(static_cast<GSSamplers&>(c));
    VSSamplers::ApplyTo(static_cast<VSSamplers&>(c));
    CSSamplers::ApplyTo(c);
    HSSamplers::ApplyTo(static_cast<HSSamplers&>(c));
    DSSamplers::ApplyTo(static_cast<DSSamplers&>(c));
  }
  void Reset()
  {
    RasterizerStateBlock11::Reset();
    BlendStateBlock11::Reset();
    DepthStencilStateBlock11::Reset();
    PSResources11::Reset();
    GSResources11::Reset();
    VSResources11::Reset();
    CSResources11::Reset();
    HSResources11::Reset();
    DSResources11::Reset();
    RenderTargets11::Reset();
    UnorderedAccessResources11::Reset();
    VertexBuffers11::Reset();
    RCData::Reset();
    PSSamplers::Reset();
    GSSamplers::Reset();
    VSSamplers::Reset();
    CSSamplers::Reset();
    HSSamplers::Reset();
    DSSamplers::Reset();
  }
  finline bool operator== (const RenderContext11& a) const
  {
    return RasterizerStateBlock11::operator==(a) &&
           BlendStateBlock11::operator==(a) &&
           DepthStencilStateBlock11::operator==(a) &&
           PSResources11::operator==(a) &&
           GSResources11::operator==(static_cast<const GSResources11&>(a)) &&
           VSResources11::operator==(static_cast<const VSResources11&>(a)) &&
           CSResources11::operator==(a) &&
           HSResources11::operator==(static_cast<const HSResources11&>(a)) &&
           DSResources11::operator==(static_cast<const DSResources11&>(a)) &&
           RenderTargets11::operator==(a) &&
           UnorderedAccessResources11::operator==(a) &&
           VertexBuffers11::operator==(a) &&
           RCData::operator==(a) &&
           PSSamplers::operator==(a) &&
           GSSamplers::operator==(static_cast<const GSSamplers&>(a)) &&
           VSSamplers::operator==(static_cast<const VSSamplers&>(a)) &&
           CSSamplers::operator==(a) &&
           HSSamplers::operator==(static_cast<const HSSamplers&>(a)) &&
           DSSamplers::operator==(static_cast<const DSSamplers&>(a));
  }
  finline void Unbind(const ShaderResource11* pRes)
  {
    UnbindItem<PSResources11>(pRes);
    UnbindItem<GSResources11>(pRes);
    UnbindItem<VSResources11>(pRes);
    UnbindItem<CSResources11>(pRes);
    UnbindItem<HSResources11>(pRes);
    UnbindItem<DSResources11>(pRes);
  }
  finline void Unbind(const Buffer11* pBuffer)
  {
    UnbindItemPairFirst<VertexBuffers11>(pBuffer);
    if(IsIndexBufferSet() && GetIndexBuffer()==pBuffer)
      UnbindIndexBuffer();
  }
  finline void Unbind(const RenderTarget11* pRenderTarget)
  {
    UnbindItemPairFirst<RenderTargets11>(pRenderTarget);
  }
  finline void Unbind(const UnorderedAccessResource11* pUA)
  {
    UnbindItemPairFirst<UnorderedAccessResources11>(pUA);
  }
  finline void Unbind(const DepthStencil11* pDepthStencil)
  {
    if(IsDepthStencilSet() && GetDepthStencil().first==pDepthStencil)
      UnbindDepthStencil();
  }
  UNBIND_RESOURCE(PixelShader);
  UNBIND_RESOURCE(GeometryShader);
  UNBIND_RESOURCE(VertexShader);
  UNBIND_RESOURCE(ComputeShader);
  UNBIND_RESOURCE(HullShader);
  UNBIND_RESOURCE(DomainShader);
  UNBIND_RESOURCE(InputLayout);
  UNBIND_RESOURCE(RasterizerState);
  UNBIND_RESOURCE(BlendState);
  UNBIND_RESOURCE(DepthStencilState);

protected:
  template<class T, class P> finline void UnbindItem(const P& a)
  {
    for(unsigned i=0; i<T::N; ++i)
      if(T::IsSet(i) && T::Get(i)==a)
        T::Unset(i);
  }
  template<class T, class P> finline void UnbindItemPairFirst(const P& a)
  {
    for(unsigned i=0; i<T::N; ++i)
      if(T::IsSet(i) && T::Get(i).first==a)
        T::Unset(i);
  }
};

#endif //#ifndef __RENDER_CONTEXT11_H
