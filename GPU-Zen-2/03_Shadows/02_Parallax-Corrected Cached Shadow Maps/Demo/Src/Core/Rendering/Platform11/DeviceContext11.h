#ifndef __DEVICE_CONTEXT11
#define __DEVICE_CONTEXT11

#include <d3d11.h>
#include "RenderContext11.h"
#include "ConstantBuffersPool11.h"

class DeviceContext11 : public RenderContext11
{
public:
  static const unsigned VERTEX_INPUT_SLOTS = 16;
  static const unsigned PS_RESOURCES_SLOTS = 16;
  static const unsigned GS_RESOURCES_SLOTS = 8;
  static const unsigned VS_RESOURCES_SLOTS = 8;
  static const unsigned CS_RESOURCES_SLOTS = 12;
  static const unsigned HS_RESOURCES_SLOTS = 1;
  static const unsigned DS_RESOURCES_SLOTS = 3;
  static const unsigned PS_SAMPLERS_SLOTS = PS_RESOURCES_SLOTS;
  static const unsigned GS_SAMPLERS_SLOTS = GS_RESOURCES_SLOTS;
  static const unsigned VS_SAMPLERS_SLOTS = VS_RESOURCES_SLOTS;
  static const unsigned CS_SAMPLERS_SLOTS = CS_RESOURCES_SLOTS;
  static const unsigned HS_SAMPLERS_SLOTS = HS_RESOURCES_SLOTS;
  static const unsigned DS_SAMPLERS_SLOTS = DS_RESOURCES_SLOTS;
  static const unsigned RENDER_TARGETS = 4;
  static const unsigned UNORDERED_ACCESS_RESOURCES = 4;
  static const unsigned PS_CONSTANT_BUFFER_SLOTS = 4;
  static const unsigned GS_CONSTANT_BUFFER_SLOTS = 4;
  static const unsigned VS_CONSTANT_BUFFER_SLOTS = 4;
  static const unsigned CS_CONSTANT_BUFFER_SLOTS = 4;
  static const unsigned HS_CONSTANT_BUFFER_SLOTS = 4;
  static const unsigned DS_CONSTANT_BUFFER_SLOTS = 4;
  static const unsigned VIEWPORTS = 8;

  HRESULT Init(ID3D11Device*, ID3D11DeviceContext*);
  void Clear();
  void InvalidateContextCache();
  ID3D11DeviceContext* FlushToDevice();
  ID3D11DeviceContext* DoNotFlushToDevice() const { return m_Context; }
  finline ConstantBuffersPool11& GetConstantBuffers() { return m_ConstantBuffers; }

  finline void ClearRenderTarget(unsigned Slot, const Vec4& ClearColor)
  {
    ApplyRenderContext();
    m_Context->ClearRenderTargetView(m_ToSet.RenderTargets[Slot], &ClearColor.x);
  }
  finline void ClearDepth(float Depth)
  {
    ApplyRenderContext();
    m_Context->ClearDepthStencilView(m_ToSet.DepthStencilView, D3D11_CLEAR_DEPTH, Depth, 0);
  }
  finline void ClearStencil(unsigned char Stencil)
  {
    ApplyRenderContext();
    m_Context->ClearDepthStencilView(m_ToSet.DepthStencilView, D3D11_CLEAR_STENCIL, 0, Stencil);
  }
  finline void ClearDepthStencil(float Depth, unsigned char Stencil)
  {
    ApplyRenderContext();
    m_Context->ClearDepthStencilView(m_ToSet.DepthStencilView, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, Depth, Stencil);
  }
  finline void OMSetBlendStateBlendFactors(const float* pBlendFactors)
  {
    Vec4(pBlendFactors).Store(m_ToSet.BlendFactors);
  }
  finline void OMSetBlendStateSampleMask(unsigned nSampleMask)
  {
    m_ToSet.SampleMask = nSampleMask;
  }
  finline void RSSetViewports(unsigned NViewports, const D3D11_VIEWPORT* pViewports, bool doNotCheckRC = false)
  {
    _ASSERT(NViewports<VIEWPORTS);
    _ASSERT(doNotCheckRC || !IsViewportSet() || !"RSSetViewports() is being overruled by RenderContext11::SetViewport(), call RenderContext11::UnsetViewport() first");
    m_ToSet.NViewports = NViewports;
    memcpy(m_ToSet.Viewports, pViewports, NViewports*sizeof(D3D11_VIEWPORT));
  }
  finline const DeviceContext11& operator = (const RenderContext11& a)
  {
    *static_cast<RenderContext11*>(this) = a;
    return *this;
  }
  finline void PushRC()
  {
    _ASSERT(m_RCStack.size()<64 && "the stack is suspiciously large");
    m_RCStack.push_back(*this);
  }
  finline void PopRC()
  {
    *this = m_RCStack.back();
    m_RCStack.pop_back();
  }
  finline void RestoreRC()
  {
    *this = m_RCStack.back();
  }
  template<class T> void Unbind(const T& a)
  {
    for(RCStack::iterator it=m_RCStack.begin(); it!=m_RCStack.end(); ++it)
      it->Unbind(a);
    RenderContext11::Unbind(a);
  }
  template<class T> finline void BindRT(unsigned Slot, T* pRes)
  {
    __super::BindRT(Slot, pRes, pRes);
  }
  template<class T> finline void BindDepthStencil(T* pRes)
  {
    __super::BindDepthStencil(pRes, pRes);
  }
  template<class T> finline void BindUA(unsigned Slot, T* pRes)
  {
    __super::BindUA(Slot, pRes, pRes);
  }

  finline void PSSetConstantBuffer(unsigned Slot, ID3D11Buffer* pBuffer) { _ASSERT(Slot<PS_CONSTANT_BUFFER_SLOTS); m_ToSet.PSConstantBuffers[Slot] = pBuffer; }
  finline void GSSetConstantBuffer(unsigned Slot, ID3D11Buffer* pBuffer) { _ASSERT(Slot<GS_CONSTANT_BUFFER_SLOTS); m_ToSet.GSConstantBuffers[Slot] = pBuffer; }
  finline void VSSetConstantBuffer(unsigned Slot, ID3D11Buffer* pBuffer) { _ASSERT(Slot<VS_CONSTANT_BUFFER_SLOTS); m_ToSet.VSConstantBuffers[Slot] = pBuffer; }
  finline void CSSetConstantBuffer(unsigned Slot, ID3D11Buffer* pBuffer) { _ASSERT(Slot<CS_CONSTANT_BUFFER_SLOTS); m_ToSet.CSConstantBuffers[Slot] = pBuffer; }
  finline void HSSetConstantBuffer(unsigned Slot, ID3D11Buffer* pBuffer) { _ASSERT(Slot<HS_CONSTANT_BUFFER_SLOTS); m_ToSet.HSConstantBuffers[Slot] = pBuffer; }
  finline void DSSetConstantBuffer(unsigned Slot, ID3D11Buffer* pBuffer) { _ASSERT(Slot<DS_CONSTANT_BUFFER_SLOTS); m_ToSet.DSConstantBuffers[Slot] = pBuffer; }

protected:
  ID3D11DeviceContext* m_Context;
  ConstantBuffersPool11 m_ConstantBuffers;
  typedef std::vector<RenderContext11> RCStack;
  RCStack m_RCStack;

  struct States
  {
    ID3D11RasterizerState* RasterizerState;
    ID3D11BlendState* BlendState;
    float BlendFactors[4];
    unsigned SampleMask;
    ID3D11DepthStencilState* DepthStencilState;
    unsigned StencilRef;
    ID3D11Buffer* VBBuffer[VERTEX_INPUT_SLOTS];
    unsigned VBStride[VERTEX_INPUT_SLOTS];
    unsigned VBOffset[VERTEX_INPUT_SLOTS];
    ID3D11Buffer* IBBuffer;
    DXGI_FORMAT IBFormat;
    unsigned IBOffset;
    ID3D11ShaderResourceView* PSResources[PS_RESOURCES_SLOTS];
    ID3D11ShaderResourceView* GSResources[GS_RESOURCES_SLOTS];
    ID3D11ShaderResourceView* VSResources[VS_RESOURCES_SLOTS];
    ID3D11ShaderResourceView* CSResources[CS_RESOURCES_SLOTS];
    ID3D11ShaderResourceView* HSResources[HS_RESOURCES_SLOTS];
    ID3D11ShaderResourceView* DSResources[DS_RESOURCES_SLOTS];
    ID3D11SamplerState* PSSamplers[PS_SAMPLERS_SLOTS];
    ID3D11SamplerState* GSSamplers[GS_SAMPLERS_SLOTS];
    ID3D11SamplerState* VSSamplers[VS_SAMPLERS_SLOTS];
    ID3D11SamplerState* CSSamplers[CS_SAMPLERS_SLOTS];
    ID3D11SamplerState* HSSamplers[HS_SAMPLERS_SLOTS];
    ID3D11SamplerState* DSSamplers[DS_SAMPLERS_SLOTS];
    D3D11_PRIMITIVE_TOPOLOGY PrimitiveTopology;
    ID3D11UnorderedAccessView* UAResources[UNORDERED_ACCESS_RESOURCES];
    ID3D11RenderTargetView* RenderTargets[RENDER_TARGETS];
    ID3D11DepthStencilView* DepthStencilView;
    ID3D11PixelShader* PixelShader;
    ID3D11GeometryShader* GeometryShader;
    ID3D11VertexShader* VertexShader;
    ID3D11ComputeShader* ComputeShader;
    ID3D11HullShader* HullShader;
    ID3D11DomainShader* DomainShader;
    ID3D11InputLayout* InputLayout;
    unsigned NViewports;
    D3D11_VIEWPORT Viewports[VIEWPORTS];
    ID3D11Buffer* PSConstantBuffers[PS_CONSTANT_BUFFER_SLOTS];
    ID3D11Buffer* GSConstantBuffers[GS_CONSTANT_BUFFER_SLOTS];
    ID3D11Buffer* VSConstantBuffers[VS_CONSTANT_BUFFER_SLOTS];
    ID3D11Buffer* CSConstantBuffers[CS_CONSTANT_BUFFER_SLOTS];
    ID3D11Buffer* HSConstantBuffers[HS_CONSTANT_BUFFER_SLOTS];
    ID3D11Buffer* DSConstantBuffers[DS_CONSTANT_BUFFER_SLOTS];
  } m_ToSet, m_InDevice;

  finline void UnbindFromDevice(const ShaderResource11* pRes)
  {
    if(pRes!=NULL && pRes->GetShaderResourceView()!=NULL)
    {
      UnbindFromDevice<ID3D11ShaderResourceView, &ID3D11DeviceContext::PSSetShaderResources>(pRes->GetShaderResourceView(), m_ToSet.PSResources, m_InDevice.PSResources);
      UnbindFromDevice<ID3D11ShaderResourceView, &ID3D11DeviceContext::GSSetShaderResources>(pRes->GetShaderResourceView(), m_ToSet.GSResources, m_InDevice.GSResources);
      UnbindFromDevice<ID3D11ShaderResourceView, &ID3D11DeviceContext::VSSetShaderResources>(pRes->GetShaderResourceView(), m_ToSet.VSResources, m_InDevice.VSResources);
      UnbindFromDevice<ID3D11ShaderResourceView, &ID3D11DeviceContext::CSSetShaderResources>(pRes->GetShaderResourceView(), m_ToSet.CSResources, m_InDevice.CSResources);
      UnbindFromDevice<ID3D11ShaderResourceView, &ID3D11DeviceContext::HSSetShaderResources>(pRes->GetShaderResourceView(), m_ToSet.HSResources, m_InDevice.HSResources);
      UnbindFromDevice<ID3D11ShaderResourceView, &ID3D11DeviceContext::DSSetShaderResources>(pRes->GetShaderResourceView(), m_ToSet.DSResources, m_InDevice.DSResources);
    }
  }
  template<class T, void (STDMETHODCALLTYPE ID3D11DeviceContext::*Set)(unsigned, unsigned, T* const*), size_t N>
    finline void UnbindFromDevice(T* a, T* (&toSet)[N], T* (&inDevice)[N])
  {
    for(size_t i=0; i<N; ++i)
    {
      if(a==toSet[i])
      {
        toSet[i] = NULL;
      }
      if(a==inDevice[i])
      {
        inDevice[i] = NULL;
        (m_Context->*Set)(i, 1, &inDevice[i]);
      }
    }
  }
  template<size_t N>
    finline void RemoveResourcesBoundToOutput(ID3D11ShaderResourceView* (&toSet)[N])
  {
    for(unsigned i=0; i<RenderTargets11::N; ++i)
      if(RenderTargets11::IsSet(i) && RenderTargets11::Get(i).second!=NULL)
        RemoveView(toSet, RenderTargets11::Get(i).second->GetShaderResourceView());
    if(IsDepthStencilSet() && GetDepthStencil().second!=NULL)
      RemoveView(toSet, GetDepthStencil().second->GetShaderResourceView());
  }
  template<size_t N>
    finline void RemoveView(ID3D11ShaderResourceView* (&toSet)[N], ID3D11ShaderResourceView* pView)
  {
    if(pView!=NULL)
      for(size_t i=0; i<N; ++i)
        if(toSet[i]==pView)
          toSet[i] = NULL;
  }
  template<class T, void (STDMETHODCALLTYPE ID3D11DeviceContext::*Set)(unsigned, unsigned, T* const*), size_t N>
    finline void SetArray(T* (&toSet)[N], T* (&inDevice)[N])
  {
    for(size_t i = 0, j; i < N; i = j)
    {
      j = i + 1;
      if(toSet[i]!=inDevice[i])
      {
        for(; j < N; ++j)
          if(toSet[j]==inDevice[j])
            break;
        (m_Context->*Set)(i, j - i, &toSet[i]);
      }
    }
  }
  finline void RSSetState(ID3D11RasterizerState* pRasterizerState)
  {
    m_ToSet.RasterizerState = pRasterizerState;
  }
  finline void OMSetBlendState(ID3D11BlendState* pBlendState)
  {
    m_ToSet.BlendState = pBlendState;
  }
  finline void OMSetDepthStencilState(ID3D11DepthStencilState* pDepthStencilState)
  {
    m_ToSet.DepthStencilState = pDepthStencilState;
  }
  finline void OMSetStencilRef(unsigned StencilRef)
  {
    m_ToSet.StencilRef = StencilRef;
  }
  finline void IASetVertexBuffer(unsigned Slot, ID3D11Buffer* Buffer, unsigned Stride, unsigned Offset)
  {
    _ASSERT(Slot<VERTEX_INPUT_SLOTS);
    m_ToSet.VBBuffer[Slot] = Buffer;
    m_ToSet.VBStride[Slot] = Stride;
    m_ToSet.VBOffset[Slot] = Offset;
  }
  finline void IASetIndexBuffer(ID3D11Buffer* Buffer, DXGI_FORMAT Format, unsigned Offset)
  {
    m_ToSet.IBBuffer = Buffer;
    m_ToSet.IBFormat = Format;
    m_ToSet.IBOffset = Offset;
  }
  finline void IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY Topology)
  {
    m_ToSet.PrimitiveTopology = Topology;
  }
  finline void OMSetRenderTarget(unsigned Slot, ID3D11RenderTargetView* pRenderTarget)
  {
    _ASSERT(Slot<RENDER_TARGETS);
    m_ToSet.RenderTargets[Slot] = pRenderTarget;
  }
  finline void OMSetDepthStencil(ID3D11DepthStencilView* pDepthStencilView)
  {
    m_ToSet.DepthStencilView = pDepthStencilView;
  }
  finline void SetInputLayout(ID3D11InputLayout* pInputLayout)
  {
    m_ToSet.InputLayout = pInputLayout;
  }

  finline void PSSetShaderResource(unsigned Slot, ID3D11ShaderResourceView* pRes) { _ASSERT(Slot<PS_RESOURCES_SLOTS); m_ToSet.PSResources[Slot] = pRes; }
  finline void GSSetShaderResource(unsigned Slot, ID3D11ShaderResourceView* pRes) { _ASSERT(Slot<GS_RESOURCES_SLOTS); m_ToSet.GSResources[Slot] = pRes; }
  finline void VSSetShaderResource(unsigned Slot, ID3D11ShaderResourceView* pRes) { _ASSERT(Slot<VS_RESOURCES_SLOTS); m_ToSet.VSResources[Slot] = pRes; }
  finline void CSSetShaderResource(unsigned Slot, ID3D11ShaderResourceView* pRes) { _ASSERT(Slot<CS_RESOURCES_SLOTS); m_ToSet.CSResources[Slot] = pRes; }
  finline void HSSetShaderResource(unsigned Slot, ID3D11ShaderResourceView* pRes) { _ASSERT(Slot<HS_RESOURCES_SLOTS); m_ToSet.HSResources[Slot] = pRes; }
  finline void DSSetShaderResource(unsigned Slot, ID3D11ShaderResourceView* pRes) { _ASSERT(Slot<DS_RESOURCES_SLOTS); m_ToSet.DSResources[Slot] = pRes; }
  finline void PSSetSampler(unsigned Slot, ID3D11SamplerState* pSampler) { _ASSERT(Slot<PS_SAMPLERS_SLOTS); m_ToSet.PSSamplers[Slot] = pSampler; }
  finline void GSSetSampler(unsigned Slot, ID3D11SamplerState* pSampler) { _ASSERT(Slot<GS_SAMPLERS_SLOTS); m_ToSet.GSSamplers[Slot] = pSampler; }
  finline void VSSetSampler(unsigned Slot, ID3D11SamplerState* pSampler) { _ASSERT(Slot<VS_SAMPLERS_SLOTS); m_ToSet.VSSamplers[Slot] = pSampler; }
  finline void CSSetSampler(unsigned Slot, ID3D11SamplerState* pSampler) { _ASSERT(Slot<CS_SAMPLERS_SLOTS); m_ToSet.CSSamplers[Slot] = pSampler; }
  finline void HSSetSampler(unsigned Slot, ID3D11SamplerState* pSampler) { _ASSERT(Slot<HS_SAMPLERS_SLOTS); m_ToSet.HSSamplers[Slot] = pSampler; }
  finline void DSSetSampler(unsigned Slot, ID3D11SamplerState* pSampler) { _ASSERT(Slot<DS_SAMPLERS_SLOTS); m_ToSet.DSSamplers[Slot] = pSampler; }
  finline void PSSetShader(ID3D11PixelShader* pShader)    { m_ToSet.PixelShader = pShader; }
  finline void GSSetShader(ID3D11GeometryShader* pShader) { m_ToSet.GeometryShader = pShader; }
  finline void VSSetShader(ID3D11VertexShader* pShader)   { m_ToSet.VertexShader = pShader; }
  finline void CSSetShader(ID3D11ComputeShader* pShader)  { m_ToSet.ComputeShader = pShader; }
  finline void HSSetShader(ID3D11HullShader* pShader)     { m_ToSet.HullShader = pShader; }
  finline void DSSetShader(ID3D11DomainShader* pShader)   { m_ToSet.DomainShader = pShader; }
  finline void CSSetUnorderedAccessView(unsigned Slot, ID3D11UnorderedAccessView* pUAV) { _ASSERT(Slot<UNORDERED_ACCESS_RESOURCES); m_ToSet.UAResources[Slot] = pUAV; }

  void ApplyRenderContext();
};

#endif //#ifndef __DEVICE_CONTEXT11
