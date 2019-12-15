#include "PreCompile.h"
#include "DeviceContext11.h"
#include "Platform11.h"
#include "DeferredContext11.h"

HRESULT DeviceContext11::Init(ID3D11Device* Device11, ID3D11DeviceContext* Context11)
{
  HRESULT hr = m_ConstantBuffers.Init(Device11);
  if(SUCCEEDED(hr))
  {
    m_Context = Context11; Context11->AddRef();
    InvalidateContextCache();
    m_ToSet = m_InDevice;
    RasterizerStateBlock11::Set(RasterizerDesc11());
    BlendStateBlock11::Set(BlendDesc11());
    DepthStencilStateBlock11::Set(DepthStencilDesc11(true, D3D11_DEPTH_WRITE_MASK_ALL, D3D11_COMPARISON_LESS_EQUAL));
    for(unsigned i=0; i<PSSamplers::N; ++i) PSSamplers::Set(i, &Platform::GetSamplerCache().ConcurrentGetByIndex(Platform::Sampler_Linear_Clamp));
    for(unsigned i=0; i<GSSamplers::N; ++i) GSSamplers::Set(i, &Platform::GetSamplerCache().ConcurrentGetByIndex(Platform::Sampler_Linear_Clamp));
    for(unsigned i=0; i<VSSamplers::N; ++i) VSSamplers::Set(i, &Platform::GetSamplerCache().ConcurrentGetByIndex(Platform::Sampler_Linear_Clamp));
    for(unsigned i=0; i<CSSamplers::N; ++i) CSSamplers::Set(i, &Platform::GetSamplerCache().ConcurrentGetByIndex(Platform::Sampler_Linear_Clamp));
    for(unsigned i=0; i<HSSamplers::N; ++i) HSSamplers::Set(i, &Platform::GetSamplerCache().ConcurrentGetByIndex(Platform::Sampler_Linear_Clamp));
    for(unsigned i=0; i<DSSamplers::N; ++i) DSSamplers::Set(i, &Platform::GetSamplerCache().ConcurrentGetByIndex(Platform::Sampler_Linear_Clamp));
  }
  return hr;
}

void DeviceContext11::Clear()
{
  m_ConstantBuffers.Clear();
  SAFE_RELEASE(m_Context);
  RenderContext11::Reset();
}

template<class T> void Release(T* p)
{
  if(p!=NULL)
   p->Release();
}

template<class T, unsigned N> void ReleaseArray(T* (&a)[N])
{
  for(unsigned i=0; i<N; ++i)
    if(a[i]!=NULL) a[i]->Release();
}

void DeviceContext11::InvalidateContextCache()
{
  m_Context->PSGetShaderResources(0, PS_RESOURCES_SLOTS, m_InDevice.PSResources);
  m_Context->GSGetShaderResources(0, GS_RESOURCES_SLOTS, m_InDevice.GSResources);
  m_Context->VSGetShaderResources(0, VS_RESOURCES_SLOTS, m_InDevice.VSResources);
  m_Context->CSGetShaderResources(0, CS_RESOURCES_SLOTS, m_InDevice.CSResources);
  m_Context->HSGetShaderResources(0, HS_RESOURCES_SLOTS, m_InDevice.HSResources);
  m_Context->DSGetShaderResources(0, DS_RESOURCES_SLOTS, m_InDevice.DSResources);
  ReleaseArray(m_InDevice.PSResources);
  ReleaseArray(m_InDevice.GSResources);
  ReleaseArray(m_InDevice.VSResources);
  ReleaseArray(m_InDevice.CSResources);
  ReleaseArray(m_InDevice.HSResources);
  ReleaseArray(m_InDevice.DSResources);
  m_Context->PSGetSamplers(0, PS_SAMPLERS_SLOTS, m_InDevice.PSSamplers);
  m_Context->GSGetSamplers(0, GS_SAMPLERS_SLOTS, m_InDevice.GSSamplers);
  m_Context->VSGetSamplers(0, VS_SAMPLERS_SLOTS, m_InDevice.VSSamplers);
  m_Context->CSGetSamplers(0, CS_SAMPLERS_SLOTS, m_InDevice.CSSamplers);
  m_Context->HSGetSamplers(0, HS_SAMPLERS_SLOTS, m_InDevice.HSSamplers);
  m_Context->DSGetSamplers(0, DS_SAMPLERS_SLOTS, m_InDevice.DSSamplers);
  ReleaseArray(m_InDevice.PSSamplers);
  ReleaseArray(m_InDevice.GSSamplers);
  ReleaseArray(m_InDevice.VSSamplers);
  ReleaseArray(m_InDevice.CSSamplers);
  ReleaseArray(m_InDevice.HSSamplers);
  ReleaseArray(m_InDevice.DSSamplers);
  m_Context->PSGetConstantBuffers(0, PS_CONSTANT_BUFFER_SLOTS, m_InDevice.PSConstantBuffers);
  m_Context->GSGetConstantBuffers(0, GS_CONSTANT_BUFFER_SLOTS, m_InDevice.GSConstantBuffers);
  m_Context->VSGetConstantBuffers(0, VS_CONSTANT_BUFFER_SLOTS, m_InDevice.VSConstantBuffers);
  m_Context->CSGetConstantBuffers(0, CS_CONSTANT_BUFFER_SLOTS, m_InDevice.CSConstantBuffers);
  m_Context->HSGetConstantBuffers(0, HS_CONSTANT_BUFFER_SLOTS, m_InDevice.HSConstantBuffers);
  m_Context->DSGetConstantBuffers(0, DS_CONSTANT_BUFFER_SLOTS, m_InDevice.DSConstantBuffers);
  ReleaseArray(m_InDevice.PSConstantBuffers);
  ReleaseArray(m_InDevice.GSConstantBuffers);
  ReleaseArray(m_InDevice.VSConstantBuffers);
  ReleaseArray(m_InDevice.CSConstantBuffers);
  ReleaseArray(m_InDevice.HSConstantBuffers);
  ReleaseArray(m_InDevice.DSConstantBuffers);
  m_Context->RSGetState(&m_InDevice.RasterizerState);
  m_Context->OMGetBlendState(&m_InDevice.BlendState, m_InDevice.BlendFactors, &m_InDevice.SampleMask);
  m_Context->OMGetDepthStencilState(&m_InDevice.DepthStencilState, &m_InDevice.StencilRef);
  Release(m_InDevice.RasterizerState);
  Release(m_InDevice.BlendState);
  Release(m_InDevice.DepthStencilState);
  m_Context->IAGetIndexBuffer(&m_InDevice.IBBuffer, &m_InDevice.IBFormat, &m_InDevice.IBOffset);
  m_Context->IAGetVertexBuffers(0, VERTEX_INPUT_SLOTS, m_InDevice.VBBuffer, m_InDevice.VBStride, m_InDevice.VBOffset);
  m_Context->IAGetPrimitiveTopology(&m_InDevice.PrimitiveTopology);
  m_Context->IAGetInputLayout(&m_InDevice.InputLayout);
  Release(m_InDevice.IBBuffer);
  ReleaseArray(m_InDevice.VBBuffer);
  Release(m_InDevice.InputLayout);
  m_Context->CSGetUnorderedAccessViews(0, UNORDERED_ACCESS_RESOURCES, m_InDevice.UAResources);
  m_Context->OMGetRenderTargets(RENDER_TARGETS, m_InDevice.RenderTargets, &m_InDevice.DepthStencilView);
  ReleaseArray(m_InDevice.UAResources);
  ReleaseArray(m_InDevice.RenderTargets);
  Release(m_InDevice.DepthStencilView);
  m_Context->PSGetShader(&m_InDevice.PixelShader, NULL, 0);
  m_Context->GSGetShader(&m_InDevice.GeometryShader, NULL, 0);
  m_Context->VSGetShader(&m_InDevice.VertexShader, NULL, 0);
  m_Context->CSGetShader(&m_InDevice.ComputeShader, NULL, 0);
  m_Context->HSGetShader(&m_InDevice.HullShader, NULL, 0);
  m_Context->DSGetShader(&m_InDevice.DomainShader, NULL, 0);
  Release(m_InDevice.PixelShader);
  Release(m_InDevice.GeometryShader);
  Release(m_InDevice.VertexShader);
  Release(m_InDevice.ComputeShader);
  Release(m_InDevice.HullShader);
  Release(m_InDevice.DomainShader);
  m_InDevice.NViewports = VIEWPORTS;
  m_Context->RSGetViewports(&m_InDevice.NViewports, m_InDevice.Viewports);
}

ID3D11DeviceContext* DeviceContext11::FlushToDevice()
{
  ApplyRenderContext();

  if(m_InDevice.RasterizerState!=m_ToSet.RasterizerState)
  {
    m_Context->RSSetState(m_ToSet.RasterizerState);
  }
  if((m_InDevice.BlendState!=m_ToSet.BlendState) | (Vec4(m_InDevice.BlendFactors)!=Vec4(m_ToSet.BlendFactors)) | (m_InDevice.SampleMask!=m_ToSet.SampleMask))
  {
    m_Context->OMSetBlendState(m_ToSet.BlendState, m_ToSet.BlendFactors, m_ToSet.SampleMask);
  }
  if((m_InDevice.DepthStencilState!=m_ToSet.DepthStencilState) | (m_InDevice.StencilRef!=m_ToSet.StencilRef))
  {
    m_Context->OMSetDepthStencilState(m_ToSet.DepthStencilState, m_ToSet.StencilRef);
  }
  if((m_ToSet.IBBuffer!=m_InDevice.IBBuffer) | (m_ToSet.IBFormat!=m_InDevice.IBFormat) | (m_ToSet.IBOffset!=m_InDevice.IBOffset))
  {
    m_Context->IASetIndexBuffer(m_ToSet.IBBuffer, m_ToSet.IBFormat, m_ToSet.IBOffset);
  }
  if(m_ToSet.PrimitiveTopology!=m_InDevice.PrimitiveTopology)
  {
    m_Context->IASetPrimitiveTopology(m_ToSet.PrimitiveTopology);
  }
  if(memcmp(m_ToSet.RenderTargets, m_InDevice.RenderTargets, sizeof(m_ToSet.RenderTargets)) | (m_ToSet.DepthStencilView!=m_InDevice.DepthStencilView))
  {
    unsigned n; for(n=0; n<RENDER_TARGETS; ++n) if(m_ToSet.RenderTargets[n]==NULL) break;
    m_Context->OMSetRenderTargets(n, n>0 ? m_ToSet.RenderTargets : NULL, m_ToSet.DepthStencilView);
  }
  if(m_ToSet.NViewports!=m_InDevice.NViewports ||
     memcmp(m_ToSet.Viewports, m_InDevice.Viewports, m_ToSet.NViewports*sizeof(D3D11_VIEWPORT)))
  {
    m_Context->RSSetViewports(m_ToSet.NViewports, m_ToSet.Viewports);
  }
  for(unsigned i=0; i<VERTEX_INPUT_SLOTS; ++i)
  {
    if((m_ToSet.VBBuffer[i]!=m_InDevice.VBBuffer[i]) | (m_ToSet.VBStride[i]!=m_InDevice.VBStride[i]) | (m_ToSet.VBOffset[i]!=m_InDevice.VBOffset[i]))
      m_Context->IASetVertexBuffers(i, 1, &m_ToSet.VBBuffer[i], &m_ToSet.VBStride[i], &m_ToSet.VBOffset[i]);
  }
  for(unsigned i=0; i<UNORDERED_ACCESS_RESOURCES; ++i)
  {
    if(m_ToSet.UAResources[i]!=m_InDevice.UAResources[i])
      m_Context->CSSetUnorderedAccessViews(i, 1, &m_ToSet.UAResources[i], NULL);
  }

  if(m_ToSet.InputLayout!=m_InDevice.InputLayout) m_Context->IASetInputLayout(m_ToSet.InputLayout);
  if(m_ToSet.PixelShader!=m_InDevice.PixelShader) m_Context->PSSetShader(m_ToSet.PixelShader, NULL, 0);
  if(m_ToSet.GeometryShader!=m_InDevice.GeometryShader) m_Context->GSSetShader(m_ToSet.GeometryShader, NULL, 0);
  if(m_ToSet.VertexShader!=m_InDevice.VertexShader) m_Context->VSSetShader(m_ToSet.VertexShader, NULL, 0);
  if(m_ToSet.ComputeShader!=m_InDevice.ComputeShader) m_Context->CSSetShader(m_ToSet.ComputeShader, NULL, 0);
  if(m_ToSet.HullShader!=m_InDevice.HullShader) m_Context->HSSetShader(m_ToSet.HullShader, NULL, 0);
  if(m_ToSet.DomainShader!=m_InDevice.DomainShader) m_Context->DSSetShader(m_ToSet.DomainShader, NULL, 0);

  RemoveResourcesBoundToOutput(m_ToSet.PSResources);
  RemoveResourcesBoundToOutput(m_ToSet.VSResources);

  SetArray<ID3D11ShaderResourceView, &ID3D11DeviceContext::PSSetShaderResources>(m_ToSet.PSResources, m_InDevice.PSResources);
  SetArray<ID3D11ShaderResourceView, &ID3D11DeviceContext::GSSetShaderResources>(m_ToSet.GSResources, m_InDevice.GSResources);
  SetArray<ID3D11ShaderResourceView, &ID3D11DeviceContext::VSSetShaderResources>(m_ToSet.VSResources, m_InDevice.VSResources);
  SetArray<ID3D11ShaderResourceView, &ID3D11DeviceContext::CSSetShaderResources>(m_ToSet.CSResources, m_InDevice.CSResources);
  SetArray<ID3D11ShaderResourceView, &ID3D11DeviceContext::HSSetShaderResources>(m_ToSet.HSResources, m_InDevice.HSResources);
  SetArray<ID3D11ShaderResourceView, &ID3D11DeviceContext::DSSetShaderResources>(m_ToSet.DSResources, m_InDevice.DSResources);
  SetArray<ID3D11SamplerState, &ID3D11DeviceContext::PSSetSamplers>(m_ToSet.PSSamplers, m_InDevice.PSSamplers);
  SetArray<ID3D11SamplerState, &ID3D11DeviceContext::GSSetSamplers>(m_ToSet.GSSamplers, m_InDevice.GSSamplers);
  SetArray<ID3D11SamplerState, &ID3D11DeviceContext::VSSetSamplers>(m_ToSet.VSSamplers, m_InDevice.VSSamplers);
  SetArray<ID3D11SamplerState, &ID3D11DeviceContext::CSSetSamplers>(m_ToSet.CSSamplers, m_InDevice.CSSamplers);
  SetArray<ID3D11SamplerState, &ID3D11DeviceContext::HSSetSamplers>(m_ToSet.HSSamplers, m_InDevice.HSSamplers);
  SetArray<ID3D11SamplerState, &ID3D11DeviceContext::DSSetSamplers>(m_ToSet.DSSamplers, m_InDevice.DSSamplers);
  SetArray<ID3D11Buffer, &ID3D11DeviceContext::PSSetConstantBuffers>(m_ToSet.PSConstantBuffers, m_InDevice.PSConstantBuffers);
  SetArray<ID3D11Buffer, &ID3D11DeviceContext::GSSetConstantBuffers>(m_ToSet.GSConstantBuffers, m_InDevice.GSConstantBuffers);
  SetArray<ID3D11Buffer, &ID3D11DeviceContext::VSSetConstantBuffers>(m_ToSet.VSConstantBuffers, m_InDevice.VSConstantBuffers);
  SetArray<ID3D11Buffer, &ID3D11DeviceContext::CSSetConstantBuffers>(m_ToSet.CSConstantBuffers, m_InDevice.CSConstantBuffers);
  SetArray<ID3D11Buffer, &ID3D11DeviceContext::HSSetConstantBuffers>(m_ToSet.HSConstantBuffers, m_InDevice.HSConstantBuffers);
  SetArray<ID3D11Buffer, &ID3D11DeviceContext::DSSetConstantBuffers>(m_ToSet.DSConstantBuffers, m_InDevice.DSConstantBuffers);

  m_InDevice = m_ToSet;
  return m_Context;
}

void DeviceContext11::ApplyRenderContext()
{
  for(unsigned i=0; i<VertexBuffers11::N; ++i)
  {
    if(VertexBuffers11::IsSet(i))
    {
      const Buffer11* pBuffer = VertexBuffers11::Get(i).first;
      unsigned Offset = VertexBuffers11::Get(i).second;
      IASetVertexBuffer(i, pBuffer->GetBuffer(), pBuffer->GetElementSize(), Offset);
    }
    else
      IASetVertexBuffer(i, NULL, 0, 0);
  }
  for(unsigned i=0; i<RenderTargets11::N; ++i)
  {
    if(RenderTargets11::IsSet(i))
    {
      UnbindFromDevice(RenderTargets11::Get(i).second);
      OMSetRenderTarget(i, RenderTargets11::Get(i).first->GetRenderTargetView());
    }
    else
      OMSetRenderTarget(i, NULL);
  }
  for(unsigned i=0; i<UnorderedAccessResources11::N; ++i)
  {
    if(UnorderedAccessResources11::IsSet(i))
    {
      UnbindFromDevice(UnorderedAccessResources11::Get(i).second);
      CSSetUnorderedAccessView(i, UnorderedAccessResources11::Get(i).first->GetUnorderedAccessView());
    }
    else
      CSSetUnorderedAccessView(i, NULL);
  }

  if(IsDepthStencilSet())
  {
    UnbindFromDevice(GetDepthStencil().second);
    OMSetDepthStencil(GetDepthStencil().first->GetDepthStencilView());
  }
  else 
    OMSetDepthStencil(NULL);

  if(IsIndexBufferSet())
  {
    const Buffer11* pBuffer = GetIndexBuffer();
    DXGI_FORMAT Format = DXGI_FORMAT_UNKNOWN;
    switch(pBuffer->GetElementSize())
    {
    case 2: Format = DXGI_FORMAT_R16_UINT; break;
    case 4: Format = DXGI_FORMAT_R32_UINT; break;
    default: _ASSERT(!"invalid element size");
    }
    IASetIndexBuffer(pBuffer->GetBuffer(), Format, GetIndexBufferOffset());
  }
  else
    IASetIndexBuffer(NULL, DXGI_FORMAT_UNKNOWN, 0);

  RSSetState(IsRasterizerStateSet() ? GetRasterizerState()->GetRasterizerState() : Platform::GetRasterizerCache().ConcurrentGet(RasterizerStateBlock11::Get()).GetRasterizerState());
  OMSetBlendState(IsBlendStateSet() ? GetBlendState()->GetBlendState() : Platform::GetBlendCache().ConcurrentGet(BlendStateBlock11::Get()).GetBlendState());
  OMSetDepthStencilState(IsDepthStencilStateSet() ? GetDepthStencilState()->GetDepthStencilState() : Platform::GetDepthStencilCache().ConcurrentGet(DepthStencilStateBlock11::Get()).GetDepthStencilState());
  for(unsigned i=0; i<PSResources11::N; ++i) PSSetShaderResource(i, PSResources11::IsSet(i) ? PSResources11::Get(i)->GetShaderResourceView() : NULL);
  for(unsigned i=0; i<GSResources11::N; ++i) GSSetShaderResource(i, GSResources11::IsSet(i) ? GSResources11::Get(i)->GetShaderResourceView() : NULL);
  for(unsigned i=0; i<VSResources11::N; ++i) VSSetShaderResource(i, VSResources11::IsSet(i) ? VSResources11::Get(i)->GetShaderResourceView() : NULL);
  for(unsigned i=0; i<CSResources11::N; ++i) CSSetShaderResource(i, CSResources11::IsSet(i) ? CSResources11::Get(i)->GetShaderResourceView() : NULL);
  for(unsigned i=0; i<HSResources11::N; ++i) HSSetShaderResource(i, HSResources11::IsSet(i) ? HSResources11::Get(i)->GetShaderResourceView() : NULL);
  for(unsigned i=0; i<DSResources11::N; ++i) DSSetShaderResource(i, DSResources11::IsSet(i) ? DSResources11::Get(i)->GetShaderResourceView() : NULL);
  for(unsigned i=0; i<PSSamplers::N; ++i) PSSetSampler(i, PSSamplers::Get(i)->GetSamplerState());
  for(unsigned i=0; i<GSSamplers::N; ++i) GSSetSampler(i, GSSamplers::Get(i)->GetSamplerState());
  for(unsigned i=0; i<VSSamplers::N; ++i) VSSetSampler(i, VSSamplers::Get(i)->GetSamplerState());
  for(unsigned i=0; i<CSSamplers::N; ++i) CSSetSampler(i, CSSamplers::Get(i)->GetSamplerState());
  for(unsigned i=0; i<HSSamplers::N; ++i) HSSetSampler(i, HSSamplers::Get(i)->GetSamplerState());
  for(unsigned i=0; i<DSSamplers::N; ++i) DSSetSampler(i, DSSamplers::Get(i)->GetSamplerState());
  if(IsPrimitiveTopologySet()) IASetPrimitiveTopology(GetPrimitiveTopology());
  if(IsStencilRefSet()) OMSetStencilRef(GetStencilRef());
  PSSetShader(IsPixelShaderSet() ? GetPixelShader()->GetPixelShader() : NULL);
  GSSetShader(IsGeometryShaderSet() ? GetGeometryShader()->GetGeometryShader() : NULL);
  VSSetShader(IsVertexShaderSet() ? GetVertexShader()->GetVertexShader() : NULL);
  CSSetShader(IsComputeShaderSet() ? GetComputeShader()->GetComputeShader() : NULL);
  HSSetShader(IsHullShaderSet() ? GetHullShader()->GetHullShader() : NULL);
  DSSetShader(IsDomainShaderSet() ? GetDomainShader()->GetDomainShader() : NULL);
  SetInputLayout(IsInputLayoutSet() ? GetInputLayout()->GetInputLayout() : NULL);
  if(IsViewportSet()) { D3D11_VIEWPORT vp = GetViewport(); RSSetViewports(1, &vp, true); }
}

DeferredContext11::DeferredContext11() : m_CommandList(NULL)
{
  ID3D11DeviceContext* Context11 = NULL;
  HRESULT hr = Platform::GetD3DDevice()->CreateDeferredContext(0, &Context11);
  hr = SUCCEEDED(hr) ? Init(Platform::GetD3DDevice(), Context11) : hr;
  SAFE_RELEASE(Context11);
}

DeferredContext11::~DeferredContext11()
{
  SAFE_RELEASE(m_CommandList);
  Clear();
}

void DeferredContext11::FinishCommandList(bool RestoreDeferredContextState)
{
  SAFE_RELEASE(m_CommandList);
  m_Context->FinishCommandList(RestoreDeferredContextState, &m_CommandList);
}

void DeferredContext11::ExecuteCommandList()
{
  if(m_CommandList!=NULL)
  {
    Platform::GetImmediateContext().FlushToDevice()->ExecuteCommandList(m_CommandList, true);
    SAFE_RELEASE(m_CommandList);
  }
}
