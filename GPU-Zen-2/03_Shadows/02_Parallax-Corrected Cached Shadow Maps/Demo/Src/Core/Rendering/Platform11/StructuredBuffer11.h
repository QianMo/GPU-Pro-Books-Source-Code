#ifndef __STRUCTURED_BUFFER11
#define __STRUCTURED_BUFFER11

#include "Platform11.h"

class StructuredBuffer : public Buffer11, public ShaderResource11, public UnorderedAccessResource11
{
public:
  template<class T> HRESULT Init(size_t nElements, const T& initValue, unsigned bindFlags = D3D11_BIND_SHADER_RESOURCE,
                                 D3D11_USAGE usage = D3D11_USAGE_DEFAULT, unsigned CPUAccessFlags = 0, ID3D11Device* Device11 = Platform::GetD3DDevice())
  {
    T* pBuf = static_cast<T*>(Platform::AllocateScratchMemory(sizeof(T)*nElements));
    for(size_t i=0; i<nElements; ++i) pBuf[i] = initValue;
    HRESULT hr = Init(nElements, sizeof(T), pBuf, bindFlags, usage, CPUAccessFlags, Device11);
    Platform::FreeScratchMemory(pBuf);
    return hr;
  }
  HRESULT Init(size_t nElements, size_t elementSize, const void* pData, unsigned bindFlags = D3D11_BIND_SHADER_RESOURCE, 
               D3D11_USAGE usage = D3D11_USAGE_DEFAULT, unsigned CPUAccessFlags = 0, ID3D11Device* Device11 = Platform::GetD3DDevice())
  {
    HRESULT hr = Buffer11::Init(Device11, nElements, elementSize, pData, bindFlags, usage, CPUAccessFlags, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED);
    if(SUCCEEDED(hr) && (bindFlags&D3D11_BIND_SHADER_RESOURCE))
    {
      D3D11_SHADER_RESOURCE_VIEW_DESC desc = { };
      desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
      desc.Buffer.NumElements = nElements;
      ID3D11ShaderResourceView* pSRV;
      hr = Device11->CreateShaderResourceView(m_Buffer, &desc, &pSRV);
      if(SUCCEEDED(hr))
      {
        hr = ShaderResource11::Init(pSRV);
        pSRV->Release();
      }
    }
    if(SUCCEEDED(hr) && (bindFlags&D3D11_BIND_UNORDERED_ACCESS))
    {
      D3D11_UNORDERED_ACCESS_VIEW_DESC desc = { };
      desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
      desc.Buffer.NumElements = nElements;
      ID3D11UnorderedAccessView* pUAV;
      hr = Device11->CreateUnorderedAccessView(m_Buffer, &desc, &pUAV);
      if(SUCCEEDED(hr))
      {
        hr = UnorderedAccessResource11::Init(pUAV);
        pUAV->Release();
      }
    }
    return hr;
  }
  void Unbind()
  {
    Platform::Unbind((Buffer11*)this);
    Platform::Unbind((ShaderResource11*)this);
    Platform::Unbind((UnorderedAccessResource11*)this);
  }
  void Clear()
  {
    Unbind();
    Buffer11::Clear();
    ShaderResource11::Clear();
    UnorderedAccessResource11::Clear();
  }
  template<class T>
  finline T* Map(D3D11_MAP mapType = D3D11_MAP_WRITE_DISCARD, unsigned mapFlags = 0, const DeviceContext11& dc = Platform::GetImmediateContext()) const
  {
    D3D11_MAPPED_SUBRESOURCE msr;
    return SUCCEEDED(dc.DoNotFlushToDevice()->Map(m_Buffer, 0, mapType, mapFlags, &msr)) ? static_cast<T*>(msr.pData) : NULL;
  }
  finline void Unmap(const DeviceContext11& dc = Platform::GetImmediateContext()) const
  {
    dc.DoNotFlushToDevice()->Unmap(m_Buffer, 0);
  }
  finline void Fill(unsigned d, const DeviceContext11& dc = Platform::GetImmediateContext())
  {
    if(GetUnorderedAccessView()!=NULL)
      dc.DoNotFlushToDevice()->ClearUnorderedAccessViewUint(GetUnorderedAccessView(), &d);
  }
};

template<size_t nElements, size_t elementSize, D3D11_USAGE usage = D3D11_USAGE_IMMUTABLE, unsigned bindFlags = D3D11_BIND_SHADER_RESOURCE, unsigned CPUAccessFlags = 0>
  class StaticStructuredBuffer : public StructuredBuffer
{
public:
  StaticStructuredBuffer(const void* pData) : m_pData(pData)
  {
    Platform::Add(Platform::OnInitDelegate::from_method<StaticStructuredBuffer, &StaticStructuredBuffer::OnPlatformInit>(this), Platform::Object_Generic);
    Platform::Add(Platform::OnShutdownDelegate::from_method<StaticStructuredBuffer, &StaticStructuredBuffer::OnPlatformShutdown>(this), Platform::Object_Generic);
  }

protected:
  const void* m_pData;

  bool OnPlatformInit() { return SUCCEEDED(Init(nElements, elementSize, m_pData, bindFlags, usage, CPUAccessFlags)); }
  void OnPlatformShutdown() { Clear(); }
};

#endif //#ifndef __STRUCTURED_BUFFER11
