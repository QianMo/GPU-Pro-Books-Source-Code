#ifndef __IA_BUFFER11
#define __IA_BUFFER11

#include "Platform11.h"

class IABuffer : public Buffer11
{
public:
  HRESULT Init(size_t nElements, size_t elementSize, const void* pData, 
               D3D11_USAGE usage = D3D11_USAGE_IMMUTABLE, unsigned bindFlags = D3D11_BIND_VERTEX_BUFFER, 
               unsigned CPUAccessFlags = 0, ID3D11Device* Device11 = Platform::GetD3DDevice())
  {
    return Buffer11::Init(Device11, nElements, elementSize, pData, bindFlags, usage, CPUAccessFlags);
  }
  void Unbind()
  {
    Platform::Unbind(this);
  }
  void Clear()
  {
    Unbind();
    Buffer11::Clear();
  }
  void Destruct()
  {
    Clear();
    delete this;
  }
  finline void* Map(D3D11_MAP mapType = D3D11_MAP_WRITE_DISCARD, unsigned mapFlags = 0, const DeviceContext11& dc = Platform::GetImmediateContext()) const
  {
    D3D11_MAPPED_SUBRESOURCE msr;
    return SUCCEEDED(dc.DoNotFlushToDevice()->Map(m_Buffer, 0, mapType, mapFlags, &msr)) ? msr.pData : NULL;
  }
  finline void Unmap(const DeviceContext11& dc = Platform::GetImmediateContext()) const
  {
    dc.DoNotFlushToDevice()->Unmap(m_Buffer, 0);
  }
};

template<size_t nElements, size_t elementSize, D3D11_USAGE usage = D3D11_USAGE_IMMUTABLE, unsigned bindFlags = D3D11_BIND_VERTEX_BUFFER, unsigned CPUAccessFlags = 0>
  class StaticIABuffer : public IABuffer
{
public:
  StaticIABuffer(const void* pData) : m_pData(pData)
  {
    Platform::Add(Platform::OnInitDelegate::from_method<StaticIABuffer, &StaticIABuffer::OnPlatformInit>(this), Platform::Object_Generic);
    Platform::Add(Platform::OnShutdownDelegate::from_method<StaticIABuffer, &StaticIABuffer::OnPlatformShutdown>(this), Platform::Object_Generic);
  }

protected:
  const void* m_pData;

  bool OnPlatformInit() { return SUCCEEDED(Init(nElements, elementSize, m_pData, usage, bindFlags, CPUAccessFlags)); }
  void OnPlatformShutdown() { Clear(); }
};

#endif //#ifndef __IA_BUFFER11
