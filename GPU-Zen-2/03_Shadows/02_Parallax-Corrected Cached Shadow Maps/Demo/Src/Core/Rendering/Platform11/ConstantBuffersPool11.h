#ifndef __CONSTANT_BUFFERS_POOL11
#define __CONSTANT_BUFFERS_POOL11

#include "Objects11.h"
#include "../../Util/BuffersPool.h"

class ConstantBuffersPool11 : public BuffersPool<ConstantBuffersPool11, Buffer11, ID3D11Device*>
{
public:
  HRESULT Init(ID3D11Device* Device11, size_t maxBufferSize = 4096, size_t minBufferSize = 64, size_t bufferSizeStep = 64)
  {
    return BuffersPool::Init(Device11, maxBufferSize, minBufferSize, bufferSizeStep);
  }
  ID3D11Buffer* Allocate(size_t Size, const void* pData = NULL, ID3D11DeviceContext* Context11 = NULL)
  {
    const Buffer11* pBuffer = BuffersPool::Allocate(Size);
    if(pBuffer==NULL || pData==NULL || Context11==NULL)
      return NULL;
    D3D11_MAPPED_SUBRESOURCE msr;
    HRESULT hr = Context11->Map(pBuffer->GetBuffer(), 0, D3D11_MAP_WRITE_DISCARD, 0, &msr);
    _ASSERT(SUCCEEDED(hr) && "unable to map buffer");
    if(SUCCEEDED(hr))
    {
      memcpy(msr.pData, pData, Size);
      Context11->Unmap(pBuffer->GetBuffer(), 0);
    }
    return pBuffer->GetBuffer();
  }
  finline void Free(ID3D11Buffer* pBuffer)
  {
    for(auto it=m_Allocated.begin(); it!=m_Allocated.end(); ++it)
      if((*it)->GetBuffer()==pBuffer) { m_Allocated.erase(it); break; }
  }
  HRESULT InitBuffer(ID3D11Device* Device11, Buffer11& buf, size_t size)
  {
    return buf.Init(Device11, size, 1, NULL, D3D11_BIND_CONSTANT_BUFFER, D3D11_USAGE_DYNAMIC, D3D11_CPU_ACCESS_WRITE);
  }
};

#endif //#ifndef __CONSTANT_BUFFERS_POOL11
