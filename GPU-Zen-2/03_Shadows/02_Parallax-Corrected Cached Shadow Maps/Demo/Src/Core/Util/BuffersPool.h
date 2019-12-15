#ifndef __BUFFERS_POOL_H
#define __BUFFERS_POOL_H

#include <algorithm>
#include "../Math/Math.h"

template<class T, class Buffer, class UserParam> class BuffersPool
{
public:
  BuffersPool() { }
  ~BuffersPool() { Clear(); }
  finline size_t GetMaxBufferSize() const { return m_MaxBufferSize; }
  finline size_t GetMinBufferSize() const { return m_MinBufferSize; }

  void Clear()
  {
    for(auto it = m_Pool.begin(); it!=m_Pool.end(); ++it)
      it->Clear();
    m_Pool.clear();
    m_Allocated.clear();
  }
  const Buffer* Allocate(size_t Size)
  {
    size_t startIndex = (std::max(Size, m_MinBufferSize) - m_MinBufferSize + m_BufferSizeStep - 1)/m_BufferSizeStep;
    if(startIndex<m_Pool.size())
    {
      for(auto it = (m_Pool.cbegin() + startIndex); it!=m_Pool.cend(); ++it)
      {
        const Buffer* pBuffer = &*it;
        if(std::find(m_Allocated.cbegin(), m_Allocated.cend(), pBuffer)==m_Allocated.cend())
        {
          m_Allocated.push_back(pBuffer);
          return pBuffer;
        }
      }
    }
    _ASSERT(!"unable to find free feasible buffer");
    return NULL;
  }
  void Free(const Buffer* pBuffer)
  {
    auto it = std::find(m_Allocated.begin(), m_Allocated.end(), pBuffer);
    if(it!=m_Allocated.end())
      m_Allocated.erase(it);
  }
  void FreeAll()
  {
    m_Allocated.clear();
  }

protected:
  size_t m_MaxBufferSize, m_BufferSizeStep, m_MinBufferSize;
  std::vector<Buffer> m_Pool;
  std::vector<const Buffer*> m_Allocated;

  HRESULT Init(const UserParam& userParam, size_t maxBufferSize, size_t minBufferSize, size_t bufferSizeStep)
  {
    m_MinBufferSize = minBufferSize;
    m_MaxBufferSize = maxBufferSize;
    m_BufferSizeStep = bufferSizeStep;
    size_t poolSize = (maxBufferSize - m_MinBufferSize + bufferSizeStep - 1)/bufferSizeStep + 1;
    m_Allocated.reserve(poolSize);
    m_Pool.resize(poolSize);
    HRESULT hr = S_OK;
    for(size_t i=0; i<poolSize && SUCCEEDED(hr); ++i)
      hr = static_cast<T*>(this)->InitBuffer(userParam, m_Pool[i], std::min(i*m_BufferSizeStep + minBufferSize, maxBufferSize));
    return hr;
  }
};

#endif //#ifndef __BUFFERS_POOL_H
