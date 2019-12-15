#ifndef __MEMORY_POOL
#define __MEMORY_POOL

#include <malloc.h>
#include <algorithm>

class MemoryPool
{
public:
  MemoryPool(size_t elementsPerPage, size_t elementSize, size_t alignment = 16) :
    m_ElementSize(std::max(sizeof(void*), ((elementSize + alignment - 1)/alignment)*alignment)),
    m_NextPage(NULL), m_ElementsPerPage(elementsPerPage), m_Alignment(alignment)
  {
    size_t bufferSize = m_ElementsPerPage*m_ElementSize;
    m_Buffer = (char*)_aligned_malloc(bufferSize, m_Alignment);
    Clear();
  }
  ~MemoryPool()
  {
    size_t freeElements;
    for(freeElements=0; m_FreeElement!=NULL; ++freeElements)
      m_FreeElement = reinterpret_cast<void**>(*m_FreeElement);
    _ASSERT(freeElements==m_ElementsPerPage && "some elements are still allocated");
    delete m_NextPage;
    _aligned_free(m_Buffer);
  }
  void* Allocate()
  {
    if(m_FreeElement==NULL)
    {
      if(m_NextPage==NULL)
        m_NextPage = new MemoryPool(m_ElementsPerPage, m_ElementSize, m_Alignment);
      return m_NextPage->Allocate();
    }
    void* p = m_FreeElement;
    m_FreeElement = reinterpret_cast<void**>(*m_FreeElement);
    return p;
  }
  void Free(void* p)
  {
    if(!InThisPage(p))
    {
      _ASSERT(m_NextPage!=NULL && "memory was not allocated from this pool");
      m_NextPage->Free(p);
      return;
    }
    _ASSERT((((char*)p - m_Buffer) % m_ElementSize)==0 && "invalid address");
    *(void**)p = m_FreeElement;
    m_FreeElement = (void**)p;
  }
  bool Contains(const void* p) const
  {
    return InThisPage(p) ? true : (m_NextPage!=NULL ? m_NextPage->Contains(p) : false);
  }

  size_t Capacity() const    { return m_ElementsPerPage + (m_NextPage!=NULL ? m_NextPage->Capacity() : 0); };
  size_t ElementSize() const { return m_ElementSize; }
  size_t Alignment() const   { return m_Alignment; }

protected:
  MemoryPool* m_NextPage;
  char* m_Buffer;
  void** m_FreeElement;
  size_t m_ElementsPerPage;
  size_t m_ElementSize;
  size_t m_Alignment;

  bool InThisPage(const void* p) const
  {
    return (size_t((char*)p - m_Buffer) < m_ElementsPerPage*m_ElementSize);
  }
  void Clear()
  {
    size_t bufferSize = m_ElementsPerPage*m_ElementSize;
    for(size_t i=m_ElementSize; i<bufferSize; i+=m_ElementSize)
      *(void**)(m_Buffer + i - m_ElementSize) = m_Buffer + i;
    *(void**)(m_Buffer + bufferSize - m_ElementSize) = NULL;
    m_FreeElement = (void**)m_Buffer;
  }
};

#define DECLARE_MEMORY_POOL() \
  public: void* operator new(size_t n) { _ASSERT(n==s_MemoryPool.ElementSize()); return s_MemoryPool.Allocate(); } \
  public: void  operator delete(void* p) { s_MemoryPool.Free(p); } \
  private: static MemoryPool s_MemoryPool;

#define IMPLEMENT_MEMORY_POOL(t, n) \
  MemoryPool t::s_MemoryPool(n, sizeof(t), __alignof(t));
  
#endif //#ifndef __MEMORY_POOL
