#ifndef __MEMORY_BUFFER
#define __MEMORY_BUFFER

#include <algorithm>
#include <string>
#include <stdio.h>
#include <stdarg.h>
#include "../Math/Math.h"

class MemoryBuffer
{
public:
  MemoryBuffer() : m_Position(0), m_Size(0), m_Capacity(0), m_IsExternal(true), m_Buffer(NULL)
  {
  }
  MemoryBuffer(size_t capacity, void* pBuffer = NULL) : m_Position(0), m_Size(0), m_Capacity(capacity), m_IsExternal(pBuffer!=NULL)
  {
    m_Buffer = (unsigned char*)(m_IsExternal ? pBuffer : _aligned_malloc(m_Capacity, 16));
  }
  MemoryBuffer(MemoryBuffer& a)
  {
    m_IsExternal = true;
    *this = a;
  }
  ~MemoryBuffer()
  {
    if(!m_IsExternal)
      _aligned_free(m_Buffer);
  }
  template<class T> finline void Write(const T& a)
  {
    if(m_Capacity<(m_Position + sizeof(T)))
      GrowTo(m_Position + sizeof(T));
    memcpy(m_Buffer + m_Position, &a, sizeof(T));
    m_Position += sizeof(T);
    m_Size = std::max(m_Size, m_Position);
  }
  finline void Write(const void* pData, size_t dataSize)
  {
    if(m_Capacity<(m_Position + dataSize))
      GrowTo(m_Position + dataSize);
    memcpy(m_Buffer + m_Position, pData, dataSize);
    m_Position += dataSize;
    m_Size = std::max(m_Size, m_Position);
  }
  finline void Write(const char* str)
  {
    Write(str, strlen(str) + 1);
  }
  void Print(const char* fmt, ...)
  {
    char str[4096] = { };
    va_list argptr;
    va_start(argptr, fmt);
    vsnprintf_s(str, sizeof(str) - 1, fmt, argptr);
    va_end(argptr);
    Write(str, strlen(str));
  }
  template<class T> finline size_t Read(T& a)
  {
    size_t toRead = std::min(sizeof(T), m_Size - m_Position);
    memcpy(&a, m_Buffer + m_Position, toRead);
    m_Position += toRead;
    return toRead;
  }
  finline size_t Read(void* pData, size_t dataSize)
  {
    size_t toRead = std::min(dataSize, m_Size - m_Position);
    memcpy(pData, m_Buffer + m_Position, toRead);
    m_Position += toRead;
    return toRead;
  }
  finline size_t Read(std::string& a)
  {
    size_t len = 0, pos = m_Position;
    for(size_t i=m_Position; i<m_Size; ++i, ++len)
      if(m_Buffer[i]==0) break;
    a.assign((char*)(m_Buffer + m_Position), len);
    m_Position = std::min(m_Size, m_Position + len + 1);
    return m_Position - pos;
  }
  finline size_t Size() const
  {
    return m_Size;
  }
  finline void Resize(size_t newSize)
  {
    if(m_Capacity<newSize)
      GrowTo(newSize);
    m_Size = newSize;
    m_Position = std::min(m_Position, m_Size);
  }
  finline size_t Capacity() const
  {
    return m_Capacity;
  }
  finline void Reserve(size_t newCapacity)
  {
    if(m_Capacity<newCapacity)
      GrowTo(newCapacity);
  }
  template<class T> finline const T Read()
  {
    T a;
#ifndef NDEBUG
    size_t n = 
#endif
    Read(a);
    _ASSERT(n==sizeof(T));
    return a;
  }
  template<class T> finline T* Ptr() const
  {
    return (T*)&m_Buffer[m_Position];
  }
  template<class T> finline T* Ptr(size_t offset) const
  {
    return (T*)&m_Buffer[offset];
  }
  finline size_t Position() const
  {
    return m_Position;
  }
  finline void Seek(size_t pos)
  {
    m_Position = std::min(pos, m_Size);
  }
  finline const MemoryBuffer& operator= (const MemoryBuffer& a)
  {
    if(!m_IsExternal)
      _aligned_free(m_Buffer);
    m_Position = a.m_Position;
    m_Size = a.m_Size;
    m_Capacity = a.m_Capacity;
    m_IsExternal = false;
    m_Buffer = (unsigned char*)_aligned_malloc(a.m_Capacity, 16);
    memcpy(m_Buffer, a.m_Buffer, a.m_Size);
    return *this;
  }
  bool Load(const char* pszFileName)
  {
    FILE* in;
    if(fopen_s(&in, pszFileName, "rb"))
      return false;
    fseek(in, 0, SEEK_END);
    Resize(ftell(in));
    fseek(in, 0, SEEK_SET);
    fread(m_Buffer, m_Size, 1, in);
    fclose(in);
    m_Position = 0;
    return true;
  }
  void Save(const char* pszFileName)
  {
    FILE* out;
    if(!fopen_s(&out, pszFileName, "w+b"))
    {
      fwrite(m_Buffer, m_Size, 1, out);
      fclose(out);
    }
  }
  finline int compare(const MemoryBuffer& a) const
  {
    int r = (Size()>a.Size()) - (Size()<a.Size());
    if(!r) r = memcmp(Ptr<void>(0), a.Ptr<void>(0), Size());
    return r;
  }

protected:
  unsigned char* m_Buffer;
  size_t m_Position;
  size_t m_Size;
  size_t m_Capacity;
  bool m_IsExternal;

  void GrowTo(size_t newCapacity)
  {
    m_Capacity = std::max(newCapacity, m_Capacity + m_Capacity/2);
    if(m_IsExternal)
    {
      m_IsExternal = false;
      unsigned char *p = (unsigned char*)_aligned_malloc(m_Capacity, 16);
      memcpy(p, m_Buffer, m_Size);
      m_Buffer = p;
    }
    else
    {
      m_Buffer = (unsigned char*)_aligned_realloc(m_Buffer, m_Capacity, 16);
    }
  }
};

#endif //#ifndef __MEMORY_BUFFER
