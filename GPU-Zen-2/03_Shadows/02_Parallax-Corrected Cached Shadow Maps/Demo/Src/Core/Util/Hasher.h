#ifndef __HASHER_H
#define __HASHER_H

template<class H, H c_InitVal, H c_Magic> class _Hasher
{
public:
  finline _Hasher() : m_Hash(c_InitVal) { }
  finline H GetHash() const { return m_Hash; }

  template<class T> finline H Hash(const T& a)
  {
    const unsigned char * restrict p = (unsigned char*)&a;
    for(size_t i=0; i<sizeof(T); ++i)
    {
      m_Hash *= c_Magic;
      m_Hash ^= p[i];
    }
    return m_Hash;
  }
  finline H Hash(const void* pData, size_t dataSize)
  {
    const unsigned char * restrict p = (unsigned char*)pData;
    for(size_t i=0; i<dataSize; ++i)
    {
      m_Hash *= c_Magic;
      m_Hash ^= p[i];
    }
    return m_Hash;
  }
  finline H Hash(const char* pszString)
  {
    return Hash(pszString, strlen(pszString));
  }
  finline int Compare(const _Hasher& h) const
  {
    return (m_Hash>h.m_Hash) - (m_Hash<h.m_Hash);
  }

protected:
  H m_Hash;
};

typedef _Hasher<unsigned, 0x811c9dc5U, 0x01000193U> Hasher32;
typedef _Hasher<unsigned __int64, 0xcbf29ce484222325ULL, 0x100000001b3ULL> Hasher64;
typedef Hasher32 Hasher;

#endif //#ifndef __HASHER_H
