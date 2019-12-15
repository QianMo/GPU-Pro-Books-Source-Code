// IMPORTANT: you have to use a cache with pre-defined capacity if you want to use
// cache's internal copies of ShaderObject. If the capacity is not limited, 
// YOU HAVE TO MAKE OWN COPY via ShaderObject::Clone() before doing anything.

#ifndef __SHADER_CACHE
#define __SHADER_CACHE

#include "ShaderObject11.h"
#include "../../Util/Cache.h"

inline unsigned __int64 GetLastWriteFileTime(const char* pszFile)
{
  union { unsigned __int64 t64; FILETIME t; };
  t64 = 0;
  HANDLE hFile = CreateFileA(pszFile, 0, 0, NULL, OPEN_EXISTING, 0, NULL);
  if(hFile!=NULL)
  {
    GetFileTime(hFile, NULL, NULL, &t);
    CloseHandle(hFile);
  }
  return t64;
}

template<class T> class ShaderCacheTraits
{
public:
  typedef T Description;
  typedef ShaderObject CacheEntry;
  typedef MemoryBuffer* UserParam;

  static finline void Allocate(const UserParam& store, const Description& d, CacheEntry& e)
  {
    if(FAILED(d.InitShader(e, store)))
      e.Clear();
  }
  static finline void Free(const UserParam&, CacheEntry& e)
  {
    e.Clear();
  }

protected:
  static finline const char* A2F(unsigned i)
  {
    static const char* const s[] = { "0", "1", "2", "3", "4", "5", "6", "7" };
    _ASSERT(i<ARRAYSIZE(s));
    return s[i];
  }
};

template<class T, class COMPARE = PODCompare<typename T::Description, 8, 64> >
  class ShaderCache : public Cache<T, COMPARE>
{
public:
  ShaderCache(size_t sizeLimit = 0) : Cache(sizeLimit), m_LastUpdateTime(0) { }

  void Init()
  {
    m_Code.Resize(0);
    __super::Init(&m_Code, false);
  }
  bool Load(const char* pszFile)
  {
    char fullPath[256];
    MemoryBuffer in;
    if(!in.Load(Platform::GetPath(Platform::File_Shader, fullPath, pszFile)))
    {
      Log::Info("Cache does not exist: %s\n", pszFile);
      return true;
    }
    std::string sig; 
    in.Read(sig);
    if(sig!=CacheFileSignature())
    {
      Log::Info("Ignoring due to an unrecognized signature: %s\n", pszFile);
      return true;
    }

    unsigned nShaders = in.Read<unsigned>();
    std::vector<size_t> indexMap(nShaders);
    for(unsigned i=0; i<nShaders; ++i)
      indexMap[i] = GetIndex(Description(in));

    std::vector<size_t> dataOffset(nShaders);
    for(unsigned i=0; i<nShaders; ++i)
    {
      dataOffset[i] = in.Position();
      in.Seek(in.Position() + *in.Ptr<unsigned>());
    }
    size_t nLoaded = 0;
    size_t nDesc = m_Descriptions.size();
    for(size_t i=0; i<nDesc; ++i)
    {
      auto it = std::find(indexMap.cbegin(), indexMap.cend(), i);
      if(it!=indexMap.cend())
      {
        size_t shaderIndex = it - indexMap.cbegin();
        in.Seek(dataOffset[shaderIndex]);
        m_CacheEntries.resize(m_CacheEntries.size() + 1);
        HRESULT hr = m_CacheEntries.back().Init(in, &m_Code);
        if(FAILED(hr))
        {
          m_CacheEntries.back().Clear();
          m_CacheEntries.pop_back();
          Log::Error("Error (HRESULT=0x%x) loading shader (%d): %s\n", hr, shaderIndex, pszFile);
          return false;
        }
        ++nLoaded;
      }
      else
        GetByIndex(i);
    }
//    Log::Info("Shader cache \"%s\": %d objects\n", pszFile, nLoaded);

    m_LastUpdateTime = GetLastWriteFileTime(fullPath);
    Update();
    return true;
  }
  void Save(const char* pszFile)
  {
    size_t nShaders = m_Descriptions.size();
    if(nShaders>0)
      GetByIndex(nShaders - 1); // ensure the entire cache is compiled

    Update();

    MemoryBuffer out;
    out.Write(CacheFileSignature());
    out.Write<unsigned>(nShaders);
    for(size_t i=0; i<nShaders; ++i)
    {
      auto it = std::find_if(m_Descriptions.cbegin(), m_Descriptions.cend(), [&] (const Descriptions::value_type& d) -> bool { return d.second==i; });
      _ASSERT(it!=m_Descriptions.cend() && "internal structure is broken");
      it->first.Serialize(out);
    }
    out.Write(m_Code.Ptr<void>(0), m_Code.Size());

    char fullPath[256];
    out.Save(Platform::GetPath(Platform::File_Shader, fullPath, pszFile));
  }
  void Update()
  {
    MemoryBuffer newCode;
    newCode.Reserve(m_Code.Size());
    m_Code.Seek(0);

    unsigned __int64 latestTime = m_LastUpdateTime;
    size_t nShaders = m_Descriptions.size();
    for(size_t i=0; i<nShaders; ++i)
    {
      auto it = std::find_if(m_Descriptions.cbegin(), m_Descriptions.cend(), [&] (const Descriptions::value_type& d) -> bool { return d.second==i; });
      _ASSERT(it!=m_Descriptions.cend() && "internal structure is broken");

      size_t startPos = m_Code.Position();
      unsigned blockSize = m_Code.Read<unsigned>();
      m_Code.Seek(startPos + blockSize);

      unsigned __int64 t = it->first.GetLastUpdateTime();
      if(m_LastUpdateTime<t)
      {
        ShaderObject& obj = m_CacheEntries[i];
        obj.Clear();
        T::Allocate(&newCode, it->first, obj);
        latestTime = std::max(latestTime, t);
      }
      else
        newCode.Write(m_Code.Ptr<void>(startPos), blockSize);
    }
    m_Code = newCode;
    m_LastUpdateTime = latestTime;
  }
  const char* CacheFileSignature() const
  {
    return "Dociousaliexpilisticfragicalirupus";
  }

protected:
  unsigned __int64 m_LastUpdateTime;
  MemoryBuffer m_Code;
};

#endif //#ifndef __SHADER_MANAGER11
