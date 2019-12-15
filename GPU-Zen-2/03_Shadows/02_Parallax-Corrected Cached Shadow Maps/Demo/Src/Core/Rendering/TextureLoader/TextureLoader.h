// IMPORTANT: you have to use a cache with pre-defined capacity if you want to use
// cache's internal copies of Texture2D. If the capacity is not limited, 
// YOU HAVE TO MAKE OWN COPY via Texture2D::Clone() before doing anything.

#ifndef __TEXTURE_LOADER_H
#define __TEXTURE_LOADER_H

#include "Texture11.h"

class FileTextureDesc
{
public:
  FileTextureDesc(const char* pszFileName, D3D11_USAGE usage = D3D11_USAGE_IMMUTABLE, unsigned bindFlags = D3D11_BIND_SHADER_RESOURCE, unsigned CPUAccessFlags = 0, unsigned miscFlags = 0) :
    m_FileName(pszFileName), m_Usage(usage), m_BindFlags(bindFlags), m_CPUAccessFlags(CPUAccessFlags), m_MiscFlags(miscFlags) { }

  const std::string& GetFileName() const { return m_FileName; }

  typedef FileTextureDesc Description;
  typedef Texture2D CacheEntry;
  typedef void* UserParam;

  static finline void Allocate(const UserParam&, const Description& d, CacheEntry& e)
  {
    if(FAILED(e.Init(d.m_FileName.c_str(), d.m_Usage, d.m_BindFlags, d.m_CPUAccessFlags, d.m_MiscFlags)))
      e.Clear();
  }
  static finline void Free(const UserParam&, CacheEntry& e)
  {
    e.Clear();
  }

protected:
  std::string m_FileName;
  D3D11_USAGE m_Usage;
  unsigned m_BindFlags;
  unsigned m_CPUAccessFlags;
  unsigned m_MiscFlags;
};

template<class T> class FileResourceDescCompare
{
public:
  enum { bucket_size = 4, min_buckets = 32 };

  finline size_t operator() (const T& d) const
  {
    Hasher h;
    return h.Hash(d.GetFileName().c_str());
  }
  finline bool operator() (const T& a, const T& b) const
  {
    return a.GetFileName().compare(b.GetFileName())<0;
  }
};

class TextureLoader : public Cache<FileTextureDesc, FileResourceDescCompare<FileTextureDesc> >
{
public:
  TextureLoader(size_t sizeLimit = 0) : Cache(sizeLimit) { }
};

class SystemTextureLoader : public TextureLoader
{
public:
  SystemTextureLoader(size_t sizeLimit = 0) : TextureLoader(sizeLimit)
  {
    Platform::Add(Platform::OnInitDelegate::from_method<SystemTextureLoader, &SystemTextureLoader::OnPlatformInit>(this), Platform::Object_Texture);
    Platform::Add(Platform::OnShutdownDelegate::from_method<SystemTextureLoader, &SystemTextureLoader::OnPlatformShutdown>(this), Platform::Object_Texture);
  }

protected:
  bool OnPlatformInit() { Init(NULL, true); return true; }
  void OnPlatformShutdown() { Clear(); }
};

extern SystemTextureLoader g_SystemTextureLoader;

#endif //#ifndef __TEXTURE_LOADER_H
