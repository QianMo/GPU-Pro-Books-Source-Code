// IMPORTANT: you have to use a cache with pre-defined capacity if you want to use
// cache's internal copies of ShaderObject. If the capacity is not limited, 
// YOU HAVE TO MAKE OWN COPY via ShaderObject::Clone() before doing anything.

#ifndef __SIMPLE_SHADER
#define __SIMPLE_SHADER

#include "ShaderCache.h"

class SimpleShaderDesc : public ShaderCacheTraits<SimpleShaderDesc>
{
public:
  SimpleShaderDesc(const char* pszPSH,
                   const char* pszGSH,
                   const char* pszVSH,
                   const char* pszCSH,
                   const char* pszHSH,
                   const char* pszDSH,
                   const D3D11_INPUT_ELEMENT_DESC* pInputDesc = NULL, 
                   size_t inputDescElements = 0) : m_InputDesc(pInputDesc, inputDescElements)
  {
    if(pszPSH!=NULL) m_PSH = pszPSH;
    if(pszGSH!=NULL) m_GSH = pszGSH;
    if(pszVSH!=NULL) m_VSH = pszVSH;
    if(pszCSH!=NULL) m_CSH = pszCSH;
    if(pszHSH!=NULL) m_HSH = pszHSH;
    if(pszDSH!=NULL) m_DSH = pszDSH;
  }
  SimpleShaderDesc(MemoryBuffer& in)
  {
    in.Read(m_PSH);
    in.Read(m_GSH);
    in.Read(m_VSH);
    in.Read(m_CSH);
    in.Read(m_HSH);
    in.Read(m_DSH);
    m_InputDesc.Deserialize(in);
  }
  void SetFlags(unsigned NFlags, const char** ppFlags)
  {
    char pszFlags[256] = { };
    for(unsigned i=0; i<NFlags; ++i)
    {
      const char* str = ppFlags[i];
      if(str!=NULL)
      {
        strcat(pszFlags, "?");
        strcat(pszFlags, str);
      }
    }
    SetFlags(m_PSH, pszFlags);
    SetFlags(m_GSH, pszFlags);
    SetFlags(m_VSH, pszFlags);
    SetFlags(m_CSH, pszFlags);
    SetFlags(m_HSH, pszFlags);
    SetFlags(m_DSH, pszFlags);
  }
  void Serialize(MemoryBuffer& out) const
  {
    out.Write(m_PSH.c_str());
    out.Write(m_GSH.c_str());
    out.Write(m_VSH.c_str());
    out.Write(m_CSH.c_str());
    out.Write(m_HSH.c_str());
    out.Write(m_DSH.c_str());
    m_InputDesc.Serialize(out);
  }
  unsigned __int64 GetLastUpdateTime() const
  {
    char fullPath[256];
    unsigned __int64 t = 0;
    if(m_PSH.length()) t = std::max(t, GetLastWriteFileTime(GetFullShaderPath(fullPath, m_PSH.c_str())));
    if(m_GSH.length()) t = std::max(t, GetLastWriteFileTime(GetFullShaderPath(fullPath, m_GSH.c_str())));
    if(m_VSH.length()) t = std::max(t, GetLastWriteFileTime(GetFullShaderPath(fullPath, m_VSH.c_str())));
    if(m_CSH.length()) t = std::max(t, GetLastWriteFileTime(GetFullShaderPath(fullPath, m_CSH.c_str())));
    if(m_HSH.length()) t = std::max(t, GetLastWriteFileTime(GetFullShaderPath(fullPath, m_HSH.c_str())));
    if(m_DSH.length()) t = std::max(t, GetLastWriteFileTime(GetFullShaderPath(fullPath, m_DSH.c_str())));
    return t;
  }
  HRESULT InitShader(ShaderObject& shader, MemoryBuffer* store) const
  {
    std::vector<D3D10_SHADER_MACRO> macro;
    char* pszPSH = NULL; if(m_PSH.length()) { pszPSH = (char*)alloca(m_PSH.length() + 1); strcpy(pszPSH, m_PSH.c_str()); ExtractMacro(pszPSH, macro); }
    char* pszGSH = NULL; if(m_GSH.length()) { pszGSH = (char*)alloca(m_GSH.length() + 1); strcpy(pszGSH, m_GSH.c_str()); ExtractMacro(pszGSH, macro); }
    char* pszVSH = NULL; if(m_VSH.length()) { pszVSH = (char*)alloca(m_VSH.length() + 1); strcpy(pszVSH, m_VSH.c_str()); ExtractMacro(pszVSH, macro); }
    char* pszCSH = NULL; if(m_CSH.length()) { pszCSH = (char*)alloca(m_CSH.length() + 1); strcpy(pszCSH, m_CSH.c_str()); ExtractMacro(pszCSH, macro); }
    char* pszHSH = NULL; if(m_HSH.length()) { pszHSH = (char*)alloca(m_HSH.length() + 1); strcpy(pszHSH, m_HSH.c_str()); ExtractMacro(pszHSH, macro); }
    char* pszDSH = NULL; if(m_DSH.length()) { pszDSH = (char*)alloca(m_DSH.length() + 1); strcpy(pszDSH, m_DSH.c_str()); ExtractMacro(pszDSH, macro); }
    D3D10_SHADER_MACRO* pMacro = NULL; if(macro.size()>0) { D3D10_SHADER_MACRO m = { }; macro.push_back(m); pMacro = &macro.front(); }
    return shader.Init(pszPSH, pszGSH, pszVSH, pszCSH, pszHSH, pszDSH, pMacro, &m_InputDesc, store);
  }

protected:
  std::string m_PSH;
  std::string m_GSH;
  std::string m_VSH;
  std::string m_CSH;
  std::string m_HSH;
  std::string m_DSH;
  VertexFormatDesc m_InputDesc;

  static void ExtractMacro(char* str, std::vector<D3D10_SHADER_MACRO>& macro)
  {
    char *pContext, *pToken;
    while((pToken=strtok_s(str, "?", &pContext)) != NULL)
    {
      if(str==NULL && macro.cend()==std::find_if(macro.cbegin(), macro.cend(), [&] (const D3D10_SHADER_MACRO& m) -> bool { return !strcmp(m.Name, pToken); }))
      {
        D3D10_SHADER_MACRO m = { pToken, "1" };
        macro.push_back(m);
      }
      str = NULL;
    }
  }
  static const char* GetFullShaderPath(char* fullPath, const char* fileName)
  {
    Platform::GetPath(Platform::File_Shader, fullPath, fileName);
    char* p = strchr(fullPath, '?'); if(p!=NULL) *p = 0;
    return fullPath;
  }
  static void SetFlags(std::string& desc, const char* f)
  {
    if(!desc.empty())
    {
      size_t pos = desc.find('?');
      if(pos!=std::string::npos) desc.resize(pos);
      desc.append(f);
    }
  }

  friend class SimpleShaderCompare;
};

class SimpleShaderCompare
{
public:
  enum { bucket_size = 4, min_buckets = 32 };

  finline size_t operator() (const SimpleShaderDesc& d) const
  {
    Hasher h;
    h.Hash(d.m_PSH.c_str());
    h.Hash(d.m_GSH.c_str());
    h.Hash(d.m_VSH.c_str());
    h.Hash(d.m_CSH.c_str());
    h.Hash(d.m_HSH.c_str());
    h.Hash(d.m_DSH.c_str());
    h.Hash(d.m_InputDesc.ptr(), d.m_InputDesc.size()*sizeof(D3D11_INPUT_ELEMENT_DESC));
    return h.GetHash();
  }
  finline bool operator() (const SimpleShaderDesc& a, const SimpleShaderDesc& b) const
  {
    int r = a.m_InputDesc.compare(b.m_InputDesc);
    if(!r) r = a.m_PSH.compare(b.m_PSH);
    if(!r) r = a.m_GSH.compare(b.m_GSH);
    if(!r) r = a.m_VSH.compare(b.m_VSH);
    if(!r) r = a.m_CSH.compare(b.m_CSH);
    if(!r) r = a.m_HSH.compare(b.m_HSH);
    if(!r) r = a.m_DSH.compare(b.m_DSH);
    return r<0;
  }
};

class SimpleShaderCache : public ShaderCache<SimpleShaderDesc, SimpleShaderCompare>
{
public:
  SimpleShaderCache(size_t sizeLimit) : ShaderCache(sizeLimit)
  {
    Platform::Add(Platform::OnInitDelegate::from_method<SimpleShaderCache, &SimpleShaderCache::OnPlatformInit>(this), Platform::Object_Shader);
    Platform::Add(Platform::OnShutdownDelegate::from_method<SimpleShaderCache, &SimpleShaderCache::OnPlatformShutdown>(this), Platform::Object_Shader);
  }

protected:
  bool OnPlatformInit() { Init(); return Load(GetFileName()); }
  void OnPlatformShutdown() { Save(GetFileName()); Clear(); }
  static const char* GetFileName() { return "Simple.bin"; }
};

extern SimpleShaderCache g_SimpleShaderCache;

#endif //#ifndef __SIMPLE_SHADER
