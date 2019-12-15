#ifndef __MESH_SHADERS_H
#define __MESH_SHADERS_H

#include "ShaderCache/ShaderCache.h"
#include "ShaderCache/VertexFormat.h"

struct PrePassShaderFlags
{
  unsigned DIFFUSEMAP  : 1;
  unsigned NORMALMAP   : 1;
  unsigned GLOSSMAP    : 1;
  unsigned DETAILMAP   : 1;
  unsigned ALPHATESTED : 1;
  unsigned VERTEXCOLOR : 1;
  unsigned BILLBOARD   : 1;
  unsigned SPECULAR    : 1;

  finline PrePassShaderFlags() { memset(this, 0, sizeof(*this)); }
};

class PrePassShaderDesc : public ShaderCacheTraits<PrePassShaderDesc>, public PrePassShaderFlags
{
public:
  PrePassShaderDesc(const PrePassShaderFlags& f, const VertexFormatDesc& inputDesc) : 
    PrePassShaderFlags(f), m_InputDesc(inputDesc)
  {
  }
  PrePassShaderDesc(MemoryBuffer& in)
  {
    in.Read(*static_cast<PrePassShaderFlags*>(this));
    m_InputDesc.Deserialize(in);
  }
  void Serialize(MemoryBuffer& out) const
  {
    out.Write(*static_cast<const PrePassShaderFlags*>(this));
    m_InputDesc.Serialize(out);
  }
  unsigned __int64 GetLastUpdateTime() const
  {
    char fullPath[256];
    return GetLastWriteFileTime(Platform::GetPath(Platform::File_Shader, fullPath, GetShaderFileName()));
  }
  HRESULT InitShader(ShaderObject& shader, MemoryBuffer* store) const
  {
    D3D10_SHADER_MACRO macro[] =
    {
      { "DIFFUSEMAP",  A2F(DIFFUSEMAP)  },
      { "NORMALMAP",   A2F(NORMALMAP)   },
      { "GLOSSMAP",    A2F(GLOSSMAP)    },
      { "DETAILMAP",   A2F(DETAILMAP)   },
      { "ALPHATESTED", A2F(ALPHATESTED) },
      { "VERTEXCOLOR", A2F(VERTEXCOLOR) },
      { "BILLBOARD",   A2F(BILLBOARD)   },
      { "SPECULAR",    A2F(SPECULAR)    },
      { NULL, NULL }
    };
    return shader.Init(GetShaderFileName(), GetShaderFileName(), GetShaderFileName(), NULL, NULL, NULL,
                       macro, &m_InputDesc, store);
  }

  const PrePassShaderFlags& GetShaderFlags() const { return *this; }
  const VertexFormatDesc& GetInputDesc() const { return m_InputDesc; }

protected:
  VertexFormatDesc m_InputDesc;

  static const char* GetShaderFileName() { return "_Shaders\\MeshPrePass.shader"; }
};

template<class T> class MeshShaderCompare
{
public:
  enum { bucket_size = 4, min_buckets = 32 };

  finline size_t operator() (const T& d) const
  {
    Hasher h;
    h.Hash(d.GetInputDesc().ptr(), d.GetInputDesc().size()*sizeof(D3D11_INPUT_ELEMENT_DESC));
    return h.Hash(d.GetShaderFlags());
  }
  finline bool operator() (const T& a, const T& b) const
  {
    int r = a.GetInputDesc().compare(b.GetInputDesc());
    if(!r) r = memcmp(&a.GetShaderFlags(), &b.GetShaderFlags(), sizeof(a.GetShaderFlags()));
    return r<0;
  }
};

class PrePassShaderCache : public ShaderCache<PrePassShaderDesc, MeshShaderCompare<PrePassShaderDesc> >
{
public:
  PrePassShaderCache(size_t sizeLimit) : ShaderCache(sizeLimit)
  {
    Platform::Add(Platform::OnInitDelegate::from_method<PrePassShaderCache, &PrePassShaderCache::OnPlatformInit>(this), Platform::Object_Shader);
    Platform::Add(Platform::OnShutdownDelegate::from_method<PrePassShaderCache, &PrePassShaderCache::OnPlatformShutdown>(this), Platform::Object_Shader);
  }

protected:
  bool OnPlatformInit() { Init(); return Load(GetFileName()); }
  void OnPlatformShutdown() { Save(GetFileName()); Clear(); }
  static const char* GetFileName() { return "MeshPrePass.bin"; }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct DepthOnlyShaderFlags
{
  unsigned ALPHATESTED    : 1;
  unsigned BILLBOARD      : 1;
  unsigned CUBEMAP        : 1;
  unsigned CUBEMAPS_ARRAY : 1;
  unsigned PARABOLIC      : 1;
  unsigned ASM_LAYER      : 1;

  finline DepthOnlyShaderFlags() { memset(this, 0, sizeof(*this)); }
};

class DepthOnlyShaderDesc : public ShaderCacheTraits<DepthOnlyShaderDesc>, public DepthOnlyShaderFlags
{
public:
  DepthOnlyShaderDesc(const DepthOnlyShaderFlags& f, const VertexFormatDesc& inputDesc) : 
    DepthOnlyShaderFlags(f), m_InputDesc(inputDesc)
  {
  }
  DepthOnlyShaderDesc(MemoryBuffer& in)
  {
    in.Read(*static_cast<DepthOnlyShaderFlags*>(this));
    m_InputDesc.Deserialize(in);
  }
  void Serialize(MemoryBuffer& out) const
  {
    out.Write(*static_cast<const DepthOnlyShaderFlags*>(this));
    m_InputDesc.Serialize(out);
  }
  unsigned __int64 GetLastUpdateTime() const
  {
    char fullPath[256];
    unsigned __int64 t = GetLastWriteFileTime(Platform::GetPath(Platform::File_Shader, fullPath, GetDepthOnlyShaderFileName()));
    t = std::max(t, GetLastWriteFileTime(Platform::GetPath(Platform::File_Shader, fullPath, GetDepthOnlyTessShaderFileName())));
    return t;
  }
  HRESULT InitShader(ShaderObject& shader, MemoryBuffer* store) const
  {
    D3D10_SHADER_MACRO macro[] =
    {
      { "ALPHATESTED",    A2F(ALPHATESTED)    },
      { "BILLBOARD",      A2F(BILLBOARD)      },
      { "CUBEMAP",        A2F(CUBEMAP)        },
      { "CUBEMAPS_ARRAY", A2F(CUBEMAPS_ARRAY) },
      { "ASM_LAYER",      A2F(ASM_LAYER)      },
      { NULL, NULL }
    };
    return shader.Init(PARABOLIC ? GetDepthOnlyTessShaderFileName() : GetDepthOnlyShaderFileName() ,
                       PARABOLIC ? NULL : (CUBEMAP ? GetDepthOnlyShaderFileName() : NULL),
                       PARABOLIC ? GetDepthOnlyTessShaderFileName() : GetDepthOnlyShaderFileName(),
                       NULL,
                       PARABOLIC ? GetDepthOnlyTessShaderFileName() : NULL,
                       PARABOLIC ? GetDepthOnlyTessShaderFileName() : NULL,
                       macro, &m_InputDesc, store);
  }

  const DepthOnlyShaderFlags& GetShaderFlags() const { return *this; }
  const VertexFormatDesc& GetInputDesc() const { return m_InputDesc; }

protected:
  VertexFormatDesc m_InputDesc;

  static const char* GetDepthOnlyShaderFileName() { return "_Shaders\\MeshDepthOnly.shader"; }
  static const char* GetDepthOnlyTessShaderFileName() { return "_Shaders\\MeshDepthOnlyTess.shader"; }
};

class DepthOnlyShaderCache : public ShaderCache<DepthOnlyShaderDesc, MeshShaderCompare<DepthOnlyShaderDesc> >
{
public:
  DepthOnlyShaderCache(size_t sizeLimit) : ShaderCache(sizeLimit)
  {
    Platform::Add(Platform::OnInitDelegate::from_method<DepthOnlyShaderCache, &DepthOnlyShaderCache::OnPlatformInit>(this), Platform::Object_Shader);
    Platform::Add(Platform::OnShutdownDelegate::from_method<DepthOnlyShaderCache, &DepthOnlyShaderCache::OnPlatformShutdown>(this), Platform::Object_Shader);
  }

protected:
  bool OnPlatformInit() { Init(); return Load(GetFileName()); }
  void OnPlatformShutdown() { Save(GetFileName()); Clear(); }
  static const char* GetFileName() { return "MeshDepthOnly.bin"; }
};

#endif //#ifndef __MESH_SHADERS_H
