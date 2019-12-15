#ifndef __LIGHTING_SHADERS_H
#define __LIGHTING_SHADERS_H

#include "ShaderCache/ShaderCache.h"
#include "ShaderCache/VertexFormat.h"

struct LightingShaderFlags
{
  unsigned SHADOWS_CUBEMAP     : 1;
  unsigned FAST_RENDER         : 1;
  unsigned BLENDING            : 1;
  unsigned HEMISPHERICAL_LIGHT : 1;
  unsigned USE_PCF9            : 1;
  unsigned USE_CBM_ARRAY       : 1;

  finline LightingShaderFlags() { memset(this, 0, sizeof(*this)); }
};

class LightingShaderDesc : public ShaderCacheTraits<LightingShaderDesc>, public LightingShaderFlags
{
public:
  LightingShaderDesc(const LightingShaderFlags& f) : LightingShaderFlags(f)
  {
  }
  LightingShaderDesc(MemoryBuffer& in)
  {
    in.Read(*static_cast<LightingShaderFlags*>(this));
  }
  void Serialize(MemoryBuffer& out) const
  {
    out.Write(*static_cast<const LightingShaderFlags*>(this));
  }
  unsigned __int64 GetLastUpdateTime() const
  {
    char fullPath[256];
    return GetLastWriteFileTime(Platform::GetPath(Platform::File_Shader, fullPath, "_Shaders\\Lighting.shader"));
  }
  HRESULT InitShader(ShaderObject& shader, MemoryBuffer* store) const
  {
    D3D10_SHADER_MACRO macro[] =
    {
      { "SHADOWS_CUBEMAP",     A2F(SHADOWS_CUBEMAP)     },
      { "FAST_RENDER",         A2F(FAST_RENDER)         },
      { "BLENDING",            A2F(BLENDING)            },
      { "HEMISPHERICAL_LIGHT", A2F(HEMISPHERICAL_LIGHT) },
      { "USE_PCF9",            A2F(USE_PCF9)            },
      { "USE_CBM_ARRAY",       A2F(USE_CBM_ARRAY)       },
      { NULL, NULL }
    };
    return shader.Init(NULL, NULL, NULL, "_Shaders\\Lighting.shader", NULL, NULL,
                       macro, NULL, store, NULL, NULL, NULL, USE_CBM_ARRAY ? "cs_5_0" : "cs_4_0");
  }
};

class LightingShaderCache : public ShaderCache<LightingShaderDesc>
{
public:
  LightingShaderCache(size_t sizeLimit) : ShaderCache(sizeLimit)
  {
    Platform::Add(Platform::OnInitDelegate::from_method<LightingShaderCache, &LightingShaderCache::OnPlatformInit>(this), Platform::Object_Shader);
    Platform::Add(Platform::OnShutdownDelegate::from_method<LightingShaderCache, &LightingShaderCache::OnPlatformShutdown>(this), Platform::Object_Shader);
  }

protected:
  bool OnPlatformInit() { Init(); return Load(GetFileName()); }
  void OnPlatformShutdown() { Save(GetFileName()); Clear(); }
  static const char* GetFileName() { return "Lighting.bin"; }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct LightingQuadShaderFlags
{
  unsigned SHADOWS_CUBEMAP     : 1;
  unsigned USE_CBM_ARRAY       : 1;
  unsigned TGSM_WORKAROUND     : 1;
  unsigned HEMISPHERICAL_LIGHT : 1;

  finline LightingQuadShaderFlags() { memset(this, 0, sizeof(*this)); }
};

class LightingQuadShaderDesc : public ShaderCacheTraits<LightingQuadShaderDesc>, public LightingQuadShaderFlags
{
public:
  LightingQuadShaderDesc(const LightingQuadShaderFlags& f) : LightingQuadShaderFlags(f)
  {
  }
  LightingQuadShaderDesc(MemoryBuffer& in)
  {
    in.Read(*static_cast<LightingQuadShaderFlags*>(this));
  }
  void Serialize(MemoryBuffer& out) const
  {
    out.Write(*static_cast<const LightingQuadShaderFlags*>(this));
  }
  unsigned __int64 GetLastUpdateTime() const
  {
    char fullPath[256];
    return GetLastWriteFileTime(Platform::GetPath(Platform::File_Shader, fullPath, "_Shaders\\LightCulling.shader"));
  }
  HRESULT InitShader(ShaderObject& shader, MemoryBuffer* store) const
  {
    D3D10_SHADER_MACRO macro[] =
    {
      { "SHADOWS_CUBEMAP", A2F(SHADOWS_CUBEMAP) },
      { "USE_CBM_ARRAY",   A2F(USE_CBM_ARRAY)   },
      { "TGSM_WORKAROUND", A2F(TGSM_WORKAROUND) },
      { "HEMISPHERICAL_LIGHT", A2F(HEMISPHERICAL_LIGHT) },
      { NULL, NULL }
    };
    return shader.Init(NULL, NULL, NULL, "_Shaders\\LightCulling.shader", NULL, NULL,
                       macro, NULL, store, NULL, NULL, NULL, USE_CBM_ARRAY ? "cs_5_0" : "cs_4_0");
  }
};

class LightingQuadShaderCache : public ShaderCache<LightingQuadShaderDesc>
{
public:
  LightingQuadShaderCache(size_t sizeLimit) : ShaderCache(sizeLimit)
  {
    Platform::Add(Platform::OnInitDelegate::from_method<LightingQuadShaderCache, &LightingQuadShaderCache::OnPlatformInit>(this), Platform::Object_Shader);
    Platform::Add(Platform::OnShutdownDelegate::from_method<LightingQuadShaderCache, &LightingQuadShaderCache::OnPlatformShutdown>(this), Platform::Object_Shader);
  }

protected:
  bool OnPlatformInit() { Init(); return Load(GetFileName()); }
  void OnPlatformShutdown() { Save(GetFileName()); Clear(); }
  static const char* GetFileName() { return "LightingQuad.bin"; }
};

#endif //#ifndef __LIGHTING_SHADERS_H
