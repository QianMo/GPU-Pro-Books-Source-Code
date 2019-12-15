#ifndef __PLATFORM11
#define __PLATFORM11

#include "RenderContext11.h"
#include "DeviceContext11.h"
#include "delegate/Delegate.h"

class RenderTarget2D;

class Platform
{
public:
  typedef Delegate<bool ()> OnInitDelegate;
  typedef Delegate<void ()> OnShutdownDelegate;

  enum PredefinedSamplers
  {
    Sampler_Linear_Clamp = 0,
    Sampler_Point_Clamp,
    Sampler_Linear_Wrap,
    Sampler_Point_Wrap,
    Sampler_ShadowMap,
    Sampler_ShadowMap_PCF,
  };
  enum FileType
  {
    File_Shader = 0,
    File_Texture,
    File_Mesh,
    NumberOfFileTypes
  };
  enum ObjectHint
  {
    Object_Generic = 0,
    Object_Shader,
    Object_Texture,
    NumberOfObjectTypes
  };

  static HRESULT Init(ID3D11Device*, ID3D11DeviceContext*);
  static void Shutdown();
  static HRESULT SetFrameBuffer(RenderTarget2D*, RenderTarget2D*);
  static RenderTarget2D* GetBackBufferRT();
  static RenderTarget2D* GetBackBufferDS();
  static D3D_FEATURE_LEVEL GetFeatureLevel() { return s_FeatureLevel; }

  static bool Add(const OnInitDelegate&, ObjectHint objHint = Object_Generic);
  static void Remove(const OnInitDelegate&);
  static void Add(const OnShutdownDelegate&, ObjectHint objHint = Object_Generic);
  static void Remove(const OnShutdownDelegate&);
  static bool InvokeOnInit(ObjectHint);
  static void InvokeOnShutdown(ObjectHint);
  static void RemoveAllCallbacks();

  static unsigned GetFormatBitsPerPixel(DXGI_FORMAT);
  static void SetDir(FileType fileType, const char* pszDir) { s_Directories[fileType] = pszDir; }
  static char* GetPath(FileType, char*, const char*);
  static void* AllocateScratchMemory(size_t);
  static void FreeScratchMemory(void*);

  static finline ID3D11Device* GetD3DDevice() { return s_Device11; }
  static finline DeviceContext11& GetImmediateContext() { return s_ImmediateContext; }
  static finline BlendCache11& GetBlendCache() { return s_BlendCache; }
  static finline RasterizerCache11& GetRasterizerCache() { return s_RasterizerCache; }
  static finline DepthStencilCache11& GetDepthStencilCache() { return s_DepthStencilCache; }
  static finline SamplerCache11& GetSamplerCache() { return s_SamplerCache; }

  static finline const SamplerState11* GetSamplerState(const SamplerDesc11& desc)
  {
    return &s_SamplerCache.Get(desc);
  }
  template<class T> static void Unbind(const T& a)
  {
    s_ImmediateContext.Unbind(a);
  }

protected:
  static ID3D11Device* s_Device11;
  static D3D_FEATURE_LEVEL s_FeatureLevel;
  static DeviceContext11 s_ImmediateContext;
  static BlendCache11 s_BlendCache;
  static RasterizerCache11 s_RasterizerCache;
  static DepthStencilCache11 s_DepthStencilCache;
  static SamplerCache11 s_SamplerCache;
  static std::string s_Directories[NumberOfFileTypes];
  static std::vector<OnInitDelegate> s_OnInit[NumberOfObjectTypes];
  static std::vector<OnShutdownDelegate> s_OnShutdown[NumberOfObjectTypes];
};

#endif //#ifndef __PLATFORM11
