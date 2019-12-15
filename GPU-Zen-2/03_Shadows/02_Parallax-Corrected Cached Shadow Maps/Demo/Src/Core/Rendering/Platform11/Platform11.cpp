#include "PreCompile.h"
#include "Platform11.h"
#include "TextureLoader/TextureLoader.h"
#include "ShaderCache/SimpleShader.h"
#include "../../Util/MemoryPool.h"

std::vector<Platform::OnInitDelegate> Platform::s_OnInit[Platform::NumberOfObjectTypes];
std::vector<Platform::OnShutdownDelegate> Platform::s_OnShutdown[Platform::NumberOfObjectTypes];

ID3D11Device* Platform::s_Device11;
DeviceContext11 Platform::s_ImmediateContext;
D3D_FEATURE_LEVEL Platform::s_FeatureLevel;

BlendCache11 Platform::s_BlendCache(2048);
RasterizerCache11 Platform::s_RasterizerCache(2048);
DepthStencilCache11 Platform::s_DepthStencilCache(2048);
SamplerCache11 Platform::s_SamplerCache(256);

static RenderTarget2D s_BackBufferRT;
static RenderTarget2D s_BackBufferDS;

RenderTarget2D* Platform::GetBackBufferRT() { return &s_BackBufferRT; }
RenderTarget2D* Platform::GetBackBufferDS() { return &s_BackBufferDS; }

HRESULT Platform::Init(ID3D11Device* Device11, ID3D11DeviceContext* Context11)
{
#ifndef RUNTIME_CHECKS_OFF
  ID3D11InfoQueue* pInfoQueue;
  if(SUCCEEDED(Device11->QueryInterface(__uuidof(ID3D11InfoQueue), (void**)&pInfoQueue)))
  {
    static const D3D11_MESSAGE_ID c_DisableMsg[] = 
    {
      D3D11_MESSAGE_ID_DEVICE_DRAW_CONSTANT_BUFFER_TOO_SMALL, // this is how constant buffers pool works
      D3D11_MESSAGE_ID_SETPRIVATEDATA_CHANGINGPARAMS, // DXUT code triggers this
    };
    D3D11_INFO_QUEUE_FILTER queueFilter;
    memset(&queueFilter, 0, sizeof(queueFilter));
    queueFilter.DenyList.NumIDs = ARRAYSIZE(c_DisableMsg);
    queueFilter.DenyList.pIDList = const_cast<D3D11_MESSAGE_ID*>(c_DisableMsg);
    pInfoQueue->AddStorageFilterEntries(&queueFilter);
    pInfoQueue->SetBreakOnCategory(D3D11_MESSAGE_CATEGORY_INITIALIZATION, true);
    pInfoQueue->SetBreakOnCategory(D3D11_MESSAGE_CATEGORY_CLEANUP, true);
    pInfoQueue->SetBreakOnCategory(D3D11_MESSAGE_CATEGORY_COMPILATION, true);
    pInfoQueue->SetBreakOnCategory(D3D11_MESSAGE_CATEGORY_STATE_CREATION, true);
    pInfoQueue->SetBreakOnCategory(D3D11_MESSAGE_CATEGORY_STATE_SETTING, true);
    pInfoQueue->SetBreakOnCategory(D3D11_MESSAGE_CATEGORY_STATE_GETTING, true);
    pInfoQueue->SetBreakOnCategory(D3D11_MESSAGE_CATEGORY_RESOURCE_MANIPULATION, true);
    pInfoQueue->SetBreakOnCategory(D3D11_MESSAGE_CATEGORY_EXECUTION, true);
    pInfoQueue->SetMuteDebugOutput(false);
    pInfoQueue->Release();
  }
#endif
  s_Device11 = Device11; Device11->AddRef();
  s_RasterizerCache.Init(s_Device11, true);
  s_BlendCache.Init(s_Device11, true);
  s_DepthStencilCache.Init(s_Device11, true);
  s_SamplerCache.Init(s_Device11, true);
  if(s_SamplerCache.IsEmpty())
  {
    s_SamplerCache.GetIndex(SamplerDesc11(D3D11_FILTER_MIN_MAG_MIP_LINEAR, D3D11_TEXTURE_ADDRESS_CLAMP, D3D11_TEXTURE_ADDRESS_CLAMP, D3D11_TEXTURE_ADDRESS_CLAMP));
    s_SamplerCache.GetIndex(SamplerDesc11(D3D11_FILTER_MIN_MAG_MIP_POINT, D3D11_TEXTURE_ADDRESS_CLAMP, D3D11_TEXTURE_ADDRESS_CLAMP, D3D11_TEXTURE_ADDRESS_CLAMP));
    s_SamplerCache.GetIndex(SamplerDesc11(D3D11_FILTER_MIN_MAG_MIP_LINEAR, D3D11_TEXTURE_ADDRESS_WRAP, D3D11_TEXTURE_ADDRESS_WRAP, D3D11_TEXTURE_ADDRESS_WRAP));
    s_SamplerCache.GetIndex(SamplerDesc11(D3D11_FILTER_MIN_MAG_MIP_POINT, D3D11_TEXTURE_ADDRESS_WRAP, D3D11_TEXTURE_ADDRESS_WRAP, D3D11_TEXTURE_ADDRESS_WRAP));
    s_SamplerCache.GetIndex(SamplerDesc11(D3D11_FILTER_COMPARISON_MIN_MAG_MIP_POINT, D3D11_TEXTURE_ADDRESS_BORDER, D3D11_TEXTURE_ADDRESS_BORDER, D3D11_TEXTURE_ADDRESS_BORDER, D3D11_COMPARISON_LESS_EQUAL, Vec4(1.0f)));
    s_SamplerCache.GetIndex(SamplerDesc11(D3D11_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR, D3D11_TEXTURE_ADDRESS_BORDER, D3D11_TEXTURE_ADDRESS_BORDER, D3D11_TEXTURE_ADDRESS_BORDER, D3D11_COMPARISON_LESS_EQUAL, Vec4(1.0f)));
  }
  HRESULT hr = s_ImmediateContext.Init(Device11, Context11);
  if(SUCCEEDED(hr))
  {
    for(int i=0; i<NumberOfObjectTypes; ++i)
      if(InvokeOnInit((ObjectHint)i)==false)
        return E_FAIL;
  }
  s_FeatureLevel = Device11->GetFeatureLevel();
  return hr;
}

void Platform::Shutdown()
{
  for(int i=0; i<NumberOfObjectTypes; ++i)
    InvokeOnShutdown((ObjectHint)i);
  s_BackBufferRT.Clear();
  s_BackBufferDS.Clear();
  s_BlendCache.Clear();
  s_RasterizerCache.Clear();
  s_DepthStencilCache.Clear();
  s_SamplerCache.Clear();
  s_ImmediateContext.Clear();
  SAFE_RELEASE(s_Device11);
}

HRESULT Platform::SetFrameBuffer(RenderTarget2D* pBackBufferRT, RenderTarget2D* pBackBufferDS)
{
  s_BackBufferRT.Clear();
  if(pBackBufferRT!=NULL)
  {
    pBackBufferRT->Clone(s_BackBufferRT);
    s_ImmediateContext.BindRT(0, &s_BackBufferRT);
  }
  s_BackBufferDS.Clear();
  if(pBackBufferDS!=NULL)
  {
    pBackBufferDS->Clone(s_BackBufferDS);
    s_ImmediateContext.BindDepthStencil(&s_BackBufferDS);
    s_BackBufferDS.SetViewport();
  }
  return S_OK;
}

unsigned Platform::GetFormatBitsPerPixel(DXGI_FORMAT fmt)
{
  switch(fmt)
  {
    case DXGI_FORMAT_R32G32B32A32_TYPELESS:
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
    case DXGI_FORMAT_R32G32B32A32_UINT:
    case DXGI_FORMAT_R32G32B32A32_SINT:
      return 128;
    case DXGI_FORMAT_R32G32B32_TYPELESS:
    case DXGI_FORMAT_R32G32B32_FLOAT:
    case DXGI_FORMAT_R32G32B32_UINT:
    case DXGI_FORMAT_R32G32B32_SINT:
      return 96;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R16G16B16A16_UNORM:
    case DXGI_FORMAT_R16G16B16A16_UINT:
    case DXGI_FORMAT_R16G16B16A16_SNORM:
    case DXGI_FORMAT_R16G16B16A16_SINT:
    case DXGI_FORMAT_R32G32_TYPELESS:
    case DXGI_FORMAT_R32G32_FLOAT:
    case DXGI_FORMAT_R32G32_UINT:
    case DXGI_FORMAT_R32G32_SINT:
    case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
    case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
    case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:
      return 64;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R10G10B10A2_UINT:
    case DXGI_FORMAT_R11G11B10_FLOAT:
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_R8G8B8A8_UINT:
    case DXGI_FORMAT_R8G8B8A8_SNORM:
    case DXGI_FORMAT_R8G8B8A8_SINT:
    case DXGI_FORMAT_R16G16_TYPELESS:
    case DXGI_FORMAT_R16G16_FLOAT:
    case DXGI_FORMAT_R16G16_UNORM:
    case DXGI_FORMAT_R16G16_UINT:
    case DXGI_FORMAT_R16G16_SNORM:
    case DXGI_FORMAT_R16G16_SINT:
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT:
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_R32_UINT:
    case DXGI_FORMAT_R32_SINT:
    case DXGI_FORMAT_R24G8_TYPELESS:
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
    case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
    case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_UNORM:
    case DXGI_FORMAT_R8G8_B8G8_UNORM:
    case DXGI_FORMAT_G8R8_G8B8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8X8_TYPELESS:
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
    case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM:
    case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
      return 32;
    case DXGI_FORMAT_R8G8_TYPELESS:
    case DXGI_FORMAT_R8G8_UNORM:
    case DXGI_FORMAT_R8G8_UINT:
    case DXGI_FORMAT_R8G8_SNORM:
    case DXGI_FORMAT_R8G8_SINT:
    case DXGI_FORMAT_R16_TYPELESS:
    case DXGI_FORMAT_R16_FLOAT:
    case DXGI_FORMAT_D16_UNORM:
    case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_R16_UINT:
    case DXGI_FORMAT_R16_SNORM:
    case DXGI_FORMAT_R16_SINT:
    case DXGI_FORMAT_B5G6R5_UNORM:
    case DXGI_FORMAT_B5G5R5A1_UNORM:
      return 16;
    case DXGI_FORMAT_R8_TYPELESS:
    case DXGI_FORMAT_R8_UNORM:
    case DXGI_FORMAT_R8_UINT:
    case DXGI_FORMAT_R8_SNORM:
    case DXGI_FORMAT_R8_SINT:
    case DXGI_FORMAT_A8_UNORM:
    case DXGI_FORMAT_BC2_TYPELESS:   // DXT3
    case DXGI_FORMAT_BC2_UNORM:      // DXT3
    case DXGI_FORMAT_BC2_UNORM_SRGB: // DXT3
    case DXGI_FORMAT_BC3_TYPELESS:   // DTX5
    case DXGI_FORMAT_BC3_UNORM:      // DTX5
    case DXGI_FORMAT_BC3_UNORM_SRGB: // DTX5
    case DXGI_FORMAT_BC5_TYPELESS:   // ATI2N
    case DXGI_FORMAT_BC5_UNORM:      // ATI2N
    case DXGI_FORMAT_BC5_SNORM:      // ATI2N
      return 8;
    case DXGI_FORMAT_BC1_TYPELESS:   // DXT1
    case DXGI_FORMAT_BC1_UNORM:      // DXT1
    case DXGI_FORMAT_BC1_UNORM_SRGB: // DXT1
    case DXGI_FORMAT_BC4_TYPELESS:   // ATI1N
    case DXGI_FORMAT_BC4_UNORM:      // ATI1N
    case DXGI_FORMAT_BC4_SNORM:      // ATI1N
      return 4;
    case DXGI_FORMAT_R1_UNORM:
      return 1;
    case DXGI_FORMAT_BC6H_TYPELESS:
    case DXGI_FORMAT_BC6H_UF16:
    case DXGI_FORMAT_BC6H_SF16:
    case DXGI_FORMAT_BC7_TYPELESS:
    case DXGI_FORMAT_BC7_UNORM:
    case DXGI_FORMAT_BC7_UNORM_SRGB:
      return 0;
  }
  return 0;
}

std::string Platform::s_Directories[] =
{
  "Shaders\\",
  "Texture\\",
  "Mesh\\",
};

char* Platform::GetPath(FileType fileType, char* buf, const char* file)
{
  strcpy(buf, s_Directories[fileType].c_str());
  strcat(buf, file);
  return buf;
}

bool Platform::Add(const OnInitDelegate& d, ObjectHint objHint)
{
  bool bOK = true;
  auto it = std::find(s_OnInit[objHint].begin(), s_OnInit[objHint].end(), d);
  if(it==s_OnInit[objHint].end())
  {
    s_OnInit[objHint].push_back(d);
    if(s_Device11!=NULL)
      bOK = d();
  }
  return bOK;
}

void Platform::Add(const OnShutdownDelegate& d, ObjectHint objHint)
{
  s_OnShutdown[objHint].push_back(d);
}

template<class T> inline void EraseCallback(std::vector<T> a[Platform::NumberOfObjectTypes], const T& d)
{
  for(int i=0; i<Platform::NumberOfObjectTypes; ++i)
  {
    auto it = std::find(a[i].begin(), a[i].end(), d);
    if(it!=a[i].end())
    {
      a[i].erase(it);
      return;
    }
  }
  _ASSERT(!"the callback was not found");
}

void Platform::Remove(const OnInitDelegate& d)
{
  EraseCallback(s_OnInit, d);
}

void Platform::Remove(const OnShutdownDelegate& d)
{
  EraseCallback(s_OnShutdown, d);
  if(s_Device11!=NULL) d();
}

bool Platform::InvokeOnInit(ObjectHint objHint)
{
  for(auto it=s_OnInit[objHint].begin(); it!=s_OnInit[objHint].end(); ++it)
    if((*it)()==false)
      return false;
  return true;
}

void Platform::InvokeOnShutdown(ObjectHint objHint)
{
  for(auto it=s_OnShutdown[objHint].begin(); it!=s_OnShutdown[objHint].end(); ++it)
    (*it)();
}

void Platform::RemoveAllCallbacks()
{
  for(int i=0; i<Platform::NumberOfObjectTypes; ++i)
  {
    if(s_Device11!=NULL)
      InvokeOnShutdown((ObjectHint)i);
    s_OnInit[i].clear();
    s_OnShutdown[i].clear();
  }
}

static tbb::spin_mutex g_MemoryPoolsListMutex;

static class MemoryPoolsList : public std::vector<MemoryPool*>
{
public:
  ~MemoryPoolsList() { std::for_each(begin(), end(), [] (MemoryPool* p) { delete p; }); }
} g_MemoryPoolsList;

static __declspec(thread) struct TLSData
{
  MemoryPool* pMP;
  size_t NAllocations;
} g_TLS;

void* Platform::AllocateScratchMemory(size_t n)
{
  const size_t c_ElementSize = 2097152;
  const size_t c_Alignment = 16;
  if(g_TLS.pMP==NULL)
  {
    g_TLS.pMP = new MemoryPool(5, c_ElementSize, c_Alignment);
    tbb::spin_mutex::scoped_lock lock(g_MemoryPoolsListMutex);
    g_MemoryPoolsList.push_back(g_TLS.pMP);
  }
  if(++g_TLS.NAllocations>16)
    Log::Error("Scratch memory: %d-th allocation request!\n", g_TLS.NAllocations);
  if(n<=c_ElementSize)
    return g_TLS.pMP->Allocate();
  Log::Error("Scratch memory: requested buffer is too large (%d bytes)!\n", n);
  return _aligned_malloc(n, c_Alignment);
}

void Platform::FreeScratchMemory(void* p)
{
  --g_TLS.NAllocations;
  if(g_TLS.pMP->Contains(p)) g_TLS.pMP->Free(p);
  else _aligned_free(p);
}

// Global objects that use Platform must be initialized here at the bottom of the file,
// after Platform's internal static objects were initialized.
SimpleShaderCache g_SimpleShaderCache(128);
SystemTextureLoader g_SystemTextureLoader(128);
