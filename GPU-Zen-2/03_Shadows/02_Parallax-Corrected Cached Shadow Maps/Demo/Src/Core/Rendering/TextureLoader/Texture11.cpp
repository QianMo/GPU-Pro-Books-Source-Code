#include "PreCompile.h"
#include "Texture11.h"
#include "DXUT11/DDS.h"
#include <d3d9.h>
#include "../../Util/Log.h"
#include "../../Util/MemoryBuffer.h"

static unsigned GetBlockSize(DXGI_FORMAT fmt)
{
  return (fmt==DXGI_FORMAT_BC1_TYPELESS || fmt==DXGI_FORMAT_BC1_UNORM || fmt==DXGI_FORMAT_BC1_UNORM_SRGB || 
          fmt==DXGI_FORMAT_BC2_TYPELESS || fmt==DXGI_FORMAT_BC2_UNORM || fmt==DXGI_FORMAT_BC2_UNORM_SRGB ||
          fmt==DXGI_FORMAT_BC3_TYPELESS || fmt==DXGI_FORMAT_BC3_UNORM || fmt==DXGI_FORMAT_BC3_UNORM_SRGB ||
          fmt==DXGI_FORMAT_BC4_TYPELESS || fmt==DXGI_FORMAT_BC4_UNORM || fmt==DXGI_FORMAT_BC4_SNORM ||
          fmt==DXGI_FORMAT_BC5_TYPELESS || fmt==DXGI_FORMAT_BC5_UNORM || fmt==DXGI_FORMAT_BC5_SNORM) ? 4 : 1;
}

static bool IsTypeless(DXGI_FORMAT fmt)
{
  return fmt==DXGI_FORMAT_R32G32B32A32_TYPELESS || fmt==DXGI_FORMAT_R32G32B32_TYPELESS ||
         fmt==DXGI_FORMAT_R16G16B16A16_TYPELESS || fmt==DXGI_FORMAT_R32G32_TYPELESS    ||
         fmt==DXGI_FORMAT_R32G8X24_TYPELESS     || fmt==DXGI_FORMAT_R16G16_TYPELESS    ||
         fmt==DXGI_FORMAT_R10G10B10A2_TYPELESS  || fmt==DXGI_FORMAT_R8G8B8A8_TYPELESS  || 
         fmt==DXGI_FORMAT_R32_TYPELESS          || fmt==DXGI_FORMAT_R24G8_TYPELESS     ||
         fmt==DXGI_FORMAT_B8G8R8A8_TYPELESS     || fmt==DXGI_FORMAT_B8G8R8X8_TYPELESS  ||
         fmt==DXGI_FORMAT_R8G8_TYPELESS         || fmt==DXGI_FORMAT_R16_TYPELESS ||
         fmt==DXGI_FORMAT_R8_TYPELESS           || fmt==DXGI_FORMAT_BC2_TYPELESS ||
         fmt==DXGI_FORMAT_BC3_TYPELESS          || fmt==DXGI_FORMAT_BC5_TYPELESS ||
         fmt==DXGI_FORMAT_BC1_TYPELESS          || fmt==DXGI_FORMAT_BC4_TYPELESS ||
         fmt==DXGI_FORMAT_BC6H_TYPELESS         || fmt==DXGI_FORMAT_BC7_TYPELESS;
}

finline bool IsBitMask(const DDS_PIXELFORMAT& ddpf, unsigned r, unsigned g, unsigned b, unsigned a)
{
  return ddpf.dwRBitMask==r && ddpf.dwGBitMask==g && ddpf.dwBBitMask==b && ddpf.dwABitMask==a;
}

static DXGI_FORMAT GetDXGIFormat(const DDS_PIXELFORMAT& ddpf)
{
  if(ddpf.dwFlags & DDS_RGB)
  {
    switch(ddpf.dwRGBBitCount)
    {
    case 32:
      if(IsBitMask(ddpf, 0x00ff0000, 0x0000ff00, 0x000000ff, 0x00000000)) return DXGI_FORMAT_B8G8R8X8_UNORM;
      if(IsBitMask(ddpf, 0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000)) return DXGI_FORMAT_B8G8R8A8_UNORM;
      if(IsBitMask(ddpf, 0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000)) return DXGI_FORMAT_R8G8B8A8_UNORM;
      if(IsBitMask(ddpf, 0x000000ff, 0x0000ff00, 0x00ff0000, 0x00000000)) return DXGI_FORMAT_R8G8B8A8_UNORM;
      if(IsBitMask(ddpf, 0x000003ff, 0x000ffc00, 0x3ff00000, 0xc0000000)) return DXGI_FORMAT_R10G10B10A2_UNORM;
      if(IsBitMask(ddpf, 0x0000ffff, 0xffff0000, 0x00000000, 0x00000000)) return DXGI_FORMAT_R16G16_UNORM;
      if(IsBitMask(ddpf, 0xffffffff, 0x00000000, 0x00000000, 0x00000000)) return DXGI_FORMAT_R32_FLOAT;
      break;
    case 16:
      if(IsBitMask(ddpf, 0x0000f800, 0x000007e0, 0x0000001f, 0x00000000)) return DXGI_FORMAT_B5G6R5_UNORM;
      if(IsBitMask(ddpf, 0x00007c00, 0x000003e0, 0x0000001f, 0x00008000)) return DXGI_FORMAT_B5G5R5A1_UNORM;
      if(IsBitMask(ddpf, 0x00007c00, 0x000003e0, 0x0000001f, 0x00000000)) return DXGI_FORMAT_B5G5R5A1_UNORM;
      break;
    }
  }
  else if(ddpf.dwFlags & DDS_LUMINANCE)
  {
    switch(ddpf.dwRGBBitCount)
    {
    case 8:
      if(IsBitMask(ddpf, 0x000000ff, 0x00000000, 0x00000000, 0x00000000)) return DXGI_FORMAT_R8_UNORM;
      break;
    case 16:
      if(IsBitMask(ddpf, 0x0000ffff, 0x00000000, 0x00000000, 0x00000000)) return DXGI_FORMAT_R16_UNORM;
      if(IsBitMask(ddpf, 0x000000ff, 0x00000000, 0x00000000, 0x0000ff00)) return DXGI_FORMAT_R8G8_UNORM;
      break;
    }
  }
  else if(ddpf.dwFlags & DDS_ALPHA)
  {
    if(ddpf.dwRGBBitCount==8) return DXGI_FORMAT_A8_UNORM;
  }
  else if(ddpf.dwFlags & DDS_FOURCC)
  {
    switch(ddpf.dwFourCC)
    {
    case MAKEFOURCC('D', 'X', 'T', '1'): return DXGI_FORMAT_BC1_UNORM;
    case MAKEFOURCC('D', 'X', 'T', '3'): return DXGI_FORMAT_BC2_UNORM;
    case MAKEFOURCC('D', 'X', 'T', '5'): return DXGI_FORMAT_BC3_UNORM;
    case MAKEFOURCC('B', 'C', '4', 'U'): return DXGI_FORMAT_BC4_UNORM;
    case MAKEFOURCC('B', 'C', '4', 'S'): return DXGI_FORMAT_BC4_SNORM;
    case MAKEFOURCC('A', 'T', 'I', '1'): return DXGI_FORMAT_BC4_UNORM;
    case MAKEFOURCC('A', 'T', 'I', '2'): return DXGI_FORMAT_BC5_UNORM;
    case MAKEFOURCC('B', 'C', '5', 'S'): return DXGI_FORMAT_BC5_SNORM;
    case MAKEFOURCC('R', 'G', 'B', 'G'): return DXGI_FORMAT_R8G8_B8G8_UNORM;
    case MAKEFOURCC('G', 'R', 'G', 'B'): return DXGI_FORMAT_G8R8_G8B8_UNORM;
    case D3DFMT_A16B16G16R16:  return DXGI_FORMAT_R16G16B16A16_UNORM;
    case D3DFMT_Q16W16V16U16:  return DXGI_FORMAT_R16G16B16A16_SNORM;
    case D3DFMT_R16F:          return DXGI_FORMAT_R16_FLOAT;
    case D3DFMT_G16R16F:       return DXGI_FORMAT_R16G16_FLOAT;
    case D3DFMT_A16B16G16R16F: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case D3DFMT_R32F:          return DXGI_FORMAT_R32_FLOAT;
    case D3DFMT_G32R32F:       return DXGI_FORMAT_R32G32_FLOAT;
    case D3DFMT_A32B32G32R32F: return DXGI_FORMAT_R32G32B32A32_FLOAT;
    }
  }
  return DXGI_FORMAT_UNKNOWN;
}

HRESULT Texture2D::QuickLoad(MemoryBuffer& in, D3D11_USAGE usage, unsigned bindFlags, unsigned CPUAccessFlags, unsigned miscFlags, ID3D11Device* Device11)
{
  if((in.Size() - in.Position())>4 && in.Read<DWORD>()==DDS_MAGIC)
  {
    DDS_HEADER header;
    in.Read(header);
    if(header.dwSize==sizeof(DDS_HEADER))
    {
      unsigned arraySize;
      DXGI_FORMAT fmt;
      if(!memcmp(&header.ddspf, &DDSPF_DX10, sizeof(DDS_PIXELFORMAT)))
      {
        DDS_HEADER_DXT10 header10;
        in.Read(header10);
        fmt = header10.dxgiFormat;
        arraySize = header10.arraySize;
        miscFlags |= header10.miscFlag;
      }
      else
      {
        fmt = GetDXGIFormat(header.ddspf);
        arraySize = 1;
        if(header.dwCubemapFlags==DDS_CUBEMAP_ALLFACES)
        {
          arraySize = 6;
          miscFlags |= D3D11_RESOURCE_MISC_TEXTURECUBE;
        }
      }
      bool bMipMapCountSane = std::max(header.dwWidth, header.dwHeight)>=(1U<<(header.dwMipMapCount - 1)); // D3DX does not use DDSD_MIPMAPCOUNT to indicate validity of the field
      if(bMipMapCountSane && fmt!=DXGI_FORMAT_UNKNOWN)
      {
        return Init(header.dwWidth, header.dwHeight, fmt, header.dwMipMapCount, in.Ptr<void>(), usage, bindFlags, CPUAccessFlags, miscFlags, arraySize, Device11);
      }
    }
  }
  return E_FAIL;
}

HRESULT Texture2D::Init(unsigned w, unsigned h, DXGI_FORMAT fmt, unsigned mipLevels, const void* pData, 
                        D3D11_USAGE usage, unsigned bindFlags, unsigned CPUAccessFlags, unsigned miscFlags, 
                        unsigned arraySize, ID3D11Device* Device11)
{
  if(mipLevels==0)
    mipLevels = FloorLog2(std::max(w, h)) + 1;
  D3D11_TEXTURE2D_DESC desc = { };
  desc.Width            = w;
  desc.Height           = h;
  desc.MipLevels        = mipLevels;
  desc.ArraySize        = arraySize;
  desc.Format           = fmt;
  desc.SampleDesc.Count = 1;
  desc.Usage            = usage;
  desc.BindFlags        = bindFlags;
  desc.CPUAccessFlags   = CPUAccessFlags;
  desc.MiscFlags        = miscFlags;
  D3D11_SUBRESOURCE_DATA* pSRD = NULL;
  if(pData!=NULL)
  {
    unsigned blockSize = GetBlockSize(fmt);
    unsigned bpp = Platform::GetFormatBitsPerPixel(fmt);
    const char* pSrc = (char*)pData;
    pSRD = (D3D11_SUBRESOURCE_DATA*)alloca(sizeof(D3D11_SUBRESOURCE_DATA)*mipLevels*arraySize);
    D3D11_SUBRESOURCE_DATA* pDst = pSRD;
    for(unsigned i=0; i<arraySize; ++i)
    {
      for(unsigned j=0; j<mipLevels; ++j, ++pDst)
      {
        pDst->pSysMem = pSrc;
        pDst->SysMemPitch = (w*blockSize*bpp)>>3;
        pDst->SysMemSlicePitch = 0;
        pSrc += (w*h*bpp)>>3;
        w = std::max(w>>1, blockSize);
        h = std::max(h>>1, blockSize);
      }
      w = desc.Width;
      h = desc.Height;
    }
  }
  ID3D11Texture2D* pTexture = NULL;
  HRESULT hr = Device11->CreateTexture2D(&desc, pSRD, &pTexture);
  hr = SUCCEEDED(hr) ? Init(pTexture, Device11) : hr;
  SAFE_RELEASE(pTexture);
  return hr;
}

HRESULT Texture2D::Init(const char* pszFileName, D3D11_USAGE usage, unsigned bindFlags, unsigned CPUAccessFlags, unsigned miscFlags, ID3D11Device* Device11)
{
  HRESULT hr = D3D11_ERROR_FILE_NOT_FOUND;
  MemoryBuffer in;
  char texFullPath[256];
  if(in.Load(Platform::GetPath(Platform::File_Texture, texFullPath, pszFileName)))
  {
    hr = QuickLoad(in, usage, bindFlags, CPUAccessFlags, miscFlags, Device11);
    if(FAILED(hr))
    {
      Clear();
      Log::Info("Unable to quickload: %s\n", pszFileName);
      D3DX11_IMAGE_LOAD_INFO loadInfo; // D3DX11_IMAGE_LOAD_INFO is a class with ctor
      loadInfo.Usage = usage;
      loadInfo.BindFlags = bindFlags;
      loadInfo.CpuAccessFlags = CPUAccessFlags;
      loadInfo.MiscFlags = miscFlags;
      ID3D11Texture2D* pTexture = NULL;
      hr = D3DX11CreateTextureFromMemory(Device11, in.Ptr<void>(0), in.Size(), &loadInfo, NULL, (ID3D11Resource**)&pTexture, NULL);
      hr = SUCCEEDED(hr) ? Init(pTexture, Device11) : hr;
      SAFE_RELEASE(pTexture);
    }
  }
  if(FAILED(hr)) Log::Error("Error (HRESULT=0x%x) loading texture: %s\n", hr, pszFileName);
//  else Log::Info("Loaded: %s\n", pszFileName);
  return hr;
}

template<class D3DOBJ, class DESC, class WRAPPER, HRESULT (STDMETHODCALLTYPE ID3D11Device::*Create)(ID3D11Resource*, const DESC*, D3DOBJ**)>
  HRESULT CreateView(DXGI_FORMAT fmt, ID3D11Resource* pRes, unsigned arraySize, int Slice, DESC desc, WRAPPER& obj, ID3D11Device* Device11)
{
  desc.Format = fmt;
  if(arraySize>1 && Slice>=0) // individual slice in an array
  {
    desc.Texture2DArray.FirstArraySlice = Slice;
    desc.Texture2DArray.ArraySize = 1;
  }
  else if(arraySize>1) // ... or view for the whole array
  {
    desc.Texture2DArray.ArraySize = arraySize;
  }
  D3DOBJ* pObj = NULL;
  HRESULT hr = (Device11->*Create)(pRes, &desc, &pObj);
  hr = SUCCEEDED(hr) ? obj.Init(pObj) : hr;
  SAFE_RELEASE(pObj);
  return hr;
}

static HRESULT CreateSRV(DXGI_FORMAT fmt, ID3D11Resource* pRes, unsigned arraySize, int Slice, ShaderResource11& obj, ID3D11Device* Device11, bool isCubeMap)
{
  D3D11_SHADER_RESOURCE_VIEW_DESC desc = { };
  if(isCubeMap && arraySize>6) // CBMs array
  {
    desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURECUBEARRAY;
    desc.TextureCubeArray.MipLevels = (unsigned)-1;
    if(Slice>=0) // view for an individual CBM
    {
      _ASSERT(!(Slice % 6) && "slice index is probably invalid");
      desc.TextureCubeArray.First2DArrayFace = Slice;
      desc.TextureCubeArray.NumCubes = 1;
    }
    else // ... or view for the whole array
    {
      desc.TextureCubeArray.NumCubes = arraySize/6;
    }
    arraySize = 0;
  }
  else if(isCubeMap && Slice<1) // view for a single CBM
  {
    desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURECUBE;
    desc.TextureCube.MipLevels = (unsigned)-1;
    arraySize = 0;
  }
  else if(arraySize>1) // treat as generic array
  {
    desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
    desc.Texture2DArray.MipLevels = (unsigned)-1;
  }
  else // not an array
  {
    desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    desc.Texture2D.MipLevels = (unsigned)-1;
  }
  return CreateView<ID3D11ShaderResourceView, D3D11_SHADER_RESOURCE_VIEW_DESC, ShaderResource11, &ID3D11Device::CreateShaderResourceView>(fmt, pRes, arraySize, Slice, desc, obj, Device11);
}

HRESULT Texture2D::Init(ID3D11Texture2D* pTexture, ID3D11Device* Device11)
{
  m_Texture = pTexture; pTexture->AddRef();
  m_Texture->GetDesc(&m_Desc);
  HRESULT hr = S_OK;
  if(m_Desc.BindFlags & D3D11_BIND_SHADER_RESOURCE)
  {
    DXGI_FORMAT fmt = m_Desc.Format;
    if(m_Desc.BindFlags & D3D11_BIND_DEPTH_STENCIL)
    {
      switch(fmt)
      {
      // default view formats for D24S8 and D16 are hardcoded for convenience of use
      case DXGI_FORMAT_R24G8_TYPELESS: fmt = DXGI_FORMAT_R24_UNORM_X8_TYPELESS; break;
      case DXGI_FORMAT_R16_TYPELESS: fmt = DXGI_FORMAT_R16_UNORM; break;
      case DXGI_FORMAT_R32_TYPELESS: fmt = DXGI_FORMAT_R32_FLOAT; break;
      }
    }
    if(!IsTypeless(fmt))
      hr = CreateSRV(fmt, m_Texture, m_Desc.ArraySize, -1, *this, Device11, !!(m_Desc.MiscFlags & D3D11_RESOURCE_MISC_TEXTURECUBE));
  }
  return hr;
}

HRESULT Texture2D::AddShaderResourceView(DXGI_FORMAT fmt, int Slice, unsigned* pIndex, ID3D11Device* Device11)
{
 // _ASSERT(IsTypeless(m_Desc.Format) && (m_Desc.BindFlags & D3D11_BIND_SHADER_RESOURCE));
  if(pIndex!=NULL) *pIndex = m_AuxSRV.size();
  m_AuxSRV.push_back(ShaderResource11());
  return CreateSRV(fmt, m_Texture, m_Desc.ArraySize, Slice, m_AuxSRV.back(), Device11, !!(m_Desc.MiscFlags & D3D11_RESOURCE_MISC_TEXTURECUBE));
}

static HRESULT CreateRTV(DXGI_FORMAT fmt, ID3D11Resource* pRes, unsigned arraySize, int Slice, RenderTarget11& obj, ID3D11Device* Device11)
{
  D3D11_RENDER_TARGET_VIEW_DESC desc = { };
  desc.ViewDimension = arraySize>1 ? D3D11_RTV_DIMENSION_TEXTURE2DARRAY : D3D11_RTV_DIMENSION_TEXTURE2D;
  if(arraySize==1 && Slice>=0) desc.Texture2D.MipSlice = Slice;
  return CreateView<ID3D11RenderTargetView, D3D11_RENDER_TARGET_VIEW_DESC, RenderTarget11, &ID3D11Device::CreateRenderTargetView>(fmt, pRes, arraySize, Slice, desc, obj, Device11);
}

static HRESULT CreateDSV(DXGI_FORMAT fmt, ID3D11Resource* pRes, unsigned arraySize, int Slice, DepthStencil11& obj, ID3D11Device* Device11)
{
  D3D11_DEPTH_STENCIL_VIEW_DESC desc = { };
  desc.ViewDimension = arraySize>1 ? D3D11_DSV_DIMENSION_TEXTURE2DARRAY : D3D11_DSV_DIMENSION_TEXTURE2D;
  if(arraySize==1 && Slice>=0) desc.Texture2D.MipSlice = Slice;
  return CreateView<ID3D11DepthStencilView, D3D11_DEPTH_STENCIL_VIEW_DESC, DepthStencil11, &ID3D11Device::CreateDepthStencilView>(fmt, pRes, arraySize, Slice, desc, obj, Device11);
}

static HRESULT CreateUAV(DXGI_FORMAT fmt, ID3D11Resource* pRes, unsigned arraySize, int Slice, UnorderedAccessResource11& obj, ID3D11Device* Device11)
{
  D3D11_UNORDERED_ACCESS_VIEW_DESC desc = { };
  desc.ViewDimension = arraySize>1 ? D3D11_UAV_DIMENSION_TEXTURE2DARRAY : D3D11_UAV_DIMENSION_TEXTURE2D;
  return CreateView<ID3D11UnorderedAccessView, D3D11_UNORDERED_ACCESS_VIEW_DESC, UnorderedAccessResource11, &ID3D11Device::CreateUnorderedAccessView>(fmt, pRes, arraySize, Slice, desc, obj, Device11);
}

HRESULT RenderTarget2D::RenderTargetInit(ID3D11Device* Device11)
{
  bool typelessResource = IsTypeless(m_Desc.Format);
  HRESULT hr = S_OK;
  if((m_Desc.BindFlags & D3D11_BIND_RENDER_TARGET) && !typelessResource)
  {
    hr = CreateDefaultViews<RenderTarget11, &CreateRTV>(DXGI_FORMAT_UNKNOWN, m_AuxRTV, Device11);
  }
  if(SUCCEEDED(hr) && (m_Desc.BindFlags & D3D11_BIND_DEPTH_STENCIL))
  {
    DXGI_FORMAT fmt = m_Desc.Format;
    switch(fmt)
    {
    // default view formats for D24S8 and D16 are hardcoded for convenience of use
    case DXGI_FORMAT_R24G8_TYPELESS: fmt = DXGI_FORMAT_D24_UNORM_S8_UINT; break;
    case DXGI_FORMAT_R16_TYPELESS: fmt = DXGI_FORMAT_D16_UNORM; break;
    case DXGI_FORMAT_R32_TYPELESS: fmt = DXGI_FORMAT_D32_FLOAT; break;
    }
    if(!typelessResource || fmt!=m_Desc.Format)
      hr = CreateDefaultViews<DepthStencil11, &CreateDSV>(fmt, m_AuxDSV, Device11);
  }
  if(SUCCEEDED(hr) && (m_Desc.BindFlags & D3D11_BIND_UNORDERED_ACCESS) && !typelessResource)
  {
    hr = CreateDefaultViews<UnorderedAccessResource11, &CreateUAV>(DXGI_FORMAT_UNKNOWN, m_AuxUAV, Device11);
  }
  return hr;
}

HRESULT RenderTarget2D::AddRenderTargetView(DXGI_FORMAT fmt, int Slice, unsigned* pIndex, ID3D11Device* Device11)
{
  //_ASSERT(IsTypeless(m_Desc.Format) && (m_Desc.BindFlags & D3D11_BIND_RENDER_TARGET));
  if(pIndex!=NULL) *pIndex = m_AuxRTV.size();
  m_AuxRTV.push_back(RenderTarget11());
  return CreateRTV(fmt, m_Texture, m_Desc.ArraySize, Slice, m_AuxRTV.back(), Device11);
}

HRESULT RenderTarget2D::AddDepthStencilView(DXGI_FORMAT fmt, int Slice, unsigned* pIndex, ID3D11Device* Device11)
{
  _ASSERT(IsTypeless(m_Desc.Format) && (m_Desc.BindFlags & D3D11_BIND_DEPTH_STENCIL));
  if(pIndex!=NULL) *pIndex = m_AuxDSV.size();
  m_AuxDSV.push_back(DepthStencil11());
  return CreateDSV(fmt, m_Texture, m_Desc.ArraySize, Slice, m_AuxDSV.back(), Device11);
}

HRESULT RenderTarget2D::AddUnorderedAccessView(DXGI_FORMAT fmt, int Slice, unsigned* pIndex, ID3D11Device* Device11)
{
  _ASSERT(IsTypeless(m_Desc.Format) && (m_Desc.BindFlags & D3D11_BIND_UNORDERED_ACCESS));
  if(pIndex!=NULL) *pIndex = m_AuxUAV.size();
  m_AuxUAV.push_back(UnorderedAccessResource11());
  return CreateUAV(fmt, m_Texture, m_Desc.ArraySize, Slice, m_AuxUAV.back(), Device11);
}

void RenderTarget2D::Unbind()
{
  __super::Unbind();
  ForEachView(m_AuxRTV, *this, [] (const RenderTarget11& a) { Platform::Unbind(&a); });
  ForEachView(m_AuxDSV, *this, [] (const DepthStencil11& a) { Platform::Unbind(&a); });
  ForEachView(m_AuxUAV, *this, [] (const UnorderedAccessResource11& a) { Platform::Unbind(&a); });
}

void RenderTarget2D::Clear()
{
  Unbind();
  Texture2D::Clear();
  ForEachView(m_AuxRTV, *this, [] (RenderTarget11& a) { a.Clear(); });
  ForEachView(m_AuxDSV, *this, [] (DepthStencil11& a) { a.Clear(); });
  ForEachView(m_AuxUAV, *this, [] (UnorderedAccessResource11& a) { a.Clear(); });
  m_AuxRTV.clear();
  m_AuxDSV.clear();
  m_AuxUAV.clear();
}

void RenderTarget2D::Clone(RenderTarget2D& a) const
{
  a = *this;
  __super::Clone(a);
  ForEachView(m_AuxRTV, *this, [] (const RenderTarget11& a) { a.AddRef(); });
  ForEachView(m_AuxDSV, *this, [] (const DepthStencil11& a) { a.AddRef(); });
  ForEachView(m_AuxUAV, *this, [] (const UnorderedAccessResource11& a) { a.AddRef(); });
}

HRESULT Texture3D::Init(unsigned w, unsigned h, unsigned d, DXGI_FORMAT fmt, unsigned mipLevels, const void* pData, D3D11_USAGE usage, 
                        unsigned bindFlags, unsigned CPUAccessFlags, unsigned miscFlags, ID3D11Device* Device11)
{
  if(mipLevels==0)
    mipLevels = FloorLog2(std::max(std::max(w, h), d)) + 1;
  D3D11_TEXTURE3D_DESC desc = { };
  desc.Width          = w;
  desc.Height         = h;
  desc.Depth          = d;
  desc.MipLevels      = mipLevels;
  desc.Format         = fmt;
  desc.Usage          = usage;
  desc.BindFlags      = bindFlags;
  desc.CPUAccessFlags = CPUAccessFlags;
  desc.MiscFlags      = miscFlags;
  D3D11_SUBRESOURCE_DATA* pSRD = NULL;
  if(pData!=NULL)
  {
    unsigned blockSize = GetBlockSize(fmt);
    unsigned bpp = Platform::GetFormatBitsPerPixel(fmt);
    const char* pSrc = reinterpret_cast<const char*>(pData);
    pSRD = (D3D11_SUBRESOURCE_DATA*)alloca(sizeof(D3D11_SUBRESOURCE_DATA)*mipLevels);
    D3D11_SUBRESOURCE_DATA* pDst = pSRD;
    for(unsigned i=0; i<mipLevels; ++i, ++pDst)
    {
      pDst->pSysMem = pSrc;
      pDst->SysMemPitch = (w*blockSize*bpp)>>3;
      pDst->SysMemSlicePitch = pDst->SysMemPitch*h;
      pSrc += pDst->SysMemSlicePitch*d;
      w = std::max(w>>1, blockSize);
      h = std::max(h>>1, blockSize);
      d = std::max(d>>1, 1U);
    }
  }
  ID3D11Texture3D* pTexture = NULL;
  HRESULT hr = Device11->CreateTexture3D(&desc, pSRD, &pTexture);
  hr = SUCCEEDED(hr) ? Init(pTexture, Device11) : hr;
  SAFE_RELEASE(pTexture);
  return hr;
}

HRESULT Texture3D::Init(ID3D11Texture3D* pTexture, ID3D11Device* Device11)
{
  m_Texture = pTexture; pTexture->AddRef();
  m_Texture->GetDesc(&m_Desc);
  HRESULT hr = S_OK;
  if(m_Desc.BindFlags & D3D11_BIND_SHADER_RESOURCE)
  {
     D3D11_SHADER_RESOURCE_VIEW_DESC desc = { };
     desc.Format = m_Desc.Format;
     desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE3D;
     desc.TextureCubeArray.MipLevels = m_Desc.MipLevels;
     ID3D11ShaderResourceView* pView = NULL;
     hr = SUCCEEDED(hr) ? Device11->CreateShaderResourceView(pTexture, &desc, &pView) : hr;
     ShaderResource11::Init(pView);
     SAFE_RELEASE(pView);
  }
  if(m_Desc.BindFlags & D3D11_BIND_RENDER_TARGET)
  {
     D3D11_RENDER_TARGET_VIEW_DESC desc = { };
     desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE3D;
     desc.Format = m_Desc.Format;
     desc.Texture3D.WSize = m_Desc.Depth;
     ID3D11RenderTargetView* pView = NULL;
     hr = SUCCEEDED(hr) ? Device11->CreateRenderTargetView(pTexture, &desc, &pView) : hr;
     RenderTarget11::Init(pView);
     SAFE_RELEASE(pView);
  }
  if(m_Desc.BindFlags & D3D11_BIND_UNORDERED_ACCESS)
  {
     D3D11_UNORDERED_ACCESS_VIEW_DESC desc = { };
     desc.Format = m_Desc.Format;
     desc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE3D;
     desc.Texture3D.WSize = m_Desc.Depth;
     ID3D11UnorderedAccessView* pView = NULL;
     hr = SUCCEEDED(hr) ? Device11->CreateUnorderedAccessView(pTexture, &desc, &pView) : hr;
     UnorderedAccessResource11::Init(pView);
     SAFE_RELEASE(pView);
  }
  return hr;
}

void Texture3D::Unbind()
{
  Platform::Unbind(static_cast<const UnorderedAccessResource11*>(this));
  Platform::Unbind(static_cast<const ShaderResource11*>(this));
  Platform::Unbind(static_cast<const RenderTarget11*>(this));
}

void Texture3D::Clear()
{
  Unbind();
  ShaderResource11::Clear();
  RenderTarget11::Clear();
  UnorderedAccessResource11::Clear();
  SAFE_RELEASE(m_Texture);
}
