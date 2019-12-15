#ifndef __TEXTURE11
#define __TEXTURE11

#include "Platform11/Platform11.h"

class MemoryBuffer;

class Texture2D : public ShaderResource11
{
public:
  Texture2D() : m_Texture(NULL) { memset(&m_Desc, 0, sizeof(m_Desc)); }
  ID3D11Texture2D* GetTexture2D() const { return m_Texture; }

  HRESULT Init(const char* pszFileName, D3D11_USAGE usage = D3D11_USAGE_IMMUTABLE, 
               unsigned bindFlags = D3D11_BIND_SHADER_RESOURCE, unsigned CPUAccessFlags = 0, 
               unsigned miscFlags = 0, ID3D11Device* Device11 = Platform::GetD3DDevice());
  HRESULT Init(unsigned w, unsigned h, DXGI_FORMAT fmt, unsigned mipLevels = 1, const void* pData = NULL, D3D11_USAGE usage = D3D11_USAGE_DEFAULT, 
               unsigned bindFlags = D3D11_BIND_SHADER_RESOURCE, unsigned CPUAccessFlags = 0, unsigned miscFlags = 0, unsigned arraySize = 1,
               ID3D11Device* Device11 = Platform::GetD3DDevice());
  HRESULT Init(ID3D11Texture2D* pTexture, ID3D11Device* Device11 = Platform::GetD3DDevice());

  HRESULT AddShaderResourceView(DXGI_FORMAT, int Slice = -1, unsigned* pIndex = NULL, ID3D11Device* Device11 = Platform::GetD3DDevice());

  void Unbind()
  {
    ForEachView(m_AuxSRV, *this, [] (const ShaderResource11& a) { Platform::Unbind(&a); });
  }
  void Clear()
  {
    Unbind();
    ForEachView(m_AuxSRV, *this, [] (ShaderResource11& a) { a.Clear(); });
    m_AuxSRV.clear();
    SAFE_RELEASE(m_Texture);
  }
  finline const D3D11_TEXTURE2D_DESC& GetDesc() const
  {
    return m_Desc;
  }
  Texture2D* Clone() const
  {
    Texture2D* p = new Texture2D();
    Clone(*p); return p;
  }
  void Clone(Texture2D& a) const
  {
    a = *this;
    ForEachView(m_AuxSRV, *this, [] (const ShaderResource11& a) { a.AddRef(); });
    if(m_Texture!=NULL) m_Texture->AddRef();
  }
  void Destruct()
  {
    Clear();
    delete this;
  }
  finline void CopyTo(Texture2D& dst, DeviceContext11& dc = Platform::GetImmediateContext())
  {
    dc.DoNotFlushToDevice()->CopyResource(dst.m_Texture, m_Texture);
  }
  finline const ShaderResource11* GetShaderResourceView(unsigned i) const
  {
    return &m_AuxSRV[i];
  }

protected:
  ID3D11Texture2D* m_Texture;
  D3D11_TEXTURE2D_DESC m_Desc;
  std::vector<ShaderResource11> m_AuxSRV;

  HRESULT QuickLoad(MemoryBuffer&, D3D11_USAGE, unsigned, unsigned, unsigned, ID3D11Device*);

  template<class A, class B, class F> static finline void ForEachView(std::vector<A>& a, B& b, F f)
  {
    std::for_each(a.begin(), a.end(), f);
    f(b);
  }
  template<class A, class B, class F> static finline void ForEachView(const std::vector<A>& a, const B& b, F f)
  {
    std::for_each(a.begin(), a.end(), f);
    f(b);
  }
};

class RenderTarget2D : public Texture2D, public RenderTarget11, public DepthStencil11, public UnorderedAccessResource11
{
public:
  HRESULT Init(int w, int h, DXGI_FORMAT fmt, int mipLevels = 1, const void* pData = NULL, D3D11_USAGE usage = D3D11_USAGE_DEFAULT,
               unsigned bindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET, unsigned CPUAccessFlags = 0,
               unsigned miscFlags = 0, unsigned arraySize = 1, ID3D11Device* Device11 = Platform::GetD3DDevice())
  {
    HRESULT hr = Texture2D::Init(w, h, fmt, mipLevels, pData, usage, bindFlags, CPUAccessFlags, miscFlags, arraySize, Device11);
    return SUCCEEDED(hr) ? RenderTargetInit(Device11) : hr;
  }
  HRESULT Init(ID3D11Texture2D* pTexture, ID3D11Device* Device11 = Platform::GetD3DDevice())
  {
    HRESULT hr = Texture2D::Init(pTexture, Device11);
    return SUCCEEDED(hr) ? RenderTargetInit(Device11) : hr;
  }
  HRESULT Init(ID3D11View* pView, ID3D11Device* Device11 = Platform::GetD3DDevice())
  {
    ID3D11Texture2D* pTexture;
    pView->GetResource((ID3D11Resource**)&pTexture);
    HRESULT hr = Init(pTexture, Device11);
    SAFE_RELEASE(pTexture);
    return hr;
  }
  void SetViewport(RenderContext11& rc = Platform::GetImmediateContext()) const
  {
    D3D11_VIEWPORT vp = { };
    vp.Width = (float)m_Desc.Width;
    vp.Height = (float)m_Desc.Height;
    vp.MaxDepth = 1.0f;
    rc.SetViewport(vp);
  }
  RenderTarget2D* Clone() const
  {
    RenderTarget2D* p = new RenderTarget2D();
    Clone(*p); return p;
  }
  finline void GenerateMips(DeviceContext11& dc = Platform::GetImmediateContext()) const
  {
    _ASSERT(m_Desc.MiscFlags&D3D11_RESOURCE_MISC_GENERATE_MIPS);
    dc.DoNotFlushToDevice()->GenerateMips(ShaderResource11::GetShaderResourceView());
  }

  HRESULT AddRenderTargetView(DXGI_FORMAT, int Slice = -1, unsigned* pIndex = NULL, ID3D11Device* Device11 = Platform::GetD3DDevice());
  HRESULT AddDepthStencilView(DXGI_FORMAT, int Slice = -1, unsigned* pIndex = NULL, ID3D11Device* Device11 = Platform::GetD3DDevice());
  HRESULT AddUnorderedAccessView(DXGI_FORMAT, int Slice = -1, unsigned* pIndex = NULL, ID3D11Device* Device11 = Platform::GetD3DDevice());

  finline const RenderTarget11* GetRenderTargetView(unsigned i) const { return &m_AuxRTV[i]; }
  finline const DepthStencil11* GetDepthStencilView(unsigned i) const { return &m_AuxDSV[i]; }
  finline const UnorderedAccessResource11* GetUnorderedAccessView(unsigned i) const { return &m_AuxUAV[i]; }

  void Unbind();
  void Clear();
  void Clone(RenderTarget2D&) const;

protected:
  std::vector<RenderTarget11> m_AuxRTV;
  std::vector<DepthStencil11> m_AuxDSV;
  std::vector<UnorderedAccessResource11> m_AuxUAV;

  HRESULT RenderTargetInit(ID3D11Device*);

  template<class WRAPPER, HRESULT (*CreateView)(DXGI_FORMAT, ID3D11Resource*, unsigned, int, WRAPPER&, ID3D11Device*)>
    HRESULT CreateDefaultViews(DXGI_FORMAT fmt, std::vector<WRAPPER>& a, ID3D11Device* Device11)
  {
    HRESULT hr = S_OK;
    if(m_Desc.ArraySize>1)
    {
      a.resize(m_Desc.ArraySize);
      for(unsigned i=0; i<m_Desc.ArraySize; ++i)
        hr = SUCCEEDED(hr) ? (*CreateView)(fmt, m_Texture, m_Desc.ArraySize, i, a[i], Device11) : hr;
    }
    hr = SUCCEEDED(hr) ? (*CreateView)(fmt, m_Texture, m_Desc.ArraySize, -1, *this, Device11) : hr;
    return hr;
  }
};

template<unsigned w, unsigned h, DXGI_FORMAT fmt, unsigned mipLevels = 1, D3D11_USAGE usage = D3D11_USAGE_IMMUTABLE, unsigned bindFlags = D3D11_BIND_SHADER_RESOURCE, unsigned CPUAccessFlags = 0,
  unsigned miscFlags = 0, unsigned arraySize = 1, class T = Texture2D> class StaticTexture2D : public T
{
public:
  StaticTexture2D(const void* pData) : m_pData(pData)
  {
    Platform::Add(Platform::OnInitDelegate::from_method<StaticTexture2D, &StaticTexture2D::OnPlatformInit>(this), Platform::Object_Texture);
    Platform::Add(Platform::OnShutdownDelegate::from_method<StaticTexture2D, &StaticTexture2D::OnPlatformShutdown>(this), Platform::Object_Texture);
  }

protected:
  const void* m_pData;

  bool OnPlatformInit() { return SUCCEEDED(Init(w, h, fmt, mipLevels, m_pData, usage, bindFlags, CPUAccessFlags, miscFlags, arraySize)); }
  void OnPlatformShutdown() { Clear(); }
};

class Texture3D : public ShaderResource11, public RenderTarget11, public UnorderedAccessResource11
{
public:
  Texture3D() : m_Texture(NULL) { memset(&m_Desc, 0, sizeof(m_Desc)); }

  HRESULT Init(unsigned w, unsigned h, unsigned d, DXGI_FORMAT fmt, unsigned mipLevels = 1, const void* pData = NULL,
               D3D11_USAGE usage = D3D11_USAGE_DEFAULT, unsigned bindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
               unsigned CPUAccessFlags = 0, unsigned miscFlags = 0, ID3D11Device* Device11 = Platform::GetD3DDevice());
  HRESULT Init(ID3D11Texture3D* pTexture, ID3D11Device* Device11 = Platform::GetD3DDevice());

  void Unbind();
  void Clear();

  finline ID3D11Texture3D* GetTexture3D() const { return m_Texture; }
  finline const D3D11_TEXTURE3D_DESC & GetDesc() const { return m_Desc; }

  Texture3D* Clone() const
  {
    Texture3D* p = new Texture3D();
    Clone(*p); return p;
  }
  void Clone(Texture3D& a) const
  {
    a = *this;
    if(m_Texture!=NULL) m_Texture->AddRef();
  }
  finline void CopyTo(Texture3D& dst, DeviceContext11& dc = Platform::GetImmediateContext())
  {
    dc.DoNotFlushToDevice()->CopyResource(dst.m_Texture, m_Texture);
  }
  finline void SetViewport(RenderContext11& rc = Platform::GetImmediateContext()) const
  {
    D3D11_VIEWPORT vp = { };
    vp.Width = (float)m_Desc.Width;
    vp.Height = (float)m_Desc.Height;
    vp.MaxDepth = 1.0f;
    rc.SetViewport(vp);
  }
  finline void GenerateMips(DeviceContext11& dc = Platform::GetImmediateContext()) const
  {
    _ASSERT(m_Desc.MiscFlags&D3D11_RESOURCE_MISC_GENERATE_MIPS);
    dc.DoNotFlushToDevice()->GenerateMips(ShaderResource11::GetShaderResourceView());
  }

protected:
  D3D11_TEXTURE3D_DESC m_Desc;
  ID3D11Texture3D* m_Texture;
};

#endif //#ifndef __TEXTURE11
