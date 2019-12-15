#ifndef __BASE_OBJECTS11
#define __BASE_OBJECTS11

#include <d3d11.h>

template<class T> class D3DObjectPointer11
{
public:
  D3DObjectPointer11() : m_pObject(NULL) { }
  inline HRESULT Init(T* pObject) { m_pObject = pObject; pObject->AddRef(); return S_OK; }
  inline void AddRef() const { if(m_pObject!=NULL) m_pObject->AddRef(); }
  inline void Clear() { SAFE_RELEASE(m_pObject); }

protected:
  T* m_pObject;
};

#define D3D_OBJECT_WRAPPER(a, b, c) class a : public D3DObjectPointer11<b> { \
  public: inline b* Get##c() const { return m_pObject; } }

D3D_OBJECT_WRAPPER(ShaderResource11, ID3D11ShaderResourceView, ShaderResourceView);
D3D_OBJECT_WRAPPER(RenderTarget11, ID3D11RenderTargetView, RenderTargetView);
D3D_OBJECT_WRAPPER(DepthStencil11, ID3D11DepthStencilView, DepthStencilView);
D3D_OBJECT_WRAPPER(PixelShader11, ID3D11PixelShader, PixelShader);
D3D_OBJECT_WRAPPER(GeometryShader11, ID3D11GeometryShader, GeometryShader);
D3D_OBJECT_WRAPPER(VertexShader11, ID3D11VertexShader, VertexShader);
D3D_OBJECT_WRAPPER(ComputeShader11, ID3D11ComputeShader, ComputeShader);
D3D_OBJECT_WRAPPER(HullShader11, ID3D11HullShader, HullShader);
D3D_OBJECT_WRAPPER(DomainShader11, ID3D11DomainShader, DomainShader);
D3D_OBJECT_WRAPPER(InputLayout11, ID3D11InputLayout, InputLayout);
D3D_OBJECT_WRAPPER(SamplerState11, ID3D11SamplerState, SamplerState);
D3D_OBJECT_WRAPPER(RasterizerState11, ID3D11RasterizerState, RasterizerState);
D3D_OBJECT_WRAPPER(BlendState11, ID3D11BlendState, BlendState);
D3D_OBJECT_WRAPPER(DepthStencilState11, ID3D11DepthStencilState, DepthStencilState);
D3D_OBJECT_WRAPPER(UnorderedAccessResource11, ID3D11UnorderedAccessView, UnorderedAccessView);

class Buffer11
{
public:
  Buffer11() : m_Buffer(NULL), m_ElementSize(0)
  {
  }
  HRESULT Init(ID3D11Device* Device11, size_t nElements, size_t elementSize, const void* pData,
               unsigned bindFlags = D3D11_BIND_CONSTANT_BUFFER, D3D11_USAGE usage = D3D11_USAGE_IMMUTABLE,
               unsigned CPUAccessFlags = 0, unsigned miscFlags = 0)
  {
    m_ElementSize = elementSize;
    D3D11_BUFFER_DESC desc = { };
    desc.BindFlags = bindFlags;
    desc.ByteWidth = nElements*m_ElementSize;
    desc.MiscFlags = miscFlags;
    desc.Usage = usage;
    desc.CPUAccessFlags = CPUAccessFlags;
    if(miscFlags & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED)
      desc.StructureByteStride = m_ElementSize;
    D3D11_SUBRESOURCE_DATA Data = { };
    Data.pSysMem = pData;
    return Device11->CreateBuffer(&desc, pData!=NULL ? &Data : NULL, &m_Buffer);
  }
  HRESULT Init(ID3D11Buffer* pBuf, size_t n)
  {
    m_Buffer = pBuf; pBuf->AddRef();
    m_ElementSize = n;
    return S_OK;
  }
  void Clear()
  {
    SAFE_RELEASE(m_Buffer);
    m_ElementSize = 0;
  }

  finline void AddRef() const { if(m_Buffer!=NULL) m_Buffer->AddRef(); }
  finline ID3D11Buffer* GetBuffer() const { return m_Buffer; }
  finline size_t GetElementSize() const { return m_ElementSize; }

protected:
  ID3D11Buffer* m_Buffer;
  size_t m_ElementSize;
};

#endif //#ifndef __BASE_OBJECTS11
