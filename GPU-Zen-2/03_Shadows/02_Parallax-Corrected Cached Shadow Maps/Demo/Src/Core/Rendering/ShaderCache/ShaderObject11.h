#ifndef __SHADER_OBJECT11     
#define __SHADER_OBJECT11

#include <d3dx11.h>
#include <D3Dcompiler.h>
#include "Platform11/Platform11.h"
#include "VertexFormat.h"
#include "../../Util/Log.h"
#include "../../Util/MemoryBuffer.h"

#pragma pack(push, 1)

struct ShaderObjectDesc
{
  unsigned RequiredShaderModel;
  unsigned ThreadGroupSizeX;
  unsigned ThreadGroupSizeY;
  unsigned ThreadGroupSizeZ;
};

#pragma pack(pop)

class ShaderObject : public PixelShader11, public GeometryShader11, public VertexShader11,
                     public ComputeShader11, public HullShader11, public DomainShader11, public InputLayout11
{
public:
  ShaderObject()
  {
    memset(&m_Desc, 0, sizeof(m_Desc));
  }
  HRESULT Init(const char* pszPSH,
               const char* pszGSH,
               const char* pszVSH,
               const char* pszCSH,
               const char* pszHSH,
               const char* pszDSH,
               D3D10_SHADER_MACRO* pMacro = NULL,
               const VertexFormatDesc* pInputDesc = NULL,
               MemoryBuffer* pBuffer = NULL,
               const char* pszShaderModelPSH = "ps_5_0",
               const char* pszShaderModelGSH = "gs_5_0",
               const char* pszShaderModelVSH = "vs_5_0",
               const char* pszShaderModelCSH = "cs_5_0",
               const char* pszShaderModelHSH = "hs_5_0",
               const char* pszShaderModelDSH = "ds_5_0",
               ID3D11Device* Device11 = Platform::GetD3DDevice())
  {
    memset(&m_Desc, 0, sizeof(m_Desc));
    size_t startPos = 0;
    size_t descPos = 0;
    if(pBuffer!=NULL)
    {
      startPos = pBuffer->Position();
      pBuffer->Write<unsigned>(0);
      pBuffer->Write<unsigned>(DescData);
      descPos = pBuffer->Position();
      pBuffer->Write(m_Desc);
    }
    ID3D10Blob* pCode = NULL;
    HRESULT hr = S_OK;
    if(pszCSH!=NULL && SUCCEEDED(hr))
    {
      hr = Compile<ID3D11ComputeShader*, ComputeShader11, &ID3D11Device::CreateComputeShader, &ShaderObject::GetComputeShaderInfo>(&pCode, pszCSH, pszShaderModelCSH, "mainCS", pMacro, Device11, CSCode, pBuffer);
      SAFE_RELEASE(pCode);
    }
    if(pszPSH!=NULL && SUCCEEDED(hr))
    {
      hr = Compile<ID3D11PixelShader*, PixelShader11, &ID3D11Device::CreatePixelShader, NULL>(&pCode, pszPSH, pszShaderModelPSH, "mainPS", pMacro, Device11, PSCode, pBuffer);
      SAFE_RELEASE(pCode);
    }
    if(pszGSH!=NULL && SUCCEEDED(hr))
    {
      hr = Compile<ID3D11GeometryShader*, GeometryShader11, &ID3D11Device::CreateGeometryShader, NULL>(&pCode, pszGSH, pszShaderModelGSH, "mainGS", pMacro, Device11, GSCode, pBuffer);
      SAFE_RELEASE(pCode);
    }
    if(pszVSH!=NULL && SUCCEEDED(hr))
    {
      if(pInputDesc!=NULL && pInputDesc->size()>0 && pBuffer!=NULL)
      {
        pBuffer->Write<unsigned>(InputDescData);
        pInputDesc->Serialize(*pBuffer);
      }
      hr = Compile<ID3D11VertexShader*, VertexShader11, &ID3D11Device::CreateVertexShader, NULL>(&pCode, pszVSH, pszShaderModelVSH, "mainVS", pMacro, Device11, VSCode, pBuffer);
      if(pInputDesc!=NULL && pInputDesc->size()>0 && SUCCEEDED(hr))
      {
        ID3D11InputLayout* pInputLayout = NULL;
        hr = SUCCEEDED(hr) ? Device11->CreateInputLayout(pInputDesc->ptr(), pInputDesc->size(), pCode->GetBufferPointer(), pCode->GetBufferSize(), &pInputLayout) : hr;
        hr = SUCCEEDED(hr) ? InputLayout11::Init(pInputLayout) : hr;
        SAFE_RELEASE(pInputLayout);
      }
      SAFE_RELEASE(pCode);
    }
    if(pszHSH!=NULL && SUCCEEDED(hr))
    {
      hr = Compile<ID3D11HullShader*, HullShader11, &ID3D11Device::CreateHullShader, NULL>(&pCode, pszHSH, pszShaderModelHSH, "mainHS", pMacro, Device11, HSCode, pBuffer);
      SAFE_RELEASE(pCode);
    }
    if(pszDSH!=NULL && SUCCEEDED(hr))
    {
      hr = Compile<ID3D11DomainShader*, DomainShader11, &ID3D11Device::CreateDomainShader, NULL>(&pCode, pszDSH, pszShaderModelDSH, "mainDS", pMacro, Device11, DSCode, pBuffer);
      SAFE_RELEASE(pCode);
    }
    if(pBuffer!=NULL)
    {
      *(pBuffer->Ptr<unsigned>(startPos)) = pBuffer->Position() - startPos;
      *(pBuffer->Ptr<ShaderObjectDesc>(descPos)) = m_Desc;
    }
    return hr;
  }
  HRESULT Init(MemoryBuffer& in, MemoryBuffer* pBuffer, ID3D11Device* Device11 = Platform::GetD3DDevice())
  {
    memset(&m_Desc, 0, sizeof(m_Desc));
    size_t startPos = in.Position();
    unsigned blockSize = in.Read<unsigned>();
    VertexFormatDesc inputDesc;
    HRESULT hr = S_OK;
    while(in.Position()<(startPos + blockSize) && SUCCEEDED(hr))
    {
      switch(in.Read<unsigned>())
      {
      case DescData:
        in.Read(m_Desc);
        if(m_Desc.RequiredShaderModel>GetSupportedShaderModel())
        {
          in.Seek(startPos + blockSize);
          Log::Info("unable to deserialize shader object: requires SM%x while device supports SM%x\n", m_Desc.RequiredShaderModel, GetSupportedShaderModel());
        }
        break;
      case VSCode:
        if(inputDesc.size()>0)
        {
          size_t pos = in.Position();
          ID3D11InputLayout* pInputLayout = NULL;
          unsigned dataSize = in.Read<unsigned>();
          hr = Device11->CreateInputLayout(inputDesc.ptr(), inputDesc.size(), in.Ptr<void>(), dataSize, &pInputLayout);
          hr = SUCCEEDED(hr) ? InputLayout11::Init(pInputLayout) : hr;
          SAFE_RELEASE(pInputLayout);
          in.Seek(pos);
        }
        hr = SUCCEEDED(hr) ? Deserialize<ID3D11VertexShader*, VertexShader11, &ID3D11Device::CreateVertexShader>(in, Device11) : hr;
        break;
      case PSCode: hr = Deserialize<ID3D11PixelShader*, PixelShader11, &ID3D11Device::CreatePixelShader>(in, Device11); break;
      case GSCode: hr = Deserialize<ID3D11GeometryShader*, GeometryShader11, &ID3D11Device::CreateGeometryShader>(in, Device11); break;
      case CSCode: hr = Deserialize<ID3D11ComputeShader*, ComputeShader11, &ID3D11Device::CreateComputeShader>(in, Device11); break;
      case HSCode: hr = Deserialize<ID3D11HullShader*, HullShader11, &ID3D11Device::CreateHullShader>(in, Device11); break;
      case DSCode: hr = Deserialize<ID3D11DomainShader*, DomainShader11, &ID3D11Device::CreateDomainShader>(in, Device11); break;
      case InputDescData: _ASSERT(GetVertexShader()==NULL); inputDesc.Deserialize(in); break;
      default: _ASSERT(!"corrupt data"); hr = E_FAIL; break;
      }
    }
    _ASSERT(in.Position()==(startPos + blockSize) && "corrupt data");
    if(SUCCEEDED(hr) && pBuffer!=NULL)
    {
      pBuffer->Write(in.Ptr<void>(startPos), blockSize);
    }
    return hr;
  }
  void Clear()
  {
    Unbind();
    PixelShader11::Clear();
    GeometryShader11::Clear();
    VertexShader11::Clear();
    ComputeShader11::Clear();
    HullShader11::Clear();
    DomainShader11::Clear();
    InputLayout11::Clear();
    memset(&m_Desc, 0, sizeof(m_Desc));
  }
  void Unbind() const
  {
    Platform::Unbind((PixelShader11*)this);
    Platform::Unbind((GeometryShader11*)this);
    Platform::Unbind((VertexShader11*)this);
    Platform::Unbind((ComputeShader11*)this);
    Platform::Unbind((HullShader11*)this);
    Platform::Unbind((DomainShader11*)this);
    Platform::Unbind((InputLayout11*)this);
  }
  finline void Bind(RenderContext11& rc = Platform::GetImmediateContext()) const
  {
    if(GetPixelShader()!=NULL)    rc.BindPS(this); else rc.UnbindPixelShader();
    if(GetGeometryShader()!=NULL) rc.BindGS(this); else rc.UnbindGeometryShader();
    if(GetVertexShader()!=NULL)   rc.BindVS(this); else rc.UnbindVertexShader();
    if(GetComputeShader()!=NULL)  rc.BindCS(this); else rc.UnbindComputeShader();
    if(GetHullShader()!=NULL)     rc.BindHS(this); else rc.UnbindHullShader();
    if(GetDomainShader()!=NULL)   rc.BindDS(this); else rc.UnbindDomainShader();
    if(GetInputLayout()!=NULL) rc.BindInputLayout(this); else rc.UnbindInputLayout();
  }
  ShaderObject* Clone() const
  {
    ShaderObject* p = new ShaderObject();
    Clone(*p); return p;
  }
  void Clone(ShaderObject& a) const
  {
    a = *this;
    PixelShader11::AddRef();
    GeometryShader11::AddRef();
    VertexShader11::AddRef();
    ComputeShader11::AddRef();
    HullShader11::AddRef();
    DomainShader11::AddRef();
    InputLayout11::AddRef();
  }
  void Destruct()
  {
    Clear();
    delete this;
  }
  finline const ShaderObjectDesc& GetDesc() const
  {
    return m_Desc;
  }

protected:
  enum Signature
  {
    PSCode = MAKEFOURCC('P', 'S', 'H', 'C'),
    GSCode = MAKEFOURCC('G', 'S', 'H', 'C'),
    VSCode = MAKEFOURCC('V', 'S', 'H', 'C'),
    CSCode = MAKEFOURCC('C', 'S', 'H', 'C'),
    HSCode = MAKEFOURCC('H', 'S', 'H', 'C'),
    DSCode = MAKEFOURCC('D', 'S', 'H', 'C'),
    InputDescData = MAKEFOURCC('I', 'N', 'P', 'T'),
    DescData = MAKEFOURCC('D', 'E', 'S', 'C'),
  };

  ShaderObjectDesc m_Desc;

  template<class Interface, class BaseObject, HRESULT (STDMETHODCALLTYPE ID3D11Device::*Create)(const void*, SIZE_T, ID3D11ClassLinkage*, Interface*), void (ShaderObject::*GetInfo)(ID3D11ShaderReflection*)>
    HRESULT Compile(ID3D10Blob** ppCode, const char* pszFile, const char* pszShaderModel, const char* pszMain, const D3D10_SHADER_MACRO* pMacro,
                    ID3D11Device* Device11, Signature sig, MemoryBuffer* pBuffer)
  {
    char buf[256];
    MemoryBuffer msg(sizeof(buf), buf);
    if(pMacro!=NULL)
      for(const D3D10_SHADER_MACRO* p = pMacro; p->Name!=NULL; ++p)
        if(p->Definition[0]!='0' || p->Definition[1]!=0)
          msg.Print(" /D%s=%s", p->Name, p->Definition);
    msg.Write((char)0);
    Log::Info("Compiling: %s /T%s /E%s%s", pszFile, pszShaderModel, pszMain, msg.Ptr<char>(0));

    unsigned compileFlags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef PIX_DEBUG
    compileFlags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif
    char shaderFullPath[256];
    HRESULT hr = D3DX11CompileFromFileA(Platform::GetPath(Platform::File_Shader, shaderFullPath, pszFile), pMacro, NULL, pszMain, pszShaderModel, compileFlags, 0, NULL, ppCode , NULL, NULL);
    Interface pObj = NULL;
    hr = SUCCEEDED(hr) ? (Device11->*Create)(ppCode[0]->GetBufferPointer(), ppCode[0]->GetBufferSize(), NULL, &pObj) : hr;
    hr = SUCCEEDED(hr) ? BaseObject::Init(pObj) : hr;
    SAFE_RELEASE(pObj);

    if(SUCCEEDED(hr)) Log::Info(" <OK>\n");
    else Log::Error("\nError creating shader: HRESULT=0x%x\n", hr);

    if(SUCCEEDED(hr))
    {
      ID3D11ShaderReflection* pReflection;
      if(SUCCEEDED(D3DReflect(ppCode[0]->GetBufferPointer(), ppCode[0]->GetBufferSize(), IID_ID3D11ShaderReflection, (void**)&pReflection)))
      {
        D3D11_SHADER_DESC desc;
        pReflection->GetDesc(&desc);
        m_Desc.RequiredShaderModel = std::max(m_Desc.RequiredShaderModel, desc.Version & 0xffff);
        if(GetInfo!=NULL)
          (this->*GetInfo)(pReflection);
        pReflection->Release();
      }
    }

    if(SUCCEEDED(hr) && pBuffer!=NULL)
    {
#ifndef PIX_DEBUG
      ID3D10Blob* pSmallerCode;
      if(SUCCEEDED(D3DStripShader(ppCode[0]->GetBufferPointer(), ppCode[0]->GetBufferSize(), D3DCOMPILER_STRIP_REFLECTION_DATA | D3DCOMPILER_STRIP_DEBUG_INFO | D3DCOMPILER_STRIP_TEST_BLOBS, &pSmallerCode)))
      {
        ppCode[0]->Release();
        ppCode[0] = pSmallerCode;
      }
#endif
      pBuffer->Write<unsigned>(sig);
      pBuffer->Write<unsigned>(ppCode[0]->GetBufferSize());
      pBuffer->Write(ppCode[0]->GetBufferPointer(), ppCode[0]->GetBufferSize());
    }
    _ASSERT(SUCCEEDED(hr));
    return hr;
  }
  template<class Interface, class BaseObject, HRESULT (STDMETHODCALLTYPE ID3D11Device::*Create)(const void*, SIZE_T, ID3D11ClassLinkage*, Interface*)>
    HRESULT Deserialize(MemoryBuffer& in, ID3D11Device* Device11)
  {
    Interface pObj = NULL;
    unsigned dataSize = in.Read<unsigned>();
    HRESULT hr = (Device11->*Create)(in.Ptr<void>(), dataSize, NULL, &pObj);
    hr = SUCCEEDED(hr) ? BaseObject::Init(pObj) : hr;
    SAFE_RELEASE(pObj);
    in.Seek(in.Position() + dataSize);
    return hr;
  }
  void GetComputeShaderInfo(ID3D11ShaderReflection* pReflection)
  {
    pReflection->GetThreadGroupSize(&m_Desc.ThreadGroupSizeX, &m_Desc.ThreadGroupSizeY, &m_Desc.ThreadGroupSizeZ);
  }
  static unsigned GetSupportedShaderModel()
  {
    switch(Platform::GetFeatureLevel())
    {
    case D3D_FEATURE_LEVEL_9_1:  return 0x20;
    case D3D_FEATURE_LEVEL_9_2:  return 0x20;
    case D3D_FEATURE_LEVEL_9_3:  return 0x30;
    case D3D_FEATURE_LEVEL_10_0: return 0x40;
    case D3D_FEATURE_LEVEL_10_1: return 0x41;
    }
    return 0x50;
  }
};

#endif //#ifndef __SHADER_OBJECT11
