/* DX9 compiled pixel-vertex shaders pair.

   Pavlo Turchyn <pavlo@nichego-novogo.net> Aug 2010 */

#ifndef __SHADER_OBJECT9
#define __SHADER_OBJECT9

#include <malloc.h>

class ShaderObject9
{
public:
  ShaderObject9() : m_VS(NULL), m_PS(NULL), m_Device9(NULL)
  {
  }
  ~ShaderObject9()
  {
    Release();
  }
  HRESULT Init(IDirect3DDevice9* Device9, const char *pszPath, const char* MacroStr = NULL)
  {
    char StrBuf[256];
    D3DXMACRO *pMacro = NULL;
    if(MacroStr)
    {
      const int MaxMacro = 128;
      pMacro = (D3DXMACRO*)alloca(sizeof(D3DXMACRO)*(MaxMacro + 1));
      int nMacro = 0;
      const char* SeparatorStr = " ";
      char *pContext;
      strcpy_s(StrBuf, sizeof(StrBuf), MacroStr);
      char *p = strtok_s(StrBuf, SeparatorStr, &pContext);
      while(p!=NULL && nMacro<MaxMacro)
      {
        pMacro[nMacro].Name = p;
        pMacro[nMacro].Definition = "1";
        ++nMacro;
        p = strtok_s(NULL, SeparatorStr, &pContext);
      }
      pMacro[nMacro].Name = NULL;
      pMacro[nMacro].Definition = NULL;
    }

    m_Device9 = Device9;
    ID3DXBuffer* pCode = NULL;
    char FileName[256];
    sprintf_s(FileName, sizeof(FileName), "%s.vsh", pszPath);
    HRESULT hr = D3DXCompileShaderFromFileA(FileName, pMacro, 0, "main", "vs_3_0", 0, &pCode, 0, 0);
    hr = SUCCEEDED(hr) ? m_Device9->CreateVertexShader((DWORD*)pCode->GetBufferPointer(), &m_VS) : hr;
    SAFE_RELEASE(pCode);
    sprintf_s(FileName, sizeof(FileName), "%s.psh", pszPath);
    hr = SUCCEEDED(hr) ? D3DXCompileShaderFromFileA(FileName, pMacro, 0, "main", "ps_3_0", 0, &pCode, 0, 0) : hr;
    hr = SUCCEEDED(hr) ? m_Device9->CreatePixelShader((DWORD*)pCode->GetBufferPointer(), &m_PS) : hr;
    SAFE_RELEASE(pCode);
    return hr;
  }
  void Release()
  {
    SAFE_RELEASE(m_VS);
    SAFE_RELEASE(m_PS);
    m_Device9 = NULL;
  }
  void Bind()
  {
    m_Device9->SetVertexShader(m_VS);
    m_Device9->SetPixelShader(m_PS);
  }

private:
  IDirect3DVertexShader9* m_VS;
  IDirect3DPixelShader9* m_PS;
  IDirect3DDevice9* m_Device9;
};

#endif //#ifndef __SHADER_OBJECT9
