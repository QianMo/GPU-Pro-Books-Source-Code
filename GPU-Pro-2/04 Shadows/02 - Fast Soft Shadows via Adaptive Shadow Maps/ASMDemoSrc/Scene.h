#ifndef __SCENE
#define __SCENE

#include <algorithm>
#include <string>
#include "Math/Math.h"
#include "Util/AlignedVector.h"
#include "Util/XMesh9.h"
#include "Util/Frustum.h"
#include "CameraPath.h"

class Scene
{
public:
  Scene(const char* pszSceneFileName, const char* pszCameraPathFileName) : m_Device9(NULL), m_CameraPath(pszCameraPathFileName)
  {
    m_ColorPassShaderFlags[0] = 0;
    m_SceneFileName = pszSceneFileName;
    m_Meshes.reserve(64);
    m_Instances.reserve(128);
  }
  HRESULT Init(IDirect3DDevice9* Device9)
  {
    m_Device9 = Device9;
    HRESULT hr = S_OK;
    V_RETURN(m_DepthPassShader.Init(Device9, "Shaders\\DepthPass"));
    V_RETURN(m_DepthPassAlphaTestShader.Init(Device9, "Shaders\\DepthPass", "ALPHATEST"));
    V_RETURN(m_LayerShader.Init(Device9, "Shaders\\DepthPass", "LAYER"));
    V_RETURN(m_LayerAlphaTestShader.Init(Device9, "Shaders\\DepthPass", "LAYER ALPHATEST"));
    V_RETURN(LoadScene(m_SceneFileName.c_str()));
    return hr;
  }
  void Release()
  {
    m_DepthPassShader.Release();
    m_DepthPassAlphaTestShader.Release();
    m_LayerShader.Release();
    m_LayerAlphaTestShader.Release();
    m_ColorPassShader.Release();
    m_ColorPassAlphaTestShader.Release();
    m_ColorPassShaderFlags[0] = 0;
    Clear();
    m_Device9 = NULL;
  }
  void RenderColorPass(const Mat4x4& ViewProj, const Vec3& LightDir, const Vec3& ViewPos, const Mat4x4& SMapProj, const Vec4& AtlasSize, 
                       const Vec4& BiasParams, bool bMRF, bool bUseLayer, bool bShowShaderCost)
  {
    CompileColorPassShaders(bMRF, bUseLayer, bShowShaderCost);

    Vec4 ps[8];
    ps[0] = LightDir;
    ps[1] = ViewPos;
    ps[2] = SMapProj.r[0];
    ps[3] = SMapProj.r[1];
    ps[4] = SMapProj.r[2];
    ps[5] = SMapProj.r[3];
    ps[6] = AtlasSize;
    ps[7] = BiasParams;
    m_Device9->SetPixelShaderConstantF(0, &ps[0].x, ARRAYSIZE(ps));

    Vec4 vs[4];
    vs[0] = ViewProj.r[0];
    vs[1] = ViewProj.r[1];
    vs[2] = ViewProj.r[2];
    vs[3] = ViewProj.r[3];
    m_Device9->SetVertexShaderConstantF(4, &vs[0].x, ARRAYSIZE(vs));
    Render(ViewProj, ~MATERIAL_NON_SHADOW_CASTER_FLAG, m_ColorPassShader, m_ColorPassAlphaTestShader);
  }
  void RenderDepthPass(const Mat4x4& ViewProj)
  {
    Vec4 vs[4];
    vs[0] = ViewProj.r[0];
    vs[1] = ViewProj.r[1];
    vs[2] = ViewProj.r[2];
    vs[3] = ViewProj.r[3];
    m_Device9->SetVertexShaderConstantF(4, &vs[0].x, ARRAYSIZE(vs));
    Render(ViewProj, ~MATERIAL_NON_SHADOW_CASTER_FLAG, m_DepthPassShader, m_DepthPassAlphaTestShader);
  }
  void RenderShadowMap(const Mat4x4& ViewProj)
  {
    Vec4 vs[4];
    vs[0] = ViewProj.r[0];
    vs[1] = ViewProj.r[1];
    vs[2] = ViewProj.r[2];
    vs[3] = ViewProj.r[3];
    m_Device9->SetVertexShaderConstantF(4, &vs[0].x, ARRAYSIZE(vs));
    Render(ViewProj, ~0UL, m_DepthPassShader, m_DepthPassAlphaTestShader);
  }
  void RenderShadowMapLayer(const Mat4x4& ViewProj, const Vec4& VSParam, const Vec4& PSParam)
  {
    Vec4 vs[5];
    vs[0] = ViewProj.r[0];
    vs[1] = ViewProj.r[1];
    vs[2] = ViewProj.r[2];
    vs[3] = ViewProj.r[3];
    vs[4] = VSParam;
    m_Device9->SetVertexShaderConstantF(4, &vs[0].x, ARRAYSIZE(vs));
    Vec4 ps[1];
    ps[0] = PSParam;
    m_Device9->SetPixelShaderConstantF(0, &ps[0].x, ARRAYSIZE(ps));
    Render(ViewProj, ~0UL, m_LayerShader, m_LayerAlphaTestShader);
  }
  const CameraPath& GetCameraPath() const
  {
    return m_CameraPath;
  }

protected:
  static const unsigned MATERIAL_DEFAULT   = 0;
  static const unsigned MATERIAL_ALPHATEST = 1;
  static const unsigned MATERIAL_NON_SHADOW_CASTER_FLAG = 128;

  ShaderObject9 m_ColorPassShader;
  ShaderObject9 m_ColorPassAlphaTestShader;
  ShaderObject9 m_DepthPassShader;
  ShaderObject9 m_DepthPassAlphaTestShader;
  ShaderObject9 m_LayerShader;
  ShaderObject9 m_LayerAlphaTestShader;
  IDirect3DDevice9* m_Device9;
  char m_ColorPassShaderFlags[256];

  CameraPath m_CameraPath;
  std::string m_SceneFileName;

  void Render(const Mat4x4& ViewProj, unsigned MaterialIDMask, ShaderObject9& DefaultMaterialShader, ShaderObject9& AlphaTestMaterialShader)
  {
    Frustum Frustum = Frustum::FromViewProjectionMatrixD3D(ViewProj);

    DefaultMaterialShader.Bind();
    Draw(MATERIAL_DEFAULT, MaterialIDMask, Frustum);

    AlphaTestMaterialShader.Bind();
    m_Device9->SetSamplerState(0, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
    m_Device9->SetSamplerState(0, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
    m_Device9->SetSamplerState(0, D3DSAMP_MIPFILTER, D3DTEXF_LINEAR);
    m_Device9->SetSamplerState(0, D3DSAMP_ADDRESSU, D3DTADDRESS_WRAP);
    m_Device9->SetSamplerState(0, D3DSAMP_ADDRESSV, D3DTADDRESS_WRAP);
    Draw(MATERIAL_ALPHATEST, MaterialIDMask, Frustum);
  }
  HRESULT LoadScene(const char* FileName)
  {
    HRESULT hr = S_OK;
    FILE* pFile;
    V_RETURN((fopen_s(&pFile, FileName, "r") ? E_FAIL : S_OK));
    while(!feof(pFile))
    {
      char pBuf[4096];
      if(fgets(pBuf, sizeof(pBuf), pFile))
      {
        char* pLine = pBuf;
        char* pTokens[256];
        char* pContext;
        int nTokens = 0;
        while(pTokens[nTokens] = strtok_s(pLine, " \n", &pContext))
        {
          ++nTokens;
          pLine = NULL;
        }
        V_RETURN(nTokens==18 ? S_OK : E_FAIL);
        float m[16];
        for(int i=0; i<16; ++i)
          m[i] = MathIO::ReadFloat(pTokens[2 + i]);
        V_RETURN(AddObject(MathIO::ReadInteger(pTokens[0]), pTokens[1], Mat4x4(m)));
      }
    }
    fclose(pFile);
    return hr;
  }

private:
  struct Object
  {
    Mat4x4 Transform;
    Mat4x4 OBB;
    float BSphereRadius;
    unsigned MeshIndex;
    unsigned MaterialID;
  };
  struct Mesh
  {
    std::string FileName;
    XMesh9 *pMesh;
    inline bool operator == (const Mesh& a) { return !FileName.compare(a.FileName); }
  };

  std::vector<Mesh> m_Meshes;
  AlignedPODVector<Object> m_Instances;

  HRESULT AddObject(unsigned MaterialID, const char *pszFileName, const Mat4x4& Transform)
  {
    HRESULT hr = S_OK;
    Mesh msh;
    msh.pMesh = NULL;
    msh.FileName = pszFileName;
    std::vector<Mesh>::iterator it = std::find(m_Meshes.begin(), m_Meshes.end(), msh);
    if(it==m_Meshes.end())
    {
      msh.pMesh = new XMesh9();
      hr = msh.pMesh->Init(m_Device9, pszFileName);
      if(FAILED(hr))
      {
        return hr;
      }
      m_Meshes.push_back(msh);
      it = m_Meshes.end() - 1;
    }
    Object obj;
    obj.Transform = Transform;
    obj.OBB = it->pMesh->GetAABB()*Transform;
    obj.BSphereRadius = GetBSphereRadius(obj.OBB);
    obj.MeshIndex = it - m_Meshes.begin();
    obj.MaterialID = MaterialID;
    m_Instances.push_back(obj);
    return hr;
  }
  void Clear()
  {
    m_Instances.clear();
    for(size_t i=0; i<m_Meshes.size(); ++i)
      delete m_Meshes[i].pMesh;
    m_Meshes.clear();
    m_Device9 = NULL;
  }
  void Draw(unsigned MaterialID, unsigned MaterialIDMask, Frustum& Frustum)
  {
    const size_t n = m_Instances.size();
    for(size_t i=0; i<n; ++i)
    {
      const Object& obj = m_Instances[i];
      if((obj.MaterialID & MaterialIDMask)==MaterialID && 
         Frustum.IsIntersecting(obj.OBB, obj.BSphereRadius))
      {
        m_Device9->SetVertexShaderConstantF(0, &obj.Transform.e11, 4);
        m_Meshes[obj.MeshIndex].pMesh->Draw();
      }
    }
  }
  HRESULT CompileColorPassShaders(bool bMRF, bool bUseLayer, bool bShowShaderCost)
  {
    HRESULT hr = S_OK;
    char Flags[256] = "DX9";
    if(bMRF)            strcat_s(Flags, sizeof(Flags), " MRF");
    if(bUseLayer)       strcat_s(Flags, sizeof(Flags), " LAYER");
    if(bShowShaderCost) strcat_s(Flags, sizeof(Flags), " SHOW_SHADER_COST");
    if(strcmp(m_ColorPassShaderFlags, Flags))
    {
      strcpy_s(m_ColorPassShaderFlags, sizeof(m_ColorPassShaderFlags), Flags);
      m_ColorPassShader.Release();
      m_ColorPassAlphaTestShader.Release();
      V_RETURN(m_ColorPassShader.Init(m_Device9, "Shaders\\ColorPass", Flags));
      strcat_s(Flags, sizeof(Flags), " ALPHATEST");
      V_RETURN(m_ColorPassAlphaTestShader.Init(m_Device9, "Shaders\\ColorPass", Flags));
    }
    return hr;
  }
};

#endif //#ifndef __SCENE
