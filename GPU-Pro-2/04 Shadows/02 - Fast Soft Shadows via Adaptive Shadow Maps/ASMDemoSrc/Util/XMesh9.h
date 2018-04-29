/* DX9 ID3DXMesh wrapper.

   Pavlo Turchyn <pavlo@nichego-novogo.net> Aug 2010 */

#ifndef __XMESH9
#define __XMESH9

#include <vector>
#include "../Math/Math.h"

class XMesh9 : public MathLibObject
{
public:
  XMesh9() : m_pMesh(NULL), m_Device9(NULL), m_AABB(Mat4x4::Identity())
  {
  }
  ~XMesh9()
  {
    Release();
  }
  HRESULT Init(IDirect3DDevice9* Device9, const char *pszFileName)
  {
    m_Device9 = Device9;
    ID3DXBuffer *pMaterials = NULL;
    ID3DXBuffer *pAdjacency = NULL;
    DWORD nMaterials;
    HRESULT hr = D3DXLoadMeshFromXA(pszFileName, D3DXMESH_IB_MANAGED | D3DXMESH_VB_MANAGED , m_Device9, &pAdjacency, &pMaterials, NULL, &nMaterials, &m_pMesh);
    if(SUCCEEDED(hr))
    {
      m_pMesh->OptimizeInplace(D3DXMESHOPT_COMPACT|D3DXMESHOPT_VERTEXCACHE, (DWORD*)pAdjacency->GetBufferPointer(), NULL, NULL, NULL);
      m_Textures.resize(nMaterials);
      D3DXMATERIAL* Materials = (D3DXMATERIAL*)pMaterials->GetBufferPointer();
      char Drive[_MAX_DRIVE];
      char Dir[_MAX_DIR];
       _splitpath_s(pszFileName, Drive, sizeof(Drive), Dir, sizeof(Dir), NULL, 0, NULL, 0);
      for(unsigned i=0; i<nMaterials; ++i)
      {
        if(Materials[i].pTextureFilename)
        {
          for(unsigned j=0; j<i; ++j)
          {
            if(Materials[j].pTextureFilename && !_stricmp(Materials[j].pTextureFilename, Materials[i].pTextureFilename))
            {
              m_Textures[i] = m_Textures[j];
              m_Textures[i]->AddRef();
              break;
            }
          }
          if(m_Textures[i]==NULL)
          {
            char TexturePath[256];
            sprintf_s(TexturePath, sizeof(TexturePath), "%s%s%s", Drive, Dir, Materials[i].pTextureFilename);
            hr = D3DXCreateTextureFromFileExA(m_Device9, TexturePath, D3DX_DEFAULT, D3DX_DEFAULT, D3DX_DEFAULT, 0, D3DFMT_UNKNOWN, D3DPOOL_DEFAULT, D3DX_DEFAULT, D3DX_DEFAULT, 0, NULL, NULL, &m_Textures[i]);
            if(FAILED(hr)) break;
          }
        }
      }

      D3DVERTEXELEMENT9 Decl[MAX_FVF_DECL_SIZE] = {};
      m_pMesh->GetDeclaration(Decl);
      int nPositionDataOffset = 0;
      for(int i=0; Decl[i].Stream!=0xff && i<MAX_FVF_DECL_SIZE; ++i)
      {
        if(Decl[i].Usage==D3DDECLUSAGE_POSITION)
        {
          nPositionDataOffset = Decl[i].Offset;
          break;
        }
      }
      unsigned char *pVB;
      if(SUCCEEDED(m_pMesh->LockVertexBuffer(D3DLOCK_READONLY | D3DLOCK_NOSYSLOCK, (void**)&pVB)))
      {
        Vec4 AABBMin(+FLT_MAX), AABBMax(-FLT_MAX);
        const int nVBStride = m_pMesh->GetNumBytesPerVertex();
        const int nVertices = m_pMesh->GetNumVertices();
        for(int i=0; i<nVertices; ++i)
        {
          Vec3 Position((float*)&pVB[nPositionDataOffset]);
          AABBMin = Vec3::Min(AABBMin, Position);
          AABBMax = Vec3::Max(AABBMax, Position);
          pVB += nVBStride;
        }
        m_pMesh->UnlockVertexBuffer();
        m_AABB = Mat4x4::ScalingTranslationD3D(0.5f*(AABBMax - AABBMin), 0.5f*(AABBMax + AABBMin));
      }
    }
    SAFE_RELEASE(pAdjacency);
    SAFE_RELEASE(pMaterials);
    return hr;
  }
  void Release()
  {
    SAFE_RELEASE(m_pMesh);
    for(unsigned i=0; i<m_Textures.size(); ++i)
      SAFE_RELEASE(m_Textures[i]);
    m_Textures.clear();
    m_Device9 = NULL;
  }
  void Draw() const
  {
    for(unsigned i=0; i<m_Textures.size(); ++i)
    {
      if(m_Textures[i]!=NULL)
        m_Device9->SetTexture(0, m_Textures[i]);
      m_pMesh->DrawSubset(i);
    }
  }
  const Mat4x4& GetAABB() const 
  {
    return m_AABB;
  }

private:
  Mat4x4 m_AABB;
  ID3DXMesh *m_pMesh;
  std::vector<IDirect3DTexture9*> m_Textures;
  IDirect3DDevice9* m_Device9;
};

#endif //#ifndef __XMESH9
