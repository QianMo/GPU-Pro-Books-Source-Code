/* DX9 utility code for transparent primitives rendering.

   Pavlo Turchyn <pavlo@nichego-novogo.net> Aug 2010 */

#ifndef __DEBUG_RENDERER9
#define __DEBUG_RENDERER9

#include <vector>
#include "../Math/Math.h"
#include "../Util/ShaderObject9.h"

class DebugRenderer9 : public MathLibObject
{
public:
  DebugRenderer9() : m_DynamicVB(NULL), m_Decl(NULL), m_Device9(NULL), m_VertexTransform(Mat4x4::Identity()), m_FillColor(~0U), m_ContourColor(~0U)
  {
    m_Triangles.reserve(1024);
    m_Lines.reserve(1024);
  }
  ~DebugRenderer9()
  {
    Release();
  }
  HRESULT Init(IDirect3DDevice9* Device9)
  {
    static const D3DVERTEXELEMENT9 c_VertexElems[] =
    {
      { 0,  0, D3DDECLTYPE_FLOAT4,   D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
      { 0, 16, D3DDECLTYPE_D3DCOLOR, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_COLOR,    0 },
      D3DDECL_END()
    };
    m_Device9 = Device9;
    HRESULT hr = Device9->CreateVertexBuffer(c_DynamicVBSize, D3DUSAGE_DYNAMIC|D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT, &m_DynamicVB, NULL);
    hr = SUCCEEDED(hr) ? Device9->CreateVertexDeclaration(c_VertexElems, &m_Decl) : hr;
    hr = SUCCEEDED(hr) ? m_Shader.Init(Device9, "Shaders\\Debug") : hr;
    return hr;
  }
  void Release()
  {
    SAFE_RELEASE(m_DynamicVB);
    SAFE_RELEASE(m_Decl);
    m_Shader.Release();
    m_Device9 = NULL;
  }
  void SetViewportTransform(unsigned w, unsigned h)
  {
    m_VertexTransform = Mat4x4::ScalingTranslationD3D(Vec2(2.0f/(float)w, -2.0f/(float)h), Vec2(-1, 1));
  }
  void SetTransform(const Mat4x4& a)
  {
    m_VertexTransform = a;
  }
  void SetFillColor(const Vec4& c)
  {
    m_FillColor = ToD3DColor(c);
  }
  void SetContourColor(const Vec4& c)
  {
    m_ContourColor = ToD3DColor(c);
  }
  void PushLine(const Vec4& a, const Vec4& b)
  {
    m_Lines.push_back(MakeVertex(a, m_ContourColor));
    m_Lines.push_back(MakeVertex(b, m_ContourColor));
  }
  void PushTriangle(const Vec4& a, const Vec4& b, const Vec4& c)
  {
    m_Triangles.push_back(MakeVertex(a, m_FillColor));
    m_Triangles.push_back(MakeVertex(b, m_FillColor));
    m_Triangles.push_back(MakeVertex(c, m_FillColor));
    Vertex lv0 = MakeVertex(a, m_ContourColor);
    Vertex lv1 = MakeVertex(b, m_ContourColor);
    Vertex lv2 = MakeVertex(c, m_ContourColor);
    m_Lines.push_back(lv0); m_Lines.push_back(lv1);
    m_Lines.push_back(lv1); m_Lines.push_back(lv2);
    m_Lines.push_back(lv2); m_Lines.push_back(lv0);
  }
  void PushQuad(const Vec4& a, const Vec4& b, const Vec4& c, const Vec4& d)
  {
    Vertex tv0 = MakeVertex(a, m_FillColor);
    Vertex tv1 = MakeVertex(b, m_FillColor);
    Vertex tv2 = MakeVertex(c, m_FillColor);
    Vertex tv3 = MakeVertex(d, m_FillColor);
    m_Triangles.push_back(tv0); m_Triangles.push_back(tv1); m_Triangles.push_back(tv2);
    m_Triangles.push_back(tv0); m_Triangles.push_back(tv2); m_Triangles.push_back(tv3);
    Vertex lv0 = MakeVertex(a, m_ContourColor);
    Vertex lv1 = MakeVertex(b, m_ContourColor);
    Vertex lv2 = MakeVertex(c, m_ContourColor);
    Vertex lv3 = MakeVertex(d, m_ContourColor);
    m_Lines.push_back(lv0); m_Lines.push_back(lv1);
    m_Lines.push_back(lv1); m_Lines.push_back(lv2);
    m_Lines.push_back(lv2); m_Lines.push_back(lv3);
    m_Lines.push_back(lv3); m_Lines.push_back(lv0);
  }
  void Render()
  {
    m_Device9->SetVertexDeclaration(m_Decl);
    m_Device9->SetStreamSource(0, m_DynamicVB, 0, sizeof(Vertex));

    m_Shader.Bind();

    m_Device9->SetRenderState(D3DRS_ALPHABLENDENABLE, TRUE);
    m_Device9->SetRenderState(D3DRS_BLENDOP, D3DBLENDOP_ADD);
    m_Device9->SetRenderState(D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA);
    m_Device9->SetRenderState(D3DRS_SRCBLEND, D3DBLEND_SRCALPHA);

    if(m_Triangles.size())
    {
      size_t nSrcDataSize = m_Triangles.size()*sizeof(Vertex);
      unsigned char* pSrcData = (unsigned char*)&m_Triangles[0];
      while(nSrcDataSize>0)
      {
        size_t ToRender = _cpp_min(nSrcDataSize, c_DynamicVBSize);
        void* pLocked;
        if(SUCCEEDED(m_DynamicVB->Lock(0, 0, &pLocked, D3DLOCK_DISCARD)))
        {
          memcpy(pLocked, pSrcData, ToRender);
          m_DynamicVB->Unlock();
          const size_t nTriangleSize = sizeof(Vertex)*3;
          m_Device9->DrawPrimitive(D3DPT_TRIANGLELIST, 0, ToRender/nTriangleSize);
        }
        nSrcDataSize -= ToRender;
        pSrcData += ToRender;
      }
      m_Triangles.clear();
    }

    if(m_Lines.size())
    {
      size_t nSrcDataSize = m_Lines.size()*sizeof(Vertex);
      unsigned char* pSrcData = (unsigned char*)&m_Lines[0];
      while(nSrcDataSize>0)
      {
        size_t ToRender = _cpp_min(nSrcDataSize, c_DynamicVBSize);
        void* pLocked;
        if(SUCCEEDED(m_DynamicVB->Lock(0, 0, &pLocked, D3DLOCK_DISCARD)))
        {
          memcpy(pLocked, pSrcData, ToRender);
          m_DynamicVB->Unlock();
          const size_t nLineSize = sizeof(Vertex)*2;
          m_Device9->DrawPrimitive(D3DPT_LINELIST, 0, ToRender/nLineSize);
        }
        nSrcDataSize -= ToRender;
        pSrcData += ToRender;
      }
      m_Lines.clear();
    }

    m_Device9->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
  }

  finline void PushLine(const Vec2& a, const Vec2& b) { PushLine(Vec2::Point(a), Vec2::Point(b)); }
  finline void PushLine(const Vec3& a, const Vec3& b) { PushLine(Vec3::Point(a), Vec3::Point(b)); }
  finline void PushTriangle(const Vec2& a, const Vec2& b, const Vec2& c) { PushTriangle(Vec2::Point(a), Vec2::Point(b), Vec2::Point(c)); }
  finline void PushTriangle(const Vec3& a, const Vec3& b, const Vec3& c) { PushTriangle(Vec3::Point(a), Vec3::Point(b), Vec3::Point(c)); }
  finline void PushQuad(const Vec2& a, const Vec2& b, const Vec2& c, const Vec2& d) { PushQuad(Vec2::Point(a), Vec2::Point(b), Vec2::Point(c), Vec2::Point(d)); }
  finline void PushQuad(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d) { PushQuad(Vec3::Point(a), Vec3::Point(b), Vec3::Point(c), Vec3::Point(d)); }

private:
  Mat4x4 m_VertexTransform;
  unsigned m_FillColor, m_ContourColor;
  struct Vertex { float Position[4]; unsigned Color; };
  std::vector<Vertex> m_Triangles;
  std::vector<Vertex> m_Lines;
  ShaderObject9 m_Shader;
  IDirect3DVertexBuffer9* m_DynamicVB;
  IDirect3DVertexDeclaration9 *m_Decl;
  IDirect3DDevice9* m_Device9;
  static const int c_VerticesPerTriangle = 3;
  static const int c_VerticesPerLine = 2;
  static const size_t c_DynamicVBSize = 256*c_VerticesPerTriangle*c_VerticesPerLine*sizeof(Vertex);

  finline Vertex MakeVertex(const Vec4& Coord, unsigned Color)
  {
    Vec4 a = Coord*m_VertexTransform;
    Vertex v = { { a.x, a.y, a.z, a.w } , Color };
    return v;
  }
};

#endif //#ifndef __DEBUG_RENDERER9
