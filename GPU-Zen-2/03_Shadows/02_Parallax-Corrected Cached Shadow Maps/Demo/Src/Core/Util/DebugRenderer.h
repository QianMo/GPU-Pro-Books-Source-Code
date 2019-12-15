#ifndef __DEBUG_RENDERER
#define __DEBUG_RENDERER

#include <vector>
#include <algorithm>
#include "ShaderCache/SimpleShader.h"
#include "Platform11/IABuffer11.h"

static const D3D11_INPUT_ELEMENT_DESC c_DebugRendererInputDesc[] =
{
  {"POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0},
  {"COLOR",    0, DXGI_FORMAT_B8G8R8A8_UNORM,     0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0},
};

class DebugRenderer : public MathLibObject
{
public:
  struct Vertex { float Position[4]; unsigned Color; };

  DebugRenderer() : m_VertexTransform(Mat4x4::Identity()), m_FillColor(~0U), m_ContourColor(~0U)
  {
    m_Triangles.reserve(4096);
    m_Lines.reserve(4096);
  }
  HRESULT Init()
  {
    HRESULT hr = m_DynamicVB.Init(c_DynamicVBSize/sizeof(Vertex), sizeof(Vertex), NULL, D3D11_USAGE_DYNAMIC, D3D11_BIND_VERTEX_BUFFER, D3D11_CPU_ACCESS_WRITE);
    return hr;
  }
  void Clear()
  {
    m_DynamicVB.Clear();
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
    m_FillColor = PackVectorU32(Vec4::Swizzle<z,y,x,w>(c));
  }
  void SetContourColor(const Vec4& c)
  {
    m_ContourColor = PackVectorU32(Vec4::Swizzle<z,y,x,w>(c));
  }
  void PushLine(const Vec4& a, const Vec4& b)
  {
    if(m_ContourColor&0xff000000)
    {
      m_Lines.push_back(MakeVertex(a, m_ContourColor));
      m_Lines.push_back(MakeVertex(b, m_ContourColor));
    }
  }
  void PushTriangle(const Vec4& a, const Vec4& b, const Vec4& c)
  {
    if(m_FillColor&0xff000000)
    {
      m_Triangles.push_back(MakeVertex(a, m_FillColor));
      m_Triangles.push_back(MakeVertex(b, m_FillColor));
      m_Triangles.push_back(MakeVertex(c, m_FillColor));
    }
    if(m_ContourColor&0xff000000)
    {
      Vertex lv0 = MakeVertex(a, m_ContourColor);
      Vertex lv1 = MakeVertex(b, m_ContourColor);
      Vertex lv2 = MakeVertex(c, m_ContourColor);
      m_Lines.push_back(lv0); m_Lines.push_back(lv1);
      m_Lines.push_back(lv1); m_Lines.push_back(lv2);
      m_Lines.push_back(lv2); m_Lines.push_back(lv0);
    }
  }
  void PushTriangle(const Vec4& a, const Vec4& b, const Vec4& c, const Vec4& ca, const Vec4& cb, const Vec4& cc)
  {
    m_Triangles.push_back(MakeVertex(a, PackVectorU32(Vec4::Swizzle<z,y,x,w>(ca))));
    m_Triangles.push_back(MakeVertex(b, PackVectorU32(Vec4::Swizzle<z,y,x,w>(cb))));
    m_Triangles.push_back(MakeVertex(c, PackVectorU32(Vec4::Swizzle<z,y,x,w>(cc))));
  }
  void PushLine(const Vec4& a, const Vec4& b, const Vec4& ca, const Vec4& cb)
  {
    m_Lines.push_back(MakeVertex(a, PackVectorU32(Vec4::Swizzle<z,y,x,w>(ca))));
    m_Lines.push_back(MakeVertex(b, PackVectorU32(Vec4::Swizzle<z,y,x,w>(cb))));
  }
  void PushQuad(const Vec4& a, const Vec4& b, const Vec4& c, const Vec4& d)
  {
    if(m_FillColor&0xff000000)
    {
      Vertex tv0 = MakeVertex(a, m_FillColor);
      Vertex tv1 = MakeVertex(b, m_FillColor);
      Vertex tv2 = MakeVertex(c, m_FillColor);
      Vertex tv3 = MakeVertex(d, m_FillColor);
      m_Triangles.push_back(tv0); m_Triangles.push_back(tv1); m_Triangles.push_back(tv2);
      m_Triangles.push_back(tv0); m_Triangles.push_back(tv2); m_Triangles.push_back(tv3);
    }
    if(m_ContourColor&0xff000000)
    {
      Vertex lv0 = MakeVertex(a, m_ContourColor);
      Vertex lv1 = MakeVertex(b, m_ContourColor);
      Vertex lv2 = MakeVertex(c, m_ContourColor);
      Vertex lv3 = MakeVertex(d, m_ContourColor);
      m_Lines.push_back(lv0); m_Lines.push_back(lv1);
      m_Lines.push_back(lv1); m_Lines.push_back(lv2);
      m_Lines.push_back(lv2); m_Lines.push_back(lv3);
      m_Lines.push_back(lv3); m_Lines.push_back(lv0);
    }
  }
  void Render(const ShaderObject* pShader = NULL, DeviceContext11& dc = Platform::GetImmediateContext())
  {
    dc.PushRC();
    dc.BindVertexBuffer(0, &m_DynamicVB, 0);
    dc.SetRenderStateB<RS_BLEND_ENABLE>(true);
    dc.SetRenderState<RS_SRC_BLEND>(D3D11_BLEND_SRC_ALPHA);
    dc.SetRenderState<RS_DEST_BLEND>(D3D11_BLEND_INV_SRC_ALPHA);
    dc.SetRenderState<RS_BLEND_OP>(D3D11_BLEND_OP_ADD);

    if(pShader==NULL)
    {
      static size_t s_ShaderIndex = g_SimpleShaderCache.GetIndex(SimpleShaderDesc("_Shaders\\DebugRenderer.shader", NULL, "_Shaders\\DebugRenderer.shader", NULL, NULL, NULL, c_DebugRendererInputDesc, ARRAYSIZE(c_DebugRendererInputDesc)));
      g_SimpleShaderCache.GetByIndex(s_ShaderIndex).Bind();
    }
    else
    {
      pShader->Bind(dc);
    }

    if(m_Triangles.size())
    {
      dc.SetPrimitiveTopology(/*D3D11_PRIMITIVE_TOPOLOGY_3_CONTROL_POINT_PATCHLIST*/D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      size_t nSrcDataSize = m_Triangles.size()*sizeof(Vertex);
      unsigned char* pSrcData = (unsigned char*)&m_Triangles[0];
      while(nSrcDataSize>0)
      {
        size_t ToRender = std::min(nSrcDataSize, c_DynamicVBSize);
        void* pDstData = m_DynamicVB.Map();
        if(pDstData!=NULL)
        {
          memcpy(pDstData, pSrcData, ToRender);
          m_DynamicVB.Unmap();
          dc.FlushToDevice()->Draw(ToRender/sizeof(Vertex), 0);
        }
        nSrcDataSize -= ToRender;
        pSrcData += ToRender;
      }
      m_Triangles.clear();
    }

    if(m_Lines.size())
    {
      dc.SetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINELIST);
      size_t nSrcDataSize = m_Lines.size()*sizeof(Vertex);
      unsigned char* pSrcData = (unsigned char*)&m_Lines[0];
      while(nSrcDataSize>0)
      {
        size_t ToRender = std::min(nSrcDataSize, c_DynamicVBSize);
        void* pDstData = m_DynamicVB.Map();
        if(pDstData!=NULL)
        {
          memcpy(pDstData, pSrcData, ToRender);
          m_DynamicVB.Unmap();
          dc.FlushToDevice()->Draw(ToRender/sizeof(Vertex), 0);
        }
        nSrcDataSize -= ToRender;
        pSrcData += ToRender;
      }
      m_Lines.clear();
    }
    dc.PopRC();
  }
  void PushSphere(int Nsubdiv)
  {
    struct F
    {
      static finline Vec4 Midpoint(const Vec3& a, const Vec3& b)
      {
        return 0.5f*(Vec4::Min(a, b) + Vec4::Max(a, b));
      }
      static void DrawTriangle(DebugRenderer& r, const Vec3& pt0, const Vec3& pt1, const Vec3& pt2, int lvl)
      {
        if(lvl>0)
        {
          Mat4x4 t = Mat4x4::OBBSetScalingD3D(Mat4x4(Midpoint(pt0, pt2), Midpoint(pt0, pt1), Midpoint(pt1, pt2), Vec4::Zero()), Vec3(1.0f));
          Vec3 a = t.r[0];
          Vec3 b = t.r[1];
          Vec3 c = t.r[2];
          DrawTriangle(r, pt0, b,   a,  --lvl);
          DrawTriangle(r, b,   pt1, c,    lvl);
          DrawTriangle(r, a,   c,   pt2,  lvl);
          DrawTriangle(r, a,   b,   c,    lvl);
        }
        else
        {
          r.PushTriangle(pt2, pt1, pt0);
        }
      }
    };

    static const Vec3 c_XPLUS( 1,  0,  0);
    static const Vec3 c_XMIN (-1,  0,  0);
    static const Vec3 c_YPLUS( 0,  1,  0);
    static const Vec3 c_YMIN ( 0, -1,  0);
    static const Vec3 c_ZPLUS( 0,  0,  1);
    static const Vec3 c_ZMIN ( 0,  0, -1);

    F::DrawTriangle(*this, c_XPLUS, c_ZPLUS, c_YPLUS, Nsubdiv);
    F::DrawTriangle(*this, c_YPLUS, c_ZPLUS, c_XMIN,  Nsubdiv);
    F::DrawTriangle(*this, c_XMIN,  c_ZPLUS, c_YMIN,  Nsubdiv);
    F::DrawTriangle(*this, c_YMIN,  c_ZPLUS, c_XPLUS, Nsubdiv);
    F::DrawTriangle(*this, c_XPLUS, c_YPLUS, c_ZMIN,  Nsubdiv);
    F::DrawTriangle(*this, c_YPLUS, c_XMIN,  c_ZMIN,  Nsubdiv);
    F::DrawTriangle(*this, c_XMIN,  c_YMIN,  c_ZMIN,  Nsubdiv);
    F::DrawTriangle(*this, c_YMIN,  c_XPLUS, c_ZMIN,  Nsubdiv);
  }
  void PushCone(const Vec3& symAxis, int Nsubdiv)
  {
    Vec3 b = GetArbitraryOrthogonalVector(symAxis);
    Quat q(symAxis, 2.0f*c_PI/(float)Nsubdiv);
    for(int i=0; i<Nsubdiv; ++i)
    {
      Vec3 a = b;
      b = Quat::Transform(b, q);
      PushTriangle<Vec3>(a, b, symAxis);
      PushTriangle<Vec3>(b, a, Vec3::Zero());
    }
  }
  void PushCircle(const Vec3& symAxis, int Nsubdiv)
  {
    Vec3 b = GetArbitraryOrthogonalVector(symAxis);
    Quat q(symAxis, 2.0f*c_PI/(float)Nsubdiv);
    for(int i=0; i<Nsubdiv; ++i)
    {
      Vec3 a = b;
      b = Quat::Transform(b, q);
      PushLine<Vec3>(b, a);
    }
  }

  template<class T> void PushLine(const T&, const T&) { static_assert(!"not implemented"); }
  template<> finline void PushLine<Vec2>(const Vec2& a, const Vec2& b) { PushLine(Vec2::Point(a), Vec2::Point(b)); }
  template<> finline void PushLine<Vec3>(const Vec3& a, const Vec3& b) { PushLine(Vec3::Point(a), Vec3::Point(b)); }

  template<class T> void PushTriangle(const T&, const T&, const T&) { static_assert(!"not implemented"); }
  template<> finline void PushTriangle<Vec2>(const Vec2& a, const Vec2& b, const Vec2& c) { PushTriangle(Vec2::Point(a), Vec2::Point(b), Vec2::Point(c)); }
  template<> finline void PushTriangle<Vec3>(const Vec3& a, const Vec3& b, const Vec3& c) { PushTriangle(Vec3::Point(a), Vec3::Point(b), Vec3::Point(c)); }

  template<class T> void PushLine(const T&, const T&, const Vec4&, const Vec4&) { static_assert(!"not implemented"); }
  template<> finline void PushLine<Vec2>(const Vec2& a, const Vec2& b, const Vec4& ca, const Vec4& cb) { PushLine(Vec2::Point(a), Vec2::Point(b), ca, cb); }
  template<> finline void PushLine<Vec3>(const Vec3& a, const Vec3& b, const Vec4& ca, const Vec4& cb) { PushLine(Vec3::Point(a), Vec3::Point(b), ca, cb); }

  template<class T> void PushTriangle(const T&, const T&, const T&, const Vec4&, const Vec4&, const Vec4&) { static_assert(!"not implemented"); }
  template<> finline void PushTriangle<Vec2>(const Vec2& a, const Vec2& b, const Vec2& c, const Vec4& ca, const Vec4& cb, const Vec4& cc) { PushTriangle(Vec2::Point(a), Vec2::Point(b), Vec2::Point(c), ca, cb, cc); }
  template<> finline void PushTriangle<Vec3>(const Vec3& a, const Vec3& b, const Vec3& c, const Vec4& ca, const Vec4& cb, const Vec4& cc) { PushTriangle(Vec3::Point(a), Vec3::Point(b), Vec3::Point(c), ca, cb, cc); }

  template<class T> void PushQuad(const T& a, const T& b, const T& c, const T& d) { static_assert(!"not implemented"); }
  template<> finline void PushQuad<Vec2>(const Vec2& a, const Vec2& b, const Vec2& c, const Vec2& d) { PushQuad(Vec2::Point(a), Vec2::Point(b), Vec2::Point(c), Vec2::Point(d)); }
  template<> finline void PushQuad<Vec3>(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d) { PushQuad(Vec3::Point(a), Vec3::Point(b), Vec3::Point(c), Vec3::Point(d)); }

  const Mat4x4& GetTransform() const { return m_VertexTransform; }

private:
  Mat4x4 m_VertexTransform;
  unsigned m_FillColor, m_ContourColor;
  std::vector<Vertex> m_Triangles;
  std::vector<Vertex> m_Lines;

  IABuffer m_DynamicVB;

  static const int c_VerticesPerTriangle = 3;
  static const int c_VerticesPerLine = 2;
  static const size_t c_DynamicVBSize = 256*c_VerticesPerTriangle*c_VerticesPerLine*sizeof(Vertex);

  finline Vertex MakeVertex(const Vec4& Coord, unsigned Color)
  {
    Vertex v;
    (Coord*m_VertexTransform).Store(v.Position);
    v.Color = Color;
    return v;
  }
};

#endif //#ifndef __DEBUG_RENDERER
