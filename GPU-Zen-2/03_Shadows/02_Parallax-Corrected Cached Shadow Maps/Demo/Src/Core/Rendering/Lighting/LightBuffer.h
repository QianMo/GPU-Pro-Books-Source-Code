#ifndef __LIGHT_BUFFER_H
#define __LIGHT_BUFFER_H

#include "Platform11/StructuredBuffer11.h"
#include "TextureLoader/Texture11.h"
#include "../../Util/AlignedVector.h"

class Camera;
class DebugRenderer;
class LightBatch;
class PushBuffer;

class LightBuffer : public StructuredBuffer
{
public:
  HRESULT Init(unsigned, unsigned);
  void Clear();
  bool Render(RenderTarget2D*, RenderTarget2D*, RenderTarget2D*, Camera*, DebugRenderer* pDRM = NULL, DeviceContext11& dc = Platform::GetImmediateContext());
  finline unsigned GetWidthInQuads() const { return m_QuadsWidth; }
  finline unsigned GetHeightInQuads() const { return m_QuadsHeight; }
protected:
  struct Tile
  {
    AABB2D BBox;
    unsigned FirstLightIndex;
    unsigned NLights;
    unsigned CurrentPassNLights;
    unsigned MergeBufferOffset;
  };

  RenderTarget2D m_DepthBounds;
  StructuredBuffer m_VisibilityBuffer;
  Texture2D m_IndexTexture;
  unsigned m_QuadsWidth, m_QuadsHeight;
  unsigned m_Width, m_Height;
  unsigned m_VisibilityQuadsWidth, m_VisibilityQuadsHeight;

  void DepthReduction(RenderTarget2D*, RenderTarget2D*, Camera*, DeviceContext11&);
  void BuildTiles(unsigned, const AABB2D*, AlignedPODVector<Tile>&, std::vector<unsigned>&, DebugRenderer*);
  bool BuildIndex(unsigned, const Vec4*, AlignedPODVector<Tile>&, std::vector<unsigned>&, const StructuredBuffer**, DeviceContext11&);
  void RunCulling(PushBuffer&, const StructuredBuffer*, DeviceContext11&);
  void RunLighting(PushBuffer&, RenderTarget2D*, RenderTarget2D*, RenderTarget2D*, Camera*, DeviceContext11&);

  static void SplitTile(Tile&, AlignedPODVector<Tile>&, std::vector<unsigned>&, const AABB2D*);

  friend class PointLightBatch;
  friend class CubeShadowMapPointLightBatch;
};

#endif //#ifndef __LIGHT_BUFFER_H
