/* Implementation of adaptive shadow maps.

   Pavlo Turchyn <pavlo@nichego-novogo.net> Aug 2010 */

#ifndef __TILE_CACHE
#define __TILE_CACHE

#include "../Math/Math.h"
#include "../Util/ptr_set.h"

class AABB2D : public MathLibObject
{
public:
  Vec2 Min;
  Vec2 Max;

  static AABB2D Zero()
  {
    AABB2D r;
    r.Min = r.Max = Vec2::Zero();
    return r;
  }
  finline bool operator== (const AABB2D& a) const
  {
    return (a.Min==Min) & (a.Max==Max);
  }
};

class Tile : public MathLibObject
{
public:
  Mat4x4 m_ViewProj;
  Mat4x4 m_InvView;
  struct Viewport { int x, y, w, h; } m_Viewport;
  unsigned char m_Refinement;

  Tile()
  {
    m_InvView = m_ViewProj = Mat4x4::Identity();
    m_Viewport = Viewport();
    m_Refinement = ASM_MAX_REFINEMENT;
  }
};

class QTreeNode;
class ShadowFrustum;

class TileCacheEntry : public Tile
{
public:
  AABB2D m_BBox;
  QTreeNode *m_pOwner;
  ShadowFrustum *m_pFrustum;
  unsigned m_LastFrameUsed;
  unsigned m_FrustumID;
  unsigned m_RenderMask;
  bool m_IsLayer;

  TileCacheEntry(int, int, int, int);
  ~TileCacheEntry();
  void Allocate(QTreeNode*, ShadowFrustum*, bool);
  void Free();
  void Render();
  void MarkReady();
  void MarkNotReady();
  bool IsReady();
  void CreateDEM();
  void CopyDEM(int, int);

protected:
  ptr_set_handle m_TilesPos;
  ptr_set_handle m_FreeTilesPos;
  ptr_set_handle m_RenderQueuePos;
  ptr_set_handle m_ReadyTilesPos;

  ptr_set_handle& GetTilesPos()       { return m_TilesPos;       }
  ptr_set_handle& GetFreeTilesPos()   { return m_FreeTilesPos;   }
  ptr_set_handle& GetRenderQueuePos() { return m_RenderQueuePos; }
  ptr_set_handle& GetReadyTilesPos()  { return m_ReadyTilesPos;  }

  void RenderShadowmap();

  friend class TileCache;
};

class TileCache
{
public:
  static ptr_set<TileCacheEntry, &TileCacheEntry::GetTilesPos>       s_Tiles;
  static ptr_set<TileCacheEntry, &TileCacheEntry::GetFreeTilesPos>   s_FreeTiles;
  static ptr_set<TileCacheEntry, &TileCacheEntry::GetRenderQueuePos> s_RenderQueue;
  static ptr_set<TileCacheEntry, &TileCacheEntry::GetReadyTilesPos>  s_ReadyTiles;
  static unsigned s_CacheHits, s_TileAllocs;
  static bool s_Initialized;

  static void Initialize(int, int, int, int);
  static void Shutdown();
  static TileCacheEntry* Allocate(QTreeNode*, ShadowFrustum*, bool);
};

#endif //#ifndef __TILE_CACHE
