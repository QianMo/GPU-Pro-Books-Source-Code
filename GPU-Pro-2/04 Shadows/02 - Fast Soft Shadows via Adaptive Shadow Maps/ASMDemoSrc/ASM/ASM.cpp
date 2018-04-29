/* Implementation of adaptive shadow maps.

   Pavlo Turchyn <pavlo@nichego-novogo.net> Aug 2010 */

#define WIN32_LEAN_AND_MEAN

#define SAFE_RELEASE(x) if(x) { (x)->Release(); (x) = NULL; }

#include <windows.h>
#include <d3d9.h>
#include <d3dx9.h>

#pragma warning(disable:4324)
#pragma warning(disable:4100)

#include "../Math/Math.h"
#include "../Util/DebugRenderer9.h"
#include "../Util/OBB.h"
#include "../Util/Frustum.h"
#include "ASM.h"
#include "TileCache.h"
#include <algorithm>

static unsigned g_FrameCounter = 0;
static unsigned g_nGPUs = 1; // On AFR multi-GPU systems this should be initialized with the number of GPUs.

finline unsigned GetGPUMask()
{
  return 1<<(g_FrameCounter%g_nGPUs);
}

/////////////////////////////////////////////////////////////////////////////

class TilesHierarchy
{
public:
  std::vector<QTreeNode*> m_Roots;

  TilesHierarchy()  { m_Roots.reserve(32); }
  ~TilesHierarchy() { Reset(); }
  void Reset();
  void RemoveNonIntersectingNodes(int, const Vec2*, const Vec2&);
  QTreeNode* FindRoot(const AABB2D&);
protected:
  void RemoveNonIntersectingNodes(QTreeNode*, int, const Vec2*, const Vec2&);
};

#define FRUSTUM_VERTICES_CNT 5

static const Vec2 c_DebugViewCenter(250, 450);
static const Vec2 c_DebugViewScale(3, 3);

class ShadowFrustum : public MathLibObject
{
public:
  Vec3 m_LightDir;
  Mat4x4 m_LightRotMat;
  AABB2D m_FrustumBBox;
  Vec2 m_FrustumTop;
  Vec2 m_FrustumHullVB[FRUSTUM_VERTICES_CNT];
  int m_FrustumHullSize;
  unsigned m_ID;

  TilesHierarchy m_Tiles;
  Tile m_IndexTile;
  unsigned char* m_IndexTextureData;
  unsigned char* m_LayerIndexTextureData;

  ShadowFrustum();
  ~ShadowFrustum();
  bool IsValid() const { return m_ID>0; }
  void Reset();
  void Set(const Vec3&);
  void CreateTiles(const Mat4x4&);
  int RenderTile(int, bool);
  void BuildIndex();
  Vec3 Unproject(const Vec2&);
  void DebugDraw(DebugRenderer9&);

protected:
  void CreateChildren(QTreeNode*);
  void AllocateTiles(QTreeNode*);

  finline Vec2 MakeDebugTriangleVertex(const Vec2& a) { return (a - m_FrustumTop)*c_DebugViewScale + c_DebugViewCenter; }
  void DrawBBox(DebugRenderer9&, int, QTreeNode*);
};

ShadowFrustum::ShadowFrustum()
{
  Reset();
  m_IndexTextureData = (unsigned char*)_aligned_malloc(ASM_INDEX_TEXTURE_DATA_SIZE, 16);
  m_LayerIndexTextureData = (unsigned char*)_aligned_malloc(ASM_INDEX_TEXTURE_DATA_SIZE, 16);
}

ShadowFrustum::~ShadowFrustum()
{
  Reset();
  _aligned_free(m_IndexTextureData);
  _aligned_free(m_LayerIndexTextureData);
}

void ShadowFrustum::Reset()
{
  m_Tiles.Reset();
  m_LightDir = Vec3::Zero();
  m_LightRotMat = Mat4x4::Identity();
  m_FrustumBBox = AABB2D::Zero();
  memset(m_FrustumHullVB, 0, sizeof(m_FrustumHullVB));
  m_FrustumTop = Vec2::Zero();
  m_FrustumHullSize = 0;
  m_ID = 0;
}

void ShadowFrustum::Set(const Vec3& LightDir)
{
  Reset();
  m_LightDir = LightDir;
  m_LightRotMat = Mat4x4::LookAtD3D(Vec3::Zero(), m_LightDir, ASM_UP_VECTOR);
  static unsigned s_IDGen = 1;
  m_ID = s_IDGen; s_IDGen += 2;
}

Vec3 ShadowFrustum::Unproject(const Vec2& pt)
{
  return pt*Mat4x4::Transpose(m_LightRotMat);
}

/////////////////////////////////////////////////////////////////////////////

static Vec2 s_VB1[256];
static Vec2 s_VB2[256];

static const Vec2 c_Normals[] =
{
  Vec2( 1, 0),
  Vec2( 0, 1),
  Vec2(-1, 0),
  Vec2( 0,-1),
};

bool DoesPolygonIntersectBBox(int n, const Vec2 *v, const AABB2D& bbox)
{
  const Vec2* minmax = &bbox.Min;
  int j, index[2];
  float d[2];
  for(int i=0; i<4; ++i)
  {
    const Vec2& pn = c_Normals[i];
    float pw = - Vec2::Dot(pn, minmax[i>>1]);
    index[1] = n - 1;
    d[1] = Vec2::Dot(pn, v[index[1]]) + pw;
    for(j=0; j<n; j++)
    {
      index[0] = index[1];
      index[1] = j;
      d[0] = d[1];
      d[1] = Vec2::Dot(pn, v[index[1]]) + pw;
      if(d[1]>0 && d[0]<0) break;
    }
    if(j<n)
    {
      int k = 0;
      Vec2* tmp = (v==s_VB1) ? s_VB2 : s_VB1;
      tmp[k++] = Vec2::Lerp(v[index[1]], v[index[0]], d[1]/(d[1] - d[0]));
      do
      {
        index[0] = index[1];
        index[1] = (index[1]+1)%n;
        d[0] = d[1];
        d[1] = Vec2::Dot(pn, v[index[1]]) + pw;
        tmp[k++] = v[index[0]];
      } while(d[1]>0);
      tmp[k++] = Vec2::Lerp(v[index[1]], v[index[0]], d[1]/(d[1] - d[0]));
      n = k;
      v = tmp;
    }
    else
    {
      if(d[1]<0) return false;
    }
  }
  return n>0;
}

static const Vec2 c_QuadrantOffsets[] = 
{
  Vec2(0, 0),
  Vec2(1, 0),
  Vec2(1, 1),
  Vec2(0, 1),
};

finline float GetTileDistSq(const AABB2D& bbox, const Vec2& ftop)
{
  return Vec2::LengthSq(0.5f*(bbox.Min + bbox.Max) - ftop);
}

class QTreeNode : public MathLibObject
{
public:
  AABB2D m_BBox;
  unsigned m_LastFrameVerified;

  QTreeNode *m_pParent;
  QTreeNode *m_Children[4];
  TileCacheEntry *m_pTile;
  TileCacheEntry *m_pLayerTile;
  unsigned char m_Refinement, m_nChildren;

  QTreeNode(TilesHierarchy *t, QTreeNode *p) : m_pQTree(t), m_pParent(p)
  {
    m_BBox = AABB2D::Zero();
    memset(m_Children, 0, sizeof(m_Children));
    m_nChildren = 0;
    m_pTile = m_pLayerTile = NULL;
    if(m_pParent!=NULL)
    {
      m_Refinement = m_pParent->m_Refinement + 1;
      m_RootNodesIndex = -1;
    }
    else
    {
      m_Refinement = 0;
      m_RootNodesIndex = m_pQTree->m_Roots.size();
      m_pQTree->m_Roots.push_back(this);
    }
    m_LastFrameVerified = 0;
  }
  ~QTreeNode()
  {
    if(m_pTile!=NULL)
      m_pTile->Free();
    if(m_pLayerTile!=NULL)
      m_pLayerTile->Free();
    for(int i=0; i<4; ++i)
      delete m_Children[i];
    if(m_pParent!=NULL)
    {
      for(int i=0; i<4; ++i)
      {
        if(m_pParent->m_Children[i]==this)
        {
          m_pParent->m_Children[i] = NULL;
          m_pParent->m_nChildren--;
          break;
        }
      }
    }
    else
    {
      QTreeNode *pLast = m_pQTree->m_Roots.back();
      pLast->m_RootNodesIndex = m_RootNodesIndex;
      m_pQTree->m_Roots[m_RootNodesIndex] = pLast;
      m_pQTree->m_Roots.pop_back();
    }
  }
  AABB2D GetChildBBox(int i)
  {
    AABB2D bbox;
    Vec2 hsize = 0.5f*(m_BBox.Max - m_BBox.Min);
    bbox.Min = m_BBox.Min + c_QuadrantOffsets[i]*hsize;
    bbox.Max = bbox.Min + hsize;
    return bbox;
  }
  QTreeNode* AddChild(int i)
  {
    if(m_Children[i]==NULL)
    {
      QTreeNode *pNode = new QTreeNode(m_pQTree, this);
      m_Children[i] = pNode;
      ++m_nChildren;
      return pNode;
    }
    return 0;
  }

protected:
  TilesHierarchy *m_pQTree;
  int m_RootNodesIndex;
};

void TilesHierarchy::Reset()
{
  while(m_Roots.size())
    delete m_Roots[0];
}

void TilesHierarchy::RemoveNonIntersectingNodes(int n, const Vec2 *v, const Vec2& ftop)
{
  for(int i = m_Roots.size() - 1; i>=0; --i)
    RemoveNonIntersectingNodes(m_Roots[i], n, v, ftop);
}

QTreeNode* TilesHierarchy::FindRoot(const AABB2D& BBox)
{
  for(int i = m_Roots.size() - 1; i>=0; --i)
    if(m_Roots[i]->m_BBox==BBox)
      return m_Roots[i];
  return 0;
}

#define SQR(a) ((a)*(a))

static const float c_RefinementDistanceSq[] = { FLT_MAX, SQR(32), SQR(24), SQR(8), SQR(4) };

void TilesHierarchy::RemoveNonIntersectingNodes(QTreeNode *pNode, int n, const Vec2 *v, const Vec2& ftop)
{
  for(int i=0; i<4; ++i)
    if(pNode->m_Children[i])
      RemoveNonIntersectingNodes(pNode->m_Children[i], n, v, ftop);
  if(pNode->m_LastFrameVerified!=g_FrameCounter)
  {
    pNode->m_LastFrameVerified = g_FrameCounter;
    if(GetTileDistSq(pNode->m_BBox, ftop)<c_RefinementDistanceSq[pNode->m_Refinement])
    {
      if(DoesPolygonIntersectBBox(n, v, pNode->m_BBox))
      {
        if(pNode->m_pParent)
          pNode->m_pParent->m_LastFrameVerified = g_FrameCounter;
        return;
      }
    }
    delete pNode;
  }
}

void ShadowFrustum::CreateChildren(QTreeNode *pParent)
{
  for(int i=0; i<4; ++i)
  {
    if(pParent->m_Children[i])
    {
      CreateChildren(pParent->m_Children[i]);
    }
    else if(pParent->m_Refinement<ASM::s_MaxRefinment)
    {
      AABB2D BBox = pParent->GetChildBBox(i);
      if(GetTileDistSq(BBox, m_FrustumTop)<c_RefinementDistanceSq[pParent->m_Refinement + 1])
      {
        if(DoesPolygonIntersectBBox(m_FrustumHullSize, m_FrustumHullVB, BBox))
        {
          QTreeNode *pNode = pParent->AddChild(i);
          pNode->m_BBox = BBox;
          CreateChildren(pNode);
        }
      }
    }
  }
}

void ShadowFrustum::AllocateTiles(QTreeNode *pNode)
{
  for(int i=0; i<4; ++i)
    if(pNode->m_Children[i]!=NULL)
      AllocateTiles(pNode->m_Children[i]);
  if(pNode->m_pTile==NULL)
    TileCache::Allocate(pNode, this, false);
  if(pNode->m_pLayerTile==NULL && pNode->m_Refinement>=ASM_MIN_REFINEMENT_FOR_LAYER)
    TileCache::Allocate(pNode, this, true);
}

static const Vec3 c_FrustumPPS[FRUSTUM_VERTICES_CNT] = 
{
  Vec3( 0,  0, 0),
  Vec3(-1, -1, 1),
  Vec3(+1, -1, 1),
  Vec3(+1, +1, 1),
  Vec3(-1, +1, 1),
};

void ShadowFrustum::CreateTiles(const Mat4x4& ViewProj)
{
  if(!m_ID)
  {
    return;
  }
  Mat4x4 InvViewProj = Mat4x4::Inverse(ViewProj);
  Vec3 ScreenCenter = Vec3::Project(Vec3::Zero(), InvViewProj);
  Vec3 ViewDir = Vec3::Normalize(Vec3::Project(c_ZAxis, InvViewProj) - ScreenCenter);
  Vec3 EndPos = Vec3::Project(ScreenCenter + ASM::s_ShadowDistance*ViewDir, ViewProj);
  Vec3 ZScale = Vec3(1, 1, EndPos.z);

  Vec2 FrustumLS[FRUSTUM_VERTICES_CNT];
  Mat4x4 PPStoLight = InvViewProj*m_LightRotMat;
  for(int i=0; i<FRUSTUM_VERTICES_CNT; ++i)
    FrustumLS[i] = Vec3::Project(ZScale*c_FrustumPPS[i], PPStoLight);
  m_FrustumTop = FrustumLS[0];

  m_FrustumHullSize = OBB2D::FindConvexHull(FRUSTUM_VERTICES_CNT, FrustumLS, m_FrustumHullVB);

  // purge outdated qtree nodes
  m_Tiles.RemoveNonIntersectingNodes(m_FrustumHullSize, m_FrustumHullVB, m_FrustumTop);

  // create qtree nodes
  m_FrustumBBox.Min = Vec2(+FLT_MAX);
  m_FrustumBBox.Max = Vec2(-FLT_MAX);
  for(int i=0; i<m_FrustumHullSize; ++i)
  {
    m_FrustumBBox.Min = Vec2::Min(m_FrustumBBox.Min, m_FrustumHullVB[i]);
    m_FrustumBBox.Max = Vec2::Max(m_FrustumBBox.Max, m_FrustumHullVB[i]);
  }

  int min[2], max[2];
  Vec2 SMin = m_FrustumBBox.Min/(float)ASM::s_LargestTileWorldSize;
  min[0] = Vec2::Floor<x>(SMin)*ASM::s_LargestTileWorldSize;
  min[1] = Vec2::Floor<y>(SMin)*ASM::s_LargestTileWorldSize;
  Vec2 SMax = m_FrustumBBox.Max/(float)ASM::s_LargestTileWorldSize;
  max[0] = Vec2::Ceil<x>(SMax)*ASM::s_LargestTileWorldSize;
  max[1] = Vec2::Ceil<y>(SMax)*ASM::s_LargestTileWorldSize;
  for(int y=min[1]; y<max[1]; y+=ASM::s_LargestTileWorldSize)
  {
    for(int x=min[0]; x<max[0]; x+=ASM::s_LargestTileWorldSize)
    {
      AABB2D BBox;
      BBox.Min = Vec2((float)x, (float)y);
      BBox.Max = Vec2((float)(x + ASM::s_LargestTileWorldSize), (float)(y + ASM::s_LargestTileWorldSize));
      if(DoesPolygonIntersectBBox(m_FrustumHullSize, m_FrustumHullVB, BBox))
      {
        QTreeNode *pNode = m_Tiles.FindRoot(BBox);
        if(pNode==NULL)
        {
          pNode = new QTreeNode(&m_Tiles, 0);
          pNode->m_BBox = BBox;
        }
        CreateChildren(pNode);
      }
    }
  }

  const int n = m_Tiles.m_Roots.size();
  for(int i=0; i<n; ++i)
    AllocateTiles(m_Tiles.m_Roots[i]);
}

align16 class TileSortStruct
{
public:
  QTreeNode *m_pNode;
  float m_Key;

  finline const TileSortStruct& operator = (const TileSortStruct& a)
  {
    _mm_store_ps((float*)this, _mm_load_ps((float*)&a));
    return *this;
  }
  finline bool operator < (const TileSortStruct& a)
  {
    return m_Key<a.m_Key;
  }
};

static const Vec3 c_BiasScale(0.5f, -0.5f, 1.0f);
static const Vec3 c_BiasOffset(0.5f, 0.5f, 0);
static const float c_FrustumFarPlane = 2000.0f;

finline bool CanBeIndexed(TileCacheEntry *t)
{
  return t!=NULL && t->IsReady();
}

finline Vec3 Project(const Vec3& p, const Mat4x4& m)
{
  return (p*m)*c_BiasScale + c_BiasOffset;
}

void ShadowFrustum::BuildIndex()
{
  if(!m_ID)
  {
    return;
  }
  const int nRoots = m_Tiles.m_Roots.size();
  const int nTileSortStructDataSize = sizeof(TileSortStruct)*nRoots;
  TileSortStruct *pSortedTiles = (TileSortStruct*)alloca(nTileSortStructDataSize + sizeof(QTreeNode*)*nRoots*(1<<(2*ASM_MAX_REFINEMENT)));
  QTreeNode **pIndexedNodes = (QTreeNode**)((unsigned char*)pSortedTiles + nTileSortStructDataSize);
  int nTilesToSort = 0;
  for(int i=0; i<nRoots; ++i)
  {
    QTreeNode *pNode = m_Tiles.m_Roots[i];
    TileCacheEntry *pTile = pNode->m_pTile;
    if(CanBeIndexed(pTile))
    {
      TileSortStruct *p = &pSortedTiles[nTilesToSort++];
      Vec2 v = Vec2::Max(Vec2::Abs(pTile->m_BBox.Min - m_FrustumTop), Vec2::Abs(pTile->m_BBox.Max - m_FrustumTop));
      p->m_Key = _cpp_max(v.x, v.y);
      p->m_pNode = pNode;
    }
  }

  std::sort(pSortedTiles, pSortedTiles + nTilesToSort);

  AABB2D BBox = AABB2D::Zero();
  Vec2 v = m_FrustumTop/(float)ASM::s_LargestTileWorldSize;
  BBox.Min.x = (float)(Vec2::Floor<x>(v)*ASM::s_LargestTileWorldSize);
  BBox.Min.y = (float)(Vec2::Floor<y>(v)*ASM::s_LargestTileWorldSize);
  BBox.Max.x = (float)(Vec2::Ceil<x>(v)*ASM::s_LargestTileWorldSize);
  BBox.Max.y = (float)(Vec2::Ceil<y>(v)*ASM::s_LargestTileWorldSize);
  Vec2 IndexMaxSize((float)(ASM_INDEX_SIZE*ASM::s_LargestTileWorldSize));
  TileSortStruct *tss = pSortedTiles;
  float LastGoodKey = 0;
  int nIndexedNodes = 0;
  for(int i=0; i<nTilesToSort; ++i, ++tss)
  {
    TileCacheEntry *pcTile = tss->m_pNode->m_pTile;
    Vec2 min = Vec2::Min(BBox.Min, pcTile->m_BBox.Min);
    Vec2 max = Vec2::Max(BBox.Max, pcTile->m_BBox.Max);
    if(!((max - min)<=IndexMaxSize))
    {
      if(tss->m_Key>LastGoodKey)
      {
        for(++i, ++tss; i<nTilesToSort; ++i, ++tss)
          /* Do_Something_With_Tile_That_Was_Not_Indexed(tss->m_pNode) */;
      }
    }
    else
    {
      BBox.Min = min;
      BBox.Max = max;
      LastGoodKey = tss->m_Key;
      pIndexedNodes[nIndexedNodes++] = tss->m_pNode;
    }
  }
  BBox.Max = BBox.Min + IndexMaxSize;

  if(nIndexedNodes)
  {
    float maxd = -FLT_MAX;
    for(int i=0; i<nIndexedNodes; ++i)
    {
      TileCacheEntry *pTile = pIndexedNodes[i]->m_pTile;
      float d = Vec3::Dot(m_LightDir, pTile->m_InvView.r[3]);
      maxd = _cpp_max(maxd, d);
    }
    Vec3 CameraPos = Unproject(0.5f*(BBox.Min + BBox.Max)) + maxd*m_LightDir;

    Mat4x4 ViewMat = Mat4x4::LookAtD3D(CameraPos, CameraPos - m_LightDir, ASM_UP_VECTOR);
    Vec2 fh = 0.5f*(BBox.Max - BBox.Min);
    Mat4x4 ProjMat = Mat4x4::OrthoD3D(-fh.x, fh.x, -fh.y, fh.y, 0, c_FrustumFarPlane);
    m_IndexTile.m_ViewProj = ViewMat*ProjMat;
    m_IndexTile.m_InvView = Mat4x4::Inverse(ViewMat);

    Vec4 *pIndexTextureData[ASM_MAX_REFINEMENT + 1];
    pIndexTextureData[0] = (Vec4*)&m_IndexTextureData[0];
    int nMipSize = ASM_INDEX_TEXTURE_SIZE*ASM_INDEX_TEXTURE_SIZE;
    for(int i=1; i<=ASM_MAX_REFINEMENT; ++i, nMipSize/=4)
      pIndexTextureData[i] = &pIndexTextureData[i - 1][nMipSize];

    memset(m_IndexTextureData, 0, ASM_INDEX_TEXTURE_DATA_SIZE);
    memset(m_LayerIndexTextureData, 0, ASM_INDEX_TEXTURE_DATA_SIZE);
    for(int z=ASM_MAX_REFINEMENT, i=0; z>=0; --z)
    {
      const int n = nIndexedNodes;
      for(; i<n; ++i)
      {
        QTreeNode *pNode = pIndexedNodes[i];
        for(int t=0; t<2; ++t)
        {
          TileCacheEntry* pTile = pNode->m_pTile;
          bool UseRegularShadowMapAsLayer = false;
          if(t>0)
          {
            if(!CanBeIndexed(pNode->m_pLayerTile))
            {
              if(pNode->m_pParent!=NULL && CanBeIndexed(pNode->m_pParent->m_pLayerTile))
                continue;
              UseRegularShadowMapAsLayer = true;
            }
            else
            {
              _ASSERT((pNode->m_pParent->m_pLayerTile==NULL || pNode->m_pParent->m_pLayerTile->IsReady()) && "layer is ready before parent's layer (render queue is bugged)");
              pTile = pNode->m_pLayerTile;
            }
          }

          float d = Vec3::Dot(m_LightDir, pTile->m_InvView.r[3]);
          Vec3 MinPt = Unproject(pTile->m_BBox.Min) + d*m_LightDir;
          Vec3 MaxPt = Unproject(pTile->m_BBox.Max) + d*m_LightDir;
          Vec3 TileMin = Project(MinPt, pTile->m_ViewProj);
          Vec3 TileMax = Project(MaxPt, pTile->m_ViewProj);
          Vec3 IndexMin = Project(MinPt, m_IndexTile.m_ViewProj);
          Vec3 IndexMax = Project(MaxPt, m_IndexTile.m_ViewProj);

          Vec2 SIndexMax = IndexMax*Vec2(ASM_INDEX_TEXTURE_SIZE) - Vec2(0.25f);
          int x0 = Vec2::Round<x>(SIndexMax);
          int y0 = Vec2::Round<y>(SIndexMax);
          Vec2 SIndexMin = IndexMin*Vec2(ASM_INDEX_TEXTURE_SIZE) - Vec2(0.75f);
          int x1 = Vec2::Round<x>(SIndexMin);
          int y1 = Vec2::Round<y>(SIndexMin);

          _ASSERT(x0<=x1 && "index texture is broken (possibly FP precision issues)");
          _ASSERT(y0<=y1 && "index texture is broken (possibly FP precision issues)");
          _ASSERT((x1 - x0)==(y1 - y0) && "index texture is broken (possibly FP precision issues)");
          _ASSERT(x0<ASM_INDEX_TEXTURE_SIZE && "index texture is broken (possibly FP precision issues)");
          _ASSERT(x1<ASM_INDEX_TEXTURE_SIZE && "index texture is broken (possibly FP precision issues)");
          _ASSERT(y0<ASM_INDEX_TEXTURE_SIZE && "index texture is broken (possibly FP precision issues)");
          _ASSERT(y1<ASM_INDEX_TEXTURE_SIZE && "index texture is broken (possibly FP precision issues)");
          const int nMipMask = (1<<z) - 1;
          _ASSERT(!(x0&nMipMask) && "index texture is broken (possibly FP precision issues)");
          _ASSERT(!(y0&nMipMask) && "index texture is broken (possibly FP precision issues)");
          _ASSERT(!((x1 + 1)&nMipMask) && "index texture is broken (possibly FP precision issues)");
          _ASSERT(!((y1 + 1)&nMipMask) && "index texture is broken (possibly FP precision issues)");

          // index normalized cube -> tile normalized cube: Index_Cube_Coordinate*cScale1 + cOffset1
          Vec3 Scale1;
          Scale1.x = (TileMax.x - TileMin.x)/(IndexMax.x - IndexMin.x);
          Scale1.y = (TileMax.y - TileMin.y)/(IndexMax.y - IndexMin.y);
          Scale1.z = 1.0f;
          Vec3 Offset1 = TileMin - IndexMin*Scale1;

          // tile normalized cube -> shadowmap atlas: Tile_Cube_Coordinates*cScale2 + cOffset2
          Vec3 Scale2;
          Scale2.x = ((float)pTile->m_Viewport.w)*ASM::s_AtlasSize.x;
          Scale2.y = ((float)pTile->m_Viewport.h)*ASM::s_AtlasSize.y;
          Scale2.z = 1.0f;
          Vec3 Offset2;
          Offset2.x = (float)pTile->m_Viewport.x*ASM::s_AtlasSize.x;
          Offset2.y = (float)pTile->m_Viewport.y*ASM::s_AtlasSize.y;
#ifdef HALF_TEXEL_OFFSET
          Offset2.x += 0.5f*ASM::s_AtlasSize.x;
          Offset2.y += 0.5f*ASM::s_AtlasSize.y;
#endif
          Offset2.z = 0;

          // index normalized cube -> shadowmap atlas
          Vec3 Scale = Scale1*Scale2;
          Vec3 Offset = Offset1*Scale2 + Offset2;
          float C0 =  (float)(ASM::s_TileShadowMapSize - 2*ASM_TILE_BORDER_TEXELS)*(float)ASM_INDEX_SIZE;
          int TexelSize = 1<<pTile->m_Refinement;
          Vec4 Pack(Offset.x, Offset.y, Offset.z, C0*(float)(UseRegularShadowMapAsLayer ? -TexelSize : TexelSize));

          for(int j=z; j>=0; --j)
          {
            int mx0 = x0>>j, mx1 = x1>>j;
            int my0 = y0>>j, my1 = y1>>j;
            int step = ASM_INDEX_TEXTURE_SIZE>>j;
            Vec4 *pDst = &pIndexTextureData[j][my0*step];
            if(t>0)
            {
              pDst = (Vec4*)(m_LayerIndexTextureData + ((unsigned char*)pDst - m_IndexTextureData));
            }
            for(int y=my0; y<=my1; ++y)
            {
              for(int x=mx0; x<=mx1; ++x)
                pDst[x] = Pack;
              pDst += step;
            }
          }
        }

        for(int j=0; j<4; ++j)
        {
          QTreeNode *pChild = pNode->m_Children[j];
          if(pChild && CanBeIndexed(pChild->m_pTile))
            pIndexedNodes[nIndexedNodes++] = pChild;
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////

void TileCacheEntry::MarkReady()
{
  _ASSERT(m_pOwner!=NULL && !m_ReadyTilesPos.is_inserted());
  _ASSERT(m_pFrustum!=NULL && m_pFrustum->m_ID!=0 && m_pFrustum->m_ID==m_FrustumID);
  TileCache::s_ReadyTiles.insert(this);
}

void TileCacheEntry::MarkNotReady()
{
  _ASSERT(m_pOwner!=NULL && m_ReadyTilesPos.is_inserted());
  _ASSERT(m_pFrustum!=NULL && m_pFrustum->m_ID!=0 && m_pFrustum->m_ID==m_FrustumID);
  TileCache::s_ReadyTiles.remove(this);
}

bool TileCacheEntry::IsReady()
{
  return m_ReadyTilesPos.is_inserted();
}

TileCacheEntry::TileCacheEntry(int x, int y, int w, int h)
{
  m_Viewport.x = x + ASM_TILE_BORDER_TEXELS; m_Viewport.w = w - 2*ASM_TILE_BORDER_TEXELS;
  m_Viewport.y = y + ASM_TILE_BORDER_TEXELS; m_Viewport.h = h - 2*ASM_TILE_BORDER_TEXELS;
  
  m_BBox = AABB2D::Zero();
  m_pOwner = NULL;
  m_pFrustum = NULL;

  TileCache::s_Tiles.insert(this);
  TileCache::s_FreeTiles.insert(this);

  m_LastFrameUsed = 0;
  m_FrustumID = 0;
  m_RenderMask = 0;

  m_IsLayer = false;
}

TileCacheEntry::~TileCacheEntry()
{
  if(m_pOwner!=NULL)
    Free();
  TileCache::s_Tiles.remove(this);
  TileCache::s_FreeTiles.remove(this);
}

void TileCacheEntry::Allocate(QTreeNode *pNode, ShadowFrustum *pFrustum, bool bIsLayer)
{
  _ASSERT((bIsLayer ? pNode->m_pLayerTile : pNode->m_pTile)==NULL);
  _ASSERT(!bIsLayer || pNode->m_pTile!=NULL);
  TileCache::s_FreeTiles.remove(this);
  m_pOwner = pNode;
  m_pFrustum = pFrustum;
  m_Refinement = pNode->m_Refinement;
  bIsLayer ? (pNode->m_pLayerTile = this) : (pNode->m_pTile = this);
  if(m_FrustumID==pFrustum->m_ID && m_BBox==pNode->m_BBox && m_IsLayer==bIsLayer)
  {
    MarkReady();
  }
  else
  {
    m_FrustumID = pFrustum->m_ID;
    m_BBox = pNode->m_BBox;
    m_IsLayer = bIsLayer;
    TileCache::s_RenderQueue.insert(this);
    m_RenderMask = (1<<g_nGPUs) - 1;
  }
}

void TileCacheEntry::Free()
{
  if(m_RenderQueuePos.is_inserted())
  {
    m_BBox = AABB2D::Zero();
    TileCache::s_RenderQueue.remove(this);
    m_Refinement = ASM_MAX_REFINEMENT;
    m_LastFrameUsed = g_FrameCounter - 0x7fffffff;
  }
  else
  {
    MarkNotReady();
    m_LastFrameUsed = g_FrameCounter;
  }
  TileCache::s_FreeTiles.insert(this);
  m_IsLayer ? (m_pOwner->m_pLayerTile = NULL) : (m_pOwner->m_pTile = NULL);
  m_pOwner = NULL;
  m_pFrustum = NULL;
}

static char s_TileCacheStatMsg[256];

TileCacheEntry* TileCache::Allocate(QTreeNode *pNode, ShadowFrustum *pFrustum, bool bIsLayer)
{
  _ASSERT((bIsLayer ? pNode->m_pLayerTile : pNode->m_pTile)==NULL);
  // first search for tile in cache
  TileCacheEntry *e = NULL;
  if(!s_FreeTiles.size())
  {
    // try to free less important tile (the one further from viewer or deeper in hierarchy)
    unsigned char Refinement = pNode->m_Refinement;
    float DistSq = GetTileDistSq(pNode->m_BBox, pFrustum->m_FrustumTop);
    const int n = s_Tiles.size();
    for(int i=0; i<n; ++i)
    {
      const TileCacheEntry *t = s_Tiles[i];
      _ASSERT(t->m_pFrustum!=NULL);
      if(Refinement>t->m_Refinement)
      {
        continue;
      }
      float f = GetTileDistSq(t->m_BBox, t->m_pFrustum->m_FrustumTop);
      if(Refinement==t->m_Refinement)
      {
        if((DistSq==f && !t->m_IsLayer) || DistSq>f)
          continue;
      }
      Refinement = t->m_Refinement;
      DistSq = f;
      e = const_cast<TileCacheEntry*>(t);
    }
    if(e==NULL)
    {
      return NULL;
    }
    e->Free();
  }
  const int n = s_FreeTiles.size();
  for(int i=0; i<n; ++i)
  {
    const TileCacheEntry *t = s_FreeTiles[i];
    if(t->m_FrustumID==pFrustum->m_ID && t->m_BBox==pNode->m_BBox && t->m_IsLayer==bIsLayer)
    {
      e = const_cast<TileCacheEntry*>(t);
      s_CacheHits++;
      break;
    }
  }
  if(e==NULL)
  {
    // tile isn't in cache, use LRU tile or a tile the deepest in hierarchy
    unsigned Refinement=0, LRUdt=0;
    for(int i=0; i<n; ++i)
    {
      const TileCacheEntry *t = s_FreeTiles[i];
      if(t->m_Refinement<Refinement) continue;
      unsigned dt = g_FrameCounter - t->m_LastFrameUsed;
      if(t->m_Refinement==Refinement && LRUdt>dt) continue;
      e = const_cast<TileCacheEntry*>(t);
      Refinement = e->m_Refinement;
      LRUdt = dt;
    }
  }
  if(e!=NULL)
  {
    e->Allocate(pNode, pFrustum, bIsLayer);
    s_TileAllocs++;
  }
  sprintf_s(s_TileCacheStatMsg, sizeof(s_TileCacheStatMsg), "tiles: %d free out of %d\n", s_FreeTiles.size(), s_Tiles.size());
  printf(s_TileCacheStatMsg);
  return e;
}

/////////////////////////////////////////////////////////////////////////////
// Debug rendering is used, well, for debugging only. Feel free to cut it.

void ShadowFrustum::DrawBBox(DebugRenderer9& Debug, int lvl, QTreeNode *pNode)
{
  if(pNode->m_Refinement==lvl)
  {
    AABB2D& bbox = pNode->m_BBox;
    Vec2 a = MakeDebugTriangleVertex(bbox.Min);
    Vec2 b = MakeDebugTriangleVertex(Vec2(bbox.Max.x, bbox.Min.y));
    Vec2 c = MakeDebugTriangleVertex(bbox.Max);
    Vec2 d = MakeDebugTriangleVertex(Vec2(bbox.Min.x, bbox.Max.y));
    Debug.SetFillColor(Vec4(1,0,1,0.5f));
    Debug.SetContourColor(Vec4(1));
    Debug.PushQuad(a, b, c, d);
  }
  else
  {
    for(int i=0; i<4; ++i)
      if(pNode->m_Children[i]!=NULL)
        DrawBBox(Debug, lvl, pNode->m_Children[i]);
  }
}

void ShadowFrustum::DebugDraw(DebugRenderer9& Debug)
{
  for(int z=0; z<5; ++z)
  {
    const int n = m_Tiles.m_Roots.size();
    for(int i=0; i<n; ++i)
      DrawBBox(Debug, z, m_Tiles.m_Roots[i]);
  }
  Debug.SetFillColor(Vec4(1,1,0,0.5f));
  Debug.SetContourColor(Vec4(0.0f));
  for(int i=2; i<m_FrustumHullSize; ++i)
  {
    Debug.PushTriangle(MakeDebugTriangleVertex(m_FrustumHullVB[0]),
                       MakeDebugTriangleVertex(m_FrustumHullVB[i - 1]),
                       MakeDebugTriangleVertex(m_FrustumHullVB[i]));
  }
}

/////////////////////////////////////////////////////////////////////////////

ptr_set<TileCacheEntry, &TileCacheEntry::GetTilesPos>       TileCache::s_Tiles;
ptr_set<TileCacheEntry, &TileCacheEntry::GetFreeTilesPos>   TileCache::s_FreeTiles;
ptr_set<TileCacheEntry, &TileCacheEntry::GetRenderQueuePos> TileCache::s_RenderQueue;
ptr_set<TileCacheEntry, &TileCacheEntry::GetReadyTilesPos>  TileCache::s_ReadyTiles;
unsigned TileCache::s_CacheHits;
unsigned TileCache::s_TileAllocs;
bool TileCache::s_Initialized;

void TileCache::Initialize(int nAtlasWidth, int nAtlasHeight, int nTileWidth, int nTileHeight)
{
  int htiles = nAtlasWidth/nTileWidth;
  int vtiles = nAtlasHeight/nTileHeight;
  for(int y=0, i=0; i<vtiles; y+=nTileHeight, ++i)
    for(int x=0, j=0; j<htiles; x+=nTileWidth, ++j)
      new TileCacheEntry(x, y, nTileWidth, nTileHeight);
  s_Initialized = true;
}

void TileCache::Shutdown()
{
  if(s_Initialized)
  {
    for(int i=s_Tiles.size()-1; i>=0; --i)
      delete s_Tiles[i];
    _ASSERT(!TileCache::s_Tiles.size());
    _ASSERT(!TileCache::s_FreeTiles.size());
    _ASSERT(!TileCache::s_RenderQueue.size());
    _ASSERT(!TileCache::s_ReadyTiles.size());
  }
  s_Initialized = false;
}

/////////////////////////////////////////////////////////////////////////////

int ASM::s_AtlasWidth = 4096;
int ASM::s_AtlasHeight = 4096;
int ASM::s_TileShadowMapSize = 256;
int ASM::s_MaxRefinment = ASM_MAX_REFINEMENT;
float ASM::s_ShadowDistance = 40;
int ASM::s_LargestTileWorldSize = 20;
ShadowFrustum* ASM::s_Frustum = NULL;
ShadowFrustum* ASM::s_PreRender = NULL;

IDirect3DTexture9* ASM::s_pAtlasTexture = NULL;
IDirect3DSurface9* ASM::s_pAtlasSurface = NULL;
IDirect3DTexture9* ASM::s_pDEMTexture = NULL;
IDirect3DSurface9* ASM::s_pDEMSurface = NULL;
IDirect3DTexture9* ASM::s_pWorkTexture = NULL;
IDirect3DSurface9* ASM::s_pWorkSurface = NULL;
IDirect3DTexture9* ASM::s_pIndexTexture = NULL;
IDirect3DTexture9* ASM::s_pLayerIndexTexture = NULL;

ShaderObject9 ASM::s_DepthExtentShader;

IDirect3DDevice9* ASM::s_Device9;
IDirect3DVertexBuffer9* ASM::s_pQuadVB;
IDirect3DVertexDeclaration9* ASM::s_QuadDecl;

static ASM* s_ASM = NULL;

ASM::ASM()
{
  s_AtlasSize.x = 1.0f/(float)s_AtlasWidth;
  s_AtlasSize.y = 1.0f/(float)s_AtlasHeight;
  s_AtlasSize.z = (float)s_AtlasWidth;
  s_AtlasSize.w = (float)s_AtlasHeight;

  s_Bias = Mat4x4::ScalingTranslationD3D(Vec3(0.5f, -0.5f, 1), Vec3(0.5f, 0.5f, 0));

  s_Frustum = new ShadowFrustum();
  s_PreRender = new ShadowFrustum();

  s_ASM = this;
}

ASM::~ASM()
{
  delete s_Frustum;
  s_Frustum = NULL;
  delete s_PreRender;
  s_PreRender = NULL;

  s_ASM = NULL;
}

HRESULT ASM::Init(IDirect3DDevice9* Device9)
{
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  static const D3DVERTEXELEMENT9 c_QuadVertexElems[] =
  {
    { 0, 0, D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
    D3DDECL_END()
  }; 
  static const Vec4 c_ScreenQuadData[]=
  {
    Vec4(+1, -1, 1, 1),
    Vec4(-1, -1, 0, 1),
    Vec4(+1, +1, 1, 0),
    Vec4(-1, +1, 0, 0),
  };

  s_Device9 = Device9;

  HRESULT hr = s_Device9->CreateTexture(ASM_INDEX_TEXTURE_SIZE, ASM_INDEX_TEXTURE_SIZE, ASM_MAX_REFINEMENT + 1, D3DUSAGE_DYNAMIC, D3DFMT_A32B32G32R32F, D3DPOOL_DEFAULT, &s_pIndexTexture, 0);
  hr = SUCCEEDED(hr) ? s_Device9->CreateTexture(ASM_INDEX_TEXTURE_SIZE, ASM_INDEX_TEXTURE_SIZE, ASM_MAX_REFINEMENT + 1, D3DUSAGE_DYNAMIC, D3DFMT_A32B32G32R32F, D3DPOOL_DEFAULT, &s_pLayerIndexTexture, 0) : hr;
  hr = SUCCEEDED(hr) ? s_Device9->CreateTexture(s_AtlasWidth, s_AtlasHeight, 1, D3DUSAGE_DEPTHSTENCIL, D3DFMT_D16, D3DPOOL_DEFAULT, &s_pAtlasTexture, NULL) : hr;
  hr = SUCCEEDED(hr) ? s_Device9->CreateTexture(s_AtlasWidth, s_AtlasHeight, 1, D3DUSAGE_RENDERTARGET, D3DFMT_R16F, D3DPOOL_DEFAULT, &s_pDEMTexture, NULL) : hr;
  hr = SUCCEEDED(hr) ? s_Device9->CreateTexture(s_TileShadowMapSize, s_TileShadowMapSize, 1, D3DUSAGE_RENDERTARGET, D3DFMT_R16F, D3DPOOL_DEFAULT, &s_pWorkTexture, NULL) : hr;
  hr = SUCCEEDED(hr) ? s_DepthExtentShader.Init(s_Device9, "Shaders\\DepthExtentMap") : hr;
  hr = SUCCEEDED(hr) ? s_Device9->CreateVertexDeclaration(c_QuadVertexElems, &s_QuadDecl) : hr;
  if(SUCCEEDED(hr))
  {
    hr = s_Device9->CreateVertexBuffer(sizeof(c_ScreenQuadData), D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT, &s_pQuadVB, NULL);
    if(SUCCEEDED(hr))
    {
      void* pMem;
      if(SUCCEEDED(s_pQuadVB->Lock(0, 0, &pMem, 0)))
      {
        memcpy(pMem, c_ScreenQuadData, sizeof(c_ScreenQuadData));
        s_pQuadVB->Unlock();
      }
    }
  }
  if(SUCCEEDED(hr))
  {
    s_pAtlasTexture->GetSurfaceLevel(0, &s_pAtlasSurface);
    s_pDEMTexture->GetSurfaceLevel(0, &s_pDEMSurface);
    s_pWorkTexture->GetSurfaceLevel(0, &s_pWorkSurface);
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    TileCache::Initialize(s_AtlasWidth, s_AtlasHeight, s_TileShadowMapSize, s_TileShadowMapSize);
    s_Frustum->Reset();
    s_PreRender->Reset();
  }
  return hr;
}

void ASM::Release()
{
  SAFE_RELEASE(s_pAtlasTexture);
  SAFE_RELEASE(s_pAtlasSurface);
  SAFE_RELEASE(s_pDEMTexture);
  SAFE_RELEASE(s_pDEMSurface);
  SAFE_RELEASE(s_pWorkTexture);
  SAFE_RELEASE(s_pWorkSurface);
  SAFE_RELEASE(s_pIndexTexture);
  SAFE_RELEASE(s_pLayerIndexTexture);

  s_DepthExtentShader.Release();
  SAFE_RELEASE(s_QuadDecl);
  SAFE_RELEASE(s_pQuadVB);

  s_Frustum->Reset();
  s_PreRender->Reset();
  TileCache::Shutdown();
}

template<int NLINES0, int LINESIZE0, int NMIPS> void UpdateDynamicTexture(IDirect3DTexture9 *pTexture, const void *pData)
{
  int nLines = NLINES0;
  int nLineSize = LINESIZE0;
  int nFlags = D3DLOCK_DISCARD;
  for(int z=0; z<NMIPS; ++z)
  {
    D3DLOCKED_RECT LRect;
    if(SUCCEEDED(pTexture->LockRect(z, &LRect, 0, nFlags)))
    {
      if(LRect.Pitch!=nLineSize)
      {
        unsigned char *pSrc = (unsigned char*)pData;
        unsigned char *pDst = (unsigned char*)LRect.pBits;
        for(int y=0; y<nLines; ++y)
        {
          memcpy(pDst, pSrc, nLineSize);
          pDst += LRect.Pitch;
          pSrc += nLineSize;
        }
      }
      else
      {
        memcpy(LRect.pBits, pData, nLines*nLineSize);
      }
      pTexture->UnlockRect(z);
      pData = (void*)(((unsigned char*)pData) + nLines*nLineSize);
      nLines>>=1;
      nLineSize>>=1;
      nFlags = 0;
    }
  }
}

static std::vector<TileCacheEntry*> s_DEMRenderQueue;

void ASM::Render(const Mat4x4& ViewProj)
{
  s_DEMRenderQueue.clear();

  s_Frustum->CreateTiles(ViewProj);
  s_PreRender->CreateTiles(ViewProj);

  if(TileCache::s_RenderQueue.size())
  {
    IDirect3DSurface9 *pRT, *pDS;
    s_Device9->GetRenderTarget(0, &pRT);
    s_Device9->GetDepthStencilSurface(&pDS);

    s_Device9->SetRenderTarget(0, s_pDEMSurface);
    s_Device9->SetDepthStencilSurface(s_pAtlasSurface);

    if(s_Frustum->RenderTile(0, false)==0)
    {
      for(int i=0; i<10; ++i)
        if(s_Frustum->RenderTile(0, false)<0)
          break;
    }
    else if(s_PreRender->RenderTile(INT_MAX, false)>0)
    {
      std::swap(s_Frustum, s_PreRender);
      s_PreRender->Reset();
    }
    else if(s_Frustum->RenderTile(INT_MAX, false)<0)
    {
      s_Frustum->RenderTile(INT_MAX, true);
    }

    s_Device9->SetRenderTarget(0, s_pDEMSurface);
    s_Device9->SetDepthStencilSurface(NULL);

    for(int i = s_DEMRenderQueue.size() - 1; i>=0; --i)
      s_DEMRenderQueue[i]->CreateDEM();

    s_Device9->SetRenderTarget(0, pRT);
    s_Device9->SetDepthStencilSurface(pDS);
    pRT->Release();
    pDS->Release();
  }

  s_Frustum->BuildIndex();
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  UpdateDynamicTexture<ASM_INDEX_TEXTURE_SIZE, ASM_INDEX_TEXTURE_SIZE*sizeof(Vec4), ASM_MAX_REFINEMENT + 1>(s_pIndexTexture, s_Frustum->m_IndexTextureData);
  UpdateDynamicTexture<ASM_INDEX_TEXTURE_SIZE, ASM_INDEX_TEXTURE_SIZE*sizeof(Vec4), ASM_MAX_REFINEMENT + 1>(s_pLayerIndexTexture, s_Frustum->m_LayerIndexTextureData);
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

  ++g_FrameCounter;
}

int ShadowFrustum::RenderTile(int MaxRefinement, bool bIsLayer)
{
  if(!m_ID)
  {
    return -1;
  }
  const unsigned GPUMask = GetGPUMask();
  const int n = TileCache::s_RenderQueue.size();
  TileCacheEntry *pTileToRender = NULL;
  float MinDist = FLT_MAX;
  int Refinement = INT_MAX;
  for(int i=0; i<n; ++i)
  {
    const TileCacheEntry *pTile = TileCache::s_RenderQueue[i];
    if(this!=pTile->m_pFrustum ||  !(pTile->m_RenderMask&GPUMask) || 
       pTile->m_IsLayer!=bIsLayer || Refinement<pTile->m_pOwner->m_Refinement)
    {
      continue;
    }
    float f = GetTileDistSq(pTile->m_BBox, m_FrustumTop);
    if(Refinement==pTile->m_pOwner->m_Refinement && MinDist<f)
    {
      continue;
    }
    Refinement = pTile->m_pOwner->m_Refinement;
    MinDist = f;
    pTileToRender = const_cast<TileCacheEntry*>(pTile);
  }
  if(pTileToRender==NULL || Refinement>MaxRefinement)
  {
    return -1;
  }
  pTileToRender->Render();
  return Refinement;
}

void GetRectangleWithinParent(const int NUp, QTreeNode* NList[ASM_MAX_REFINEMENT + 1], RECT& ParentRect, RECT& TileRect)
{
  for(int i=0; i<NUp; ++i)
    NList[i + 1] = NList[i]->m_pParent;
  const Tile::Viewport& SrcViewport = NList[NUp]->m_pTile->m_Viewport;
  ParentRect.left = SrcViewport.x - ASM_TILE_BORDER_TEXELS;
  ParentRect.top  = SrcViewport.y - ASM_TILE_BORDER_TEXELS;
  ParentRect.right  = SrcViewport.x + SrcViewport.w + ASM_TILE_BORDER_TEXELS;
  ParentRect.bottom = SrcViewport.y + SrcViewport.h + ASM_TILE_BORDER_TEXELS;
  RECT Src = ParentRect;
  for(int i=0; i<NUp; ++i)
  {
    Vec2 d = NList[NUp - i - 1]->m_BBox.Min - NList[NUp - i]->m_BBox.Min;
    RECT Rect;
    Rect.left = d.x>0 ? Src.left : (Src.right + Src.left)/2;
    Rect.top  = d.y>0 ? Src.top  : (Src.bottom + Src.top)/2;
    Rect.right  = Rect.left + (Src.right - Src.left)/2;
    Rect.bottom = Rect.top  + (Src.bottom - Src.top)/2;
    const int Border = ASM_TILE_BORDER_TEXELS>>(i + 1);
    Rect.left   += d.x>0 ? Border : -Border;
    Rect.right  += d.x>0 ? Border : -Border;
    Rect.top    += d.y>0 ? Border : -Border;
    Rect.bottom += d.y>0 ? Border : -Border;
    Src = Rect;
  }
  TileRect = Src;
}

void TileCacheEntry::Render()
{
  _ASSERT(!m_IsLayer || m_pOwner->m_pTile->m_ReadyTilesPos.is_inserted());

  Vec3 AABBMin, AABBMax;
  s_ASM->GetSceneAABB(&AABBMin, &AABBMax);
  AABBMin = AABBMin - Vec3((float)ASM::s_LargestTileWorldSize);
  AABBMax = AABBMax + Vec3((float)ASM::s_LargestTileWorldSize);
  Mat4x4 BBoxToLight = Mat4x4::ScalingTranslationD3D(0.5f*(AABBMax - AABBMin), 0.5f*(AABBMax + AABBMin))*m_pFrustum->m_LightRotMat;
  float minZ = FLT_MAX;
  for(int i=0; i<8; ++i)
  {
    Vec4 v = TransformBoxCorner(i, BBoxToLight);
    minZ = _cpp_min(minZ, v.z);
  }
  Vec3 CamPos = m_pFrustum->Unproject(0.5f*(m_BBox.Min + m_BBox.Max)) + minZ*m_pFrustum->m_LightDir;

  const Vec4 Bounds[] =
  {
    Vec4(-1.0f, 0.0f, 0.0f, AABBMax.x),
    Vec4( 0.0f,-1.0f, 0.0f, AABBMax.y),
    Vec4( 0.0f, 0.0f,-1.0f, AABBMax.z),
    Vec4( 1.0f, 0.0f, 0.0f,-AABBMin.x),
    Vec4( 0.0f, 1.0f, 0.0f,-AABBMin.y),
    Vec4( 0.0f, 0.0f, 1.0f,-AABBMin.z),
  };
  float minF = FLT_MAX;
  for(int i=0; i<ARRAYSIZE(Bounds); ++i)
  {
    float f1 = Vec4::Dot(Bounds[i], Vec3::Point(CamPos));
    float f2 = Vec4::Dot(Bounds[i], Vec3::Vector(m_pFrustum->m_LightDir));
    if(f1>0 && f2<0)
    {
      minF = _cpp_min(minF, -f1/f2);
    }
  }
  CamPos = CamPos + minF*m_pFrustum->m_LightDir;

  D3DVIEWPORT9 Viewport;
  Viewport.X      = m_Viewport.x - ASM_TILE_BORDER_TEXELS;
  Viewport.Y      = m_Viewport.y - ASM_TILE_BORDER_TEXELS;
  Viewport.Width  = m_Viewport.w + 2*ASM_TILE_BORDER_TEXELS;
  Viewport.Height = m_Viewport.h + 2*ASM_TILE_BORDER_TEXELS;
  Viewport.MinZ   = 0;
  Viewport.MaxZ   = 1;

  Mat4x4 ViewportExtension = Mat4x4::Identity();
  ViewportExtension.e11 = (float)m_Viewport.w/(float)Viewport.Width;
  ViewportExtension.e22 = (float)m_Viewport.h/(float)Viewport.Height;
  ViewportExtension.e14 = ViewportExtension.e11 + 2.0f*(float)ASM_TILE_BORDER_TEXELS/(float)Viewport.Width - 1.0f;
  ViewportExtension.e24 = 1.0f - ViewportExtension.e22 - 2.0f*(float)ASM_TILE_BORDER_TEXELS/(float)Viewport.Height;

  Mat4x4 ViewMat = Mat4x4::LookAtD3D(CamPos, CamPos - m_pFrustum->m_LightDir, ASM_UP_VECTOR);
  Vec2 fh = 0.5f*(m_BBox.Max - m_BBox.Min);
  Mat4x4 ProjMat = Mat4x4::OrthoD3D(-fh.x, fh.x, -fh.y, fh.y, 0, c_FrustumFarPlane);

  m_ViewProj = ViewMat*ProjMat;
  m_InvView = Mat4x4::Inverse(ViewMat);

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  // In D3D9, Clear() affects viewport only; 
  // one has to emulate this by drawing a quad that outputs color/depth in D3D10.
  ASM::s_Device9->SetViewport(&Viewport);
  ASM::s_Device9->Clear(0, NULL, D3DCLEAR_TARGET|D3DCLEAR_ZBUFFER, 0xffffffff, 1.0f, 0);

  // Plausible anti-self-shadowing offset determined experimentaly. 
  // To avoid artifacs, such offsetting generally has to be supplemented 
  // with a gradient-based offsetting, see e.g. 
  // John R. Isidoro. Shadow Mapping: GPU-based Tips and Techniques.
  union { float fBias; unsigned iBias; };
  fBias = 0.00002f*(float)(1<<(ASM_MAX_REFINEMENT - m_Refinement));
  ASM::s_Device9->SetRenderState(D3DRS_DEPTHBIAS, iBias);
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

  if(m_IsLayer)
  {
    const int NUp = m_Refinement;
    QTreeNode* NList[ASM_MAX_REFINEMENT + 1] = { m_pOwner };
    RECT ParentRect, Src;
    GetRectangleWithinParent(NUp, NList, ParentRect, Src);

    // copy parent's DEM to s_pWorkSurface
    ASM::s_Device9->SetRenderTarget(0, ASM::s_pWorkSurface);
    ASM::s_Device9->SetDepthStencilSurface(NULL);

    CopyDEM(ParentRect.left, ParentRect.top);

    ASM::s_Device9->SetRenderTarget(0, ASM::s_pDEMSurface);
    ASM::s_Device9->SetDepthStencilSurface(ASM::s_pAtlasSurface);
    ASM::s_Device9->SetViewport(&Viewport);

    Vec4 PSParam;
    PSParam.x = (Vec4(m_ViewProj.r[3]) - Vec4(NList[NUp]->m_pTile->m_ViewProj.r[3])).z;
    PSParam.y = 0;
    PSParam.z = 0;
    PSParam.w = 0;

    Vec4 VSParam;
    VSParam.x = (float)(Src.right - Src.left)/(float)ASM::s_TileShadowMapSize;
    VSParam.y = (float)(Src.bottom - Src.top)/(float)ASM::s_TileShadowMapSize;
    VSParam.z = (float)(Src.left - ParentRect.left)/(float)ASM::s_TileShadowMapSize;
    VSParam.w = (float)(Src.top - ParentRect.top)/(float)ASM::s_TileShadowMapSize;
#ifdef HALF_TEXEL_OFFSET
    VSParam.z += 0.5f/(float)ASM::s_TileShadowMapSize;
    VSParam.w += 0.5f/(float)ASM::s_TileShadowMapSize;
#endif

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // DEM stored in s_pWorkTexture is used for depth peeling.
    const int c_DEMSampler = 10;
    ASM::s_Device9->SetTexture(c_DEMSampler, ASM::s_pWorkTexture);
    ASM::s_Device9->SetSamplerState(c_DEMSampler, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
    ASM::s_Device9->SetSamplerState(c_DEMSampler, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
    ASM::s_Device9->SetSamplerState(c_DEMSampler, D3DSAMP_MIPFILTER, D3DTEXF_NONE);
    ASM::s_Device9->SetSamplerState(c_DEMSampler, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
    ASM::s_Device9->SetSamplerState(c_DEMSampler, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    s_ASM->RenderScene(ViewMat, ProjMat*ViewportExtension, VSParam, PSParam);

    ASM::s_Device9->SetTexture(c_DEMSampler, NULL);
  }
  else
  {
    s_ASM->RenderScene(ViewMat, ProjMat*ViewportExtension);
  }

  ASM::s_Device9->SetRenderState(D3DRS_DEPTHBIAS, 0);

  m_RenderMask &= ~GetGPUMask();
  if(!m_RenderMask)
  {
    TileCache::s_RenderQueue.remove(this);
    MarkReady();
  }

//  if(m_Refinement==0)
  s_DEMRenderQueue.push_back(this);

  ++ASM::s_nTilesRendered;
}

void ASM::SetTextures(int nSMSamper, int nIndexSampler, int nDEMSampler, int nLayerIndexSampler)
{
  s_Device9->SetTexture(nSMSamper, s_pAtlasTexture);
  s_Device9->SetSamplerState(nSMSamper, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
  s_Device9->SetSamplerState(nSMSamper, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
  s_Device9->SetSamplerState(nSMSamper, D3DSAMP_MIPFILTER, D3DTEXF_NONE);
  s_Device9->SetSamplerState(nSMSamper, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
  s_Device9->SetSamplerState(nSMSamper, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);  

  s_Device9->SetTexture(nIndexSampler, s_pIndexTexture);
  s_Device9->SetSamplerState(nIndexSampler, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
  s_Device9->SetSamplerState(nIndexSampler, D3DSAMP_MINFILTER, D3DTEXF_POINT);
  s_Device9->SetSamplerState(nIndexSampler, D3DSAMP_MIPFILTER, D3DTEXF_POINT);
  s_Device9->SetSamplerState(nIndexSampler, D3DSAMP_ADDRESSU, D3DTADDRESS_BORDER);
  s_Device9->SetSamplerState(nIndexSampler, D3DSAMP_ADDRESSV, D3DTADDRESS_BORDER);
  s_Device9->SetSamplerState(nIndexSampler, D3DSAMP_BORDERCOLOR, 0);

  s_Device9->SetTexture(nDEMSampler, s_pDEMTexture);
  s_Device9->SetSamplerState(nDEMSampler, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
  s_Device9->SetSamplerState(nDEMSampler, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
  s_Device9->SetSamplerState(nDEMSampler, D3DSAMP_MIPFILTER, D3DTEXF_NONE);
  s_Device9->SetSamplerState(nDEMSampler, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
  s_Device9->SetSamplerState(nDEMSampler, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);

  s_Device9->SetTexture(nLayerIndexSampler, s_pLayerIndexTexture);
  s_Device9->SetSamplerState(nLayerIndexSampler, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
  s_Device9->SetSamplerState(nLayerIndexSampler, D3DSAMP_MINFILTER, D3DTEXF_POINT);
  s_Device9->SetSamplerState(nLayerIndexSampler, D3DSAMP_MIPFILTER, D3DTEXF_POINT);
  s_Device9->SetSamplerState(nLayerIndexSampler, D3DSAMP_ADDRESSU, D3DTADDRESS_BORDER);
  s_Device9->SetSamplerState(nLayerIndexSampler, D3DSAMP_ADDRESSV, D3DTADDRESS_BORDER);
  s_Device9->SetSamplerState(nLayerIndexSampler, D3DSAMP_BORDERCOLOR, 0);
}

static void FillShaderConstants(Vec4* vs, Vec4* ps, int nDstRectSize,
                                int nDstX, int nDstY, int nDstW, int nDstH, 
                                int nSrcRectSize,
                                int nSrcX, int nSrcY, int nSrcW, int nSrcH,
                                float KernelSizeFactor)
{
  vs[0].x = (float)nDstRectSize/(float)nDstW;
  vs[0].y = (float)nDstRectSize/(float)nDstH;
  vs[0].z = vs[0].x + 2.0f*(float)nDstX/(float)nDstW - 1.0f;
  vs[0].w = 1.0f - vs[0].y - 2.0f*(float)nDstY/(float)nDstH;
  vs[1].x = (float)nSrcRectSize/(float)nSrcW;
  vs[1].y = (float)nSrcRectSize/(float)nSrcH;
  vs[1].z = (float)nSrcX/(float)nSrcW;
  vs[1].w = (float)nSrcY/(float)nSrcH;
#ifdef HALF_TEXEL_OFFSET
  vs[1].z += (float)0.5f/(float)nSrcW;
  vs[1].w += (float)0.5f/(float)nSrcH;
#endif

  static const Vec4 c_Offsets[8] =
  {
    Vec4( 1, 1, 0, 0), Vec4( 0, 1, 0, 0),
    Vec4(-1, 1, 0, 0), Vec4(-1, 0, 0, 0),
    Vec4(-1,-1, 0, 0), Vec4( 0,-1, 0, 0),
    Vec4( 1,-1, 0, 0), Vec4( 1, 0, 0, 0),
  };
  Vec4 f(KernelSizeFactor/(float)nSrcW, KernelSizeFactor/(float)nSrcH, 0, 0);
  for(int i=0; i<ARRAYSIZE(c_Offsets); ++i)
    ps[i] = f*c_Offsets[i];
}

void TileCacheEntry::CreateDEM()
{
  ASM::s_Device9->SetRenderTarget(0, ASM::s_pWorkSurface);
  ASM::s_Device9->SetDepthStencilSurface(NULL);

  ASM::s_Device9->Clear(0, 0, D3DCLEAR_TARGET, 0xffffffff, 0, 0);

  ASM::s_DepthExtentShader.Bind();
  ASM::s_Device9->SetStreamSource(0, ASM::s_pQuadVB, 0, sizeof(Vec4));
  ASM::s_Device9->SetVertexDeclaration(ASM::s_QuadDecl);

  Vec4 vs[2], ps[8];
  const int nRectSize = ASM::s_TileShadowMapSize - 2;
  const int nTileSize = ASM::s_TileShadowMapSize;
  const int nAtlasX = m_Viewport.x - ASM_TILE_BORDER_TEXELS + 1;
  const int nAtlasY = m_Viewport.y - ASM_TILE_BORDER_TEXELS + 1;
  const int nAtlasW = ASM::s_AtlasWidth;
  const int nAtlasH = ASM::s_AtlasHeight;
  FillShaderConstants(vs, ps, nRectSize, 1, 1, nTileSize, nTileSize, nRectSize, nAtlasX, nAtlasY, nAtlasW, nAtlasH, 1.0f);

  ASM::s_Device9->SetVertexShaderConstantF(0, &vs[0].x, ARRAYSIZE(vs));
  ASM::s_Device9->SetPixelShaderConstantF(0, &ps[0].x, ARRAYSIZE(ps));

  ASM::s_Device9->SetTexture(0, ASM::s_pDEMTexture);
  ASM::s_Device9->SetSamplerState(0, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
  ASM::s_Device9->SetSamplerState(0, D3DSAMP_MINFILTER, D3DTEXF_POINT);
  ASM::s_Device9->SetSamplerState(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT);
  ASM::s_Device9->SetSamplerState(0, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
  ASM::s_Device9->SetSamplerState(0, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);

  ASM::s_Device9->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);

  ASM::s_Device9->SetRenderTarget(0, ASM::s_pDEMSurface);

  ASM::s_Device9->SetTexture(0, ASM::s_pWorkTexture);

  if(m_Refinement>0 && !m_IsLayer)
  {
    const int NUp = m_Refinement;
    QTreeNode* NList[ASM_MAX_REFINEMENT + 1] = { m_pOwner };
    RECT ParentRect, Src;
    GetRectangleWithinParent(NUp, NList, ParentRect, Src);
    FillShaderConstants(vs, ps, Src.right - Src.left, Src.left, Src.top, nAtlasW, nAtlasH, nTileSize, 0, 0, nTileSize, nTileSize, 1.0f);
    ps[0].z = (Vec4(NList[NUp]->m_pTile->m_ViewProj.r[3]) - Vec4(m_ViewProj.r[3])).z;

    ASM::s_Device9->SetRenderState(D3DRS_ALPHABLENDENABLE, TRUE);
    ASM::s_Device9->SetRenderState(D3DRS_BLENDOP, D3DBLENDOP_MIN);
    ASM::s_Device9->SetRenderState(D3DRS_DESTBLEND, D3DBLEND_ONE);
    ASM::s_Device9->SetRenderState(D3DRS_SRCBLEND, D3DBLEND_ONE);
  }
  else
  {
    FillShaderConstants(vs, ps, nRectSize, nAtlasX, nAtlasY, nAtlasW, nAtlasH, nRectSize, 1, 1, nTileSize, nTileSize, 1.0f);
  }

  ASM::s_Device9->SetVertexShaderConstantF(0, &vs[0].x, ARRAYSIZE(vs));
  ASM::s_Device9->SetPixelShaderConstantF(0, &ps[0].x, ARRAYSIZE(ps));

  ASM::s_Device9->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);

  ASM::s_Device9->SetTexture(0, NULL);

  if(m_Refinement>0)
  {
    ASM::s_Device9->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
  }
}

void TileCacheEntry::CopyDEM(int x, int y)
{
  ASM::s_DepthExtentShader.Bind();
  ASM::s_Device9->SetStreamSource(0, ASM::s_pQuadVB, 0, sizeof(Vec4));
  ASM::s_Device9->SetVertexDeclaration(ASM::s_QuadDecl);

  Vec4 vs[2], ps[8];
  FillShaderConstants(vs, ps, ASM::s_TileShadowMapSize, 0, 0, ASM::s_TileShadowMapSize, ASM::s_TileShadowMapSize, ASM::s_TileShadowMapSize, x, y, ASM::s_AtlasWidth, ASM::s_AtlasHeight, 1.5f);
  ASM::s_Device9->SetVertexShaderConstantF(0, &vs[0].x, ARRAYSIZE(vs));
  ASM::s_Device9->SetPixelShaderConstantF(0, &ps[0].x, ARRAYSIZE(ps));

  ASM::s_Device9->SetTexture(0, ASM::s_pDEMTexture);
  ASM::s_Device9->SetSamplerState(0, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
  ASM::s_Device9->SetSamplerState(0, D3DSAMP_MINFILTER, D3DTEXF_POINT);
  ASM::s_Device9->SetSamplerState(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT);
  ASM::s_Device9->SetSamplerState(0, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
  ASM::s_Device9->SetSamplerState(0, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);

  ASM::s_Device9->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);

  ASM::s_Device9->SetTexture(0, NULL);
}

/////////////////////////////////////////////////////////////////////////////

int ASM::s_nTilesRendered = 0;
bool ASM::s_bDrawHUD = false;
Vec4 ASM::s_AtlasSize;
Mat4x4 ASM::s_Bias;

Vec4   ASM::GetAtlasSize()                     { return s_AtlasSize; }
Mat4x4 ASM::GetViewProj()                      { return s_Frustum->m_IndexTile.m_ViewProj*s_Bias; }
Vec3   ASM::GetLightDir()                      { return s_Frustum->m_LightDir; }
Vec3   ASM::GetPreRenderLightDir()             { return s_PreRender->m_LightDir; }
int    ASM::GetTileShadowMapSize()             { return s_TileShadowMapSize; }
int    ASM::GetNumberOfRenderedTiles()         { int n = s_nTilesRendered; s_nTilesRendered = 0; return n; }
float  ASM::GetShadowDistance()                { return s_ShadowDistance;}

void ASM::ResetCache()
{
  s_Frustum->Reset();
  s_PreRender->Reset();
  TileCache::Shutdown();
  TileCache::Initialize(s_AtlasWidth, s_AtlasHeight, s_TileShadowMapSize, s_TileShadowMapSize);
}

void ASM::SetLightDir(const Vec3& LightDir)
{
  (s_Frustum->IsValid() ? s_PreRender : s_Frustum)->Set(LightDir);
}

void ASM::DebugDraw(DebugRenderer9& Debug)
{
  s_Frustum->DebugDraw(Debug);
}
