/* Implementation of adaptive shadow maps.

   ***********************************************************************
   * Callbacks
   ***********************************************************************

   ASM::GetSceneAABB(Vec3* AABBMin, Vec3* AABBMax)
     returns whole scene's AABB. The AABB is used to construct shadow 
     mapping matrices.

   ASM::RenderScene(const Mat4x4& ViewMatrix, const Mat4x4& ProjectionMatrix) 
     renders a shadow map using specified matrices. The matrices should also
     be used for culling objects since drawing the whole scene will be a waste,
     as generally only a tiny fraction of the scene is rendered.

   ASM::RenderScene(const Mat4x4& ViewMatrix, const Mat4x4& ProjectionMatrix,
   const Vec4& VSParam, const Vec4& PSParam)
     same as above, but this one is used for layer shadow map rendering.
     VSParam  and PSParam are passed down to PS/VS for accessing the DEM, which 
     is bound to c_DEMSampler.

   ***********************************************************************
   * Parameters
   ***********************************************************************

   ASM::s_TileShadowMapSize
     size of tile's shadow map; tile shadow maps are assembled into 
     a single atlas.

   ASM::s_AtlasWidth
   ASM::s_AtlasHeight
     size of atlas depth texture.

   ASM::s_MaxRefinment
     is max number of subdivisions of a tile; 
     valid range 0...ASM_MAX_REFINEMENT

   ASM::s_ShadowDistance
     max distance from the camera, in world space, at which ASM shadows 
     are rendered.

   ASM::s_LargestTileWorldSize
     the size of a non-subdivided tile in world space.

   c_RefinementDistanceSq[]
     squared threshold distances, in world space, at which a tile is refined 
     further.

   c_FrustumFarPlane
     far plane of all shadow map projection matrices

   ASM_UP_VECTOR
     world up vector for shadow mapping matrices

   ASM_MIN_REFINEMENT_FOR_LAYER
     tile refinement level starting from which layer shadow maps are 
     created */

#ifndef __ASM
#define __ASM

#define ASM_TILE_BORDER_TEXELS       8
#define ASM_MAX_REFINEMENT           3
#define ASM_INDEX_SIZE               8
#define ASM_INDEX_TEXTURE_SIZE       ((1<<ASM_MAX_REFINEMENT)*ASM_INDEX_SIZE)
#define ASM_INDEX_TEXTURE_DATA_SIZE  (ASM_INDEX_TEXTURE_SIZE*ASM_INDEX_TEXTURE_SIZE*sizeof(Vec4)*2)
#define ASM_UP_VECTOR                c_YAxis
#define ASM_MIN_REFINEMENT_FOR_LAYER 2

#define HALF_TEXEL_OFFSET // DX9-specific, should be undefined for D3D10/OpenGL

#include <d3d9.h>
#include "../Math/Math.h"
#include "../Util/ShaderObject9.h"

class ShadowFrustum;

class ASM
{
public:
  virtual void GetSceneAABB(Vec3* AABBMin, Vec3* AABBMax) PURE;
  virtual void RenderScene(const Mat4x4&, const Mat4x4&) PURE;
  virtual void RenderScene(const Mat4x4&, const Mat4x4&, const Vec4&, const Vec4&) PURE;

  static HRESULT Init(IDirect3DDevice9*);
  static void Release();

  static void SetTextures(int, int, int, int);
  static void SetLightDir(const Vec3&);
  static void ResetCache();
  static void Render(const Mat4x4&);

  static Vec4 GetAtlasSize();
  static Mat4x4 GetViewProj();
  static Vec4 GetDepthPassParamPS();
  static Vec4 GetDepthPassParamVS();
  static Vec3 GetLightDir();
  static Vec3 GetPreRenderLightDir();
  static int GetTileShadowMapSize();
  static int GetNumberOfRenderedTiles();
  static float GetShadowDistance();
  static void DebugDraw(DebugRenderer9&);

protected:
  static ShadowFrustum* s_Frustum;
  static ShadowFrustum* s_PreRender;
  static int s_AtlasWidth, s_AtlasHeight, s_TileShadowMapSize;
  static int s_MaxRefinment;
  static int s_LargestTileWorldSize;
  static float s_ShadowDistance;
  static int s_nTilesRendered;

  friend class ShadowFrustum;
  friend class TileCacheEntry;
  friend class DynamicShadowMap;
  friend class HUDRenderer;
  friend class PlainShadowMapTest;

  static Vec4 s_AtlasSize;
  static Mat4x4 s_Bias;
  static bool s_bDrawHUD;
  static Vec4 s_DepthPassParamPS;
  static Vec4 s_DepthPassParamVS;

  static IDirect3DTexture9* s_pAtlasTexture;
  static IDirect3DSurface9* s_pAtlasSurface;
  static IDirect3DTexture9* s_pDEMTexture;
  static IDirect3DSurface9* s_pDEMSurface;
  static IDirect3DTexture9* s_pWorkTexture;
  static IDirect3DSurface9* s_pWorkSurface;
  static IDirect3DTexture9* s_pIndexTexture;
  static IDirect3DTexture9* s_pLayerIndexTexture;

  static ShaderObject9 s_DepthExtentShader;

  static IDirect3DDevice9* s_Device9;
  static IDirect3DVertexBuffer9* s_pQuadVB;
  static IDirect3DVertexDeclaration9* s_QuadDecl;

  ASM();
  ~ASM();
};

#endif //#ifndef __ASM
