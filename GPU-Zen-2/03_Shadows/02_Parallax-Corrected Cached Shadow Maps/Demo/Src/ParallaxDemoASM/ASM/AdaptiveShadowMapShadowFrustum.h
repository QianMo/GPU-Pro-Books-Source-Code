#pragma once

#include "AdaptiveShadowMap.h"
#include "AdaptiveShadowMapQuadTree.h"

#include "TextureLoader/Texture11.h"

#include "../Misc/AtlasQuads.h"
#include "../Misc/ConvexHull2D.h"

class CAdaptiveShadowMap::CShadowFrustum : public MathLibObject
{
public:
    struct Config
    {
        float m_largestTileWorldSize;
        float m_shadowDistance;
        int m_maxRefinement;
        int m_minRefinementForLayer;
        int m_indexSize;
        bool m_forceImpostors;
        float m_refinementDistanceSq[ MAX_REFINEMENT + 2 ];
        float m_minExtentLS;
    };

    unsigned int m_ID;
    Vec3 m_lightDir;
    Mat4x4 m_lightRotMat;
    Mat4x4 m_invLightRotMat;
    Vec2 m_refinementPoint;

    Vec3 m_receiverWarpVector;
    Vec3 m_blockerSearchVector;
    bool m_disableWarping;

    CConvexHull2D<> m_frustumHull;
    CConvexHull2D<> m_largerHull;
    CConvexHull2D<> m_prevLargerHull;

    CQuadTree m_quadTree;

    Config m_cfg;
    int m_indirectionTextureSize;
    int m_demMinRefinement[2];

    Mat4x4 m_indexTexMat;
    Mat4x4 m_indexViewMat;

    CShadowFrustum( const Config& cfg, bool useMRF );
    ~CShadowFrustum();

    bool IsLightDirDifferent( const Vec3& lightDir ) const;
    void Set( const Vec3& lightDir );
    void Reset();

    void CreateTiles( CTileCache* pCache, const Camera& mainViewCamera );

    void BuildTextures( const SShadowMapRenderContext& context, bool isPreRender );

    void DrawDebug( DebugRenderer& debug, float scale );

    bool IsValid() const { return m_ID != 0; }
    bool UseLayers() const { return m_cfg.m_minRefinementForLayer <= m_cfg.m_maxRefinement; }
    int GetDEMMinRefinement( bool isLayer ) const { return m_demMinRefinement[ isLayer ]; }
    bool IsLightBelowHorizon() const { return false; }//IsValid() && m_lightDir.y < 0; }

    const RenderTarget2D& GetIndirectionTexture() const { return m_indirectionTexture; }
    const RenderTarget2D& GetLODClampTexture() const { return m_lodClampTexture; }
    const RenderTarget2D& GetLayerIndirectionTexture() const { return m_layerIndirectionTexture; }

    const Camera CalcCamera( const Vec3& cameraPos, const CAABBox& BBoxLS, const Vec2& viewportScaleFactor ) const;
    const Camera CalcCamera( const CAABBox& BBoxLS, const Vec3& worldCenter, const Vec2& viewportScaleFactor ) const;

    void UpdateWarpVector( const Vec3& sunDir, bool disableWarping );

    void GetIndirectionTextureData( CTileCacheEntry* pTile, Vec4& packedData, Vec4i& dstCoord );

private:
    RenderTarget2D m_indirectionTexture;
    RenderTarget2D m_lodClampTexture;
    RenderTarget2D m_layerIndirectionTexture;

    std::vector< AtlasQuads::SFillQuad > m_quads;
    unsigned int m_quadsCnt[ MAX_REFINEMENT + 1 ];

    std::vector< AtlasQuads::SFillQuad > m_lodClampQuads;

    std::vector< CQuadTreeNode* > m_indexedNodes;
    CAABBox m_indexBBox;
    Vec3 m_indexCameraPos;

    static finline bool RefineAgainstFrustum(
        const CAABBox& childBBox,
        const CQuadTreeNode* pParent,
        const CShadowFrustum& frustum );

    template< class T, bool ( *isRefinable )( const CAABBox&, const CQuadTreeNode*, const T& ) >
    static void RefineNode( CQuadTreeNode* pParent, int maxRefinement, const T& userData );

    const Vec3 GetDebugVertex( const Vec2& point, float scale );
    void DrawDebugBBox( const CAABBox& BBox, DebugRenderer& debug, float scale, const Vec4& color );
    void DrawDebugQuadTreeSubTree( CQuadTreeNode* pNode, DebugRenderer& debug, float scale );

    template< unsigned int maxVertices >
    void DrawDebugConvexHull( const CConvexHull2D< maxVertices >& convexHull, DebugRenderer& debug, float scale, const Vec4& color );

    void AllocateTiles( CTileCache* pCache, CQuadTreeNode* pNode );
    void RemoveNonIntersectingNodes( CQuadTreeNode* pNode );

    void FindIndexedNodes();
    void FillIndirectionTextureData( bool processLayers );
    void ResetIndirectionTextureData();

    const Vec3 ProjectToTS( const Vec3& pointLS, const CAABBox& BBoxLS, const Vec3& cameraOffset );

    bool ShouldNodeExist( const CAABBox& BBox, unsigned char refinement ) const;

    void FillLODClampTextureData();
    void UpdateIndirectionTexture( RenderTarget2D& indirectionTexture, const SShadowMapRenderContext& context, bool disableHierarchy );
    void UpdateLODClampTexture( RenderTarget2D& lodClampTexture, const SShadowMapRenderContext& context );
};
