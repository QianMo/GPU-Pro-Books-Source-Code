#pragma once

#include "AdaptiveShadowMap.h"

#include "TextureLoader/Texture11.h"
#include "Scene/Camera.h"

#include "../Misc/AtlasQuads.h"
#include "../Misc/IntrusiveUnorderedPtrSet.h"
#include "../Misc/AABBox.h"

class CAdaptiveShadowMap::CTileCacheEntry : public MathLibObject
{
public:
    struct SViewport { int x, y, w, h; } m_viewport;
    unsigned char m_refinement;

    CAABBox m_BBox;
    CQuadTreeNode* m_pOwner;
    CShadowFrustum* m_pFrustum;
    unsigned int m_lastFrameUsed;
    unsigned int m_frustumID;
    bool m_isLayer;
    float m_fadeInFactor;

    Camera m_renderCamera;

    CTileCacheEntry( CTileCache* pCache, int x, int y );
    ~CTileCacheEntry();

    template< CTileCacheEntry*& (CQuadTreeNode::*TileAccessor)(), bool isLayer >
    void Allocate( CQuadTreeNode* pOwner, CShadowFrustum* pFrustum );

    void Free();
    void Invalidate();
    void MarkReady();
    void MarkNotReady();

    bool IsReady() const { return m_readyTilesPos.IsInserted(); }
    CTileCache* GetCache() const { return m_pCache; }
    bool IsAllocated() const { return m_pOwner != nullptr; }
    bool IsBeingUpdated() const { return m_updateQueuePos.IsInserted(); }
    bool IsQueuedForRendering() const { return m_renderQueuePos.IsInserted(); }

    static const Vec4i GetRect( const SViewport& vp, int border ) { return Vec4i( vp.x - border, vp.y - border, vp.x + vp.w + border, vp.y + vp.h + border ); }

protected:
    CTileCache* m_pCache;

    CIntrusiveUnorderedSetItemHandle m_tilesPos;
    CIntrusiveUnorderedSetItemHandle m_freeTilesPos;
    CIntrusiveUnorderedSetItemHandle m_renderQueuePos;
    CIntrusiveUnorderedSetItemHandle m_readyTilesPos;
    CIntrusiveUnorderedSetItemHandle m_demQueuePos;
    CIntrusiveUnorderedSetItemHandle m_renderBatchPos;
    CIntrusiveUnorderedSetItemHandle m_updateQueuePos;

    CIntrusiveUnorderedSetItemHandle& GetTilesPos() { return m_tilesPos; }
    CIntrusiveUnorderedSetItemHandle& GetFreeTilesPos() { return m_freeTilesPos; }
    CIntrusiveUnorderedSetItemHandle& GetRenderQueuePos() { return m_renderQueuePos; }
    CIntrusiveUnorderedSetItemHandle& GetReadyTilesPos() { return m_readyTilesPos; }
    CIntrusiveUnorderedSetItemHandle& GetDemQueuePos() { return m_demQueuePos; }
    CIntrusiveUnorderedSetItemHandle& GetRenderBatchPos() { return m_renderBatchPos; }
    CIntrusiveUnorderedSetItemHandle& GetUpdateQueuePos() { return m_updateQueuePos; }

    void PrepareRender( const SShadowMapPrepareRenderContext& context );

    friend class CTileCache;
};

class CAdaptiveShadowMap::CTileCache
{
public:
    CTileCache();
    ~CTileCache();

    template< CTileCacheEntry*& (CQuadTreeNode::*TileAccessor)(), bool isLayer >
    CTileCacheEntry* Allocate( CQuadTreeNode* pNode, CShadowFrustum* pFrustum );

    int AddTileFromRenderQueueToRenderBatch(
        CShadowFrustum* pFrustum,
        int maxRefinement,
        bool isLayer );

    int AddTileFromUpdateQueueToRenderBatch(
        CShadowFrustum* pFrustum,
        int maxRefinement,
        bool isLayer );

    bool PrepareRenderTilesBatch( const SShadowMapPrepareRenderContext& context );

    void RenderTilesBatch(
        const RenderTarget2D& workBufferDepth,
        const RenderTarget2D& workBufferColor,
        const SShadowMapRenderContext& context );

    void CreateDEM(
        const RenderTarget2D& workBufferColor,
        const SShadowMapRenderContext& context,
        bool createDemForLayerRendering );

    void UpdateTiles(
        CShadowFrustum* pFrustum,
        const CAABBox& BBoxWS );

    const RenderTarget2D& GetDepthAtlas() const { return m_depthAtlas; }
    const RenderTarget2D& GetDEMAtlas() const { return m_demAtlas; }
    const unsigned int GetDepthAtlasWidth() const { return m_depthAtlasWidth; }
    const unsigned int GetDepthAtlasHeight() const { return m_depthAtlasHeight; }
    const unsigned int GetDEMAtlasWidth() const { return m_demAtlasWidth; }
    const unsigned int GetDEMAtlasHeight() const { return m_demAtlasHeight; }

    void Tick( float deltaTime );
    void DrawDebug( DebugRenderer& debug );

    bool NothingToRender() const { return m_renderBatch.empty(); }
    bool IsFadeInFinished( const CShadowFrustum* pFrustum ) const;

protected:
    CVectorBasedIntrusiveUnorderedPtrSet< CTileCacheEntry, &CTileCacheEntry::GetTilesPos > m_tiles;
    CVectorBasedIntrusiveUnorderedPtrSet< CTileCacheEntry, &CTileCacheEntry::GetFreeTilesPos > m_freeTiles;
    CVectorBasedIntrusiveUnorderedPtrSet< CTileCacheEntry, &CTileCacheEntry::GetRenderQueuePos > m_renderQueue;
    CVectorBasedIntrusiveUnorderedPtrSet< CTileCacheEntry, &CTileCacheEntry::GetReadyTilesPos > m_readyTiles;
    CVectorBasedIntrusiveUnorderedPtrSet< CTileCacheEntry, &CTileCacheEntry::GetDemQueuePos > m_demQueue;
    CVectorBasedIntrusiveUnorderedPtrSet< CTileCacheEntry, &CTileCacheEntry::GetRenderBatchPos > m_renderBatch;
    CVectorBasedIntrusiveUnorderedPtrSet< CTileCacheEntry, &CTileCacheEntry::GetUpdateQueuePos > m_updateQueue;

    unsigned int m_cacheHits;
    unsigned int m_tileAllocs;
    unsigned int m_numTilesRendered;
    unsigned int m_numTilesUpdated;
    unsigned int m_depthAtlasWidth;
    unsigned int m_depthAtlasHeight;
    unsigned int m_demAtlasWidth;
    unsigned int m_demAtlasHeight;

    RenderTarget2D m_depthAtlas;
    RenderTarget2D m_demAtlas;

    template< class T >
    int AddTileToRenderBatch(
        T& tilesQueue,
        CShadowFrustum* pFrustum,
        int maxRefinement,
        bool isLayer );

    void RenderTiles(
        unsigned int numTiles,
        CTileCacheEntry** tiles,
        const RenderTarget2D& workBufferDepth,
        const RenderTarget2D& workBufferColor,
        const SShadowMapRenderContext& context,
        bool allowDEM );

    void StartDEM( CTileCacheEntry* pTile, AtlasQuads::SCopyQuad& copyDEMQuad );

    friend class CTileCacheEntry;
};
