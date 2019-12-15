#pragma once

#include <vector>

#include "../../Core/Math/Math.h"

#include "_Shaders/HLSL2C.inc"
#include "../Shaders/ASMShaderData.inc"

class Camera;
class DebugRenderer;
class RenderTarget2D;
class CAABBox;
class DeviceContext11;

class CAdaptiveShadowMap : public MathLibObject
{
public:
    CAdaptiveShadowMap();
    virtual ~CAdaptiveShadowMap();

    bool PrepareRender(
        const Camera& mainViewCamera,
        bool disablePreRender );

    void Render(
        const RenderTarget2D& workBufferDepth,
        const RenderTarget2D& workBufferColor,
        DeviceContext11& dc );

    void Reset();

    void Tick( unsigned int currentTime, unsigned int dt, bool disableWarping, bool forceUpdate, unsigned int updateDeltaTime );

    void DrawDebug( DebugRenderer& debug );

    void GetResolveShaderData( ASM_ResolveShaderData& shaderData );
    void SetResolveTextures( DeviceContext11& dc );

    const Vec3& GetLightDir() const;
    const Vec3& GetPreRenderLightDir() const;

    void Update( const CAABBox& BBoxWS );

    bool NothingToRender() const;
    bool PreRenderAvailable() const;

    virtual void RenderTile(
        const Vec4i& viewport,
        const Camera& renderCamera,
        bool isLayer,
        DeviceContext11& dc ) = 0;

    virtual const Vec3 GetLightDirection(
        unsigned t ) = 0;

protected:
    static const unsigned int MAX_REFINEMENT = gs_ASMMaxRefinement;
    static const unsigned int TILE_BORDER_TEXELS = gs_ASMTileBorderTexels;
    static const unsigned int TILE_SIZE = gs_ASMTileSize;
    static const unsigned int DEM_DOWNSAMPLE_LEVEL = gs_ASMDEMDownsampleLevel;
    static const unsigned int DEM_TILE_SIZE = gs_ASMDEMTileSize;
    static const unsigned int BORDERLESS_TILE_SIZE = gs_ASMBorderlessTileSize;

    struct SShadowMapPrepareRenderContext
    {
        const Vec3* m_worldCenter;
    };

    struct SShadowMapRenderContext
    {
        DeviceContext11* m_dc;
        CAdaptiveShadowMap* m_shadowMapRenderer;
    };

    class CQuadTree;
    class CQuadTreeNode;
    class CTileCache;
    class CTileCacheEntry;
    class CShadowFrustum;

    CTileCache* m_cache;
    CShadowFrustum* m_longRangeShadows;
    CShadowFrustum* m_longRangePreRender;

    bool m_preRenderDone;

    static unsigned int s_frameCounter;
    static float s_tileFarPlane;

    static float GetRefinementDistanceSq( const CAABBox& BBox, const Vec2& refinementPos );

    static finline bool IsTileAcceptableForIndexing( const CTileCacheEntry* pTile );
    static finline bool IsNodeAcceptableForIndexing( const CQuadTreeNode* pNode );

    template< bool ( *isAcceptable )( const CQuadTreeNode* ) >
    static void SortNodes(
        const Vec2& refinementPoint,
        const Vec2& sortRegionMaxSize,
        float tileSize,
        const std::vector<CQuadTreeNode*>& nodes,
        std::vector<CQuadTreeNode*>& sortedNodes,
        CAABBox& sortedBBox );

    template< CTileCacheEntry*& ( CQuadTreeNode::*TileAccessor )() const >
    static bool GetRectangleWithinParent(
        const int NUp,
        CQuadTreeNode* NList[ MAX_REFINEMENT + 1 ],
        Vec4i& parentRect,
        Vec4i& tileRect );
};
