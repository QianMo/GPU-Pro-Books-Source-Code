#define SAFE_RELEASE(x) if(x) { (x)->Release(); (x) = NULL; }
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX

#include "AdaptiveShadowMapShadowFrustum.h"
#include "AdaptiveShadowMapTileCache.h"
#include "ShaderCache/SimpleShader.h"
#include "../Core/Util/DebugRenderer.h"
#include "../Misc/Utils.h"

//float gfx_asm_lightDirUpdateThreshold = 0.997f;
float gfx_asm_lightDirUpdateThreshold = 0.999f;
float gfx_asm_maxWarpAngleCos = 0.994f;

bool CAdaptiveShadowMap::CShadowFrustum::IsLightDirDifferent( const Vec3& lightDir ) const
{
    return Vec3::Dot( m_lightDir, lightDir ) < gfx_asm_lightDirUpdateThreshold;
}

CAdaptiveShadowMap::CShadowFrustum::CShadowFrustum( const Config& cfg, bool useMRF ) :
    m_cfg( cfg )
{
    /* Some hardcode here: we're using DEM created for the tiles with 0 refinement, but if layering 
       is enabled then we're using 1-refinmement DEM to hold a version of 0-refinement DEM but with wider-area
       min filter kernel. This is reflected in demForLayerRenderingMip constant in shader. */
    m_demMinRefinement[0] = useMRF ? ( UseLayers() ? 1 : 0 ) : -1;
    m_demMinRefinement[1] = m_cfg.m_minRefinementForLayer;

    m_indirectionTextureSize = ( 1 << m_cfg.m_maxRefinement ) * m_cfg.m_indexSize;

    Reset();

    m_indirectionTexture.Init( m_indirectionTextureSize, m_indirectionTextureSize, DXGI_FORMAT_R32G32B32A32_FLOAT, m_cfg.m_maxRefinement + 1 );
    m_layerIndirectionTexture.Init( m_indirectionTextureSize, m_indirectionTextureSize, DXGI_FORMAT_R32G32B32A32_FLOAT, m_cfg.m_maxRefinement + 1 );

    for( int i = 0; i <= m_cfg.m_maxRefinement; ++i )
    {
        m_indirectionTexture.AddRenderTargetView( DXGI_FORMAT_R32G32B32A32_FLOAT, i );
        m_layerIndirectionTexture.AddRenderTargetView( DXGI_FORMAT_R32G32B32A32_FLOAT, i );
    }

    m_lodClampTexture.Init( m_indirectionTextureSize, m_indirectionTextureSize, DXGI_FORMAT_R8_UNORM );
}

CAdaptiveShadowMap::CShadowFrustum::~CShadowFrustum()
{
    m_indirectionTexture.Clear();
    m_lodClampTexture.Clear();
    m_layerIndirectionTexture.Clear();
}

void CAdaptiveShadowMap::CShadowFrustum::Reset()
{
    m_quadTree.Reset();

    m_ID = 0;
    m_lightDir = Vec3::Zero();
    m_lightRotMat = Mat4x4::Identity();
    m_invLightRotMat = Mat4x4::Identity();
    m_refinementPoint = Vec2::Zero();

    m_frustumHull.Reset();
    m_largerHull.Reset();
    m_prevLargerHull.Reset();

    m_receiverWarpVector = Vec3::Zero();
    m_blockerSearchVector = Vec3::Zero();
    m_disableWarping = false;

    ResetIndirectionTextureData();
}

void CAdaptiveShadowMap::CShadowFrustum::Set( const Vec3& lightDir )
{
    Reset();

    m_lightDir = lightDir;

    Camera camera;
    Utils::LookAt( Vec3::Zero(), -m_lightDir, camera );

    m_lightRotMat = camera.GetViewMatrix();
    m_invLightRotMat = camera.GetViewMatrixInverse();

    static unsigned int s_IDGen = 1;
    m_ID = s_IDGen; s_IDGen += 2;
}

const Camera CAdaptiveShadowMap::CShadowFrustum::CalcCamera( const Vec3& cameraPos, const CAABBox& BBoxLS, const Vec2& viewportScaleFactor ) const
{
    Camera camera;
    Utils::LookAt( cameraPos, -m_lightDir, camera );

    float hw = 0.5f * BBoxLS.GetSizeX() * viewportScaleFactor.x;
    float hh = 0.5f * BBoxLS.GetSizeY() * viewportScaleFactor.y;
    camera.SetProjection( Mat4x4::OrthoD3D(-hw, hw,-hh, hh, 0, s_tileFarPlane ) );

    return camera;
}

const Camera CAdaptiveShadowMap::CShadowFrustum::CalcCamera( const CAABBox& BBoxLS, const Vec3& worldCenter, const Vec2& viewportScaleFactor ) const
{
    Vec3 aabbMin = worldCenter + Vec3(-800.0f,-200.0f,-800.0f );
    Vec3 aabbMax = worldCenter + Vec3( 800.0f, 500.0f, 800.0f );

    float minZ = FLT_MAX;
    for( int i = 0; i < 8; ++i )
    {
        Vec3 aabbCorner(
             i & 1 ? aabbMin.x : aabbMax.x,
             i & 2 ? aabbMin.y : aabbMax.y,
             i & 4 ? aabbMin.z : aabbMax.z );
        minZ = std::min( minZ, -Vec3::Dot( aabbCorner, m_lightDir ) );
    }
    Vec3 cameraPos = BBoxLS.GetCenter() * m_invLightRotMat - minZ * m_lightDir;

    static const Vec3 boundsN[] =
    {
         Vec3(-1.0f, 0.0f, 0.0f ),
         Vec3( 0.0f,-1.0f, 0.0f ),
         Vec3( 0.0f, 0.0f,-1.0f ),
         Vec3( 1.0f, 0.0f, 0.0f ),
         Vec3( 0.0f, 1.0f, 0.0f ),
         Vec3( 0.0f, 0.0f, 1.0f ),
    };

    float boundsD[] =
    {
         aabbMax.x,  aabbMax.y,  aabbMax.z,
        -aabbMin.x, -aabbMin.y, -aabbMin.z,
    };

    float minF = 0;
    for( unsigned int i = 0; i < 6; ++i )
    {
        float f1 = Vec3::Dot( boundsN[i], cameraPos ) + boundsD[i];
        float f2 = Vec3::Dot( boundsN[i], m_lightDir );
        if( f1 <= 0 && f2 < 0 )
        {
            minF = std::max( minF, f1 / f2 );
        }
    }

    return CalcCamera( cameraPos - minF * m_lightDir, BBoxLS, viewportScaleFactor );
}

void CAdaptiveShadowMap::CShadowFrustum::UpdateWarpVector( const Vec3& lightDir, bool disableWarping )
{
    if( !IsValid() ) return;

    m_disableWarping |= disableWarping;
    if( m_disableWarping ) return;
    
    if( Vec3::Dot( m_lightDir, lightDir ) < gfx_asm_maxWarpAngleCos ) return;

#if 0
    Vec3 shadowDir = m_lightDir;
    Vec3 dir = lightDir;

    float warpBias = 1.0f + Vec3::Length( dir - shadowDir );
    m_receiverWarpVector = warpBias * dir - shadowDir;
#else
    Vec3 shadowDir = -m_lightDir;
    Vec3 dir = lightDir - 2.0f * Vec3::Dot( shadowDir, lightDir ) * shadowDir;

    float warpBias = 1.0f - 0.9f * Vec3::Length( dir - shadowDir );
    m_receiverWarpVector = warpBias * dir - shadowDir;
#endif

    Vec3 warpDirVS = Vec3::Vector( m_receiverWarpVector ) * m_indexViewMat;

    float stepDistance = 55.0f;
    float stepBias = 10.0f;
    m_blockerSearchVector = Vec3(
        stepDistance * warpDirVS.x / gs_ASMDEMAtlasTextureWidth,
        stepDistance * warpDirVS.y / gs_ASMDEMAtlasTextureHeight,
        -stepBias / gs_ASMTileFarPlane);
}

void CAdaptiveShadowMap::CShadowFrustum::BuildTextures( const SShadowMapRenderContext& context, bool isPreRender )
{
    FindIndexedNodes();

    ResetIndirectionTextureData();
    FillIndirectionTextureData( false );
    UpdateIndirectionTexture( m_indirectionTexture, context, isPreRender );

    if( isPreRender )
    {
        FillLODClampTextureData();
        UpdateLODClampTexture( m_lodClampTexture, context );
    }

    if( UseLayers() )
    {
        ResetIndirectionTextureData();
        FillIndirectionTextureData( true );
        UpdateIndirectionTexture( m_layerIndirectionTexture, context, isPreRender );
    }
}

//-----------------------------------------------------------------------------

void CAdaptiveShadowMap::CShadowFrustum::CreateTiles( CTileCache* pCache, const Camera& mainViewCamera )
{
    if( !IsValid() || IsLightBelowHorizon() ) return;

    m_refinementPoint = m_frustumHull.FindFrustumConvexHull( mainViewCamera, m_cfg.m_shadowDistance, m_lightRotMat );

    m_prevLargerHull = m_largerHull;
    m_largerHull.FindFrustumConvexHull( mainViewCamera, 1.01f * m_cfg.m_shadowDistance, m_lightRotMat );

    for( size_t i = m_quadTree.GetRoots().size(); i > 0; --i )
        RemoveNonIntersectingNodes( m_quadTree.GetRoots().at( i -1 ) );

    CAABBox hullBBox;
    hullBBox.Reset();
    for( int i=0; i<m_frustumHull.m_size; ++i )
        hullBBox.Add( Vec3( m_frustumHull.m_vertices[i] ) );

    hullBBox.Add( m_refinementPoint + Vec3( m_cfg.m_minExtentLS, m_cfg.m_minExtentLS, 0.0f ) );
    hullBBox.Add( m_refinementPoint - Vec3( m_cfg.m_minExtentLS, m_cfg.m_minExtentLS, 0.0f ) );

    Utils::AlignBBox( hullBBox, m_cfg.m_largestTileWorldSize );

    CAABBox nodeBBox( Vec3::Zero(), Vec3::Zero() );
    for( nodeBBox.m_min.y = hullBBox.m_min.y; nodeBBox.m_min.y < hullBBox.m_max.y; nodeBBox.m_min.y += m_cfg.m_largestTileWorldSize )
    {
        for( nodeBBox.m_min.x = hullBBox.m_min.x; nodeBBox.m_min.x < hullBBox.m_max.x; nodeBBox.m_min.x += m_cfg.m_largestTileWorldSize )
        {
            nodeBBox.m_max = nodeBBox.m_min + Vec3( m_cfg.m_largestTileWorldSize, m_cfg.m_largestTileWorldSize, 0.0f );
            if( ShouldNodeExist( nodeBBox, 0 ) )
            {
                CQuadTreeNode* pNode = m_quadTree.FindRoot( nodeBBox );
                if( pNode == nullptr )
                {
                    pNode = new CQuadTreeNode( &m_quadTree, 0 );
                    pNode->m_BBox = nodeBBox;
                }
                RefineNode< CShadowFrustum, &RefineAgainstFrustum >( pNode, m_cfg.m_maxRefinement, *this );
            }
        }
    }

   for( auto it = m_quadTree.GetRoots().begin(); it != m_quadTree.GetRoots().end(); ++it )
        AllocateTiles( pCache, *it );
}

bool CAdaptiveShadowMap::CShadowFrustum::ShouldNodeExist( const CAABBox& BBox, unsigned char refinement ) const
{
    return GetRefinementDistanceSq( BBox, m_refinementPoint ) < fabsf( m_cfg.m_refinementDistanceSq[ refinement ] ) ?
        ( m_cfg.m_refinementDistanceSq[ refinement ] < 0 || m_frustumHull.Intersects( BBox ) ) : false;
}

void CAdaptiveShadowMap::CShadowFrustum::RemoveNonIntersectingNodes( CQuadTreeNode* pNode )
{
    for( int i = 0; i < 4; ++i )
        if( pNode->m_children[i] != nullptr )
            RemoveNonIntersectingNodes( pNode->m_children[i] );

    if( pNode->m_lastFrameVerified != s_frameCounter )
    {
        pNode->m_lastFrameVerified = s_frameCounter;
        if( ShouldNodeExist( pNode->m_BBox, pNode->m_refinement ) )
        {
            if( pNode->m_pParent != nullptr )
                pNode->m_pParent->m_lastFrameVerified = s_frameCounter;
            return;
        }
        delete pNode;
    }
}

bool CAdaptiveShadowMap::CShadowFrustum::RefineAgainstFrustum(
    const CAABBox& childBBox,
    const CQuadTreeNode* pParent,
    const CShadowFrustum& frustum )
{
    return frustum.ShouldNodeExist( childBBox, pParent->m_refinement + 1 );
}

template< class T, bool ( *isRefinable )( const CAABBox&, const CAdaptiveShadowMap::CQuadTreeNode*, const T& ) >
void CAdaptiveShadowMap::CShadowFrustum::RefineNode( CQuadTreeNode* pParent, int maxRefinement, const T& userData )
{
    if( pParent->m_refinement < maxRefinement )
    {
        for( int i=0; i<4; ++i )
        {
            if( pParent->m_children[i] != nullptr )
            {
                RefineNode< T, isRefinable >( pParent->m_children[i], maxRefinement, userData );
            }
            else
            {
                CAABBox childBBox = pParent->GetChildBBox(i);
                if( isRefinable( childBBox, pParent, userData ) )
                {
                    CQuadTreeNode* pNode = pParent->AddChild(i);
                    pNode->m_BBox = childBBox;
                    RefineNode< T, isRefinable >( pNode, maxRefinement, userData );
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------

const Vec3 CAdaptiveShadowMap::CShadowFrustum::GetDebugVertex( const Vec2& point, float scale )
{
    return scale * ( point - m_refinementPoint ) + Vec2( 400.0f, 400.0f );
}

template< unsigned int maxVertices >
void CAdaptiveShadowMap::CShadowFrustum::DrawDebugConvexHull( const CConvexHull2D< maxVertices >& convexHull, DebugRenderer& debug, float scale, const Vec4& color )
{
    Vec2 vertices[ maxVertices ];
    for( int i = 0; i < convexHull.m_size; ++i )
        vertices[i] = GetDebugVertex( convexHull.m_vertices[i], scale );

    debug.SetContourColor( color );
    for( int i = 0; i < convexHull.m_size; ++i )
        debug.PushLine( vertices[ i ], vertices[ ( i + 1 ) % convexHull.m_size ] );
}

void CAdaptiveShadowMap::CShadowFrustum::DrawDebugBBox( const CAABBox& BBox, DebugRenderer& debug, float scale, const Vec4& color )
{
    if( !BBox.IsInvalid() )
    {
        Vec2 quad[] =
        {
            GetDebugVertex( Vec2( BBox.m_min.x, BBox.m_min.y ), scale ),
            GetDebugVertex( Vec2( BBox.m_max.x, BBox.m_min.y ), scale ),
            GetDebugVertex( Vec2( BBox.m_max.x, BBox.m_max.y ), scale ),
            GetDebugVertex( Vec2( BBox.m_min.x, BBox.m_max.y ), scale ),
        };

        debug.SetContourColor( color );
        for( int i = 0; i < 4; ++i )
            debug.PushLine( quad[ i ], quad[ ( i + 1 ) % 4 ] );
    }
}

void CAdaptiveShadowMap::CShadowFrustum::DrawDebugQuadTreeSubTree( CQuadTreeNode* pNode, DebugRenderer& debug, float scale )
{
    static const Vec4 colors[] =
    {
        Vec4( 1.0f, 1.0f, 0.0f, 1.0f ),
        Vec4( 0.0f, 0.0f, 1.0f, 1.0f ),
        Vec4( 1.0f, 0.0f, 0.0f, 1.0f ),
        Vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
    };

    for( int i = 0; i < 4; ++i )
        if( pNode->m_children[i] != nullptr )
            DrawDebugQuadTreeSubTree( pNode->m_children[i], debug, scale );

    DrawDebugBBox( pNode->m_BBox, debug, scale,
        pNode->m_pTile!=nullptr ? ( pNode->m_pTile->IsReady() ? ( pNode->m_pTile->IsBeingUpdated() ? colors[3] : colors[0] ) : colors[1] ) : colors[2] );
}

void CAdaptiveShadowMap::CShadowFrustum::DrawDebug( DebugRenderer& debug, float scale )
{
    if( !IsValid() ) return;

    for( auto it = m_quadTree.GetRoots().begin(); it != m_quadTree.GetRoots().end(); ++it )
        DrawDebugQuadTreeSubTree( *it, debug, scale );

    DrawDebugConvexHull( m_frustumHull, debug, scale, Vec4( 0.0f, 1.0f, 0.0f, 1.0f ) );

    DrawDebugBBox( m_indexBBox, debug, scale, Vec4( 0.5f, 0.5f, 0.5f, 1.0f ) );
}

//-----------------------------------------------------------------------------

template< bool ( *isAcceptable )( const CAdaptiveShadowMap::CQuadTreeNode* ) >
void CAdaptiveShadowMap::SortNodes(
    const Vec2& refinementPoint,
    const Vec2& sortRegionMaxSize,
    float tileSize,
    const std::vector<CQuadTreeNode*>& nodes,
    std::vector<CQuadTreeNode*>& sortedNodes,
    CAABBox& sortedBBox )
{
    struct SSortStruct
    {
        CQuadTreeNode* m_pNode;
        float m_key;

        bool operator < ( const SSortStruct& a )
        {
            return m_key < a.m_key;
        }
    };

    SSortStruct* nodesToSort = static_cast< SSortStruct* >( alloca( sizeof( SSortStruct ) * nodes.size() ) );
    Vec2 distMax = sortRegionMaxSize + Vec2( tileSize, tileSize );
    float distMaxSq = Vec2::Dot( distMax, distMax );
    unsigned int numNodesToSort = 0;
    for( auto it = nodes.begin(); it != nodes.end(); ++it )
    {
        CQuadTreeNode* pNode = *it;
        if( (*isAcceptable)( pNode ) )
        {
            const CAABBox& BBox = pNode->m_BBox;
            Vec3 bboxCenter = BBox.GetCenter();
            float dx = std::max( fabsf( refinementPoint.x - bboxCenter.x ) - BBox.GetSizeX() * 0.5f, 0.0f );
            float dy = std::max( fabsf( refinementPoint.y - bboxCenter.y ) - BBox.GetSizeY() * 0.5f, 0.0f );
            float distSq = dx*dx + dy*dy;
            if( distSq < distMaxSq )
            {
                SSortStruct& ss = nodesToSort[ numNodesToSort++ ];
                ss.m_key = fabsf( BBox.m_min.x - refinementPoint.x );
                ss.m_key = std::max( fabsf( BBox.m_min.y - refinementPoint.y ), ss.m_key );
                ss.m_key = std::max( fabsf( BBox.m_max.x - refinementPoint.x ), ss.m_key );
                ss.m_key = std::max( fabsf( BBox.m_max.y - refinementPoint.y ), ss.m_key );
                ss.m_pNode = pNode;
            }
        }
    }

    std::sort( nodesToSort, nodesToSort + numNodesToSort );

    sortedBBox = CAABBox( refinementPoint, refinementPoint );
    Utils::AlignBBox( sortedBBox, tileSize );

    sortedNodes.resize( 0 );
    for( unsigned int i = 0; i < numNodesToSort; ++i )
    {
        SSortStruct& ss = nodesToSort[i];
        const CAABBox& nodeBBox = ss.m_pNode->m_BBox;
        Vec3 testMin( std::min( sortedBBox.m_min.x, nodeBBox.m_min.x ), std::min( sortedBBox.m_min.y, nodeBBox.m_min.y ), 0.0f );
        Vec3 testMax( std::max( sortedBBox.m_max.x, nodeBBox.m_max.x ), std::max( sortedBBox.m_max.y, nodeBBox.m_max.y ), 0.0f );
        if( ( testMax.x - testMin.x ) > sortRegionMaxSize.x || ( testMax.y - testMin.y ) > sortRegionMaxSize.y )
        {
            if( ss.m_key > distMax.x )
                break;
        }
        else
        {
            sortedBBox = CAABBox( testMin, testMax );
            sortedNodes.push_back( ss.m_pNode );
        }
    }
}

bool CAdaptiveShadowMap::IsTileAcceptableForIndexing( const CTileCacheEntry* pTile )
{
    return pTile != nullptr && pTile->IsReady();
}

bool CAdaptiveShadowMap::IsNodeAcceptableForIndexing( const CQuadTreeNode* pNode )
{
    return IsTileAcceptableForIndexing( pNode->m_pTile );
}

void CAdaptiveShadowMap::CShadowFrustum::FindIndexedNodes()
{
    if( !IsValid() ) return;

    float sortRegionSizeMax = float( m_cfg.m_indexSize ) * m_cfg.m_largestTileWorldSize;

    SortNodes< &IsNodeAcceptableForIndexing >(
        m_refinementPoint,
        Vec2( sortRegionSizeMax ),
        m_cfg.m_largestTileWorldSize,
        m_quadTree.GetRoots(),
        m_indexedNodes,
        m_indexBBox );

    m_indexBBox = CAABBox( m_indexBBox.m_min, m_indexBBox.m_min + Vec3( sortRegionSizeMax, sortRegionSizeMax, 0.0f ) );

    if( !m_indexedNodes.empty() )
    {
        float offset = -FLT_MAX;
        for( auto it = m_indexedNodes.begin(); it != m_indexedNodes.end(); ++it )
        {
            offset = std::max( offset, Vec3::Dot( m_lightDir, (*it)->m_pTile->m_renderCamera.GetPosition() ) );
        }
        m_indexCameraPos = m_indexBBox.GetCenter() * m_invLightRotMat + offset * m_lightDir;

        Camera camera = CalcCamera( m_indexCameraPos, m_indexBBox, 1.0f );

        m_indexViewMat = camera.GetViewMatrix();

        static const Mat4x4 screenToTexCoordMatrix = Mat4x4::ScalingTranslationD3D( Vec3( 0.5f,-0.5f, 1.0f ), Vec3( 0.5f, 0.5f, 0.0f ) );
        m_indexTexMat = camera.GetViewProjection() * screenToTexCoordMatrix;
    }
}

//-----------------------------------------------------------------------------

const Vec3 CAdaptiveShadowMap::CShadowFrustum::ProjectToTS( const Vec3& pointLS, const CAABBox& BBoxLS, const Vec3& cameraOffset )
{
    _ASSERT( BBoxLS.ContainsPoint( pointLS ) );

    return Vec3(
        ( pointLS.x - BBoxLS.m_min.x ) / BBoxLS.GetSizeX(),
        1.0f - ( pointLS.y - BBoxLS.m_min.y ) / BBoxLS.GetSizeY(),
        -Vec3::Dot( m_lightDir, pointLS * m_invLightRotMat + cameraOffset ) / s_tileFarPlane );
}

void CAdaptiveShadowMap::CShadowFrustum::ResetIndirectionTextureData()
{
    memset( m_quadsCnt, 0, sizeof( m_quadsCnt ) );
    m_quads.resize(0);

    m_lodClampQuads.resize(1);
    m_lodClampQuads[0] = AtlasQuads::SFillQuad::Get( 1.0f, m_indirectionTextureSize, m_indirectionTextureSize, 0, 0, m_indirectionTextureSize, m_indirectionTextureSize );
}

void CAdaptiveShadowMap::CShadowFrustum::GetIndirectionTextureData( CTileCacheEntry* pTile, Vec4& packedData, Vec4i& dstCoord )
{
    float invAtlasWidth = 1.0f / float( pTile->GetCache()->GetDepthAtlasWidth() );
    float invAtlasHeight = 1.0f / float( pTile->GetCache()->GetDepthAtlasHeight() );

    // Compute corners of tile's view frustum in tile and index texture spaces.
    Vec3 tileMin( 0.0f, 1.0f, 0.0f );// ProjectToTS( pTile->m_BBox.m_min, pTile->m_BBox, Vec3::Zero() );
    Vec3 tileMax( 1.0f, 0.0f, 0.0f );// ProjectToTS( pTile->m_BBox.m_max, pTile->m_BBox, Vec3::Zero() );
    Vec3 indexMin = ProjectToTS( pTile->m_BBox.m_min, m_indexBBox, pTile->m_renderCamera.GetPosition() - m_indexCameraPos );
    Vec3 indexMax = ProjectToTS( pTile->m_BBox.m_max, m_indexBBox, pTile->m_renderCamera.GetPosition() - m_indexCameraPos );

    // Integer coordinates of the corners in index texture.
    int x0 = int( indexMin.x * float( m_indirectionTextureSize ) + 0.25f );
    int y0 = int( indexMax.y * float( m_indirectionTextureSize ) + 0.25f );
    int x1 = int( indexMax.x * float( m_indirectionTextureSize ) - 0.25f );
    int y1 = int( indexMin.y * float( m_indirectionTextureSize ) - 0.25f );

    _ASSERT( x0 <= x1 && "index texture is broken (possibly FP precision issues)" );
    _ASSERT( y0 <= y1 && "index texture is broken (possibly FP precision issues)" );
    _ASSERT( ( x1 - x0 ) == ( y1 - y0 ) && "index texture is broken (possibly FP precision issues)" );
    _ASSERT( x0 >= 0 && x0 < m_indirectionTextureSize && "index texture is broken (possibly FP precision issues)" );
    _ASSERT( x1 >= 0 && x1 < m_indirectionTextureSize && "index texture is broken (possibly FP precision issues)" );
    _ASSERT( y0 >= 0 && y0 < m_indirectionTextureSize && "index texture is broken (possibly FP precision issues)" );
    _ASSERT( y1 >= 0 && y1 < m_indirectionTextureSize && "index texture is broken (possibly FP precision issues)" );

    const int mipMask = ( 1 << ( m_cfg.m_maxRefinement - pTile->m_refinement ) ) - 1;
    _ASSERT( !( x0 & mipMask ) && "index texture is broken (possibly FP precision issues)" );
    _ASSERT( !( y0 & mipMask ) && "index texture is broken (possibly FP precision issues)" );
    _ASSERT( !( ( x1 + 1 ) & mipMask ) && "index texture is broken (possibly FP precision issues)" );
    _ASSERT( !( ( y1 + 1 ) & mipMask ) && "index texture is broken (possibly FP precision issues)" );

    // Compute affine transform (scale and offset) from index normalized cube to tile normalized cube.
    Vec3 scale1(
        ( tileMax.x - tileMin.x ) / ( indexMax.x - indexMin.x ),
        ( tileMax.y - tileMin.y ) / ( indexMax.y - indexMin.y ),
        1.0f );
    Vec3 offset1 = tileMin - indexMin * scale1;

    // Compute affine transform (scale and offset) from tile normalized cube to shadowmap atlas.
    Vec3 scale2(
        float( pTile->m_viewport.w ) * invAtlasWidth,
        float( pTile->m_viewport.h ) * invAtlasHeight,
        1.0f );
    Vec3 offset2(
        ( float( pTile->m_viewport.x ) + 0.5f ) * invAtlasWidth,
        ( float( pTile->m_viewport.y ) + 0.5f ) * invAtlasHeight,
        0.0f );

    // Compute combined affine transform from index normalized cube to shadowmap atlas.
    Vec3 scale = scale1 * scale2;
    Vec3 offset = offset1 * scale2 + offset2;

    // Assemble data for indirection texture:
    //   packedData.xyz contains transform from view frustum of index texture to view frustum of individual tile
    //   packedData.w contains packed data: integer part is refinement-dependent factor for texcoords computation,
    //      fractional part is bias for smooth tile transition unpacked via getFadeInConstant() in shader,
    //      sign indicates if the tile is a layer tile or just a regular tile.
    packedData = offset;
    packedData.w = float( ( 1 << pTile->m_refinement ) * BORDERLESS_TILE_SIZE * m_cfg.m_indexSize );

    dstCoord = Vec4i( x0, y0, x1, y1 );
}

void CAdaptiveShadowMap::CShadowFrustum::FillIndirectionTextureData( bool processLayers )
{
    if( !IsValid() ) return;

    if( m_indexedNodes.empty() ) return;

    unsigned int numIndexedNodes = m_indexedNodes.size();
    unsigned int i = 0;
    for( int z = m_cfg.m_maxRefinement; z >= 0; --z )
    {
        unsigned int numNodes = m_indexedNodes.size();
        for( ; i < numNodes; ++i )
        {
            CQuadTreeNode* pNode = m_indexedNodes[i];

            CTileCacheEntry* pTile = pNode->m_pTile;
            bool useRegularShadowMapAsLayer = false;
            if( processLayers )
            {
                if( !IsTileAcceptableForIndexing( pNode->m_pLayerTile ) )
                {
                    if( pNode->m_pParent != nullptr && IsTileAcceptableForIndexing( pNode->m_pParent->m_pLayerTile ) )
                        continue;

                    useRegularShadowMapAsLayer = true;
                }
                else
                {
                    _ASSERT( ( pNode->m_pParent == nullptr || pNode->m_pParent->m_pLayerTile == nullptr || pNode->m_pParent->m_pLayerTile->IsReady() ) && 
                                "layer is ready before parent's layer (render queue is bugged)" );
                    pTile = pNode->m_pLayerTile;
                }
            }

            Vec4 packedData; Vec4i dstCoord;
            GetIndirectionTextureData( pTile, packedData, dstCoord );

            _ASSERT( floorf( packedData.w + pTile->m_fadeInFactor ) == packedData.w && "FP precision problem" );
            packedData.w += pTile->m_fadeInFactor;

            if( useRegularShadowMapAsLayer )
                packedData.w = -packedData.w;

            m_quads.push_back( AtlasQuads::SFillQuad::Get(
                packedData,
                dstCoord.z - dstCoord.x + 1,
                dstCoord.w - dstCoord.y + 1,
                dstCoord.x,
                dstCoord.y,
                m_indirectionTextureSize, m_indirectionTextureSize ) );

            ++m_quadsCnt[z];

            for( int j = 0; j < 4; ++j )
            {
                CQuadTreeNode* pChild = pNode->m_children[j];
                if( pChild != nullptr && IsNodeAcceptableForIndexing( pChild ) )
                    m_indexedNodes.push_back( pChild );
            }
        }
    }

    m_indexedNodes.resize( numIndexedNodes );
}

void CAdaptiveShadowMap::CShadowFrustum::FillLODClampTextureData()
{
    if( !IsValid() ) return;

    if( m_indexedNodes.empty() ) return;

    unsigned int numIndexedNodes = m_indexedNodes.size();
    unsigned int i = 0;
    for( int z = m_cfg.m_maxRefinement; z >= 0; --z )
    {
        float clampValue = float(z) / float(MAX_REFINEMENT);

        unsigned int numNodes = m_indexedNodes.size();
        for( ; i < numNodes; ++i )
        {
            CQuadTreeNode* pNode = m_indexedNodes[i];
            CTileCacheEntry* pTile = pNode->m_pTile;

            Vec4 packedData; Vec4i dstCoord;
            GetIndirectionTextureData( pTile, packedData, dstCoord );

            if( z < m_cfg.m_maxRefinement )
            {
                m_lodClampQuads.push_back( AtlasQuads::SFillQuad::Get(
                    Vec4( clampValue ),
                    dstCoord.z - dstCoord.x + 1,
                    dstCoord.w - dstCoord.y + 1,
                    dstCoord.x,
                    dstCoord.y,
                    m_indirectionTextureSize, m_indirectionTextureSize ) );
            }

            for( int j = 0; j < 4; ++j )
            {
                CQuadTreeNode* pChild = pNode->m_children[j];
                if( pChild != nullptr && pChild->m_pTile != nullptr )
                    m_indexedNodes.push_back( pChild );
            }
        }
    }

    m_indexedNodes.resize( numIndexedNodes );
}

//-----------------------------------------------------------------------------

void CAdaptiveShadowMap::CShadowFrustum::UpdateIndirectionTexture( RenderTarget2D& indirectionTexture, const SShadowMapRenderContext& context, bool disableHierarchy )
{
    DeviceContext11& dc = *context.m_dc;

    dc.PushRC();

    dc.UnbindRT( 1 );
    dc.UnbindRT( 2 );
    dc.UnbindRT( 3 );
    dc.UnbindDepthStencil();

    static size_t s_shaderIndex = g_SimpleShaderCache.GetIndex( SimpleShaderDesc( "ASM.shader?FILL_INDIRECTION_TEXTURE", nullptr, "ASM.shader?FILL_INDIRECTION_TEXTURE", nullptr, nullptr, nullptr ) );
    g_SimpleShaderCache.GetByIndex( s_shaderIndex ).Bind( dc );

    AtlasQuads::SFillQuad clearQuad = AtlasQuads::SFillQuad::Get( Vec4::Zero(), m_indirectionTextureSize, m_indirectionTextureSize, 0, 0, m_indirectionTextureSize, m_indirectionTextureSize );
    unsigned int firstQuad = 0;
    unsigned int numQuads = 0;

    for( int mip = m_cfg.m_maxRefinement; mip >= 0; --mip )
    {
        dc.RenderContext11::BindRT( 0, indirectionTexture.GetRenderTargetView( mip ), &indirectionTexture );

        D3D11_VIEWPORT vp = { };
        vp.Width = float( m_indirectionTextureSize >> mip );
        vp.Height = vp.Width;
        vp.MaxDepth = 1.0f;
        context.m_dc->SetViewport( vp );

        numQuads += m_quadsCnt[ mip ];

        AtlasQuads::Draw( 1, &clearQuad, dc );

        if( numQuads > 0 )
            AtlasQuads::Draw( numQuads, &m_quads[ firstQuad ], dc );

        if( disableHierarchy )
        {
            firstQuad += numQuads;
            numQuads = 0;
        }
    }

    dc.PopRC();
}

void CAdaptiveShadowMap::CShadowFrustum::UpdateLODClampTexture( RenderTarget2D& lodClampTexture, const SShadowMapRenderContext& context )
{
    DeviceContext11& dc = *context.m_dc;

    dc.PushRC();

    dc.BindRT( 0, &lodClampTexture );
    dc.UnbindRT( 1 );
    dc.UnbindRT( 2 );
    dc.UnbindRT( 3 );
    dc.UnbindDepthStencil();

    lodClampTexture.SetViewport( dc );

    static size_t s_shaderIndex = g_SimpleShaderCache.GetIndex( SimpleShaderDesc( "ASM.shader?FILL_INDIRECTION_TEXTURE", nullptr, "ASM.shader?FILL_INDIRECTION_TEXTURE", nullptr, nullptr, nullptr ) );
    g_SimpleShaderCache.GetByIndex( s_shaderIndex ).Bind( dc );

    AtlasQuads::Draw( m_lodClampQuads.size(), &m_lodClampQuads[0], dc );

    dc.PopRC();
}
