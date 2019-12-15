#define SAFE_RELEASE(x) if(x) { (x)->Release(); (x) = NULL; }
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX

#include "AdaptiveShadowMapShadowFrustum.h"
#include "AdaptiveShadowMapTileCache.h"
#include "ShaderCache/SimpleShader.h"
#include "../Misc/Utils.h"
#include "_Shaders/ASMLayerShaderData.inc"

template< CAdaptiveShadowMap::CTileCacheEntry*& ( CAdaptiveShadowMap::CQuadTreeNode::*TileAccessor )() const >
bool CAdaptiveShadowMap::GetRectangleWithinParent(
    const int NUp,
    CQuadTreeNode* NList[ MAX_REFINEMENT + 1 ],
    Vec4i& parentRect,
    Vec4i& tileRect )
{
    for( int i = 0; i < NUp; ++i )
        NList[ i + 1 ] = NList[ i ]->m_pParent;

    CTileCacheEntry* pParentTile = NList[ NUp ]->m_pTile;//( NList[ NUp ]->*TileAccessor )();
    if( pParentTile != nullptr )
    {
        parentRect = CTileCacheEntry::GetRect( pParentTile->m_viewport, TILE_BORDER_TEXELS );

        Vec4i src = parentRect;
        for( int i = 0; i < NUp; ++i )
        {
            Vec3 d = NList[ NUp - i - 1 ]->m_BBox.m_min - NList[ NUp - i ]->m_BBox.m_min;

            Vec4i rect;
            rect.x = d.x <= 0 ? src.x : ( ( src.z + src.x ) / 2 );
            rect.y = d.y >  0 ? src.y : ( ( src.w + src.y ) / 2 );
            rect.z = rect.x + ( src.z - src.x ) / 2;
            rect.w = rect.y + ( src.w - src.y ) / 2;

            const int border = TILE_BORDER_TEXELS >> ( i + 1 );
            rect.x += d.x <= 0 ? border : -border;
            rect.z += d.x <= 0 ? border : -border;
            rect.y += d.y >  0 ? border : -border;
            rect.w += d.y >  0 ? border : -border;

            src = rect;
        }

        tileRect = src;
        return true;
    }

    return false;
}

CAdaptiveShadowMap::CTileCache::CTileCache() :
    m_cacheHits( 0 ),
    m_tileAllocs( 0 ),
    m_numTilesRendered( 0 ),
    m_numTilesUpdated( 0 ), 
    m_depthAtlasWidth( 0 ),
    m_depthAtlasHeight( 0 ),
    m_demAtlasWidth( 0 ),
    m_demAtlasHeight( 0 )
{
    m_depthAtlas.Init( gs_ASMDepthAtlasTextureWidth, gs_ASMDepthAtlasTextureHeight, DXGI_FORMAT_R16_TYPELESS, 1, nullptr, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_DEPTH_STENCIL );
    m_demAtlas.Init( gs_ASMDEMAtlasTextureWidth, gs_ASMDEMAtlasTextureHeight, DXGI_FORMAT_R16_FLOAT );

    m_depthAtlasWidth = m_depthAtlas.GetDesc().Width;
    m_depthAtlasHeight = m_depthAtlas.GetDesc().Height;

    m_demAtlasWidth = m_demAtlas.GetDesc().Width;
    m_demAtlasHeight = m_demAtlas.GetDesc().Height;

    _ASSERT( m_demAtlasWidth == ( m_depthAtlasWidth >> DEM_DOWNSAMPLE_LEVEL ) );
    _ASSERT( m_demAtlasHeight == ( m_depthAtlasHeight >> DEM_DOWNSAMPLE_LEVEL ) );

    int gridWidth = m_depthAtlasWidth / TILE_SIZE;
    int gridHeight = m_depthAtlasHeight / TILE_SIZE;
    for( int i = 0; i < gridHeight; ++i )
        for( int j = 0; j < gridWidth; ++j )
            new CTileCacheEntry( this, j * TILE_SIZE, i * TILE_SIZE );
}

CAdaptiveShadowMap::CTileCache::~CTileCache()
{
    m_depthAtlas.Clear();
    m_demAtlas.Clear();

    for( size_t i = m_tiles.size(); i > 0; --i )
        delete m_tiles[ i - 1 ];

    _ASSERT( m_tiles.empty() );
    _ASSERT( m_freeTiles.empty() );
    _ASSERT( m_renderQueue.empty() );
    _ASSERT( m_readyTiles.empty() );
    _ASSERT( m_demQueue.empty() );
    _ASSERT( m_updateQueue.empty() );
}

template< CAdaptiveShadowMap::CTileCacheEntry*& (CAdaptiveShadowMap::CQuadTreeNode::*TileAccessor)(), bool isLayer >
CAdaptiveShadowMap::CTileCacheEntry* CAdaptiveShadowMap::CTileCache::Allocate( CQuadTreeNode* pNode, CShadowFrustum* pFrustum )
{
    _ASSERT( (pNode->*TileAccessor)() == nullptr );

    // first search for tile in cache
    CTileCacheEntry* pTileToAlloc = nullptr;

    if( m_freeTiles.empty() )
    {
        // try to free visually less important tile (the one further from viewer or deeper in hierarchy)
        unsigned char minRefinement = pNode->m_refinement;
        float minDistSq = GetRefinementDistanceSq( pNode->m_BBox, pFrustum->m_refinementPoint );
        for( auto it = m_tiles.begin(); it != m_tiles.end(); ++it )
        {
            CTileCacheEntry* pTile = *it;

            _ASSERT( pTile->m_pFrustum != nullptr );
            _ASSERT( pTile->m_pOwner != nullptr );

            if( pTile->m_refinement < minRefinement )
                continue;

            float distSq = GetRefinementDistanceSq( pTile->m_BBox, pTile->m_pFrustum->m_refinementPoint );
            if( pTile->m_refinement == minRefinement )
                if( ( distSq == minDistSq && !pTile->m_isLayer ) || distSq < minDistSq )
                    continue;

            pTileToAlloc = pTile;
            minRefinement = pTile->m_refinement;
            minDistSq = distSq;
        }

        if( pTileToAlloc == nullptr )
            return nullptr;

        pTileToAlloc->Free();
    }

    for( auto it = m_freeTiles.begin(); it != m_freeTiles.end(); ++it )
    {
        CTileCacheEntry* pTile = *it;

        if( pTile->m_frustumID == pFrustum->m_ID &&
            !( pTile->m_BBox != pNode->m_BBox ) &&
            pTile->m_isLayer == isLayer )
        {
            pTileToAlloc = pTile;
            ++m_cacheHits;
            break;
        }
    }

    if( pTileToAlloc == nullptr )
    {
        // tile isn't in cache, use LRU tile or a tile the deepest in hierarchy
        unsigned char refinement = 0;
        unsigned int LRUdt = 0;
        for( auto it = m_freeTiles.begin(); it != m_freeTiles.end(); ++it )
        {
            CTileCacheEntry* pTile = *it;

            if( pTile->m_refinement < refinement )
                continue;

            unsigned int dt = s_frameCounter - pTile->m_lastFrameUsed;
            if( pTile->m_refinement == refinement && dt < LRUdt )
                continue;

            pTileToAlloc = pTile;
            refinement = pTile->m_refinement;
            LRUdt = dt;
        }

        if( pTileToAlloc != nullptr )
            pTileToAlloc->Invalidate();
    }

    if( pTileToAlloc != nullptr )
    {
        pTileToAlloc->Allocate< TileAccessor, isLayer >( pNode, pFrustum );
        ++m_tileAllocs;
    }

    return pTileToAlloc;
}

void CAdaptiveShadowMap::CShadowFrustum::AllocateTiles( CTileCache* pCache, CQuadTreeNode* pNode )
{
    for( int i = 0; i < 4; ++i )
        if( pNode->m_children[i] != nullptr )
            AllocateTiles( pCache, pNode->m_children[i] );

    if( pNode->m_pTile == nullptr )
        pCache->Allocate< &CQuadTreeNode::GetTile, false >( pNode, this );

    if( pNode->m_pLayerTile == nullptr && pNode->m_refinement >= m_cfg.m_minRefinementForLayer )
        pCache->Allocate< &CQuadTreeNode::GetLayerTile, true >( pNode, this );
}

void CAdaptiveShadowMap::CTileCache::Tick( float deltaTime )
{
    for( auto it = m_readyTiles.begin(); it != m_readyTiles.end(); ++it )
    {
        CTileCacheEntry* pTile = *it;
        pTile->m_fadeInFactor = std::max( 0.0f, pTile->m_fadeInFactor - deltaTime );
    }
}

bool CAdaptiveShadowMap::CTileCache::IsFadeInFinished( const CShadowFrustum* pFrustum ) const
{
    for( auto it = m_readyTiles.begin(); it != m_readyTiles.end(); ++it )
    {
        CTileCacheEntry* pTile = *it;
        if( pTile->m_frustumID == pFrustum->m_ID && pTile->m_fadeInFactor > 0 )
            return false;
    }
    return true;
}

//-----------------------------------------------------------------------------

CAdaptiveShadowMap::CTileCacheEntry::CTileCacheEntry( CTileCache* pCache, int x, int y ) :
    m_pCache( pCache ),
    m_pOwner( nullptr ),
    m_pFrustum( nullptr ),
    m_lastFrameUsed( 0 ),
    m_frustumID( 0 ),
    m_isLayer( false ),
    m_fadeInFactor( 0.0f )
{
    m_viewport.x = x + TILE_BORDER_TEXELS; m_viewport.w = BORDERLESS_TILE_SIZE;
    m_viewport.y = y + TILE_BORDER_TEXELS; m_viewport.h = BORDERLESS_TILE_SIZE;

    Invalidate();

    m_pCache->m_tiles.Add( this );
    m_pCache->m_freeTiles.Add( this );
}

CAdaptiveShadowMap::CTileCacheEntry::~CTileCacheEntry()
{
    if( IsAllocated() ) Free();

    m_pCache->m_tiles.Remove( this );
    m_pCache->m_freeTiles.Remove( this );
}

void CAdaptiveShadowMap::CTileCacheEntry::Invalidate()
{
    m_BBox = CAABBox();
    m_refinement = MAX_REFINEMENT;
    m_lastFrameUsed = s_frameCounter - 0x7fFFffFF;
    m_frustumID = 0;
}

template< CAdaptiveShadowMap::CTileCacheEntry*& (CAdaptiveShadowMap::CQuadTreeNode::*TileAccessor)(), bool isLayer >
void CAdaptiveShadowMap::CTileCacheEntry::Allocate( CQuadTreeNode* pOwner, CShadowFrustum* pFrustum )
{
    _ASSERT( (pOwner->*TileAccessor)() == nullptr );
    _ASSERT( !isLayer || pOwner->m_pTile != nullptr );

    m_pCache->m_freeTiles.Remove( this );
    m_pOwner = pOwner;
    m_pFrustum = pFrustum;
    m_refinement = pOwner->m_refinement;
    (pOwner->*TileAccessor)() = this;

    if( m_frustumID == pFrustum->m_ID && !( m_BBox != pOwner->m_BBox ) && m_isLayer == isLayer )
    {
        MarkReady();
    }
    else
    {
        m_frustumID = pFrustum->m_ID;
        m_BBox = pOwner->m_BBox;
        m_isLayer = isLayer;

        m_pCache->m_renderQueue.Add( this );
    }
}

void CAdaptiveShadowMap::CTileCacheEntry::Free()
{
    if( m_renderQueuePos.IsInserted() || 
        m_renderBatchPos.IsInserted() || 
        m_updateQueuePos.IsInserted() || 
        m_demQueuePos.IsInserted() )
    {
        m_pCache->m_renderQueue.Remove( this, true );
        m_pCache->m_renderBatch.Remove( this, true );
        m_pCache->m_updateQueue.Remove( this, true );
        m_pCache->m_demQueue.Remove( this, true );
        m_pCache->m_readyTiles.Remove( this, true );
        Invalidate();
    }
    else
    {
        MarkNotReady();
        m_lastFrameUsed = s_frameCounter;
    }
    m_pCache->m_freeTiles.Add( this );
    ( m_isLayer ? m_pOwner->m_pLayerTile : m_pOwner->m_pTile ) = nullptr;
    m_pOwner = nullptr;
    m_pFrustum = nullptr;
}

void CAdaptiveShadowMap::CTileCacheEntry::MarkReady()
{
    _ASSERT( m_pOwner != nullptr && !m_readyTilesPos.IsInserted() );
    _ASSERT( m_pFrustum != nullptr && m_pFrustum->m_ID != 0 && m_pFrustum->m_ID == m_frustumID );

    m_pCache->m_readyTiles.Add( this );

    m_fadeInFactor = 0.5f;

//    // do not fade in the tiles that are invisible during previous frames
//    if( !m_pFrustum->m_prevLargerHull.Intersects( m_BBox ) )
//        m_fadeInFactor = 0;
}

void CAdaptiveShadowMap::CTileCacheEntry::MarkNotReady()
{
    _ASSERT( m_pOwner != nullptr && m_readyTilesPos.IsInserted() );
    _ASSERT( m_pFrustum != nullptr && m_pFrustum->m_ID != 0 && m_pFrustum->m_ID == m_frustumID );

    m_pCache->m_readyTiles.Remove( this );
}

//-----------------------------------------------------------------------------

void CAdaptiveShadowMap::CTileCache::StartDEM( CTileCacheEntry* pTile, AtlasQuads::SCopyQuad& copyDEMQuad )
{
    m_demQueue.Add( pTile, true );

    int demAtlasX = ( pTile->m_viewport.x - TILE_BORDER_TEXELS ) >> DEM_DOWNSAMPLE_LEVEL;
    int demAtlasY = ( pTile->m_viewport.y - TILE_BORDER_TEXELS ) >> DEM_DOWNSAMPLE_LEVEL;

    copyDEMQuad = AtlasQuads::SCopyQuad::Get(
        Vec4::Zero(),
        DEM_TILE_SIZE, DEM_TILE_SIZE,
        demAtlasX, demAtlasY,
        m_demAtlasWidth, m_demAtlasHeight,
        TILE_SIZE, TILE_SIZE,
        pTile->m_viewport.x - TILE_BORDER_TEXELS,
        pTile->m_viewport.y - TILE_BORDER_TEXELS,
        m_depthAtlasWidth, m_depthAtlasHeight );
}

void CAdaptiveShadowMap::CTileCache::RenderTiles(
    unsigned int numTiles,
    CTileCacheEntry** tiles,
    const RenderTarget2D& workBufferDepth,
    const RenderTarget2D& workBufferColor,
    const SShadowMapRenderContext& context,
    bool allowDEM )
{
    if( !numTiles ) return;

    DeviceContext11& dc = *context.m_dc;
    dc.PushRC();

    unsigned int workBufferWidth = workBufferDepth.GetDesc().Width;
    unsigned int workBufferHeight = workBufferDepth.GetDesc().Height;
    unsigned int numTilesW = workBufferWidth / TILE_SIZE;
    unsigned int numTilesH = workBufferHeight / TILE_SIZE;
    unsigned int maxTilesPerPass = numTilesW * numTilesH;

    // We want to work with DX9-like pixel centers, i.e. pixel centers located at integer grid nodes.
    // DX10 and onward uses (0.5, 0.5) as pixel center, hence the offset.
    Mat4x4 pixelCenterOffsetMatrix = Mat4x4::TranslationD3D( Vec3( 1.0f / float( workBufferWidth ), -1.0f / float( workBufferHeight ), 0 ) );

    AtlasQuads::SCopyQuad* copyDepthQuads = static_cast< AtlasQuads::SCopyQuad* >( alloca( sizeof( AtlasQuads::SCopyQuad ) * ( maxTilesPerPass + numTiles ) ) );
    AtlasQuads::SCopyQuad* copyDEMQuads = copyDepthQuads + maxTilesPerPass;

    float invAtlasWidth = 1.0f / float( m_depthAtlasWidth );
    float invAtlasHeight = 1.0f / float( m_depthAtlasHeight );

    unsigned int numCopyDEMQuads = 0;
    for( unsigned int i = 0; i < numTiles; )
    {
        unsigned int tilesToRender = std::min( maxTilesPerPass, numTiles - i );

        dc.BindRT( 0, &workBufferColor );
        dc.UnbindRT( 1 );
        dc.UnbindRT( 2 );
        dc.UnbindRT( 3 );
        dc.BindDepthStencil( &workBufferDepth );

        dc.ClearRenderTarget( 0, 1.0f );
        dc.ClearDepthStencil( 1.0f, 0 );

        for( unsigned int j = 0; j < tilesToRender; ++j )
        {
            CTileCacheEntry* pTile = tiles[ i + j ];

            _ASSERT( !pTile->m_isLayer || pTile->m_pOwner->m_pTile->IsReady() );

            Vec4i viewport(
                ( j % numTilesW ) * TILE_SIZE,
                ( j / numTilesW ) * TILE_SIZE,
                TILE_SIZE, TILE_SIZE );

            Camera renderCamera;
            renderCamera.SetViewMatrix( pTile->m_renderCamera.GetViewMatrix() );
            renderCamera.SetProjection( pTile->m_renderCamera.GetProjection() * pixelCenterOffsetMatrix );

            if( pTile->m_isLayer )
            {
                const int NUp = pTile->m_refinement - 1;
                CQuadTreeNode* NList[ MAX_REFINEMENT + 1 ] = { pTile->m_pOwner };
                Vec4i parentRect, src;
                bool isTileAttached = GetRectangleWithinParent< &CQuadTreeNode::GetTile >( NUp, NList, parentRect, src );
                _ASSERT( isTileAttached );
                _ASSERT( NList[ NUp ]->m_pTile->m_refinement == 1 );

                Vec4 packedData; Vec4i dstCoord;
                pTile->m_pFrustum->GetIndirectionTextureData( NList[ NUp ]->m_pTile, packedData, dstCoord );

                ASMLayerShaderData shaderData;
                shaderData.g_ASMIndirectionTexMat = pTile->m_pFrustum->m_indexTexMat;
                shaderData.g_ASMTileData = packedData;
                shaderData.g_ASMOneOverDepthAtlasSize = Vec2( invAtlasWidth, invAtlasHeight );

                dc.BindPS( 10, &m_demAtlas );
                dc.SetSamplerPS( 10, &Platform::GetSamplerCache().GetByIndex( Platform::Sampler_Linear_Clamp ) );

                ID3D11Buffer* pBuffer = dc.GetConstantBuffers().Allocate( sizeof(shaderData), &shaderData, dc.DoNotFlushToDevice() );
                dc.VSSetConstantBuffer( ASMLayerShaderData::BANK, pBuffer );

                context.m_shadowMapRenderer->RenderTile( viewport, renderCamera, true, dc );

                dc.GetConstantBuffers().Free( pBuffer );
            }
            else
            {
                context.m_shadowMapRenderer->RenderTile( viewport, renderCamera, false, dc );
            }

            float depthBias = Utils::CalcDepthBias(
                pTile->m_renderCamera.GetProjection(),
                Vec3( 3.5f, 3.5f, 1.0f ),
                TILE_SIZE,
                TILE_SIZE,
                16 );

            copyDepthQuads[ j ] = AtlasQuads::SCopyQuad::Get(
                Vec4( 0, 0, depthBias, 0 ),
                TILE_SIZE, TILE_SIZE,
                pTile->m_viewport.x - TILE_BORDER_TEXELS,
                pTile->m_viewport.y - TILE_BORDER_TEXELS,
                m_depthAtlasWidth, m_depthAtlasHeight,
                TILE_SIZE, TILE_SIZE,
                viewport.x, viewport.y,
                workBufferWidth, workBufferHeight );

            bool generateDEM = pTile->m_refinement <= pTile->m_pFrustum->GetDEMMinRefinement( pTile->m_isLayer );
            if( generateDEM && ( allowDEM || pTile->IsReady() ) )
                StartDEM( pTile, copyDEMQuads[ numCopyDEMQuads++ ] );
        }

        dc.RestoreRC();

        dc.UnbindRT( 0 );
        dc.UnbindRT( 1 );
        dc.UnbindRT( 2 );
        dc.UnbindRT( 3 );
        dc.BindDepthStencil( &m_depthAtlas );

        m_depthAtlas.SetViewport( dc );

        static size_t s_depthStencilBlockIndex = Platform::GetDepthStencilCache().ConcurrentGetIndex( DepthStencilDesc11( true, D3D11_DEPTH_WRITE_MASK_ALL, D3D11_COMPARISON_ALWAYS ) );
        dc.SetDepthStencilState( &Platform::GetDepthStencilCache().ConcurrentGetByIndex( s_depthStencilBlockIndex ) );
        
        static size_t s_shaderIndex = g_SimpleShaderCache.GetIndex( SimpleShaderDesc( "ASM.shader?COPY_DEPTH", nullptr, "ASM.shader?COPY_DEPTH", nullptr, nullptr, nullptr ) );
        g_SimpleShaderCache.GetByIndex( s_shaderIndex ).Bind( dc );

        dc.BindPS( 0, &workBufferDepth );
        dc.SetSamplerPS( 0, &Platform::GetSamplerCache().GetByIndex( Platform::Sampler_Point_Clamp ) );

        AtlasQuads::Draw( tilesToRender, copyDepthQuads, dc );

        i += tilesToRender;
    }

    if( numCopyDEMQuads > 0 )
    {
        dc.RestoreRC();

        dc.BindRT( 0, &m_demAtlas );
        dc.UnbindRT( 1 );
        dc.UnbindRT( 2 );
        dc.UnbindRT( 3 );
        dc.UnbindDepthStencil();

        m_demAtlas.SetViewport( dc );

        static size_t s_shaderIndex = g_SimpleShaderCache.GetIndex( SimpleShaderDesc( "ASM.shader?COPY_DEM", nullptr, "ASM.shader?COPY_DEM", nullptr, nullptr, nullptr ) );
        g_SimpleShaderCache.GetByIndex( s_shaderIndex ).Bind( dc );

        dc.BindPS( 0, &m_depthAtlas );
        dc.SetSamplerPS( 0, &Platform::GetSamplerCache().GetByIndex( Platform::Sampler_Point_Clamp ) );

        AtlasQuads::Draw( numCopyDEMQuads, copyDEMQuads, dc );
    }

    dc.PopRC();
}

//-----------------------------------------------------------------------------

template< class T >
int CAdaptiveShadowMap::CTileCache::AddTileToRenderBatch(
    T& tilesQueue,
    CShadowFrustum* pFrustum,
    int maxRefinement,    
    bool isLayer )
{
    if( !pFrustum->IsValid() ) return -1;

    CTileCacheEntry* pTileToRender = nullptr;
    float minDistSq = FLT_MAX;
    unsigned char refinement = UCHAR_MAX;
    for( auto it = tilesQueue.begin(); it != tilesQueue.end(); ++it )
    {
        CTileCacheEntry* pTile = *it;
        if( pFrustum == pTile->m_pFrustum && isLayer == pTile->m_isLayer &&
            ( !pTile->m_isLayer || pTile->m_pOwner->m_pTile->IsReady() ) )
        {
            float distSq = GetRefinementDistanceSq( pTile->m_BBox, pFrustum->m_refinementPoint );
            if( pTile->m_refinement < refinement || 
                ( refinement == pTile->m_refinement && distSq < minDistSq ) )
            {
                refinement = pTile->m_refinement;
                minDistSq = distSq;
                pTileToRender = pTile;
            }
        }
    }

    if( pTileToRender == nullptr || pTileToRender->m_refinement > maxRefinement ) return -1;

    tilesQueue.Remove( pTileToRender );
    m_renderBatch.Add( pTileToRender );
    return pTileToRender->m_refinement;
}

int CAdaptiveShadowMap::CTileCache::AddTileFromRenderQueueToRenderBatch(
    CShadowFrustum* pFrustum,
    int maxRefinement,
    bool isLayer )
{
    return AddTileToRenderBatch(
        m_renderQueue,
        pFrustum,
        maxRefinement,
        isLayer );
}

int CAdaptiveShadowMap::CTileCache::AddTileFromUpdateQueueToRenderBatch(
    CShadowFrustum* pFrustum,
    int maxRefinement,
    bool isLayer)
{
    return AddTileToRenderBatch(
        m_updateQueue,
        pFrustum,
        maxRefinement,
        isLayer );
}

void CAdaptiveShadowMap::CTileCacheEntry::PrepareRender( const SShadowMapPrepareRenderContext& context )
{
    m_renderCamera = m_pFrustum->CalcCamera( m_BBox, *context.m_worldCenter, float( TILE_SIZE ) / float( m_viewport.w ) );
}

bool CAdaptiveShadowMap::CTileCache::PrepareRenderTilesBatch( const SShadowMapPrepareRenderContext& context )
{
    for( auto it = m_renderBatch.begin(); it != m_renderBatch.end(); ++it )
    {
        CTileCacheEntry* pTile = *it;
        pTile->PrepareRender( context );
    }

    return !m_renderBatch.empty();
}

void CAdaptiveShadowMap::CTileCache::RenderTilesBatch(
    const RenderTarget2D& workBufferDepth,
    const RenderTarget2D& workBufferColor,
    const SShadowMapRenderContext& context )
{
    if( !m_renderBatch.empty() )
    {
        RenderTiles(
            m_renderBatch.size(),
            &m_renderBatch[0],
            workBufferDepth,
            workBufferColor,
            context,
            true );
    }

    for( size_t i = m_renderBatch.size(); i > 0; --i )
    {
        CTileCacheEntry* pTile = m_renderBatch[ i - 1 ];
        m_renderBatch.Remove( pTile );

        if( !pTile->IsReady() )
        {
            pTile->MarkReady();
            ++m_numTilesRendered;
        }
        else
        {
            ++m_numTilesRendered;
        }
    }
}

//-----------------------------------------------------------------------------

void CAdaptiveShadowMap::CTileCache::UpdateTiles(
    CShadowFrustum* pFrustum,
    const CAABBox& BBoxWS )
{
    if( !pFrustum->IsValid() ) return;

    CAABBox BBoxLS;
    BBoxWS.TransformTo( pFrustum->m_lightRotMat, BBoxLS );
    BBoxLS.m_min.z = 0;
    BBoxLS.m_max.z = 0;

    for( auto it = m_tiles.begin(); it != m_tiles.end(); ++it )
    {
        CTileCacheEntry* pTile = *it;
        if( pTile->m_frustumID == pFrustum->m_ID )
        {
            if( pTile->IsAllocated() )
            {
                if( !pTile->IsQueuedForRendering() && BBoxLS.IsIntersectBox( pTile->m_BBox ) )
                {
                    m_updateQueue.Add( pTile, true );
                }
            }
            else
            {
                if( BBoxLS.IsIntersectBox( pTile->m_BBox ) )
                    pTile->Invalidate();
            }
        }
    }
}

//-----------------------------------------------------------------------------

void CAdaptiveShadowMap::CTileCache::CreateDEM(
    const RenderTarget2D& workBufferColor,
    const SShadowMapRenderContext& context,
    bool createDemForLayerRendering )
{
    if( m_demQueue.empty() ) return;

    DeviceContext11& dc = *context.m_dc;

    unsigned int workBufferWidth = workBufferColor.GetDesc().Width;
    unsigned int workBufferHeight = workBufferColor.GetDesc().Height;
    unsigned int numTilesW = workBufferWidth / DEM_TILE_SIZE;
    unsigned int numTilesH = workBufferHeight / DEM_TILE_SIZE;
    unsigned int maxTilesPerPass = numTilesW * numTilesH;

    AtlasQuads::SCopyQuad* atlasToBulkQuads = static_cast< AtlasQuads::SCopyQuad* >(
        alloca( ( sizeof( AtlasQuads::SCopyQuad ) * 2 + sizeof( CTileCacheEntry* ) ) * maxTilesPerPass ) );
    AtlasQuads::SCopyQuad* bulkToAtlasQuads = atlasToBulkQuads + maxTilesPerPass;
    CTileCacheEntry** tilesToUpdate = reinterpret_cast< CTileCacheEntry** >( bulkToAtlasQuads + maxTilesPerPass );

    dc.PushRC();

    for( ;; )
    {
        unsigned int numTiles = 0; 
        for( auto it = m_demQueue.begin(); it != m_demQueue.end() && numTiles < maxTilesPerPass; ++it )
        {
            CTileCacheEntry* pTile = *it;

            bool isDemForLayerRendering = pTile->m_refinement > 0 && !pTile->m_isLayer;
            if( isDemForLayerRendering == createDemForLayerRendering )
            {
                static const unsigned int rectSize = DEM_TILE_SIZE - 2;

                unsigned int workX = ( numTiles % numTilesW ) * DEM_TILE_SIZE;
                unsigned int workY = ( numTiles / numTilesW ) * DEM_TILE_SIZE;

                unsigned int atlasX = ( ( pTile->m_viewport.x - TILE_BORDER_TEXELS ) >> DEM_DOWNSAMPLE_LEVEL ) + 1;
                unsigned int atlasY = ( ( pTile->m_viewport.y - TILE_BORDER_TEXELS ) >> DEM_DOWNSAMPLE_LEVEL ) + 1;

                if( createDemForLayerRendering )
                {
                    const int NUp = pTile->m_refinement;
                    CQuadTreeNode* NList[ MAX_REFINEMENT + 1 ] = { pTile->m_pOwner };
                    Vec4i parentRect, src;
                    bool isTileAttached = GetRectangleWithinParent< &CQuadTreeNode::GetTile >( NUp, NList, parentRect, src );
                    _ASSERT( isTileAttached );

                    float depthOffset = pTile->m_renderCamera.GetViewProjection().e43 - NList[ NUp ]->m_pTile->m_renderCamera.GetViewProjection().e43;

                    atlasToBulkQuads[ numTiles ] = AtlasQuads::SCopyQuad::Get(
                        Vec4( 1.0f/ float( m_demAtlasWidth ), 1.0f/ float( m_demAtlasHeight ), depthOffset, 0.0f ),
                        DEM_TILE_SIZE, DEM_TILE_SIZE, workX, workY, workBufferWidth, workBufferHeight,
                        ( src.z - src.x ) >> DEM_DOWNSAMPLE_LEVEL,
                        ( src.w - src.y ) >> DEM_DOWNSAMPLE_LEVEL,
                        src.x >> DEM_DOWNSAMPLE_LEVEL,
                        src.y >> DEM_DOWNSAMPLE_LEVEL,
                        m_demAtlasWidth, m_demAtlasHeight );
                }
                else
                {
                    atlasToBulkQuads[ numTiles ] = AtlasQuads::SCopyQuad::Get(
                        Vec4( 1.0f/ float( m_demAtlasWidth ), 1.0f/ float( m_demAtlasHeight ), 0.0f, 0.0f ),
                        rectSize, rectSize, workX + 1, workY + 1, workBufferWidth, workBufferHeight,
                        rectSize, rectSize, atlasX, atlasY, m_demAtlasWidth, m_demAtlasHeight );
                }

                bulkToAtlasQuads[ numTiles ] = AtlasQuads::SCopyQuad::Get(
                    Vec4( 1.0f/ float( workBufferWidth ), 1.0f/ float( workBufferHeight ), 0.0f, 0.0f ),
                    rectSize, rectSize, atlasX, atlasY, m_demAtlasWidth, m_demAtlasHeight,
                    rectSize, rectSize, workX + 1, workY + 1, workBufferWidth, workBufferHeight );

                tilesToUpdate[ numTiles++ ] = pTile;
            }
        }

        if( numTiles == 0 ) break;
        
        dc.BindRT( 0, &workBufferColor );
        dc.UnbindRT( 1 );
        dc.UnbindRT( 2 );
        dc.UnbindRT( 3 );
        dc.UnbindDepthStencil();

        workBufferColor.SetViewport( dc );

        dc.ClearRenderTarget( 0, Vec4( 1.0f ) );

        static size_t s_shaderIndex = g_SimpleShaderCache.GetIndex( SimpleShaderDesc( "ASM.shader?COMPUTE_DEM", nullptr, "ASM.shader?COMPUTE_DEM", nullptr, nullptr, nullptr ) );
        g_SimpleShaderCache.GetByIndex( s_shaderIndex ).Bind( dc );

        dc.BindPS( 0, &m_demAtlas );
        dc.SetSamplerPS( 0, &Platform::GetSamplerCache().GetByIndex( Platform::Sampler_Point_Clamp ) );

        AtlasQuads::Draw( numTiles, atlasToBulkQuads, dc );

        dc.BindRT( 0, &m_demAtlas );

        m_demAtlas.SetViewport( dc );

        dc.BindPS( 0, &workBufferColor );

        AtlasQuads::Draw( numTiles, bulkToAtlasQuads, dc );

        for( unsigned int i = 0; i < numTiles; ++i )
            m_demQueue.Remove( tilesToUpdate[i] );
    }

    dc.PopRC();
}

//-----------------------------------------------------------------------------

void CAdaptiveShadowMap::CTileCache::DrawDebug( DebugRenderer& debug )
{
}
