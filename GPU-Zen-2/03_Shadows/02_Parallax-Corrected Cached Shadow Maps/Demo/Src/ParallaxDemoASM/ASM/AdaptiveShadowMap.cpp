#define SAFE_RELEASE(x) if(x) { (x)->Release(); (x) = NULL; }
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX

#include "AdaptiveShadowMapShadowFrustum.h"
#include "AdaptiveShadowMapTileCache.h"

int gfx_asm_showDebug = 1;

unsigned int CAdaptiveShadowMap::s_frameCounter;
float CAdaptiveShadowMap::s_tileFarPlane = gs_ASMTileFarPlane;

//-----------------------------------------------------------------------------

float CAdaptiveShadowMap::GetRefinementDistanceSq( const CAABBox& BBox, const Vec2& refinementPos )
{
    return Vec2::LengthSq( Vec2( BBox.GetCenter() ) - refinementPos );
}

//-----------------------------------------------------------------------------

#define SQR(a) ( (a) * (a) )

CAdaptiveShadowMap::CAdaptiveShadowMap() :
    m_preRenderDone( false ),
    m_cache( new CTileCache() )
{
    static const CShadowFrustum::Config longRangeCfg =
    {
        gs_ASMLargestTileWorldSize, gs_ASMDistanceMax, MAX_REFINEMENT, INT_MAX, 8, true,
        { SQR( gs_ASMDistanceMax ), SQR( 120.0f ), SQR( 60.0f ), SQR( 30.0f ), SQR( 10.0f ) }
    };

    m_longRangeShadows = new CShadowFrustum( longRangeCfg, true );
    m_longRangePreRender = new CShadowFrustum( longRangeCfg, true );

    Reset();
}

CAdaptiveShadowMap::~CAdaptiveShadowMap()
{
    delete m_cache;
    delete m_longRangeShadows;
    delete m_longRangePreRender;
}

//-----------------------------------------------------------------------------

bool CAdaptiveShadowMap::PrepareRender( const Camera& mainViewCamera, bool disablePreRender )
{
    m_longRangeShadows->CreateTiles( m_cache, mainViewCamera );
    m_longRangePreRender->CreateTiles( m_cache, mainViewCamera );

    if( m_cache->NothingToRender() )
    {
        bool keepRendering = true;

        for( unsigned int i = 0; i <= MAX_REFINEMENT && keepRendering; ++i )
        {
            keepRendering = m_cache->AddTileFromRenderQueueToRenderBatch( m_longRangeShadows, i, false ) < 0;

            if( keepRendering )
                keepRendering = m_cache->AddTileFromRenderQueueToRenderBatch( m_longRangeShadows, i, true ) < 0;

            if( keepRendering && m_longRangePreRender->IsValid() && i == 0 && !disablePreRender )
                keepRendering = m_cache->AddTileFromRenderQueueToRenderBatch( m_longRangePreRender, i, false ) < 0;
        }

        if( keepRendering && m_longRangePreRender->IsValid() && !disablePreRender )
        {
//            static int cnt;
//            if( ( ++cnt & 0xf ) == 0 )
            {
                keepRendering = m_cache->AddTileFromRenderQueueToRenderBatch( m_longRangePreRender, INT_MAX, false ) < 0;
 
                if( keepRendering )
                    keepRendering = m_cache->AddTileFromRenderQueueToRenderBatch( m_longRangePreRender, INT_MAX, true ) < 0;

                if( keepRendering )
                    m_preRenderDone = true;
            }
        }

        if( keepRendering )
            m_cache->AddTileFromRenderQueueToRenderBatch( m_longRangeShadows, INT_MAX, false );
    }

    Vec3 mainViewCameraPosition = mainViewCamera.GetPosition();
    SShadowMapPrepareRenderContext context = { &mainViewCameraPosition };
    return m_cache->PrepareRenderTilesBatch( context );
}

void CAdaptiveShadowMap::Render(
    const RenderTarget2D& workBufferDepth,
    const RenderTarget2D& workBufferColor,
    DeviceContext11& dc )
{
    SShadowMapRenderContext context = { &dc, this };

    if( !m_cache->NothingToRender() )
    {
        m_cache->RenderTilesBatch(
            workBufferDepth,
            workBufferColor,
            context );
    }

    m_cache->CreateDEM( workBufferColor, context, false );
    m_cache->CreateDEM( workBufferColor, context, true );

    m_longRangeShadows->BuildTextures( context, false );

    if( m_longRangePreRender->IsValid() )
        m_longRangePreRender->BuildTextures( context, true );
}

//-----------------------------------------------------------------------------

void CAdaptiveShadowMap::GetResolveShaderData( ASM_ResolveShaderData& shaderData )
{
    shaderData.LongRangeIndexTexMat = m_longRangeShadows->m_indexTexMat;
    shaderData.LongRangePreRenderIndexTexMat = m_longRangePreRender->m_indexTexMat;

    shaderData.LongRangeReceiverWarpVector = m_longRangeShadows->m_receiverWarpVector;
    shaderData.LongRangePreRenderReceiverWarpVector = m_longRangePreRender->m_receiverWarpVector;

    shaderData.LongRangeBlockerSearchVector = m_longRangeShadows->m_blockerSearchVector;
    shaderData.LongRangePreRenderBlockerSearchVector = m_longRangePreRender->m_blockerSearchVector;

    shaderData.LongRangeDefaultShadowFactor = m_longRangeShadows->IsLightBelowHorizon() ? 0.0f : 1.0f;
    shaderData.LongRangePreRenderDefaultShadowFactor = m_longRangePreRender->IsLightBelowHorizon() ? 0.0f : 1.0f;

    shaderData.LongRangeLightDir = m_longRangeShadows->m_lightDir;
    shaderData.LongRangePreRenderLightDir = m_longRangePreRender->m_lightDir;
}

void CAdaptiveShadowMap::SetResolveTextures( DeviceContext11& dc )
{
    dc.BindPS( 9, &m_longRangeShadows->GetIndirectionTexture() );
    dc.BindPS( 10, &m_longRangePreRender->GetIndirectionTexture() );
    dc.BindPS( 11, &m_cache->GetDepthAtlas() );
    dc.BindPS( 12, &m_cache->GetDEMAtlas() );
    dc.BindPS( 13, &m_longRangePreRender->GetLODClampTexture() );

    dc.SetSamplerPS( 10, &Platform::GetSamplerCache().GetByIndex( Platform::Sampler_ShadowMap_PCF ) );
    dc.SetSamplerPS( 11, &Platform::GetSamplerCache().Get( SamplerDesc11( D3D11_FILTER_MIN_MAG_MIP_POINT, D3D11_TEXTURE_ADDRESS_BORDER, D3D11_TEXTURE_ADDRESS_BORDER ) ) );
    dc.SetSamplerPS( 12, &Platform::GetSamplerCache().GetByIndex( Platform::Sampler_Linear_Clamp ) );
}

//-----------------------------------------------------------------------------

void CAdaptiveShadowMap::DrawDebug( DebugRenderer& debug )
{
    if( gfx_asm_showDebug >= 0 )
        m_longRangeShadows->DrawDebug( debug, 0.5f );

    if( gfx_asm_showDebug >= 1 )
        m_longRangePreRender->DrawDebug( debug, 0.5f );

    if( gfx_asm_showDebug >= 0 )
        m_cache->DrawDebug( debug );
}

void CAdaptiveShadowMap::Reset()
{
    m_longRangeShadows->Reset();
    m_longRangePreRender->Reset();
    m_preRenderDone = false;
}

void CAdaptiveShadowMap::Update( const CAABBox& BBoxWS )
{
    m_cache->UpdateTiles( m_longRangeShadows, BBoxWS );
    m_cache->UpdateTiles( m_longRangePreRender, BBoxWS );
}

void CAdaptiveShadowMap::Tick( unsigned int currentTime, unsigned int dt, bool disableWarping, bool forceUpdate, unsigned int updateDeltaTime )
{
    Vec3 sunDir = GetLightDirection( currentTime );
    
    float deltaTime = float( dt ) * 0.001f;

    bool isUpdated = false;
    if( !m_longRangeShadows->IsValid() )
    {
        m_longRangeShadows->Set( sunDir );

        isUpdated = true;
    }
    else if( forceUpdate )
    {
        m_longRangePreRender->Reset();

        m_longRangePreRender->Set( sunDir );
        m_preRenderDone = false;

        isUpdated = true;
    }
    else if( !m_longRangePreRender->IsValid() )
    {
        Vec3 nextSunDir = GetLightDirection( currentTime + ( updateDeltaTime >> 1 ) );

        isUpdated = m_longRangeShadows->IsLightDirDifferent( nextSunDir );

        if( isUpdated )
        {
            m_longRangePreRender->Set( nextSunDir );
            m_preRenderDone = false;
        }
    }

    m_longRangeShadows->UpdateWarpVector( sunDir, disableWarping );
    m_longRangePreRender->UpdateWarpVector( sunDir, disableWarping );

    m_cache->Tick( deltaTime );

    if( m_longRangePreRender->IsValid() && m_preRenderDone && m_cache->IsFadeInFinished( m_longRangePreRender ) )
    {
        std::swap( m_longRangeShadows, m_longRangePreRender );

        m_longRangePreRender->Reset();
        m_preRenderDone = false;
    }

    ++s_frameCounter;
}

const Vec3& CAdaptiveShadowMap::GetLightDir() const
{
    return m_longRangeShadows->m_lightDir;
}

const Vec3& CAdaptiveShadowMap::GetPreRenderLightDir() const
{
    return m_longRangePreRender->m_lightDir;
}

bool CAdaptiveShadowMap::NothingToRender() const
{
    return m_cache->NothingToRender();
}

bool CAdaptiveShadowMap::PreRenderAvailable() const
{
    return m_longRangePreRender->IsValid();
}
