#define SAFE_RELEASE(x) if(x) { (x)->Release(); (x) = NULL; }
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX

#include "AdaptiveShadowMapQuadTree.h"
#include "AdaptiveShadowMapTileCache.h"

CAdaptiveShadowMap::CQuadTree::CQuadTree()
{
    m_roots.reserve( 32 );
}

CAdaptiveShadowMap::CQuadTree::~CQuadTree()
{
    Reset();
}

void CAdaptiveShadowMap::CQuadTree::Reset()
{
    while( !m_roots.empty() )
        delete m_roots.back();
}

CAdaptiveShadowMap::CQuadTreeNode* CAdaptiveShadowMap::CQuadTree::FindRoot( const CAABBox& BBox )
{
    for( auto it = m_roots.begin(); it != m_roots.end(); ++it )
    {
        if(!( (*it)->m_BBox != BBox ) )
            return *it;
    }
    return nullptr;
}

CAdaptiveShadowMap::CQuadTreeNode::CQuadTreeNode( CQuadTree* pQuadTree, CQuadTreeNode* pParent ) :
    m_pQuadTree( pQuadTree ),
    m_pParent( pParent ),
    m_lastFrameVerified( 0 ),
    m_numChildren( 0 ),
    m_pTile( nullptr ),
    m_pLayerTile( nullptr )
{
    memset( m_children, 0, sizeof( m_children ) );

    if( m_pParent != nullptr )
    {
        m_refinement = m_pParent->m_refinement + 1;
        m_rootNodesIndex = -1;
    }
    else
    {
        m_refinement = 0;
        m_rootNodesIndex = m_pQuadTree->m_roots.size();
        m_pQuadTree->m_roots.push_back( this );
    }
}

CAdaptiveShadowMap::CQuadTreeNode::~CQuadTreeNode()
{
    if( m_pTile != nullptr )
        m_pTile->Free();

    if( m_pLayerTile != nullptr )
        m_pLayerTile->Free();

    for( int i = 0; i < 4; ++i )
        delete m_children[i];

    if( m_pParent != nullptr )
    {
        for( int i=0; i<4; ++i )
        {
            if( m_pParent->m_children[i]==this )
            {
                m_pParent->m_children[i] = nullptr;
                --m_pParent->m_numChildren;
                break;
            }
        }
    }
    else
    {
        CQuadTreeNode* pLast = m_pQuadTree->m_roots.back();
        pLast->m_rootNodesIndex = m_rootNodesIndex;
        m_pQuadTree->m_roots[ m_rootNodesIndex ] = pLast;
        m_pQuadTree->m_roots.pop_back();
    }
}

const CAABBox CAdaptiveShadowMap::CQuadTreeNode::GetChildBBox( int childIndex )
{
    static const Vec3 quadrantOffsets[] =
    {
        Vec3( 0.0f, 0.0f, 0.0f ),
        Vec3( 1.0f, 0.0f, 0.0f ),
        Vec3( 1.0f, 1.0f, 0.0f ),
        Vec3( 0.0f, 1.0f, 0.0f ),
    };

    Vec3 halfSize = 0.5f * ( m_BBox.m_max - m_BBox.m_min );
    Vec3 BBoxMin = m_BBox.m_min + quadrantOffsets[ childIndex ] * halfSize;
    Vec3 BBoxMax = BBoxMin + halfSize;
    return CAABBox( BBoxMin, BBoxMax );
}

CAdaptiveShadowMap::CQuadTreeNode* CAdaptiveShadowMap::CQuadTreeNode::AddChild( int childIndex )
{
    if( m_children[ childIndex ] == nullptr )
    {
        m_children[ childIndex ] = new CQuadTreeNode( m_pQuadTree, this );
        ++m_numChildren;
    }
    return m_children[ childIndex ];
}
