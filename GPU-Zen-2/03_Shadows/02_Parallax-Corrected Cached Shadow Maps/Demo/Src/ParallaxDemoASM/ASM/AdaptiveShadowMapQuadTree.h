#pragma once

#include "AdaptiveShadowMap.h"

#include "../Misc/AABBox.h"

class CAdaptiveShadowMap::CQuadTree : public MathLibObject
{
public:
    CQuadTree();
    ~CQuadTree();
    void Reset();

    CQuadTreeNode* FindRoot( const CAABBox& BBox );

    const std::vector<CQuadTreeNode*>& GetRoots() const { return m_roots; }

protected:
    std::vector<CQuadTreeNode*> m_roots;

    friend class CQuadTreeNode;
};

class CAdaptiveShadowMap::CQuadTreeNode : public MathLibObject
{
public:
    CAABBox m_BBox;
    unsigned int m_lastFrameVerified;

    CQuadTreeNode* m_pParent;
    CQuadTreeNode* m_children[4];
    CTileCacheEntry* m_pTile;
    CTileCacheEntry* m_pLayerTile;
    unsigned char m_refinement;
    unsigned char m_numChildren;

    CQuadTreeNode( CQuadTree* pQuadTree, CQuadTreeNode* pParent );
    ~CQuadTreeNode();

    const CAABBox GetChildBBox( int childIndex );
    CQuadTreeNode* AddChild( int childIndex );

    CTileCacheEntry*& GetTile() { return m_pTile; }
    CTileCacheEntry*& GetLayerTile() { return m_pLayerTile; }

    CTileCacheEntry* GetTile( bool isLayer ) const { return isLayer ? m_pLayerTile : m_pTile; }

protected:
    CQuadTree* m_pQuadTree;
    int m_rootNodesIndex;
};
