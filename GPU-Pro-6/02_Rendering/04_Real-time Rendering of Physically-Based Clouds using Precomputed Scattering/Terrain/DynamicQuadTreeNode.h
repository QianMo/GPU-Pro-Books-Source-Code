// Copyright 2013 Intel Corporation
// All Rights Reserved
//
// Permission is granted to use, copy, distribute and prepare derivative works of this
// software for any purpose and without fee, provided, that the above copyright notice
// and this statement appear in all copies.  Intel makes no representations about the
// suitability of this software for any purpose.  THIS SOFTWARE IS PROVIDED "AS IS."
// INTEL SPECIFICALLY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, AND ALL LIABILITY,
// INCLUDING CONSEQUENTIAL AND OTHER INDIRECT DAMAGES, FOR THE USE OF THIS SOFTWARE,
// INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PROPRIETARY RIGHTS, AND INCLUDING THE
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  Intel does not
// assume any responsibility for any errors which may appear in this software nor any
// responsibility to update it.

#pragma once

#include <memory>

// Structure describing quad tree node location
struct SQuadTreeNodeLocation
{
    // Position in a tree
	int horzOrder;
	int vertOrder;
	int level;

	SQuadTreeNodeLocation(int h, int v, int l)
		: horzOrder(h)
		, vertOrder(v)
		, level(l)
	{
		assert(h < (1 << l));
		assert(v < (1 << l));
	}
	SQuadTreeNodeLocation()
		: horzOrder(0)
		, vertOrder(0)
		, level(0)
	{}

    // Gets location of a child
	inline friend SQuadTreeNodeLocation GetChildLocation(const SQuadTreeNodeLocation &parent,
		unsigned int siblingOrder)
	{
		return SQuadTreeNodeLocation(parent.horzOrder * 2 + (siblingOrder&1),
			parent.vertOrder * 2 + (siblingOrder>>1),
			parent.level + 1);
	}

    // Gets location of a parent
	inline friend SQuadTreeNodeLocation GetParentLocation(const SQuadTreeNodeLocation &node)
	{
		assert(node.level > 0);
		return SQuadTreeNodeLocation(node.horzOrder / 2, node.vertOrder / 2, node.level - 1);
	}
};

// Base class for iterators traversing the quad tree
class HierarchyIteratorBase
{
public:
	operator const SQuadTreeNodeLocation& () const { return m_current; }
	int Level() const { return m_current.level; }
	int Horz()  const { return m_current.horzOrder; }
	int Vert()  const { return m_current.vertOrder; }
protected:
	SQuadTreeNodeLocation m_current;
	int m_currentLevelSize;
};

// Iterator for recursively traversing the quad tree starting from the root up to the specified level
class HierarchyIterator : public HierarchyIteratorBase
{
public:
	HierarchyIterator(int nLevels)
		: m_nLevels(nLevels)
	{
		m_currentLevelSize = 1;
	}
	bool IsValid() const { return m_current.level < m_nLevels; }
	void Next()
	{
		if( ++m_current.horzOrder == m_currentLevelSize )
		{
			m_current.horzOrder = 0;
			if( ++m_current.vertOrder == m_currentLevelSize )
			{
				m_current.vertOrder = 0;
				m_currentLevelSize = 1 << ++m_current.level;
			}
		}
	}

private:
	int m_nLevels;
};

// Iterator for recursively traversing the quad tree starting from the specified level up to the root
class HierarchyReverseIterator : public HierarchyIteratorBase
{
public:
	HierarchyReverseIterator(int nLevels)
	{
		m_current.level = nLevels - 1;
		m_currentLevelSize = 1 << m_current.level;
	}
	bool IsValid() const { return m_current.level >= 0; }
	void Next()
	{
		if( ++m_current.horzOrder == m_currentLevelSize )
		{
			m_current.horzOrder = 0;
			if( ++m_current.vertOrder == m_currentLevelSize )
			{
				m_current.vertOrder = 0;
				m_currentLevelSize = 1 << --m_current.level;
			}
		}
	}
};

// Template class for the node of a dynamic quad tree
template<typename NodeDataType>
class CDynamicQuadTreeNode
{
public:
    CDynamicQuadTreeNode() : 
        m_pAncestor(NULL)
    {
    }

    NodeDataType &GetData(){return m_Data;}
    const NodeDataType &GetData()const{return m_Data;}

    CDynamicQuadTreeNode *GetAncestor() const        { return m_pAncestor; }
    void GetDescendants(const CDynamicQuadTreeNode* &LBDescendant,
                        const CDynamicQuadTreeNode* &RBDescendant,
                        const CDynamicQuadTreeNode* &LTDescendant,
                        const CDynamicQuadTreeNode* &RTDescendant) const
    {
        LBDescendant = m_pLBDescendant.get();
        RBDescendant = m_pRBDescendant.get();
        LTDescendant = m_pLTDescendant.get();
        RTDescendant = m_pRTDescendant.get();
    }

    void GetDescendants(CDynamicQuadTreeNode* &LBDescendant,
                        CDynamicQuadTreeNode* &RBDescendant,
                        CDynamicQuadTreeNode* &LTDescendant,
                        CDynamicQuadTreeNode* &RTDescendant)
    {
        LBDescendant = m_pLBDescendant.get();
        RBDescendant = m_pRBDescendant.get();
        LTDescendant = m_pLTDescendant.get();
        RTDescendant = m_pRTDescendant.get();
    }

    typedef std::auto_ptr<CDynamicQuadTreeNode<NodeDataType> > AutoPtrType;
    // Attahes specified descendants to the tree
    void CreateDescendants(AutoPtrType pLBDescendant,
                           AutoPtrType pRBDescendant,
                           AutoPtrType pLTDescendant,
                           AutoPtrType pRTDescendant);
    // Creates descendants UNATTACHED to the tree
    void CreateFloatingDescendants(AutoPtrType &pLBDescendant,
                                   AutoPtrType &pRBDescendant,
                                   AutoPtrType &pLTDescendant,
                                   AutoPtrType &pRTDescendant);
    // Destroys ALL descendants for the node
    void DestroyDescendants();

	const SQuadTreeNodeLocation& GetPos() const { return m_pos; }

	void SetPos(const SQuadTreeNodeLocation& pos){ m_pos = pos; }

private:
    CDynamicQuadTreeNode(CDynamicQuadTreeNode *pAncestor, int iSiblingOrder) : 
        m_pAncestor(pAncestor),
        m_pos(GetChildLocation(pAncestor->m_pos, iSiblingOrder))
    {
    }

    NodeDataType m_Data;

    std::auto_ptr< CDynamicQuadTreeNode > m_pLBDescendant;
    std::auto_ptr< CDynamicQuadTreeNode > m_pRBDescendant;
    std::auto_ptr< CDynamicQuadTreeNode > m_pLTDescendant;
    std::auto_ptr< CDynamicQuadTreeNode > m_pRTDescendant;
    CDynamicQuadTreeNode *m_pAncestor;

    SQuadTreeNodeLocation m_pos;
};

template<typename NodeDataType>
void CDynamicQuadTreeNode<NodeDataType>::CreateFloatingDescendants(AutoPtrType &pLBDescendant,
                                                                   AutoPtrType &pRBDescendant,
                                                                   AutoPtrType &pLTDescendant,
                                                                   AutoPtrType &pRTDescendant)
{
    pLBDescendant.reset(new CDynamicQuadTreeNode<NodeDataType>(this, 0));
    pRBDescendant.reset(new CDynamicQuadTreeNode<NodeDataType>(this, 1));
    pLTDescendant.reset(new CDynamicQuadTreeNode<NodeDataType>(this, 2));
    pRTDescendant.reset(new CDynamicQuadTreeNode<NodeDataType>(this, 3));
}

template<typename NodeDataType>
void CDynamicQuadTreeNode<NodeDataType>::CreateDescendants(AutoPtrType pLBDescendant,
                                                           AutoPtrType pRBDescendant,
                                                           AutoPtrType pLTDescendant,
                                                           AutoPtrType pRTDescendant)
{
    assert( !m_pLBDescendant.get() );
    assert( !m_pRBDescendant.get() );
    assert( !m_pLTDescendant.get() );
    assert( !m_pRTDescendant.get() );

    m_pLBDescendant = pLBDescendant;
    m_pRBDescendant = pRBDescendant;
    m_pLTDescendant = pLTDescendant;
    m_pRTDescendant = pRTDescendant;
}

template<typename NodeDataType>
void CDynamicQuadTreeNode<NodeDataType>::DestroyDescendants()
{
    if( m_pLBDescendant.get() )m_pLBDescendant->DestroyDescendants();
    if( m_pRBDescendant.get() )m_pRBDescendant->DestroyDescendants();
    if( m_pLTDescendant.get() )m_pLTDescendant->DestroyDescendants();
    if( m_pRTDescendant.get() )m_pRTDescendant->DestroyDescendants();
    
    m_pLBDescendant.reset();
    m_pRBDescendant.reset();
    m_pLTDescendant.reset();
    m_pRTDescendant.reset();
}
