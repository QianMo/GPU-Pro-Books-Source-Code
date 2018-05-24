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

#include <vector>
#include "DynamicQuadTreeNode.h"

// Template class implementing hierarchy array, which is a quad tree indexed by 
// quad tree node location
template <class T>
class HierarchyArray
{
public:
	T& operator [] (const SQuadTreeNodeLocation &at)
	{
		return m_data[at.level][at.horzOrder + (at.vertOrder << at.level)];
	}
	const T& operator [] (const SQuadTreeNodeLocation &at) const
	{
		return m_data[at.level][at.horzOrder + (at.vertOrder << at.level)];
	}

	void Resize(size_t numLevelsInHierarchy)
	{
		m_data.resize(numLevelsInHierarchy);
		if( numLevelsInHierarchy )
		{
			for(size_t level = numLevelsInHierarchy; level--; )
			{
				size_t numElementsInLevel = 1 << level;
				m_data[level].resize(numElementsInLevel*numElementsInLevel);
			}
		}
	}

	bool Empty() const
	{
		return m_data.empty();
	}

private:
	std::vector<std::vector<T> > m_data;
};


// end of file
