/******************************************************************************

 @File         MeshManager.cpp

 @Title        Introducing the POD 3d file format

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Shows how to use the pfx format

******************************************************************************/
#include "MeshManager.h"

namespace pvrengine
{
	

	/******************************************************************************/

	MeshManager::MeshManager()
	{
		Init();
	}

	/******************************************************************************/

	void MeshManager::addMesh(CPVRTModelPOD& sScene, SPODNode* pNode, SPODMesh* pMesh)
	{
		m_daMeshes.append(new Mesh(&sScene, pNode, pMesh));
	}

	/******************************************************************************/

	void MeshManager::setDrawMode(const Mesh::DrawMode eDrawMode)
	{
		for(unsigned int i=0;i<m_daMeshes.getSize();i++)
		{
			if(m_daMeshes[i]->getDrawMode()!=eDrawMode)
				m_daMeshes[i]->setDrawMode(eDrawMode);
		}
	}

	/******************************************************************************/

	Mesh::DrawMode MeshManager::getDrawMode(const unsigned int i32MeshNum) const
	{
		return m_daMeshes[i32MeshNum]->getDrawMode();
	}

	/******************************************************************************/

	CPVRTString MeshManager::getDrawModeName(const Mesh::DrawMode eDrawMode) const
	{
		return Mesh::g_strDrawModeNames[eDrawMode];
	}

	/******************************************************************************/

	dynamicArray<Mesh*>* MeshManager::getMeshes()
	{
		return &m_daMeshes;
	}

	/******************************************************************************/

	MeshManager::~MeshManager()
	{
		for(unsigned int i=0;i<m_daMeshes.getSize();++i)
		{
			PVRDELETE(m_daMeshes[i]);
		}
	}

	/******************************************************************************/

	void MeshManager::Init()
	{
	}


	/******************************************************************************/

	void MeshManager::sort()
	{
		m_daMeshes.bubbleSortPointers();
	}


}

/******************************************************************************
End of file (MeshManager.cpp)
******************************************************************************/
