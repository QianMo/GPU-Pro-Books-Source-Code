/******************************************************************************

 @File         MeshBase.cpp

 @Title        MeshBase

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  API independent source file for Mesh class

******************************************************************************/
#include "Mesh.h"
#include "MaterialManager.h"
#include "ConsoleLog.h"

#ifdef _UITRON_
template void PVRTswap<pvrengine::Mesh*>(pvrengine::Mesh*&,pvrengine::Mesh*&);
#endif

namespace pvrengine
{
	

	const CPVRTString Mesh::g_strDrawModeNames[Mesh::eNumDrawModes] = {
		"Normal",
			"No FX",
			"Wireframe",
			"Wireframe with No FX",
			"Boundaries"
	};


	/******************************************************************************/

	Mesh::Mesh()
	{
		Init();
	}

	/******************************************************************************/

	Mesh::Mesh(CPVRTModelPOD* psScene, SPODNode* psNode, SPODMesh* psMesh)
	{
		Init();
		m_psScene = psScene;
		m_psMesh = psMesh;
		m_psNode = psNode;

		m_pMaterial = MaterialManager::inst().getMaterial(psNode->nIdxMaterial);
		m_bSkinned = m_pMaterial->getSkinned();
		if(m_bSkinned)
			prepareSkinning();

		DoBoundaries();
		CreateBuffer();

	}

	/******************************************************************************/

	void Mesh::Init()
	{
		m_eDrawMode = eNormal;
	}

	/******************************************************************************/

	bool Mesh::setDrawMode(const DrawMode eDrawMode)
	{
		if(m_eDrawMode==eDrawMode)
			return true;

		switch (eDrawMode)
		{
		case eNormal:
		case eNoFX:
		//case eDepthComplexity:
			if(m_psMesh->ePrimitiveType!=ePODTriangles)
				if(!ConvertToTriangles())
					return false;
			break;
		case eWireframe:
		case eWireframeNoFX:
			if(m_psMesh->ePrimitiveType!=ePODLines)
				if(!ConvertToLines())
					return false;
			break;
		case eBounds:
			break;
		default:
			{	// not a recognised drawMode
				return false;
			}
		}

		m_eDrawMode = eDrawMode;
		return true;
	}

	/******************************************************************************/

	bool Mesh::ConvertToLines()
	{
		// triangles have 3 corners	* lines have 2 ends


		if(!m_psMesh->nNumStrips)
		{
			if(m_psMesh->sFaces.pData)
			{
				// Indexed Triangle list

				unsigned int u32IndexCount = m_psMesh->nNumFaces*6;

				unsigned short* pNewIndices = (unsigned short*)malloc(u32IndexCount*sizeof(unsigned short));
				if(!pNewIndices)
				{
					ConsoleLog::inst().log("Could not allocate memory for conversion to lines for mesh: %s",
						m_psNode->pszName);
					return false;
				}
				unsigned short* pSrcPointer = (unsigned short*)m_psMesh->sFaces.pData,
					*pDstPointer = pNewIndices;

				// convert indices to indexed line list
				for(unsigned int i=0;i<m_psMesh->nNumFaces;i++)
				{	// tri ABC lines ABBCCA
					*pDstPointer++=*pSrcPointer++;			// A
					*pDstPointer++=*pSrcPointer;				// B

					*pDstPointer++=*pSrcPointer++;			// B
					*pDstPointer++=*pSrcPointer;				// C

					*pDstPointer++=*pSrcPointer++;			// C
					*pDstPointer++=*(pSrcPointer-3);			// A
				}
				// change mesh to use new indices
				free(m_psMesh->sFaces.pData);
				m_psMesh->sFaces.pData = (unsigned char*)pNewIndices;
				m_psMesh->ePrimitiveType = ePODLines;
			}
			else
			{
				// Non-Indexed Triangle list
			}
		}
		else
		{
			if(m_psMesh->sFaces.pData)
			{
				//// Indexed Triangle strips
				//int offset = 0;
				//for(int i = 0; i < (int)m_psMesh->nNumStrips; i++)
				//{
				//	offset += m_psMesh->pnStripLength[i]+2;
				//}
			}
			else
			{
				//// Non-Indexed Triangle strips
				//int offset = 0;
				//for(int i = 0; i < (int)m_psMesh->nNumStrips; i++)
				//{
				//	offset += m_psMesh->pnStripLength[i]+2;
				//}
			}
		}
		return true;
	}

	/******************************************************************************/

	bool Mesh::ConvertToTriangles()
	{
		if(!m_psMesh->nNumStrips)
		{
			if(m_psMesh->sFaces.pData)
			{
				// Indexed Line list

				// Triangles a have 3 corners on the whole
				unsigned int u32IndexCount = m_psMesh->nNumFaces*3;

				unsigned char* pNewIndices = (unsigned char*)malloc(u32IndexCount*sizeof(unsigned char)*2);
				if(!pNewIndices)
				{
					ConsoleLog::inst().log("Could not allocate memory for conversion to triangles for mesh: %s",
						m_psNode->pszName);
					return false;
				}
				// convert indices to indexed triangle list
				unsigned short* pSrcPointer = (unsigned short*)m_psMesh->sFaces.pData,
					*pDstPointer = (unsigned short*)pNewIndices;

				for(unsigned int i=0;i<m_psMesh->nNumFaces;i++)
				{	// tri ABC lines ABBCCA
					*pDstPointer++=*pSrcPointer++;			// A

					*pDstPointer++=*pSrcPointer++;			// B
					pSrcPointer+=2;

					*pDstPointer++=*pSrcPointer;			// C
					pSrcPointer+=2;
				}
				// change mesh to use new indices
				free(m_psMesh->sFaces.pData);
				m_psMesh->sFaces.pData = pNewIndices;
				m_psMesh->ePrimitiveType = ePODTriangles;
			}
			else
			{
				// Non-Indexed Triangle list
			}
		}
		else
		{
			if(m_psMesh->sFaces.pData)
			{
				// Indexed Triangle strips
				int offset = 0;
				for(int i = 0; i < (int)m_psMesh->nNumStrips; i++)
				{
					offset += m_psMesh->pnStripLength[i]+2;
				}
			}
			else
			{
				// Non-Indexed Triangle strips
				int offset = 0;
				for(int i = 0; i < (int)m_psMesh->nNumStrips; i++)
				{
					offset += m_psMesh->pnStripLength[i]+2;
				}
			}
		}

		return true;
	}

	/******************************************************************************/

	void Mesh::DoBoundaries()
	{
		m_BoundingBox = BoundingBox(m_psMesh->pInterleaved,
			m_psMesh->nNumVertex,
			(int)m_psMesh->sVertex.pData,
			m_psMesh->sVertex.nStride);

		// find centre
		m_vCentreModel = PVRTVec3(f2vt(0.0f),f2vt(0.0f),f2vt(0.0f));

		for(unsigned int i=0;i<8;i++)
		{
			m_vCentreModel.x+=m_BoundingBox.vPoint[i].x;
			m_vCentreModel.y+=m_BoundingBox.vPoint[i].y;
			m_vCentreModel.z+=m_BoundingBox.vPoint[i].z;
		}
		m_vCentreModel.x = VERTTYPEDIV(m_vCentreModel.x,8);
		m_vCentreModel.y = VERTTYPEDIV(m_vCentreModel.y,8);
		m_vCentreModel.z = VERTTYPEDIV(m_vCentreModel.z,8);

		// find radius
		VERTTYPE fRadiusSquared = f2vt(0.0f);

		PVRTVec3 vec;

		for(unsigned int i=1;i<8;i++)
		{
			vec = PVRTVec3(m_BoundingBox.vPoint[i].x-m_vCentreModel.x,
				m_BoundingBox.vPoint[i].y-m_vCentreModel.y,
				m_BoundingBox.vPoint[i].z-m_vCentreModel.z);
			VERTTYPE fCandidateMagSquared = vec.lenSqr();
			if(fRadiusSquared<fCandidateMagSquared)
			{
				fRadiusSquared = fCandidateMagSquared;
			}
		}
		m_fRadius = f2vt((float) sqrt(vt2f(fRadiusSquared)));

		//make display buffer
		VERTTYPE *pfBufferPointer = m_pfBoundingBuffer;
		for(unsigned int i=0;i<8;++i)
		{
			*pfBufferPointer++ = m_BoundingBox.vPoint[i].x;
			*pfBufferPointer++ = m_BoundingBox.vPoint[i].y;
			*pfBufferPointer++ = m_BoundingBox.vPoint[i].z;
		}

		*pfBufferPointer++ = m_vCentreModel.x;
		*pfBufferPointer++ = m_vCentreModel.y;
		*pfBufferPointer++ = m_vCentreModel.z;


	}

}

/******************************************************************************
End of file (Mesh.cpp)
******************************************************************************/
