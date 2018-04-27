/******************************************************************************

 @File         Mesh.cpp

 @Title        Mesh

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Mesh class for PVREngine

******************************************************************************/
#include "Mesh.h"
#include "ContextManager.h"
#include "ConsoleLog.h"
#include "UniformHandler.h"
#include "MaterialManager.h"

namespace pvrengine
{
	

	/******************************************************************************/

	Mesh::~Mesh()
	{
		glDeleteBuffers(1,&m_gluBuffer);
	}

	/******************************************************************************/

	void Mesh::CreateBuffer()
	{
		// ask for a buffer object for this mesh
		glGenBuffers(1,&m_gluBuffer);
		// fill it with juicy data
		glBindBuffer(GL_ARRAY_BUFFER, m_gluBuffer);
		glBufferData(GL_ARRAY_BUFFER,
			m_psMesh->sVertex.nStride*m_psMesh->nNumVertex,
			m_psMesh->pInterleaved,
			GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		// buffer for centre point
		// buffer for bounding box
		// ask for a buffer object 
		glGenBuffers(1,&m_gluBoundsBuffer);
		// fill it with juicy data
		glBindBuffer(GL_ARRAY_BUFFER, m_gluBoundsBuffer);
		glBufferData(GL_ARRAY_BUFFER,
			sizeof(float)*40,
			m_pfBoundingBuffer,
			GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	/******************************************************************************/

	void Mesh::prepareSkinning()
	{
		dynamicArray<Uniform> *pdaSkinningUniforms = m_pMaterial->getSkinningUniforms();
		for(unsigned int i=0;i<pdaSkinningUniforms->getSize();i++)
		{
			switch ((*pdaSkinningUniforms)[i].getSemantic())
			{
			case eUsBONECOUNT:
				m_gliSkinningLocations[eBoneCount] = (*pdaSkinningUniforms)[i].getLocation();
				break;
			case eUsBONEMATRIXARRAY:
				m_gliSkinningLocations[eBoneMatrices] = (*pdaSkinningUniforms)[i].getLocation();
				break;
			case eUsBONEMATRIXARRAYI:
				m_gliSkinningLocations[eBoneMatricesI] = (*pdaSkinningUniforms)[i].getLocation();
				break;
			case eUsBONEMATRIXARRAYIT:
				m_gliSkinningLocations[eBoneMatricesIT] = (*pdaSkinningUniforms)[i].getLocation();
				break;
			default:
				ConsoleLog::inst().log("Unknown skinning semantic found");
			}
		}
	}

	/******************************************************************************/

	void Mesh::draw()
	{

		UniformHandler& sUniformHandler = UniformHandler::inst();


		switch(m_eDrawMode)
		{
		case eNormal:
			{
				glBindBuffer(GL_ARRAY_BUFFER, m_gluBuffer);
				// activates all frame and material uniforms required for mesh
				if(!m_pMaterial->Activate())
					ConsoleLog::inst().log("Material %s failed to activate",m_pMaterial->getName().c_str());

				dynamicArray<Uniform> *pdaMeshUniforms = m_pMaterial->getMeshUniforms();
				for(unsigned int i=0;i<pdaMeshUniforms->getSize();i++)
				{
					sUniformHandler.CalculateMeshUniform((*pdaMeshUniforms)[i],m_psMesh, m_psNode);
				}
				if(m_pMaterial->getSkinned())
					DrawSkinned();
				else
					DrawMesh();

				for(unsigned int j = 0; j < pdaMeshUniforms->getSize(); ++j)
				{
					switch((*pdaMeshUniforms)[j].getSemantic())
					{
					case eUsPosition:
					case eUsNormal:
					case eUsUV:
					case eUsTangent:
					case eUsBinormal:
					case eUsBoneIndex:
					case eUsBoneWeight:
						{
							glDisableVertexAttribArray((*pdaMeshUniforms)[j].getLocation());
						}
					}
				}
			}
			break;
		case eWireframe:
			{
				glBindBuffer(GL_ARRAY_BUFFER, m_gluBuffer);
				// activates all frame and material uniforms required for mesh
				if(!m_pMaterial->Activate())
					ConsoleLog::inst().log("Material %s failed to activate",m_pMaterial->getName().c_str());

				dynamicArray<Uniform> *pdaMeshUniforms = m_pMaterial->getMeshUniforms();
				for(unsigned int i=0;i<pdaMeshUniforms->getSize();i++)
				{
					//(*pdaMeshUniforms)[i].Calculate();
					//(*pdaMeshUniforms)[i].Load();
					sUniformHandler.CalculateMeshUniform((*pdaMeshUniforms)[i],m_psMesh);
				}
				DrawWireframeMesh();
				for(unsigned int j = 0; j < pdaMeshUniforms->getSize(); ++j)
				{
					switch((*pdaMeshUniforms)[j].getSemantic())
					{
					case eUsPosition:
					case eUsNormal:
					case eUsUV:
					case eUsTangent:
					case eUsBinormal:
						{
							glDisableVertexAttribArray((*pdaMeshUniforms)[j].getLocation());
						}
					}
				}

			}
			break;
		case eWireframeNoFX:
			{
				glBindBuffer(GL_ARRAY_BUFFER, m_gluBuffer);
				// activates all frame and material uniforms required for mesh
				Material *pMaterial = MaterialManager::inst().getFlatMaterial();
				if(!pMaterial->Activate())
					ConsoleLog::inst().log("Material %s failed to activate",m_pMaterial->getName().c_str());

				dynamicArray<Uniform> *pdaMeshUniforms = pMaterial->getMeshUniforms();
				for(unsigned int i=0;i<pdaMeshUniforms->getSize();i++)
				{
					//(*pdaMeshUniforms)[i].Calculate();
					//(*pdaMeshUniforms)[i].Load();
					sUniformHandler.CalculateMeshUniform((*pdaMeshUniforms)[i],m_psMesh);
				}
				DrawWireframeMesh();
				for(unsigned int j = 0; j < pdaMeshUniforms->getSize(); ++j)
				{
					switch((*pdaMeshUniforms)[j].getSemantic())
					{
					case eUsPosition:
						{
							glDisableVertexAttribArray((*pdaMeshUniforms)[j].getLocation());
						}
					}
				}
			}
			break;
		case eNoFX:
			{
				glBindBuffer(GL_ARRAY_BUFFER, m_gluBuffer);
				// activates all frame and material uniforms required for mesh
				Material *pMaterial = MaterialManager::inst().getFlatMaterial();
				if(!pMaterial->Activate())
					ConsoleLog::inst().log("Material %s failed to activate",m_pMaterial->getName().c_str());

				dynamicArray<Uniform> *pdaMeshUniforms = pMaterial->getMeshUniforms();
				for(unsigned int i=0;i<pdaMeshUniforms->getSize();i++)
				{
					//(*pdaMeshUniforms)[i].Calculate();
					//(*pdaMeshUniforms)[i].Load();
					sUniformHandler.CalculateMeshUniform((*pdaMeshUniforms)[i],m_psMesh);
				}
				DrawMesh();
				for(unsigned int j = 0; j < pdaMeshUniforms->getSize(); ++j)
				{
					switch((*pdaMeshUniforms)[j].getSemantic())
					{
					case eUsPosition:
						{
							glDisableVertexAttribArray((*pdaMeshUniforms)[j].getLocation());
						}
					}
				}
			}
			break;
		case eBounds:	// always uses flat material so I can hardwire stuff
			{	
				glBindBuffer(GL_ARRAY_BUFFER, m_gluBoundsBuffer);
				// activates all frame and material uniforms required for mesh
				Material *pMaterial = MaterialManager::inst().getFlatMaterial();
				if(!pMaterial->Activate())
					ConsoleLog::inst().log("Material %s failed to activate",m_pMaterial->getName().c_str());
				dynamicArray<Uniform> *pdaMeshUniforms = pMaterial->getMeshUniforms();
				for(unsigned int i=0;i<pdaMeshUniforms->getSize();i++)
				{
					//(*pdaMeshUniforms)[i].Calculate();
					//(*pdaMeshUniforms)[i].Load();
					Uniform sUniform = (*pdaMeshUniforms)[i];
					if(sUniform.getSemantic()!=eUsPosition)
					{
						sUniformHandler.CalculateMeshUniform(sUniform,m_psMesh);
					}
					else
					{
						glVertexAttribPointer(sUniform.getLocation(), 3, GL_FLOAT, GL_FALSE, 0, NULL);
						glEnableVertexAttribArray(sUniform.getLocation());
					}
				}


				DrawBounds();
				for(unsigned int j = 0; j < pdaMeshUniforms->getSize(); ++j)
				{
					if((*pdaMeshUniforms)[j].getSemantic()==eUsPosition)
					{
						glDisableVertexAttribArray((*pdaMeshUniforms)[j].getLocation());
					}
				}


			}
			break;
		}

	}

	/******************************************************************************/

	void Mesh::DrawMesh()
	{
		{
			/*
			Now we give the vertex and texture coordinates data to OpenGL ES.
			The pMesh has been exported with the "Interleave Vectors" check box on,
			so all the data starts at the address pMesh->pInterleaved but with a different offset.
			Interleaved data makes better use of the cache and thus is faster on embedded devices.
			*/

			/*
			The geometry can be exported in 4 ways:
			- Non-Indexed Triangle list
			- Indexed Triangle list
			- Non-Indexed Triangle strips
			- Indexed Triangle strips
			*/
			if(!m_psMesh->nNumStrips)
			{
				if(m_psMesh->sFaces.pData)
				{
					// Indexed Triangle list
					glDrawElements(GL_TRIANGLES, m_psMesh->nNumFaces*3,
						GL_UNSIGNED_SHORT, m_psMesh->sFaces.pData);
				}
				else
				{
					// Non-Indexed Triangle list
					glDrawArrays(GL_TRIANGLES, 0, m_psMesh->nNumFaces*3);
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
						glDrawElements(GL_TRIANGLE_STRIP, m_psMesh->pnStripLength[i]+2,
							GL_UNSIGNED_SHORT, m_psMesh->sFaces.pData + offset*2);
						offset += m_psMesh->pnStripLength[i]+2;
					}
				}
				else
				{
					// Non-Indexed Triangle strips
					int offset = 0;
					for(int i = 0; i < (int)m_psMesh->nNumStrips; i++)
					{
						glDrawArrays(GL_TRIANGLE_STRIP, offset, m_psMesh->pnStripLength[i]+2);
						offset += m_psMesh->pnStripLength[i]+2;
					}
				}
			}
		}
	}

	/******************************************************************************/

	void Mesh::DrawWireframeMesh()
	{
		{
			/*
			Now we give the vertex and texture coordinates data to OpenGL ES.
			The pMesh has been exported with the "Interleave Vectors" check box on,
			so all the data starts at the address pMesh->pInterleaved but with a different offset.
			Interleaved data makes better use of the cache and thus is faster on embedded devices.
			*/

			/*
			The geometry can be exported in 4 ways:
			- Non-Indexed Triangle list
			- Indexed Triangle list
			- Non-Indexed Triangle strips
			- Indexed Triangle strips
			*/
			// Indexed Triangle list
			glDrawElements(GL_LINES, m_psMesh->nNumFaces*6,
				GL_UNSIGNED_SHORT, m_psMesh->sFaces.pData);
		}
	}

	/******************************************************************************/

	void Mesh::DrawSkinned()
	{
		/*
		There is a limit to the number of bone matrices that you can pass to the shader so we have
		chosen to limit the number of bone matrices that affect a mesh to 8. However, this does
		not mean our character can only have a skeleton consisting of 8 bones. We can get around
		this by using bone batching where the character is split up into sub-meshes that are only
		affected by a sub set of the overal skeleton. This is why we have this for loop that
		iterates through the bone batches contained within the SPODMesh.
		*/


		for (int i32Batch = 0; i32Batch <m_psMesh->sBoneBatches.nBatchCnt; ++i32Batch)
		{
			/*
			If the current mesh has bone index and weight data then we need to
			set up some additional variables in the shaders.
			*/
			if(m_psMesh->sBoneIdx.pData && m_psMesh->sBoneWeight.pData)
			{
				// Set the number of bones that will influence each vertex in the mesh
				glUniform1i(m_gliSkinningLocations[eBoneCount], m_psMesh->sBoneIdx.n);

				// Go through the bones for the current bone batch
				PVRTMat4 amBoneWorld[8];
				PVRTMat3 amBoneWorldIT[8];
				for(int i = 0; i < m_psMesh->sBoneBatches.pnBatchBoneCnt[i32Batch]; i++)
				{
					// Get the Node of the bone
					int i32NodeID = m_psMesh->sBoneBatches.pnBatches[i32Batch * m_psMesh->sBoneBatches.nBatchBoneMax + i];

					// Get the World transformation matrix for this bone
					amBoneWorld[i] = m_psScene->GetBoneWorldMatrix(*m_psNode, m_psScene->pNode[i32NodeID]);

					// Calculate the inverse transpose of the 3x3 rotation/scale part for correct lighting
					amBoneWorldIT[i] = PVRTMat3(amBoneWorld[i].inverse().transpose());
				}

				glUniformMatrix4fv(m_gliSkinningLocations[eBoneMatrices], 8, GL_FALSE, amBoneWorld[0].ptr());
				glUniformMatrix3fv(m_gliSkinningLocations[eBoneMatricesIT], 8, GL_FALSE, amBoneWorldIT[0].f);
			}
			else
			{
				glUniform1i(m_gliSkinningLocations[eBoneCount], 0);
			}
			/*
			As we are using bone batching we don't want to draw all the faces contained within pMesh, we only want
			to draw the ones that are in the current batch. To do this we pass to the drawMesh function the offset
			to the start of the current batch of triangles (Mesh.sBoneBatches.pnBatchOffset[i32Batch]) and the
			total number of triangles to draw (i32Tris)
			*/
			int i32Tris;
			if(i32Batch+1 < m_psMesh->sBoneBatches.nBatchCnt)
				i32Tris = m_psMesh->sBoneBatches.pnBatchOffset[i32Batch+1] - m_psMesh->sBoneBatches.pnBatchOffset[i32Batch];
			else
				i32Tris = m_psMesh->nNumFaces - m_psMesh->sBoneBatches.pnBatchOffset[i32Batch];

			glDrawElements(GL_TRIANGLES, i32Tris*3,
				GL_UNSIGNED_SHORT,
				&((unsigned short*)m_psMesh->sFaces.pData)[3*m_psMesh->sBoneBatches.pnBatchOffset[i32Batch]]);
		}
	}

	const unsigned short pBoundingBoxIndices[40] =
	{
		0,1, 1,3, 3,2, 2,0, 0,4, 4,6, 6,2, 4,5, 5,7, 7,6, 5,1, 3,7,
			8,0, 8,1, 8,2, 8,3, 8,4, 8,5, 8,6, 8,7
	};

	/******************************************************************************/

	void Mesh::DrawBounds()
	{
		// Indexed Triangle list
		glDrawElements(GL_LINES,40,GL_UNSIGNED_SHORT,pBoundingBoxIndices);
	}
}

/******************************************************************************
End of file (Mesh.cpp)
******************************************************************************/
