/******************************************************************************

 @File         MeshManager.h

 @Title        A simple nesh manager for use with PVREngine

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent - OGL/OGLES/OGLES2 Specific at the moment

 @Description  A class for holding information about meshes as they are loaded so
               that duplicate meshes are not kept in memory and so the things can
               be disposed of easily at the end of execution.

******************************************************************************/
#ifndef MESHMANAGER_H
#define MESHMANAGER_H

/******************************************************************************
Includes
******************************************************************************/

#include "../PVRTools.h"
#include "Mesh.h"

namespace pvrengine
{
	/*!***************************************************************************
	* @Class MeshManager
	* @Brief A class for managing meshes.
	* @Description A class for managing meshes.
	*****************************************************************************/
	class MeshManager
	{
	public:
		/*!***************************************************************************
		@Function			MeshManager
		@Description		blank constructor.
		*****************************************************************************/
		MeshManager();

		/*!***************************************************************************
		@Function			~MaterialManager
		@Description		destructor.
		*****************************************************************************/
		~MeshManager();

		/*!***************************************************************************
		@Function			addMesh
		@Input				sScene	a PODScene
		@Input				pNode	this node
		@Input				pMesh	this mesh
		@Description		Adds the mesh specified by the parameters to the manager.
		*****************************************************************************/
		void addMesh(CPVRTModelPOD& sScene, SPODNode* pNode, SPODMesh* pMesh);

		/*!***************************************************************************
		@Function			setDrawMode
		@Input				eDrawMode	the desired drawmode
		@Description		Blanket sets the drawng modes for all the meshes held
		within the manager.
		*****************************************************************************/
		void setDrawMode(const Mesh::DrawMode eDrawMode);

		/*!***************************************************************************
		@Function			getDrawMode
		@Input				i32MeshNum index of mesh
		@Description		Retrieves the current drawing mode of the specified
		mesh.
		*****************************************************************************/
		Mesh::DrawMode getDrawMode(const unsigned int i32MeshNum=0) const;

		/*!***************************************************************************
		@Function			getDrawModeName
		@Input				eDrawMode	the desired drawmode
		@Return				A string for the passed drawmode
		@Description		Retrieves a human readable string for the given DrawMode
		*****************************************************************************/
		CPVRTString getDrawModeName(const Mesh::DrawMode eDrawMode) const;

		/*!***************************************************************************
		@Function			sort
		@Description		Sorts the meshes according to their materials
		*****************************************************************************/
		void sort();

		/*!***************************************************************************
		@Function			getMeshes
		@Return				the array of meshes
		@Description		Blanket sets the drawng modes for all the meshes held
		within the manager.
		*****************************************************************************/
		dynamicArray<Mesh*>* getMeshes();

	private:

		dynamicArray<Mesh*> m_daMeshes;	/*! the actual collection of meshes */

		/*!***************************************************************************
		@Function			sort
		@Description		shared initialisation function for constructors
		*****************************************************************************/
		void Init();

	};

}
#endif // MESHMANAGER_H

/******************************************************************************
End of file (MeshManager.h)
******************************************************************************/
