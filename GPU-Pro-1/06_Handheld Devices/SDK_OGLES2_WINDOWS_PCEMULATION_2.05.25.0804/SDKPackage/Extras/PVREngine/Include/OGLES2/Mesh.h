/******************************************************************************

 @File         Mesh.h

 @Title        Mesh.

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent - OGL/OGLES/OGLES2 Specific at the moment

 @Description  A class for holding information about renderable meshes.

******************************************************************************/
#ifndef _MESH_H_
#define _MESH_H_

#include "../PVRTools.h"
#include "Globals.h"
#include "BoundingHex.h"
#include "Material.h"

namespace pvrengine
{
	/*!***************************************************************************
	* @Class Mesh
	* @Brief A class for holding information about renderable meshes.
	* @Description A class for holding information about renderable meshes.
	*****************************************************************************/
	class Mesh
	{
	public:

		enum DrawMode
		{
			eNormal = 0,
			eNoFX,
			eWireframe,
			eWireframeNoFX,
			eBounds,
			eNumDrawModes
		}; /*! rendering modes for meshes */

		/*! user readable names for rendering modes */
		const static CPVRTString g_strDrawModeNames[eNumDrawModes];

		/*!***************************************************************************
		@Function			Mesh
		@Description		constructor
		*****************************************************************************/
		Mesh();

		/*!***************************************************************************
		@Function			Mesh
		@Input				psScene
		@Input				psNode
		@Input				psMesh
		@Description		constructor from POD structs
		*****************************************************************************/
		Mesh(CPVRTModelPOD *psScene, SPODNode* psNode, SPODMesh* psMesh);

		/*!***************************************************************************
		@Function			~Mesh
		@Description		destructor
		*****************************************************************************/
		~Mesh();

		/*!***************************************************************************
		@Function			~draw
		@Description		draws the mesh according to whatever drawmode is current
		*****************************************************************************/
		void draw();

		/*!***************************************************************************
		@Function			getNode
		@Return				POD node associated with this mesh
		@Description		Retrieves the POD node associated with this mesh object
		*****************************************************************************/
		SPODNode* getNode(){return m_psNode;}

		/*!***************************************************************************
		@Function			setWorldMatrix
		@Input				mWorld The world matrix to set to
		@Description		Sets the world matrix for this mesh object
		*****************************************************************************/
		void setWorldMatrix(const PVRTMat4& mWorld)
		{
			m_mWorld = mWorld;
		}

		/*!***************************************************************************
		@Function			overlaps
		@Input				sBoundingHex The bounding volume to test against
		@Return				true if this mesh overlaps the passed volume
		@Description		Determines whether there is an intersection between this
		mesh and the passed bounding volume. The test used is rather crude at the moment.
		*****************************************************************************/
		bool overlaps(const BoundingHex& sBoundingHex);

		/*!***************************************************************************
		@Function			setDrawMode
		@Input				eDrawMode the desired rendering method
		@Return				whether the chosen mode has been successfully set
		@Description		Sets how a mesh is rendered
		*****************************************************************************/
		bool setDrawMode(const DrawMode eDrawMode);

		/*!***************************************************************************
		@Function			getDrawMode
		@Description		Gets the draw mode for this mesh
		*****************************************************************************/
		DrawMode getDrawMode() const	{return m_eDrawMode;}

		/*!***************************************************************************
		@Function			getCentre
		@Description		gets the 'centre' point for this mesh
		*****************************************************************************/
		PVRTVec3 getCentre() { return m_vCentreModel;}

		/*!***************************************************************************
		@Function			getRadius
		@Description		Gets the radius for this mesh
		*****************************************************************************/
		VERTTYPE getRadius() { return m_fRadius;}

	private:
		GLuint				m_gluBuffer, m_gluBoundsBuffer;	/*! OpenGL Buffers */
		CPVRTModelPOD		*m_psScene;						/*! POD Scene */
		SPODNode*			m_psNode;						/*! POD node */
		SPODMesh*			m_psMesh;						/*! POD mesh */
		Material*			m_pMaterial;					/*! Material for this Mesh */
		PVRTMat4			m_mWorld;						/*! World matrix for this Mesh */
		BoundingBox			m_BoundingBox;					/*! Bounding box for this Mesh */
		VERTTYPE			m_pfBoundingBuffer[40];			/*! Vertices for bounding box model */
		BoundingHex			m_BoundingHex;					/*! Bounding hex for collisions */
		VERTTYPE			m_fRadius;						/*! Radius of Mesh within whcih all vertices of this mesh exist */
		PVRTVec3			m_vCentreModel;					/*! Centre coordinates for collisions */
		DrawMode			m_eDrawMode;					/*! Render mode for this Mesh */

		bool				m_bSkinned;						/*! Is this mesh skinned */
		GLint				m_gliSkinningLocations[4];		/*! Uniform locations for skinning uniforms */
		enum ESkinningLocations
		{
			eBoneCount = 0,
			eBoneMatrices,
			eBoneMatricesI,
			eBoneMatricesIT
		};/*! enums for skinning uniform locations */
		dynamicArray<SPVRTPFXUniform> m_daUniforms;			/*! Uniforms for this mesh */

		/*!***************************************************************************
		@Function			Init
		@Description		Shared initialisation for Mesh constructors
		*****************************************************************************/
		void	Init();

		/*!***************************************************************************
		@Function			CreateBuffer
		@Description		Creates OpenGL buffers
		*****************************************************************************/
		void	CreateBuffer();

		/*!***************************************************************************
		@Function			ConvertToLines
		@Description		Converts the geometry in this mesh to lines (from triangles)
		so that it can be rendered in a wireframe mode efficiently.
		*****************************************************************************/
		bool	ConvertToLines();

		/*!***************************************************************************
		@Function			ConvertToTriangles
		@Description		Converts the geometry in this mesh to triangles (from lines)
		so that it can be rendered in a non-wireframe mode efficiently.
		*****************************************************************************/
		bool	ConvertToTriangles();

		/*!****************************************************************************
		@Function		DrawMesh
		@Description	Draws the actual geometry for a mesh of triangles
		******************************************************************************/
		void	DrawMesh();

		/*!****************************************************************************
		@Function		DrawWireframeMesh
		@Description	Draws the actual geometry for a mesh of lines
		******************************************************************************/
		void	DrawWireframeMesh();

		/*!****************************************************************************
		@Function		DrawBounds
		@Description	Draws the actual geometry for the bounding boxes
		******************************************************************************/
		void	DrawBounds();

		/*!****************************************************************************
		@Function		DrawSkinned
		@Description	Draws the actual geometry for a mesh with bones
		******************************************************************************/
		void	DrawSkinned();

		/*!***************************************************************************
		@Function			DoBoundaries
		@Description		Calculates various values to determine the boundaries, centre
		of this mesh
		*****************************************************************************/
		void	DoBoundaries();

		/*!***************************************************************************
		@Function			prepareSkinning
		@Description		prepares the mesh for skinning
		*****************************************************************************/
		void	prepareSkinning();

		/*!***************************************************************************
		@Function			>
		@Description		Used for sorting meshes by comparing against Materials used
		*****************************************************************************/
		friend bool operator>(const Mesh& A,const Mesh& B)
		{
			return A.m_pMaterial>B.m_pMaterial;	// just compare material pointers
		}

	};
}
#endif // _MESH_H_

/******************************************************************************
End of file (Mesh.h)
******************************************************************************/
