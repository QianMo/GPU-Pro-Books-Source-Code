/******************************************************************************

 @File         UniformHandler.h

 @Title        A class for handling shader uniforms.

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent/OGLES2

 @Description  A class for handling shader uniforms.

******************************************************************************/
#ifndef UNIFORMHANDLER_H
#define UNIFORMHANDLER_H

#include "../PVRTSingleton.h"
#include "LightManager.h"
#include "BoundingBox.h"
#include "Material.h"
#include "Semantics.h"

struct SPVRTPFXUniform;
class CPVRTModelPOD;
class CPVRTPFXEffect;


namespace pvrengine
{

	class Uniform;

	const int i32NUM_LIGHTS  = 32;

	/*!***************************************************************************
	* @Class UniformHandler
	* @Brief 	A class for handling shader uniforms.
	* @Description 	A class for handling shader uniforms.
	*****************************************************************************/
	class UniformHandler : public CPVRTSingleton<UniformHandler>
	{
	public:
		/*!****************************************************************************
		@Function		UniformHandler
		@Description	Blank constructor
		******************************************************************************/
		UniformHandler()	{ResetFrameUniforms();}

		/*!****************************************************************************
		@Function		~UniformHandler
		@Description	Destructor
		******************************************************************************/
		~UniformHandler(){}

		/*!****************************************************************************
		@Function		getProjection
		@Return			Projection matrix
		@Description	Accessor for projection matrix
		******************************************************************************/
		PVRTMat4 getProjection()
		{return m_mProjection;}

		/*!****************************************************************************
		@Function		setProjection
		@Input			fFOV - the field of view
		@Input			fAspectRatio - width vs height ratio of viewport
		@Input			fNear - near plane of view frustum
		@Input			fFar - far plane of view frustum
		@Input			bRotate - true for portrait mode
		@Description	Calculates the  projection matrix from passed values.
		******************************************************************************/
		void setProjection(const VERTTYPE fFOV,
			const VERTTYPE fAspectRatio,
			const VERTTYPE fNear,
			const VERTTYPE fFar,
			const bool bRotate);

		/*!****************************************************************************
		@Function		getView
		@Return			View matrix
		@Description	Accessor for View matrix
		******************************************************************************/
		PVRTMat4 getView();

		/*!****************************************************************************
		@Function		setView
		@Input			vFrom - position coordinates of viewer
		@Input			vTo - vector indicating the centre of the view
		@Input			vUp - vector indicating the orientation of the view
		@Description	Calculates the view matrix from passed values.
		******************************************************************************/
		void setView(const PVRTVec3& vFrom, const PVRTVec3& vTo, const PVRTVec3& vUp);

		/*!****************************************************************************
		@Function		getWorld
		@Return			World matrix
		@Description	Accessor for World matrix
		******************************************************************************/
		PVRTMat4 getWorld();

		/*!****************************************************************************
		@Function		setScene
		@Input			psScene PODScene currently being rendered
		@Description	Accessor for projection matrix
		******************************************************************************/
		void setScene(CPVRTModelPOD *psScene);

		/*!****************************************************************************
		@Function		setFrame
		@Input			fFrame - this POD's current frame of animation
		@Description	Accessor for projection matrix
		******************************************************************************/
		void setFrame(float fFrame);

		/*!****************************************************************************
		@Function		DoFrameUniform
		@Input			sUniform
		@Description	Calculates and binds a uniform that applies to the entire frame
		******************************************************************************/
		void DoFrameUniform(const Uniform& sUniform);

		/*!****************************************************************************
		@Function		ResetFrameUniforms
		@Description	Clears the record of which frame uniforms have been calculated
		so that they will be calculated again for a new frame.
		******************************************************************************/
		void ResetFrameUniforms();

		/*!****************************************************************************
		@Function		CalculateMeshUniform
		@Input			sUniform
		@Input			pMesh
		@Input			pNode
		@Description	Calculates and binds a uniform that is valid for this mesh
		******************************************************************************/
		void CalculateMeshUniform(const Uniform& sUniform, SPODMesh *pMesh, SPODNode *pNode=NULL);

		/*!****************************************************************************
		@Function		CalculateFrameUniform
		@Input			sUniform
		@Description	Calculates the value of a uniform that applies to this entire frame
		******************************************************************************/
		void CalculateFrameUniform(const Uniform& sUniform);

		/*!****************************************************************************
		@Function		BindFrameUniform
		@Input			sUniform
		@Description	Binds a frame uniform
		******************************************************************************/
		void BindFrameUniform(const Uniform& sUniform);

		/*!****************************************************************************
		@Function		CalculateMaterialUniform
		@Input			sUniform
		@Input			sMaterial
		@Description	Calculates and binds a uniform that applies to any mesh using
		this material.
		******************************************************************************/
		void CalculateMaterialUniform(const Uniform& sUniform, Material& sMaterial);

		/*!****************************************************************************
		@Function		setLightManager
		@Input			pLightManager
		@Description
		******************************************************************************/
		void setLightManager(LightManager *pLightManager){m_pLightManager = pLightManager;}

		/*!****************************************************************************
		@Function		setWorld
		@Input			mWorld world matrix
		@Description	sets the world matric for current section of render. Calculates
		WorldViewProjection matrix at the same time.
		******************************************************************************/
		void setWorld(const PVRTMat4& mWorld)
		{
			m_mWorld = mWorld;
			m_mWorldView = m_mView * m_mWorld;
			m_mWorldViewProjection = m_mProjection * m_mWorldView;
		}

		/*!****************************************************************************
		@Function		isVisibleSphere
		@Input			v3Centre centre of sphere to be checked
		@Input			fRadius radius of sphere to be checked
		@Description	Checks if the sphere defined by the centre coordinates and 
		of the passed radius is visible according to the current projection matrix.
		Requires the world, view and projection matrices to defined in order to function
		correctly.
		******************************************************************************/
		bool isVisibleSphere(const PVRTVec3& v3Centre, const VERTTYPE fRadius);

	private:
		/*!* View details */
		VERTTYPE		m_fNear, m_fFar, m_fFOV, m_fAspectRatio;
		bool			m_bRotate;

		/*!* Current POD scene and frame*/
		CPVRTModelPOD	*m_psScene;
		float			m_fFrame;
		float			m_fAnimation;

		/*!* flags to store whether frame uniforms are already calculated or not */
		unsigned int	m_pu32FrameUniformFlags[eNumSemantics];


		/*!* 3d matrix values for rendering */
		PVRTMat4		m_mView, m_mViewI;
		PVRTMat3		m_mViewIT;

		PVRTMat4		m_mProjection, m_mProjectionI;
		PVRTMat3		m_mProjectionIT;

		PVRTMat4		m_mViewProjection, m_mViewProjectionI;
		PVRTMat3		m_mViewProjectionIT;

		PVRTMat4		m_mWorld, m_mWorldView, m_mWorldViewProjection;

		PVRTMat4		m_mObject, m_mObjectI;
		PVRTMat3		m_mObjectIT;

		PVRTVec3		m_vEyePositionWorld, m_vEyePositionModel;

		/*!* Information on lights */
		float*			m_pfLightColor[i32NUM_LIGHTS];
		PVRTVec4		m_vLightPosWorld[i32NUM_LIGHTS], m_vLightPosEye[i32NUM_LIGHTS],
			m_vLightDirWorld[i32NUM_LIGHTS], m_vLightDirEye[i32NUM_LIGHTS];


		/*!* permanant pointer to light manager */
		LightManager	*m_pLightManager;



		/*!****************************************************************************
		@Function		setFlag
		@Input			sUniform
		@Description	stores that a specific uniform has been calculated to avoid
		recalculation
		******************************************************************************/
		// deals with the flag system
		void setFlag(const Uniform &sUniform)
		{
			m_pu32FrameUniformFlags[sUniform.getSemantic()]|=1<<sUniform.getIdx();
		}

		/*!****************************************************************************
		@Function		setFlag
		@Input			u32Semantic
		@Input			u32Index
		@Description	stores that a specific uniform has been calculated to avoid
		recalculation
		******************************************************************************/
		void setFlag(const unsigned int u32Semantic, const unsigned int u32Index=0)
		{
			m_pu32FrameUniformFlags[u32Semantic]|=1<<u32Index;
		}

		/*!****************************************************************************
		@Function		getFlag
		@Input			sUniform
		@Description	reports whether the requested value has been calculated or not
		******************************************************************************/
		bool getFlag(const Uniform &sUniform) const
		{
			return (m_pu32FrameUniformFlags[sUniform.getSemantic()]&1<<sUniform.getIdx())==0;
		}

		/*!****************************************************************************
		@Function		getFlag
		@Input			u32Semantic
		@Input			u32Index
		@Description	reports whether the requested value has been calculated or not
		******************************************************************************/
		bool getFlag(const EUniformSemantic u32Semantic, const unsigned int u32Index=0) const
		{
			return (m_pu32FrameUniformFlags[u32Semantic]&1<<u32Index)!=0;
		}

	};

}

#endif // UNIFORMHANDLER_H

/******************************************************************************
End of file (UniformHandler.h)
******************************************************************************/
