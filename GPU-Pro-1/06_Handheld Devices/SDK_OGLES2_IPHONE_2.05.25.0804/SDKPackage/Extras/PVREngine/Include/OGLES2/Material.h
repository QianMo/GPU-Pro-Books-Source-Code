/******************************************************************************

 @File         Material.h

 @Title        A class detailing material values for meshes

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent - OGL/OGLES/OGLES2 Specific at the moment

 @Description  A class detailing material values for meshes such as colour,
               shader to be applied etc.

******************************************************************************/
#ifndef MATERIAL_H
#define MATERIAL_H

#include "Uniform.h"
#include "../PVRTString.h"
#include "dynamicArray.h"

namespace pvrengine
{

	/******************************************************************************
	Enums
	******************************************************************************/
	enum EDefaultMaterial	/*! enums denoting default material types*/
	{
		eFlat=0,
	};

	/*!***************************************************************************
	* @Class Material
	* @Brief A class detailing material values for meshes such as colour.
	* @Description A class detailing material values for meshes such as colour.
	*****************************************************************************/
	class Material
	{
	public:
		/*!***************************************************************************
		@Function			Material
		@Description		blank constructor.
		*****************************************************************************/
		Material();

		/*!***************************************************************************
		@Function			~Material
		@Description		destructor.
		*****************************************************************************/
		~Material(){}

		/*!***************************************************************************
		@Function			Material
		@Input				eMat - type of default material
		@Description		sets up a material from one of the defaults
		*****************************************************************************/
		Material(EDefaultMaterial eMat);

		/*!***************************************************************************
		@Function			Material
		@Input				original - other material
		@Description		copy constructor
		*****************************************************************************/
		Material(const Material& original);

		/*!***************************************************************************
		@Function			=
		@Input				sOtherMaterial - another material
		@Description		assignment operator
		*****************************************************************************/
		Material& operator=(const Material& sOtherMaterial);

		/*!***************************************************************************
		@Function			Material
		@Input				u32Id - identifier for material
		@Input				strPFXFile - path of PFX for this file
		@Input				strTexturePath - path of texture for this material
		@Input				sSPODMaterial - material values from POD
		@Description		sets up a material from POD/PFX values
		*****************************************************************************/
		Material(unsigned int u32Id,
			const CPVRTString& strPFXFile,
			const CPVRTString& strTexturePath,
			const SPODMaterial& sSPODMaterial);

		/*!***************************************************************************
		@Function			Material
		@Input				u32Id - identifier for material
		@Input				strTexturePath - path of texture for this material
		@Input				sSPODMaterial - material values from POD
		@Description		sets up a material from a POD texture
		*****************************************************************************/
		Material(unsigned int u32Id,
		const CPVRTString& strTexturePath,
			const SPODMaterial& sSPODMaterial);

		/*!***************************************************************************
		@Function			getEffectFileName
		@Return				path to PFX file for this material
		@Description		retrieves the path to the PFX file for this material
		*****************************************************************************/
		CPVRTString			getEffectFileName() const;

		/*!***************************************************************************
		@Function			getTextureFileName
		@Return				path to the texture file for this material
		@Description		retrieves the path to the texture file for this material
		*****************************************************************************/
		CPVRTString			getTextureFileName() const;

		/*!***************************************************************************
		@Function			getEffectName
		@Return				name of the effect for this material
		@Description		retrieves the name of the effect for this material
		*****************************************************************************/
		CPVRTString			getEffectName() const;

		/*!***************************************************************************
		@Function			getName
		@Return				name of  this material
		@Description		retrieves the name of this material
		*****************************************************************************/
		CPVRTString			getName() const;

		/*!***************************************************************************
		@Function			getSkinned
		@Return				does this material use skinning
		@Description		retrieves whether this material uses skinning or not
		*****************************************************************************/
		bool					getSkinned() const {return m_bSkinned;}

		/*!***************************************************************************
		@Function			Activate
		@Description		Activates this material for rendering
		*****************************************************************************/
		bool Activate();

		/*!***************************************************************************
		@Function			Deactivate
		@Description		Deactivates this material for rendering
		*****************************************************************************/
		void Deactivate();

		/*!***************************************************************************
		@Function			getMeshUniforms
		@Return				array of mesh uniforms
		@Description		retrieves the uniforms that must be initialised per mesh
		*****************************************************************************/
		dynamicArray<Uniform>*	getMeshUniforms();

		/*!***************************************************************************
		@Function			getSkinningUniforms
		@Return				array of skinning uniforms
		@Description		retrieves the uniforms that are required for skinning
		*****************************************************************************/
		dynamicArray<Uniform>*	getSkinningUniforms();

		/*!***************************************************************************
		@Function			getOpacity
		@Return				opacity value
		@Description		retrieves the opacity setting for this material
		*****************************************************************************/
		VERTTYPE			getOpacity(){return m_fOpacity;}

		/*!***************************************************************************
		@Function			getAmbient
		@Return				Ambient value
		@Description		retrieves the ambient colour setting for this material
		*****************************************************************************/
		PVRTVec3		getAmbient(){return m_vAmbient;}

		/*!***************************************************************************
		@Function			getDiffuse
		@Return				Diffuse value
		@Description		retrieves the diffuse colour setting for this material
		*****************************************************************************/
		PVRTVec3		getDiffuse(){return m_vDiffuse;}

		/*!***************************************************************************
		@Function			getSpecular
		@Return				Specular value
		@Description		retrieves the specular colour setting for this material
		*****************************************************************************/
		PVRTVec3		getSpecular(){return m_vSpecular;}

		/*!***************************************************************************
		@Function			getShininess
		@Return				Shininess value
		@Description		retrieves the shininess setting for this material
		*****************************************************************************/
		VERTTYPE			getShininess(){return m_fShininess;}

		/*!***************************************************************************
		@Function			getValid
		@Return				whether this material has initialised properly
		@Description		retrieves whether this material has initialised properly
		*****************************************************************************/
		bool					getValid(){return m_bValid;}
	private:
		unsigned int			m_u32Id;		/*! identifier for this material */
		bool					m_bValid, m_bActive, m_bSkinned;
		CPVRTString			m_strEffectFileName, m_strEffectName, m_strName, m_strTextureFileName;
		CPVRTPFXEffect			m_sEffect;				/*! PFX for this material */
		CPVRTPFXParser			m_sEffectParser;		/*! PFX parser for this material */
		dynamicArray<Uniform>	m_daFrameUniforms,		/*! uniforms valid per frame */
								m_daMaterialUniforms,	/*! uniforms valid per material */
								m_daMeshUniforms,		/*! uniforms valid per mesh */
								m_daSkinningUniforms,	/*! uniforms used for skinning */
								m_daCustomUniforms;		/*! other uniforms... */

		VERTTYPE	m_fOpacity;		/*! Material opacity (used with vertex alpha ?) */
		PVRTVec3	m_vAmbient;		/*! Ambient RGB value */
		PVRTVec3	m_vDiffuse;		/*! Diffuse RGB value */
		PVRTVec3	m_vSpecular;	/*! Specular RGB value */
		VERTTYPE	m_fShininess;	/*! Material shininess */

		/*!***************************************************************************
		@Function			Init
		@Description		common initalisation code for constructors
		*****************************************************************************/
		void		Init(unsigned int u32Id);
		/*!***************************************************************************
		@Function			setAsFlat
		@Description		sets this material as textured only
		*****************************************************************************/
		void		setAsFlat();
		/*!***************************************************************************
		@Function			loadPFXShaders
		@Input				sPODMaterial	-	POD material structure
		@Description		loads and compiles the PFX shaders
		*****************************************************************************/
		bool		loadPFXShaders();
		/*!***************************************************************************
		@Function			loadPODMaterialValues
		@Input				sPODMaterial	-	POD material structure
		@Description		initialises other material valus from the POD material
		structure.
		*****************************************************************************/
		void		loadPODMaterialValues(const SPODMaterial& sPODMaterial);
		/*!***************************************************************************
		@Function			loadPFXTextures
		@Input				strTexturePath	-	path to textures
		@Description		loads textures as specified by the PFX file
		*****************************************************************************/
		bool		loadPFXTextures(const CPVRTString& strTexturePath);
		/*!***************************************************************************
		@Function			buildUniformLists
		@Description		manufactures the uniform lists from the PFX file's values
		*****************************************************************************/
		bool		buildUniformLists();
		/*!***************************************************************************
		@Function			>
		@Input				A	-	a material object
		@Input				B	-	a material object
		@Description		compares the identifiers of two materials so that they
		can be sorted.
		*****************************************************************************/
		friend bool operator>(const Material& A, const Material& B)
		{
			return (A.m_u32Id>B.m_u32Id);
		}

	};
}
#endif // MATERIAL_H

/******************************************************************************
End of file (Material.h)
******************************************************************************/
