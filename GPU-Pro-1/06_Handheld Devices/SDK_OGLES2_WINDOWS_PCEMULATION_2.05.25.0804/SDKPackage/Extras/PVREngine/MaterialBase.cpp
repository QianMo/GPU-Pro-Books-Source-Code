/******************************************************************************

 @File         MaterialBase.cpp

 @Title        Material

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Texture container for OGLES2

******************************************************************************/
#include "Material.h"

namespace pvrengine
{
	

	/******************************************************************************/

	Material::Material()
	{
		Init(0);	
	}

	/******************************************************************************/

	void Material::Init(unsigned int u32Id)
	{
		m_bValid=false;
		m_bActive = false;
		m_bSkinned = false;
		m_u32Id = u32Id;
	}

	/******************************************************************************/

	void	Material::loadPODMaterialValues(const SPODMaterial& sPODMaterial)
	{
		// set material values
		m_fOpacity = sPODMaterial.fMatOpacity;		/*! Material opacity (used with vertex alpha ?) */
		m_vAmbient = PVRTVec3(sPODMaterial.pfMatAmbient);	/*! Ambient RGB value */
		m_vDiffuse = PVRTVec3(sPODMaterial.pfMatDiffuse);	/*! Diffuse RGB value */
		m_vSpecular = PVRTVec3(sPODMaterial.pfMatSpecular);	/*! Specular RGB value */
		m_fShininess = sPODMaterial.fMatShininess;		/*! Material shininess */
	}


	/******************************************************************************/

	CPVRTString Material::getEffectFileName() const
	{
		return m_strEffectFileName;
	}

	/******************************************************************************/

	CPVRTString Material::getTextureFileName() const
	{
		return m_strTextureFileName;
	}

	/******************************************************************************/

	CPVRTString Material::getEffectName() const
	{
		return m_strEffectName;
	}

	/******************************************************************************/

	CPVRTString Material::getName() const
	{
		return m_strName;
	}

	/******************************************************************************/

	void Material::Deactivate()
	{
		m_bActive = false;
	}

}

/******************************************************************************
End of file (Material.cpp)
******************************************************************************/
