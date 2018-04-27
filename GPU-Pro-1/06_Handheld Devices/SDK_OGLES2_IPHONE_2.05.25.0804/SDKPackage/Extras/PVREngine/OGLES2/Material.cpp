/******************************************************************************

 @File         Material.cpp

 @Title        Texture

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Mesh material for PVREngine

******************************************************************************/
#include "Material.h"
#include "ContextManager.h"
#include "ConsoleLog.h"
#include "Texture.h"
#include "TextureManager.h"
#include "MaterialManager.h"
#include "UniformHandler.h"
#include "Light.h"

#include "ambientdiffuse.h"
#include "ambientdiffusedirection.h"
#include "flatMaterial.h"

namespace pvrengine
{

	

	/******************************************************************************/

	Material::Material(EDefaultMaterial eMat)
	{
		Init(0xffffffff);
		switch(eMat)
		{
		case eFlat:
		default:
			{
				setAsFlat();
			}
			break;
		}
	}

	/******************************************************************************/

	Material::Material(unsigned int u32Id,
		const CPVRTString& strEffectFile,
			const CPVRTString& strTexturePath,
			const SPODMaterial& sPODMaterial)
	{
		Init(u32Id);
		m_strEffectFileName= strEffectFile;
		m_strEffectName = sPODMaterial.pszEffectName;
		m_strName = sPODMaterial.pszName;
		m_strTextureFileName = strTexturePath;

		// Parse the file
		ConsoleLog::inst().log("\nLoading effect:%s\n",strEffectFile.c_str());
		CPVRTString errorString;
		if(PVR_SUCCESS!=m_sEffectParser.ParseFromFile(strEffectFile.c_str(), &errorString))
		{
			ConsoleLog::inst().log("Effect parse failed:%s\n",errorString.c_str());
			return;
		}

		if(!loadPFXShaders())
			return;
		if(!loadPFXTextures(strTexturePath))
			return;
		loadPODMaterialValues(sPODMaterial);
		if(!buildUniformLists())
			return;
		m_bValid = true;
	}

	/******************************************************************************/

	Material::Material(unsigned int u32Id,
		const CPVRTString& strTextureFile,
			const SPODMaterial& sPODMaterial)
	{
		Init(u32Id);
		m_strEffectName = "BasicDiffuse";
		m_strName = sPODMaterial.pszName;
		m_strTextureFileName = strTextureFile;
		// Parse the file
		CPVRTString errorString;
		ConsoleLog::inst().log("\nNo PFX; Loading default ambient/diffuse + texture effect using texture %s\n",
			strTextureFile.c_str());

		// TODO: find a better way than this
		// check first light for what type it is and load appropriate 

		if (LightManager::ptr()->getLight(0)->getType()==eLight_Point)
		{
			if(PVR_SUCCESS!=m_sEffectParser.ParseFromMemory(ambdiffpointshader, &errorString))
			{
				ConsoleLog::inst().log("Effect parse failed:%s\n",errorString.c_str());
				return;
			}
		}
		else
		{
			if(PVR_SUCCESS!=m_sEffectParser.ParseFromMemory(ambdiffdirectionshader, &errorString))
			{
				ConsoleLog::inst().log("Effect parse failed:%s\n",errorString.c_str());
				return;
			}

		}

		loadPFXShaders();

			Texture* pTexture = TextureManager::inst().LoadTexture(strTextureFile);
			m_sEffect.SetTexture(0, pTexture->getHandle(), pTexture->isCubeMap()?PVRTEX_CUBEMAP:0);

		loadPODMaterialValues(sPODMaterial);
		buildUniformLists();
		m_bValid=true;
	}

	/******************************************************************************/

	bool Material::Activate()
	{
		if(m_bActive)
		{	// already active
			return true;
		}
		
		// notify manager of new active effect
		MaterialManager::inst().ReportActiveMaterial(this);
		m_bActive = true;

		if(PVR_SUCCESS!=m_sEffect.Activate())
		{
		return false;
		}

		// deal with uniforms

		UniformHandler& sUniformHandler = UniformHandler::inst();

		// for all frame uniforms check if already calculated
		for(unsigned int i=0;i<m_daFrameUniforms.getSize();i++)
		{
			sUniformHandler.DoFrameUniform(m_daFrameUniforms[i]);
		}



		// calculate and load material uniforms
		for(unsigned int i=0;i<m_daMaterialUniforms.getSize();i++)
		{
			sUniformHandler.CalculateMaterialUniform(m_daMaterialUniforms[i],*this);
		}

		// TODO: custom uniforms by app (not sure how right now)

		return true;
	}

	/******************************************************************************/

	bool Material::buildUniformLists()
	{
		unsigned int u32UniformCount, u32UnknownUniformCount;
		CPVRTString strError;

		SPVRTPFXUniform* psUniforms; 
		if(PVR_SUCCESS!=m_sEffect.BuildUniformTable(
			&psUniforms, &u32UniformCount, &u32UnknownUniformCount,
			c_psUniformSemantics, sizeof(c_psUniformSemantics) / sizeof(*c_psUniformSemantics),
			&strError))
		{
			ConsoleLog::inst().log("Couldn't build uniform table: %s.\n", strError.c_str());
			return false;
		}
		if(u32UnknownUniformCount)
		{
			ConsoleLog::inst().log("\nWarning: %d unknown uniform semantics.\n", u32UnknownUniformCount);
			//return false;
		}

		dynamicArray<Uniform>daAllUniforms;
		for(unsigned int i=0;i<u32UniformCount;i++)
		{
			daAllUniforms.append(Uniform(psUniforms[i]));
		}
		FREE(psUniforms);

		// split into frame uniforms, material uniforms, mesh uniforms and skinning uniforms

		for(unsigned int i=0;i<daAllUniforms.getSize();i++)
		{
			for(unsigned int j=0;eFrameUniforms[j];j++)
			{
				if(daAllUniforms[i].getSemantic()==eFrameUniforms[j])
				{
					m_daFrameUniforms.append(daAllUniforms[i]);
					goto next;
				}
			}
			for(unsigned int j=0;eMaterialUniforms[j];j++)
			{
				if(daAllUniforms[i].getSemantic()==eMaterialUniforms[j])
				{
					m_daMaterialUniforms.append(daAllUniforms[i]);
					goto next;
				}
			}
			for(unsigned int j=0;eMeshUniforms[j];j++)
			{
				if(daAllUniforms[i].getSemantic()==eMeshUniforms[j])
				{
					m_daMeshUniforms.append(daAllUniforms[i]);
					goto next;
				}
			}
			for(unsigned int j=0;eSkinningUniforms[j];j++)
			{
				if(daAllUniforms[i].getSemantic()==eSkinningUniforms[j])
				{
					m_bSkinned = true;
					m_daSkinningUniforms.append(daAllUniforms[i]);
					goto next;
				}
			}

			m_daCustomUniforms.append(daAllUniforms[i]);

next:;
		}
			return true;
	}

	/******************************************************************************/

	dynamicArray<Uniform>*	Material::getMeshUniforms()
	{
		return &m_daMeshUniforms;
	}

	/******************************************************************************/

	dynamicArray<Uniform>*	Material::getSkinningUniforms()
	{
		if(m_daSkinningUniforms.getSize()!=0)
			return &m_daSkinningUniforms;
		else
			return NULL;
	}

	/******************************************************************************/

	bool Material::loadPFXShaders()
	{

		// set context
		m_sEffect.m_psContext = ContextManager::inst().getCurrentContext();
		// Load an effect from the file
		CPVRTString strError;
		if (PVR_SUCCESS!=m_sEffect.Load(m_sEffectParser,
			m_strEffectName.c_str(),
			m_strEffectFileName.c_str(),
			&strError))
		{
			ConsoleLog::inst().log("Couldn't load effect: %s:\n",m_strEffectName.c_str());
			ConsoleLog::inst().log("%s\n",strError.c_str());
			return false;
		}
		return true;
	}

	/******************************************************************************/

	bool Material::loadPFXTextures(const CPVRTString& strTexturePath)
	{

		//Load the textures.

		const SPVRTPFXTexture	*psTex;
		unsigned int			u32TextureCnt;

		psTex = m_sEffect.GetTextureArray(u32TextureCnt);

		for(unsigned int u32CurrentTexture = 0; u32CurrentTexture < u32TextureCnt; ++u32CurrentTexture)
		{
			CPVRTString strTexture;
			((strTexture+=strTexturePath)+="/")+=((char*)psTex[u32CurrentTexture].p);
			ConsoleLog::inst().log("Loading Texture: '%s'\n",strTexture.c_str());
			Texture* pTexture = TextureManager::inst().LoadTexture(strTexture);
			if(!pTexture)
			{
				ConsoleLog::inst().log("ERROR: Cannot load the texture: %s\n",strTexturePath.c_str());
				return false;
			}
			m_sEffect.SetTexture(u32CurrentTexture, pTexture->getHandle(), pTexture->isCubeMap()?PVRTEX_CUBEMAP:0);
		}
		return true;
	}

	/******************************************************************************/

	void Material::setAsFlat()
	{
		SPODMaterial sPODMaterial;
		sPODMaterial.pszEffectName = (char*)"Flat";
		sPODMaterial.fMatOpacity = 1.0f;
		m_strEffectFileName= "Unset";
		m_strEffectName = sPODMaterial.pszEffectName;
		m_strName = "Default Flat Effect";
		// Load an effect from the file
		CPVRTString errorString;
		if(PVR_SUCCESS!=m_sEffectParser.ParseFromMemory(flatMaterial, &errorString))
		{
			ConsoleLog::inst().log("Effect parse failed:%s\n",errorString.c_str());
			ConsoleLog::inst().log(" Material initialisation failed outright\n");
			return;
		}

		loadPFXShaders();
		loadPODMaterialValues(sPODMaterial);
		buildUniformLists();

	}
}

/******************************************************************************
End of file (MaterialAPI.cpp)
******************************************************************************/
