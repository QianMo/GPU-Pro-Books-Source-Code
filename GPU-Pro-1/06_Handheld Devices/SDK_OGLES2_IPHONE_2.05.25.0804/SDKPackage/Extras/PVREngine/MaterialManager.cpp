/******************************************************************************

 @File         MaterialManager.cpp

 @Title        Introducing the POD 3d file format

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Shows how to use the pfx format

******************************************************************************/
#include "PVRTools.h"
#include "MaterialManager.h"
#include "Globals.h"
#include "Material.h"
#include "ConsoleLog.h"

#define NONE_STRING "*** None ***";

namespace pvrengine
{

	

	/******************************************************************************/

	MaterialManager::MaterialManager()
	{
		Init();
	}

	/******************************************************************************/

	MaterialManager::MaterialManager(int i32MaxMaterials)
	{
		Init();
		m_daMaterials = dynamicArray<Material*>(i32MaxMaterials);
	}

	/******************************************************************************/

	MaterialManager::~MaterialManager()
	{
		for(unsigned int i=0;i<m_daMaterials.getSize();++i)
		{	// check if material is already in manager
			for(unsigned int j=i+1;j<m_daMaterials.getSize();++j)
			{
				if(m_daMaterials[i]==m_daMaterials[j]
				|| m_daMaterials[i]==m_pFlatMaterial)
				{
					m_daMaterials[i]=NULL;
					break;
				}
			}
		}

		for(unsigned int i=0;i<m_daMaterials.getSize();++i)
		{
			if(m_daMaterials[i]!=m_pFlatMaterial)
				PVRDELETE(m_daMaterials[i]);
		}

		PVRDELETE(m_pFlatMaterial);
	}

	/******************************************************************************/

	void MaterialManager::Init()
	{
		m_pFlatMaterial = new Material(eFlat);
	}

	/******************************************************************************/

	Material* MaterialManager::LoadMaterial(const CPVRTString& strEffectPath,
		const CPVRTString& strTexturePath,
		const SPODMaterial& sPODMaterial,
		const SPODTexture& sPODTexture)
	{
		CPVRTString strPFXFilename = NONE_STRING;
		if(sPODMaterial.pszEffectFile)
		{
			strPFXFilename = strEffectPath;
			((strPFXFilename+="/")+=sPODMaterial.pszEffectFile);
		}
		else
		{
			strPFXFilename = NONE_STRING;
		}

		// resolve texture file name/path
		CPVRTString strTextureFile = strTexturePath;
		if(!sPODMaterial.pszEffectFile)
		{
			if(sPODMaterial.nIdxTexDiffuse!=-1 && sPODTexture.pszName)
			{
				(strTextureFile+="/") += sPODTexture.pszName;
			}
		}

		for(unsigned int i=0;i<m_daMaterials.getSize();++i)
		{	// check if material is already in manager
			// check
			if(m_daMaterials[i]->getEffectFileName().compare(strPFXFilename)==0
				&&m_daMaterials[i]->getEffectName().compare(sPODMaterial.pszEffectName)==0
				&&m_daMaterials[i]->getName().compare(sPODMaterial.pszName)==0
				&&m_daMaterials[i]->getTextureFileName().compare(strTextureFile)==0)
			{
				ConsoleLog::inst().log("\nMaterial %s already in manager.",sPODMaterial.pszName);
				m_daMaterials.append(m_daMaterials[i]);

				return m_daMaterials[i];
			}
		}

		// actually make/load material - and attempt error handle
		Material* pNewMaterial;
		if(sPODMaterial.pszEffectFile)
		{	// with effect -> m_daMaterials.getSize() gives unique id to a material
			pNewMaterial = new Material(m_daMaterials.getSize(),strPFXFilename,strTexturePath,sPODMaterial);
		}
		else
		{	// no effect
			pNewMaterial = new Material(m_daMaterials.getSize(),strTextureFile,sPODMaterial);
		}

		if((!pNewMaterial) || !pNewMaterial->getValid())
		{
			ConsoleLog::inst().log("Loading material failed. Replacing with default\n");
			PVRDELETE(pNewMaterial);
			m_daMaterials.append(m_pFlatMaterial);
			return(m_pFlatMaterial);
		}

		m_daMaterials.append(pNewMaterial);

		return pNewMaterial;
	}

	/******************************************************************************/

	Material* MaterialManager::getMaterial(const unsigned int u32Id)
	{
		return m_daMaterials[u32Id];
	}

	/******************************************************************************/

	void MaterialManager::ReportActiveMaterial(Material* pNewMaterial)
	{
		if(m_pActiveMaterial)
			m_pActiveMaterial->Deactivate();	// only one active material at a time
		m_pActiveMaterial = pNewMaterial;
	}

	/******************************************************************************/

	Material*	MaterialManager::getFlatMaterial()
	{
		return m_pFlatMaterial;
	}

}

/******************************************************************************
End of file (MaterialManager.cpp)
******************************************************************************/
