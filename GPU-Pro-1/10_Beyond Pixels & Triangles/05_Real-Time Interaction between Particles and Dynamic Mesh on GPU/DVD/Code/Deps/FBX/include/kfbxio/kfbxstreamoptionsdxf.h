/*!  \file kfbxstreamoptionsdxf.h
 */
 
#ifndef _FBXSDK_KFBXSTREAMOPTIONSDXF_H_
#define _FBXSDK_KFBXSTREAMOPTIONSDXF_H_

/**************************************************************************************

 Copyright © 2001 - 2008 Autodesk, Inc. and/or its licensors.
 All Rights Reserved.

 The coded instructions, statements, computer programs, and/or related material 
 (collectively the "Data") in these files contain unpublished information 
 proprietary to Autodesk, Inc. and/or its licensors, which is protected by 
 Canada and United States of America federal copyright law and by international 
 treaties. 
 
 The Data may not be disclosed or distributed to third parties, in whole or in
 part, without the prior written consent of Autodesk, Inc. ("Autodesk").

 THE DATA IS PROVIDED "AS IS" AND WITHOUT WARRANTY.
 ALL WARRANTIES ARE EXPRESSLY EXCLUDED AND DISCLAIMED. AUTODESK MAKES NO
 WARRANTY OF ANY KIND WITH RESPECT TO THE DATA, EXPRESS, IMPLIED OR ARISING
 BY CUSTOM OR TRADE USAGE, AND DISCLAIMS ANY IMPLIED WARRANTIES OF TITLE, 
 NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE OR USE. 
 WITHOUT LIMITING THE FOREGOING, AUTODESK DOES NOT WARRANT THAT THE OPERATION
 OF THE DATA WILL BE UNINTERRUPTED OR ERROR FREE. 
 
 IN NO EVENT SHALL AUTODESK, ITS AFFILIATES, PARENT COMPANIES, LICENSORS
 OR SUPPLIERS ("AUTODESK GROUP") BE LIABLE FOR ANY LOSSES, DAMAGES OR EXPENSES
 OF ANY KIND (INCLUDING WITHOUT LIMITATION PUNITIVE OR MULTIPLE DAMAGES OR OTHER
 SPECIAL, DIRECT, INDIRECT, EXEMPLARY, INCIDENTAL, LOSS OF PROFITS, REVENUE
 OR DATA, COST OF COVER OR CONSEQUENTIAL LOSSES OR DAMAGES OF ANY KIND),
 HOWEVER CAUSED, AND REGARDLESS OF THE THEORY OF LIABILITY, WHETHER DERIVED
 FROM CONTRACT, TORT (INCLUDING, BUT NOT LIMITED TO, NEGLIGENCE), OR OTHERWISE,
 ARISING OUT OF OR RELATING TO THE DATA OR ITS USE OR ANY OTHER PERFORMANCE,
 WHETHER OR NOT AUTODESK HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH LOSS
 OR DAMAGE. 

**************************************************************************************/


#define KFBXSTREAMOPT_DXF_DEFORMATION "DEFORMATION"
#define KFBXSTREAMOPT_DXF_TRIANGULATE "TRIANGULATE"
#define KFBXSTREAMOPT_DXF_WELD_VERTICES "WELD VERTICES"
#define KFBXSTREAMOPT_DXF_OBJECT_DERIVATION_LABEL "OBJECT DERIVATION LABEL"
#define KFBXSTREAMOPT_DXF_OBJECT_DERIVATION "OBJECT DERIVATION"
#define KFBXSTREAMOPT_DXF_REFERENCENODE "REFERENCENODE"

#include <kaydaradef.h>
#ifndef KFBX_DLL 
	#define KFBX_DLL K_DLLIMPORT
#endif

#include <kfbxio/kfbxstreamoptions.h>

#include <fbxfilesdk_nsbegin.h>

/**	\brief This class is used for accessing the Import options of Dxf files.
  * The content of KfbxStreamOptionsDxf is stored in the inherited Property of its parent (KFbxStreamOptions).
  */
class KFBX_DLL KFbxStreamOptionsDxfReader : public KFbxStreamOptions
{
	KFBXOBJECT_DECLARE(KFbxStreamOptionsDxfReader,KFbxStreamOptions);

public:

	/** Reset all options to default values
	*/
	void Reset();

	/** \enum EObjectDerivation   Shading modes
	  * - \e eBY_LAYER       
	  * - \e eBY_ENTITY
	  * - \e eBY_BLOCK
	  */
	typedef enum 
	{
		eBY_LAYER,
		eBY_ENTITY,
		eBY_BLOCK
	} EObjectDerivation;
	
	/** Sets the Create Root Node Option
	* \param pCreateRootNode     The boolean value to be set. 
	*/
	inline void SetCreateRootNode(bool pCreateRootNode) {this->GetOption(KFBXSTREAMOPT_DXF_REFERENCENODE).Set(pCreateRootNode);}
	
	/** Sets the Weld Vertices Option
	* \param pWeldVertices     The boolean value to be set. 
	*/
	inline void SetWeldVertices(bool pWeldVertices){this->GetOption(KFBXSTREAMOPT_DXF_WELD_VERTICES).Set(pWeldVertices);}
	
	/** Sets the Object Derivation
	* \param pDerivation     The object variation to be set. 
	*/
	void SetObjectDerivation(EObjectDerivation pDerivation);

	/** Gets the Object Derivation
	* \return     The object variation. 
	*/
	EObjectDerivation GetObjectDerivation();

#ifndef DOXYGEN_SHOULD_SKIP_THIS
public:
	virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;

protected:

	KFbxStreamOptionsDxfReader(KFbxSdkManager& pManager, char const* pName);

	virtual void Construct(const KFbxStreamOptionsDxfReader* pFrom );
	virtual bool ConstructProperties(bool pForceSet);
	virtual void Destruct(bool pRecursive, bool pDependents);
#endif
};



/**	\brief This class is used for accessing the Export options of Dxf files.
  * The content of KfbxStreamOptionsDxf is stored in the inherited Property of its parent (KFbxStreamOptions).
  */
class KFBX_DLL KFbxStreamOptionsDxfWriter : public KFbxStreamOptions
{
	KFBXOBJECT_DECLARE(KFbxStreamOptionsDxfWriter,KFbxStreamOptions);
#ifndef DOXYGEN_SHOULD_SKIP_THIS
public:
	virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;
	void Reset();

protected:

	KFbxStreamOptionsDxfWriter(KFbxSdkManager& pManager, char const* pName);
	virtual void Construct(const KFbxStreamOptionsDxfWriter* pFrom);
	virtual bool ConstructProperties(bool pForceSet);
	virtual void Destruct(bool pRecursive, bool pDependents);
#endif
};

#include <fbxfilesdk_nsend.h>

#endif

