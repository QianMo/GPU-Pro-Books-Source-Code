/*!  \file kfbxsurfacelambert.h
 */

#ifndef _FBXSDK_SURFACELAMBERT_H_
#define _FBXSDK_SURFACELAMBERT_H_

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

#include <kaydaradef.h>
#ifndef KFBX_DLL 
	#define KFBX_DLL K_DLLIMPORT
#endif

#include <kaydara.h>

#include <kfbxplugins/kfbxsurfacematerial.h>
#include <kfbxplugins/kfbxcolor.h>
#include <kfbxplugins/kfbxgroupname.h>

#include <klib/kerror.h>

#include <fbxfilesdk_nsbegin.h>

/** Material settings.
  * \nosubgrouping
  * A material is attached to an instance of class KFbxGeometry
  * by calling KFbxGeometry::AddMaterial(). Materials can be shared among 
  * many instances of class KFbxGeometry.
  */
class KFBX_DLL KFbxSurfaceLambert : public KFbxSurfaceMaterial
{
	KFBXOBJECT_DECLARE(KFbxSurfaceLambert,KFbxSurfaceMaterial);

public:
	/**
	 * \name Material properties
	 */
	//@{
	
	/** Get the emissive color property.
	 */
	KFbxPropertyDouble3 GetEmissiveColor() const;
	
	/** Get the emissive factor property. This factor is used to
	 * attenuate the emissive color.
	 */
	KFbxPropertyDouble1 GetEmissiveFactor() const;
	
	/** Get the ambient color property.
	 */
	KFbxPropertyDouble3 GetAmbientColor() const;
	
	/** Get the ambient factor property. This factor is used to
	 * attenuate the ambient color.
	 */
	KFbxPropertyDouble1 GetAmbientFactor() const;
	
	/** Get the diffuse color property.
	 */
	KFbxPropertyDouble3 GetDiffuseColor() const;
	
	/** Get the diffuse factor property. This factor is used to
	 * attenuate the diffuse color.
	 */
	KFbxPropertyDouble1 GetDiffuseFactor() const;
	
	/** Get the bump property. This property is used to perturb the
	 * surface normal, creating the illusion of a bumpy surface.
	 */
	KFbxPropertyDouble3 GetBump() const;
	
	/** Get the transparent color property. This property is used to make a
	 * surface more or less transparent.
	 */
	KFbxPropertyDouble3 GetTransparentColor() const;
	
	/** Get the transparency property. This property is used to make a
	 * surface more or less opaque (0 = opaque, 1 = transparent).
	 */
	KFbxPropertyDouble1 GetTransparencyFactor() const;
	
	//@}

	//////////////////////////////////////////////////////////////////////////
	// Static values
	//////////////////////////////////////////////////////////////////////////

	// Default property values
	static fbxDouble3 sEmissiveDefault;
	static fbxDouble1 sEmissiveFactorDefault;

	static fbxDouble3 sAmbientDefault;
	static fbxDouble1 sAmbientFactorDefault;

	static fbxDouble3 sDiffuseDefault;
	static fbxDouble1 sDiffuseFactorDefault;
	
	static fbxDouble3 sBumpDefault;
    static fbxDouble3 sNormalMapDefault;

	static fbxDouble3 sTransparentDefault;
	static fbxDouble1 sTransparencyFactorDefault;

///////////////////////////////////////////////////////////////////////////////
//
//  WARNING!
//
//	Anything beyond these lines may not be documented accurately and is 
// 	subject to change without notice.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef DOXYGEN_SHOULD_SKIP_THIS

	// Clone
	virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;

	bool operator==(KFbxSurfaceLambert const& pMaterial) const;

protected:
	KFbxSurfaceLambert(KFbxSdkManager& pManager, char const* pName);
	virtual ~KFbxSurfaceLambert();

	virtual bool ConstructProperties(bool pForceSet);
	virtual void Destruct(bool pRecursive, bool pDependents);

	//! Assignment operator.
    KFbxSurfaceLambert& operator=(KFbxSurfaceLambert const& pMaterial);

	// From KFbxObject
	virtual KString		GetTypeName() const;
	virtual KStringList	GetTypeFlags() const;

	// Local
	void Init();

	KFbxPropertyDouble3 Emissive;
	KFbxPropertyDouble1 EmissiveFactor;
	
	KFbxPropertyDouble3 Ambient;
	KFbxPropertyDouble1 AmbientFactor;
	
	KFbxPropertyDouble3 Diffuse;
	KFbxPropertyDouble1 DiffuseFactor;
	
	KFbxPropertyDouble3 Bump;
	KFbxPropertyDouble3 NormalMap;

	KFbxPropertyDouble3 TransparentColor;
	KFbxPropertyDouble1 TransparencyFactor;

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS 

};

typedef KFbxSurfaceMaterial* HKFbxSurfaceMaterial;

#include <fbxfilesdk_nsend.h>

#endif


