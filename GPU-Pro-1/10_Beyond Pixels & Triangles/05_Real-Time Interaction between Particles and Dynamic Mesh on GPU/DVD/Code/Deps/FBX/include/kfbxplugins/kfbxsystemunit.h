/*!  \file kfbxsystemunit.h
 */

#ifndef _KFbxSystemUnit_h
#define _KFbxSystemUnit_h

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

#include <klib/kstring.h>
#include <klib/karrayul.h>

#include <fbxfilesdk_nsbegin.h>

class KFbxGlobalSettings;
class KFCurve;
class KFbxXMatrix;
class KFbxNode;
class KFbxScene;

/**	\brief This class is used to describe the units of measurement used within a particular scene.
  * \nosubgrouping
  */
class KFBX_DLL KFbxSystemUnit 
{
public:

	/** Defines various options that can be set for converting the units of a scene
	  */
	struct KFbxUnitConversionOptions
	{
		bool mConvertLightIntensity;		/**< Convert the intensity property of lights. */
		bool mConvertRrsNodes;  /**< Convert the nodes that do not inheirit their parent's scale */
	};

	/** Constructor
	  * \param pScaleFactor The equivalent number of centimeters in the new system unit. 
	  *                     eg For an inch unit, use a scale factor of 2.54
	  * \param pMultiplier  A multiplier factor of pScaleFactor.
	  */
	KFbxSystemUnit(double pScaleFactor, double pMultiplier = 1.0);
	~KFbxSystemUnit();

	// predefined units
	static const KFbxSystemUnit mm;
    static const KFbxSystemUnit dm;
	static const KFbxSystemUnit cm;
	static const KFbxSystemUnit m;
	static const KFbxSystemUnit km;
	static const KFbxSystemUnit Inch;
	static const KFbxSystemUnit Foot;
	static const KFbxSystemUnit Mile;
	static const KFbxSystemUnit Yard;

	#define KFbxSystemUnit_sPredefinedUnitCount 9
	static const KFbxSystemUnit *sPredefinedUnits; // points to an array of KFbxSystemUnit_sPredifinedUnitCount size

	static const KFbxUnitConversionOptions DefaultConversionOptions;

	/** Convert a scene from its system units to this unit.
	  * \param pScene The scene to convert
	  * \param pOptions Various conversion options. See KFbxSystemUnit::KFbxUnitConversionOptions
	  */
	void ConvertScene( KFbxScene* pScene, const KFbxUnitConversionOptions& pOptions = DefaultConversionOptions ) const;

	/** Converts the children of the given node to this system unit.
	  * Unlike the ConvertScene() method, this method does not set the axis system 
	  * of the scene that the pRoot node belongs, nor does it adjust KFbxPoses
	  * as they are not stored under the scene, and not under a particular node.
	  */
	void ConvertChildren( KFbxNode* pRoot, const KFbxSystemUnit& pSrcUnit, const KFbxUnitConversionOptions& pOptions = DefaultConversionOptions ) const;

	/** Convert a scene from its system units to this unit, using the specified 
	  * Fbx_Root node. This method is provided for backwards compatibility only
	  * and ConvertScene( KFbxScene* , const KFbxUnitConversionOptions&  ) should 
	  * be used instead whenever possible.
	  * \param pScene The scene to convert
	  * \param pFbxRoot The Fbx_Root node to use in conversion
	  * \param pOptions Conversion options. See KFbxSystemUnit::KFbxUnitConversionOptions
	  */
	void ConvertScene( KFbxScene* pScene, KFbxNode* pFbxRoot, const KFbxUnitConversionOptions& pOptions = DefaultConversionOptions ) const;

	/** Gets the scale factor of this system unit, relative to centimeters.
	  * This factor scales values in system units to centimeters.
	  * For the purpose of scaling values to centimeters, this value should be used
	  * and the "multiplier" (returned by GetMultiplier()) should be ignored.
	  */
	double GetScaleFactor() const;

	/** Returns a unit label for the current scale factor.
	  */
	KString GetScaleFactorAsString(bool pAbbreviated = true) const;

	/** Returns a unit label for the current scale factor. Capital first letter + "s" added + foot -> feet
	  */
	KString GetScaleFactorAsString_Plurial() const;

	/** Gets the multiplier factor of this system unit.
	  */
	double GetMultiplier() const;

	bool operator ==(const KFbxSystemUnit& pOther) const;
	bool operator !=(const KFbxSystemUnit& pOther) const;

	/** Returns the conversion factor from this unit to pTarget (does not include the muliplier factor).
	  */
	double GetConversionFactorTo( const KFbxSystemUnit& pTarget ) const;

	/** Returns the conversion factor from pSource to this unit
	  */
	double GetConversionFactorFrom( const KFbxSystemUnit& pSource ) const;

///////////////////////////////////////////////////////////////////////////////
//
//  WARNING!
//
//	Anything beyond these lines may not be documented accurately and is 
// 	subject to change without notice.
//
///////////////////////////////////////////////////////////////////////////////

protected:
	double mScaleFactor;
	double mMultiplier;

	void ApplyMultiplier(KFbxNode* pRoot, bool pSubChildrenOnly) const;
	void ConvertSTProperties(KArrayTemplate<KFbxNode*>& pNodes, double pConversionFactor) const;
	void ConvertSProperty(KArrayTemplate<KFbxNode*>& pNodes, double pConversionFactor) const;
	void ConvertFCurve(KArrayTemplate<KFCurve*>& pFCurves, double pConversionFactor) const;
	double GetConversionFactor( double pTargetScaleFactor, double pSourceScaleFactor) const;
	void AdjustPivots(KFbxNode* pNode, double pConversionFactor, KFbxXMatrix& pOriginalGlobalM ) const;
	void AdjustLimits(KFbxNode* pNode, double pConversionFactor) const;
	void AdjustPoses(KFbxScene* pScene, double pConversionFactor) const;

	void AdjustCluster(KFbxNode* pNode, double pConversionFactor) const;
	void AdjustLightIntensity(KFbxNode* pNode, const double pConversionFactor) const;
	void AdjustPhotometricLightProperties(KFbxNode* pNode, const double pConversionFactor) const;
    void AdjustCameraClipPlanes(KFbxNode* pNode, const double pConversionFactor) const;

	void ConvertChildren( KFbxNode* pRoot, const KFbxSystemUnit& pSrcUnit, bool pSubChildrenOnly, const KFbxUnitConversionOptions& pOptions ) const;

	friend class KFbxGlobalSettings;
};

#include <fbxfilesdk_nsend.h>

#endif //_KFbxSystemUnit_h
