/*!  \file kfbxtexture.h
 */

#ifndef _FBXSDK_TEXTURE_H_
#define _FBXSDK_TEXTURE_H_

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

#ifdef KARCH_DEV_MACOSX_CFM
    #include <CFURL.h>
    #include <Files.h>
#endif

// FBX includes
#include <kfbxmath/kfbxvector2.h>
#include <kfbxplugins/kfbxshadingnode.h>

// FBX namespace
#include <fbxfilesdk_nsbegin.h>

// Forward declaration
class KFbxVector4;
class KFbxLayerContainer;


/** A texture is the description of the mapping of an image over a geometry.
  * \nosubgrouping
  */
class KFBX_DLL KFbxTexture : public KFbxShadingNode
{
	KFBXOBJECT_DECLARE(KFbxTexture,KFbxShadingNode);

	public:
	/**
	  * \name Texture Properties
	  */
	//@{
		typedef enum
		{ 
			eUMT_UV, 
			eUMT_XY, 
			eUMT_YZ, 
			eUMT_XZ, 
			eUMT_SPHERICAL,
			eUMT_CYLINDRICAL,
			eUMT_ENVIRONMENT,
			eUMT_PROJECTION,
			eUMT_BOX, // deprecated
			eUMT_FACE, // deprecated
			eUMT_NO_MAPPING,
		} EUnifiedMappingType;

		typedef enum 
		{
			eTEXTURE_USE_6_STANDARD,
			eTEXTURE_USE_6_SPHERICAL_REFLEXION_MAP,
			eTEXTURE_USE_6_SPHERE_REFLEXION_MAP,
			eTEXTURE_USE_6_SHADOW_MAP,
			eTEXTURE_USE_6_LIGHT_MAP,
			eTEXTURE_USE_6_BUMP_NORMAL_MAP
		} ETextureUse6;

		/** \enum EWrapMode Wrap modes.
		  * - \e eREPEAT
		  * - \e eCLAMP
		  */
		typedef enum 
		{
			eREPEAT,
			eCLAMP
		} EWrapMode;

		/** \enum EBlendMode Blend modes.
		  * - \e eTRANSLUCENT
		  * - \e eADDITIVE
		  * - \e eMODULATE
		  * - \e eMODULATE2
		  */
		typedef enum 
		{
			eTRANSLUCENT,
			eADDITIVE,
			eMODULATE,
			eMODULATE2
        } EBlendMode;

        /** \enum EAlignMode Alignment modes.
        * - \e KFBXTEXTURE_LEFT
        * - \e KFBXTEXTURE_RIGHT
        * - \e KFBXTEXTURE_TOP
        * - \e KFBXTEXTURE_BOTTOM
        */
        typedef enum  
        {
            eLEFT = 0,
            eRIGHT,
            eTOP,
            eBOTTOM
        } EAlignMode;

        /** \enum ECoordinates Texture coordinates.
        * - \e KFBXTEXTURE_U
        * - \e KFBXTEXTURE_V
        * - \e KFBXTEXTURE_W
        */
        typedef enum 
        {
            eU = 0,
            eV,
            eW
        } ECoordinates;

		// Type description
		KFbxTypedProperty<ETextureUse6>			TextureTypeUse;
		KFbxTypedProperty<fbxDouble1>			Alpha;

		// Mapping information
		KFbxTypedProperty<EUnifiedMappingType>	CurrentMappingType;
		KFbxTypedProperty<EWrapMode>			WrapModeU;
		KFbxTypedProperty<EWrapMode>			WrapModeV;
		KFbxTypedProperty<fbxBool1>				UVSwap;

		// Texture positioning
		KFbxTypedProperty<fbxDouble3>			Translation;
		KFbxTypedProperty<fbxDouble3>			Rotation;
		KFbxTypedProperty<fbxDouble3>			Scaling;
		KFbxTypedProperty<fbxDouble3>			RotationPivot;
		KFbxTypedProperty<fbxDouble3>			ScalingPivot;

		// Material management
		KFbxTypedProperty<fbxBool1>				UseMaterial;
		KFbxTypedProperty<fbxBool1>				UseMipMap;

		// Blend mode
		KFbxTypedProperty<EBlendMode>	CurrentTextureBlendMode;

		// UV set to use.
		KFbxTypedProperty<fbxString>			UVSet;

	/** Reset the texture to its default values.
	  * \remarks Texture file name is not reset.
	  */
	void Reset();

    /** Set the associated texture file. 
      * \param pName The absolute path of the texture file.   
      * \return Return \c true on success.
	  *	\remarks The texture file name must be valid.
      */
    bool SetFileName(char const* pName);

    /** Set the associated texture file. 
      * \param pName The relative path of the texture file.   
      * \return Return \c true on success.
	  *	\remarks The texture file name must be valid.
      */
    bool SetRelativeFileName(char const* pName);

	#ifdef KARCH_DEV_MACOSX_CFM
    bool SetFile(const FSSpec &pMacFileSpec);
    bool SetFile(const FSRef &pMacFileRef);
    bool SetFile(const CFURLRef &pMacURL);
	#endif

    /** Get the associated texture file path.
	  * \return The associated texture file path.
	  * \return An empty string if KFbxTexture::SetFileName() has not been called before.
	  */
    char const* GetFileName () const;

    /** Get the associated texture file path.
	  * \return The associated texture file path.
	  * \return An empty string if KFbxTexture::SetRelativeFileName() has not been called before.
	  */
    char const* GetRelativeFileName() const;

	#ifdef KARCH_DEV_MACOSX_CFM
	bool GetFile(FSSpec &pMacFileSpec) const;
    bool GetFile(FSRef &pMacFileRef) const;
    bool GetFile(CFURLRef &pMacURL) const;
	#endif

    /** Set the swap UV flag.
	  * \param pSwapUV Set to \c true if swap UV flag is enabled.
	  * \remarks If swap UV flag is enabled, the texture's width and height are swapped.
	  */
    void SetSwapUV(bool pSwapUV);

    /** Get the swap UV flag.
	  * \return \c true if swap UV flag is enabled.
	  * \remarks If swap UV flag is enabled, the texture's width and height are swapped.
	  */
    bool GetSwapUV() const;

	/** \enum EAlphaSource Alpha sources.
	  * - \e eNONE
	  * - \e eRGB_INTENSITY
	  * - \e eBLACK
	  */
    typedef enum    
    { 
        eNONE, 
        eRGB_INTENSITY, 
        eBLACK 
    } EAlphaSource;

    /** Set alpha source.
	  * \param pAlphaSource Alpha source identifier.
	  */
    void SetAlphaSource(EAlphaSource pAlphaSource);

    /** Get alpha source.
      * \return Alpha source identifier for this texture.
	  */
	EAlphaSource GetAlphaSource() const;

    /** Set cropping.
	  * \param pLeft Left cropping value.
	  * \param pTop  Top cropping value.
	  * \param pRight Right cropping value.
	  * \param pBottom Bottom cropping value.
	  * \remarks The defined rectangle is not checked for invalid values.
	  * It is the responsability of the caller to validate that the rectangle
	  * is meaningful for this texture.
	  */
    void SetCropping(int pLeft, int pTop, int pRight, int pBottom);

    /** Get left cropping.
	  * \return Left side of the cropping rectangle.
	  */
    int GetCroppingLeft() const;

    /** Get top cropping.
	  * \return Top side of the cropping rectangle.
	  */
    int GetCroppingTop() const;

    /** Get right cropping.
	  * \return Right side of the cropping rectangle.
	  */
    int GetCroppingRight() const;

    /** Get bottom cropping.
	  * \return Bottom side of the cropping rectangle.
	  */
    int GetCroppingBottom() const;
	
	/** \enum EMappingType Texture mapping types.
	  * - \e eNULL
	  * - \e ePLANAR
	  * - \e eSPHERICAL
	  * - \e eCYLINDRICAL
	  * - \e eBOX
	  * - \e eFACE
	  * - \e eUV
	  * - \e eENVIRONMENT
	  */
    typedef enum    
    { 
        eNULL, 
        ePLANAR, 
        eSPHERICAL, 
        eCYLINDRICAL, 
        eBOX, 
        eFACE,
        eUV,
		eENVIRONMENT
    } EMappingType;

    /** Set mapping type.
	  * \param pMappingType Mapping type identifier.
	  */
    void SetMappingType(EMappingType pMappingType);

    /** Get mapping type.
	  * \return Mapping type identifier.
	  */
    EMappingType GetMappingType() const;

	/** \enum EPlanarMappingNormal Planar mapping normal orientations.
	  * - \e ePLANAR_NORMAL_X
	  * - \e ePLANAR_NORMAL_Y
	  * - \e ePLANAR_NORMAL_Z
	  */
    typedef enum   
    { 
        ePLANAR_NORMAL_X, 
        ePLANAR_NORMAL_Y, 
        ePLANAR_NORMAL_Z 
    } EPlanarMappingNormal;

    /** Set planar mapping normal orientations.
	  * \param pPlanarMappingNormal Planar mapping normal orientation identifier.
	  */
    void SetPlanarMappingNormal(EPlanarMappingNormal pPlanarMappingNormal);

    /** Get planar mapping normal orientations.
	  * \return Planar mapping normal orientation identifier.
	  */
    EPlanarMappingNormal GetPlanarMappingNormal() const;

	/** \enum EMaterialUse Material usages.
	  * - \e eMODEL_MATERIAL
	  * - \e eDEFAULT_MATERIAL
	  */
    typedef enum 
    {
        eMODEL_MATERIAL,
        eDEFAULT_MATERIAL
    } EMaterialUse;

    /** Set material usage.
	  * \param pMaterialUse Material usage identifier.
	  */
    void SetMaterialUse(EMaterialUse pMaterialUse);

    /** Get material usage.
	  * \return Material usage identifier.
	  */
    EMaterialUse GetMaterialUse() const;

	/** \enum ETextureUse Texture usages.
	  * - \e eSTANDARD
	  * - \e eSHADOW_MAP
	  * - \e eLIGHT_MAP
	  * - \e eSPHERICAL_REFLEXION_MAP
	  * - \e eSPHERE_REFLEXION_MAP
	  * - \e eBUMP_NORMAL_MAP
	  */
	typedef enum 
	{
		eSTANDARD,
		eSHADOW_MAP,
		eLIGHT_MAP,
		eSPHERICAL_REFLEXION_MAP,
		eSPHERE_REFLEXION_MAP,
		eBUMP_NORMAL_MAP
	} ETextureUse;

	/** Set texture usage.
	  * \param pTextureUse Texure usage identifier.
	  */
    void SetTextureUse(ETextureUse pTextureUse);

    /** Get texture usage.
	  * \return Texture usage identifier.
	  */
    ETextureUse GetTextureUse() const;


	/** Set wrap mode in U and V.
	  * \param pWrapU Wrap mode identifier.
	  * \param pWrapV Wrap mode identifier.
	  */
    void SetWrapMode(EWrapMode pWrapU, EWrapMode pWrapV);

    /** Get wrap mode in U.
	  * \return U wrap mode identifier.
	  */
    EWrapMode GetWrapModeU() const;

	/** Get wrap mode in V.
	  * \return V wrap mode identifier.
	  */
	EWrapMode GetWrapModeV() const;


	/** Set blend mode.
	  * \param pBlendMode Blend mode identifier.
	  */
	void SetBlendMode(EBlendMode pBlendMode);

	/** Get blend mode.
	  * \return Blend mode identifier.
	  */
	EBlendMode GetBlendMode() const;

	//@}

	/**
	  * \name Default Animation Values
	  * This set of functions provide direct access to default
	  * animation values in the default take node. 
	  */
	//@{

	/** Set default translation vector. 
	  * \param pT First element is the U translation applied to 
	  * texture. A displacement of one unit is equal to the texture
	  * width after the scaling in U is applied. Second element is the
	  * V translation applied to texture. A displacement of one unit is 
	  * equal to the texture height after the scaling in V is applied.
	  * Third and fourth elements do not have an effect on texture 
	  * translation.
	  */
		inline void SetDefaultT(const KFbxVector4& pT) { Translation.Set( pT ); }

	/** Get default translation vector. 
	  * \param pT First element is the U translation applied to 
	  * texture. A displacement of one unit is equal to the texture 
	  * width after the scaling in U is applied. Second element is the
	  * V translation applied to texture. A displacement of one unit is 
	  * equal to the texture height after the scaling in V is applied.
	  * Third and fourth elements do not have an effect on the texture. 
	  * translation.
	  * \return Input parameter filled with appropriate data.
	  */
	KFbxVector4& GetDefaultT(KFbxVector4& pT) const;

	/** Set default rotation vector. 
	  * \param pR First element is the texture rotation around the 
	  * U axis in degrees. Second element is the texture rotation 
	  * around the V axis in degrees. Third element is the texture 
	  * rotation around the W axis in degrees.
	  * \remarks The W axis is oriented towards the result of the 
	  * vector product of the U axis and V axis i.e. W = U x V.
      */
	inline void SetDefaultR(const KFbxVector4& pR) { Rotation.Set( fbxDouble3(pR[0],pR[1],pR[2]) ); }

	/** Get default rotation vector. 
	  * \param pR First element is the texture rotation around the 
	  * U axis in degrees. Second element is the texture rotation 
	  * around the V axis in degrees. Third element is the texture 
	  * rotation around the W axis in degrees.
	  * \return Input parameter filled with appropriate data.
	  * \remarks The W axis is oriented towards the result of the 
	  * vector product of the U axis and V axis i.e. W = U x V.
	  */
	KFbxVector4& GetDefaultR(KFbxVector4& pR) const;

	/** Set default scale vector. 
	  * \param pS First element is scale applied to texture width. 
	  * Second element is scale applied to texture height. Third 
	  * and fourth elements do not have an effect on the texture. 
	  * \remarks A scale value inferior to 1 means the texture is stretched.
	  * A scale value superior to 1 means the texture is compressed.
	  */
	inline void SetDefaultS(const KFbxVector4& pS) { Scaling.Set( fbxDouble3(pS[0],pS[1],pS[2]) ); }

	/** Get default scale vector. 
	  * \param pS First element is scale applied to texture width. 
	  * Second element is scale applied to texture height. Third 
	  * and fourth elements do not have an effect on the texture. 
	  * \return Input parameter filled with appropriate data.
	  * \remarks A scale value inferior to 1 means the texture is stretched.
	  * A scale value superior to 1 means the texture is compressed.
	  */
	KFbxVector4& GetDefaultS(KFbxVector4& pS) const;

	/** Set default alpha.
	  *	\param pAlpha A value on a scale from 0 to 1, 0 meaning transparent.
      */
	void SetDefaultAlpha(double pAlpha);

	/** Get default alpha.
	  *	\return A value on a scale from 0 to 1, 0 meaning transparent.
	  */
	double GetDefaultAlpha() const;

	//@}

	/**
	  * \name Obsolete Functions
	  * This set of functions is obsolete since animated parameters
	  * are now supported. U, V and W coordinates are mapped to X, Y and Z
	  * coordinates of the default vectors found in section "Default Animation 
	  * Values".
	  */
	//@{

    /** Set translation.
	  * \param pU Horizontal translation applied to texture. A displacement 
	  * of one unit is equal to the texture's width after the scaling in 
	  * U is applied.
	  * \param pV Vertical translation applied to texture. A displacement 
	  * of one unit is equal to the texture's height after the scaling in 
	  * V is applied.
	  */
	void SetTranslation(double pU,double pV);

    /** Get translation applied to texture width.
      * \remarks A displacement of one unit is equal to the texture's width 
	  * after the scaling in U is applied.
	  */
    double GetTranslationU() const;

    /** Get translation applied to texture height.
      * \remarks A displacement of one unit is equal to the texture's height 
	  * after the scaling in V is applied.
	  */
    double GetTranslationV() const;

    /** Set rotation.
	  * \param pU Texture rotation around the U axis in degrees.
	  * \param pV Texture rotation around the V axis in degrees.
	  * \param pW Texture rotation around the W axis in degrees.
	  * \remarks The W axis is oriented towards the result of the vector product of 
	  * the U axis and V axis i.e. W = U x V.
	  */
    void SetRotation(double pU, double pV, double pW = 0.0);

    //! Get texture rotation around the U axis in degrees.
    double GetRotationU() const;

    //! Get texture rotation around the V axis in degrees.
    double GetRotationV() const;

    //! Get texture rotation around the W axis in degrees.
    double GetRotationW() const;

    /** Set scale.
	  * \param pU Scale applied to texture width. 
	  * \param pV Scale applied to texture height. 
	  * \remarks A scale value inferior to 1 means the texture is stretched.
	  * A scale value superior to 1 means the texture is compressed.
	  */
	void SetScale(double pU,double pV);

    /** Get scale applied to texture width. 
	  * \remarks A scale value inferior to 1 means the texture is stretched.
	  * A scale value superior to 1 means the texture is compressed.
	  */
    double GetScaleU() const;

    /** Get scale applied to texture height. 
	  * \remarks A scale value inferior to 1 means the texture is stretched.
	  * A scale value superior to 1 means the texture is compressed.
	  */
    double GetScaleV() const;

///////////////////////////////////////////////////////////////////////////////
//
//  WARNING!
//
//	Anything beyond these lines may not be documented accurately and is 
// 	subject to change without notice.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef DOXYGEN_SHOULD_SKIP_THIS

	bool operator==(KFbxTexture const& pTexture) const;

	KString& GetMediaName();
	void SetMediaName(char const* pMediaName);

	void SetUVTranslation(KFbxVector2& pT);
	KFbxVector2& GetUVTranslation();
	void SetUVScaling(KFbxVector2& pS);
	KFbxVector2& GetUVScaling();

	KString GetTextureType();


	// Clone
	virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;

protected:
    KFbxTexture(KFbxSdkManager& pManager, char const* pName);  
	virtual ~KFbxTexture();

	virtual void Construct(const KFbxTexture* pFrom);
	virtual bool ConstructProperties(bool pForceSet);
	virtual void Destruct(bool pRecursive, bool pDependents);

	//! Assignment operator.
	KFbxTexture& operator=(KFbxTexture const& pTexture);

	virtual KStringList	GetTypeFlags() const;
	
	void Init();
	void SyncVideoFileName(char const* pFileName);
	void SyncVideoRelativeFileName(char const* pFileName);

	int mTillingUV[2]; // not a prop
	int mCropping[4]; // not a prop

    EAlphaSource mAlphaSource; // now unused in MB (always set to None); not a prop
	EMappingType mMappingType; // CurrentMappingType
	EPlanarMappingNormal mPlanarMappingNormal; // CurrentMappingType

	KString mFileName;
	KString mRelativeFileName;
	KString mMediaName; // not a prop

	static KError smError;

	// Unsupported parameters in the FBX SDK, these are declared but not accessible.
	// They are used to keep imported and exported data identical.

	KFbxVector2 mUVScaling; // not a prop
	KFbxVector2 mUVTranslation; // not a prop

	friend class KFbxWriterFbx6;

	friend class KFbxLayerContainer;

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

};

inline EFbxType FbxTypeOf( KFbxTexture::EUnifiedMappingType const &pItem )		{ return eENUM; }
inline EFbxType FbxTypeOf( KFbxTexture::ETextureUse6 const &pItem )				{ return eENUM; }
inline EFbxType FbxTypeOf( KFbxTexture::EWrapMode const &pItem )				{ return eENUM; }
inline EFbxType FbxTypeOf( KFbxTexture::EBlendMode const &pItem )				{ return eENUM; }

typedef KFbxTexture* HKFbxTexture;

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_TEXTURE_H_



