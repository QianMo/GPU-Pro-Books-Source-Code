/*!  \file kfbxlight.h
 */
 
#ifndef _FBXSDK_LIGHT_H_
#define _FBXSDK_LIGHT_H_

/**************************************************************************************

 Copyright ?2001 - 2008 Autodesk, Inc. and/or its licensors.
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

#include <kfbxplugins/kfbxnodeattribute.h>
#include <fbxfilesdk_nsbegin.h>

class KFbxColor;
class KFbxSdkManager;
class KFbxTexture;

/** \brief This node attribute contains methods for accessing the properties of a light.
  * \nosubgrouping
  */
class KFBX_DLL KFbxLight : public KFbxNodeAttribute
{
	KFBXOBJECT_DECLARE(KFbxLight,KFbxNodeAttribute);

public:

	//! Return the type of node attribute which is EAttributeType::eLIGHT.
	virtual EAttributeType GetAttributeType() const;

	/**
	  * \name Light Properties
	  */
	//@{

    /** \enum ELightType Light types.
	  * - \e ePOINT
	  * - \e eDIRECTIONAL
	  * - \e eSPOT
	  */
    typedef enum  
    {
	    ePOINT = 0, 
	    eDIRECTIONAL, 
	    eSPOT
    } ELightType;

    /** \enum EDecayType     Decay types. Used for setting the attenuation of the light.
	  * - \e eNONE          No decay. The light's intensity will not diminish with distance.		
	  * - \e eLINEAR        Linear decay. The light's intensity will diminish linearly with the distance from the light.
	  * - \e eQUADRATIC     Quadratic decay. The light's intensity will diminish with the squared distance from the light.
	  *                     This is the most physically accurate decay rate.
	  * - \e eCUBIC         Cubic decay. The light's intensity will diminish with the cubed distance from the light.
	  */
    typedef enum  
    {
	    eNONE = 0,
	    eLINEAR,
	    eQUADRATIC,
		eCUBIC
    } EDecayType;

    /** Set the light type. 
	  * \param pLightType     The light type.
	  *
	  * \remarks This function is deprecated. Use property LightType.Set(pLightType) instead.
	  *
	  */
    K_DEPRECATED void SetLightType(ELightType pLightType);

	/** Get the light type.
	  * \return     The current light type.
	  *
	  * \remarks This function is deprecated. Use property LightType.Get() instead.
	  *
	  */
    K_DEPRECATED ELightType GetLightType() const;

	/** Activate or disable the light.
	  * \param pCastLight     Set to \c true to enable the light.
	  *
	  * \remarks This function is deprecated. Use property CastLight.Set(pCastLight) instead.
	  *
	  */
    K_DEPRECATED void SetCastLight(bool pCastLight);

	/** Get the light state.
	  * \return     \c true if the light is currently active.
	  *
	  * \remarks This function is deprecated. Use property CastLight.Get() instead.
	  *
	  */
    K_DEPRECATED bool GetCastLight() const;

	/** Set the shadow state for the light.
	  * \param pCastShadows    Set to \c true to have the light cast shadows.
	  */
	/** Activate or de-activate the shadow casting.
	  * \param pCastShadows If \c true the casting is active.
	  *
	  * \remarks This function is deprecated. Use property CastShadows.Set(pCastShadows) instead.
	  *
	  */
	K_DEPRECATED void SetCastShadows(bool pCastShadows);

	/** Get the shadow state for the light.
	  * \return     \c true if the light is currently casting shadows.
	  */
	/** Get shadow casting state.
	  * \return \c true if shadow casting is active.
	  *
	  * \remarks This function is deprecated. Use property CastShadows.Get() instead.
	  *
	  */
	K_DEPRECATED bool GetCastShadows() const;

	/** Set the shadow color for the light.
	  * \param pColor     The shadow color for the light expressed as kFbxColor.
	  */
	/** Set Shadow Color.
	  * \param pColor Shadow color.
	  *
	  * \remarks This function is deprecated. Use property ShadowColor.Set(pShadowColor) instead.
	  *
	  */
	K_DEPRECATED void SetShadowColor( KFbxColor& pColor );

	/** Get the shadow color for the light.
	  * \return     The shadow color of the light expressed as kFbxColor.
	  */
	/** Get Shadow Color.
	  * \return Shadow Color.
	  *
	  * \remarks This function is deprecated. Use property ShadowColor.Get() instead.
	  *
	  */
	K_DEPRECATED KFbxColor GetShadowColor() const;

	/** Set the shadow texture for the light.
	  * \param pTexture     The texture cast by the light shadow.
	  */
	void SetShadowTexture( KFbxTexture* pTexture );

	/** Get the light state.
	  * \return     Pointer to the texture cast by the light shadow, or \c NULL if the shadow texture has not been set.
	  */
	KFbxTexture* GetShadowTexture() const;

	//@}

	/**
	  * \name Gobo properties
	  */
	//@{

	/** Set the associated gobo file. 
	  * \param pFileName     The path of the gobo file.   
	  * \return              \c false if the pointer is null.
	  * \remarks             The gobo file name must be valid. In addition, the gobo file must be an 8 bit grayscale TIFF image file with 
	  *                      height and width dimensions must equal a power of two, and it cannot exceed 256 pixels.
	  *
	  * \remarks This function is deprecated. Use property FileName.Set() instead.
	  *
	  */
    K_DEPRECATED bool SetFileName(char const* pFileName);
    
#ifdef KARCH_DEV_MACOSX_CFM
    K_DEPRECATED bool SetFile(const FSSpec &pMacFileSpec);
    K_DEPRECATED bool SetFile(const FSRef &pMacFileRef);
    K_DEPRECATED bool SetFile(const CFURLRef &pMacURL);
#endif

    /** Get the associated gobo file path.
	  * \return     The associated gobo file path, or an empty string if the gobo file has not been set (see KFbxLight::SetFileName())
	  *
	  * \remarks This function is deprecated. Use property FileName.Get() instead.
	  *
	  */
    K_DEPRECATED char const* GetFileName() const;

#ifdef KARCH_DEV_MACOSX_CFM
	bool GetFile(FSSpec &pMacFileSpec) const;
    bool GetFile(FSRef &pMacFileRef) const;
    bool GetFile(CFURLRef &pMacURL) const;
#endif


	/** Sets the decay type
	  * \param pDecayType     The decay type
	  *
	  * \remarks This function is deprecated. Use property DecayType.Set(pDecayType) instead.
	  *
	  */
	K_DEPRECATED void SetDecayType( EDecayType pDecayType );

	/** Gets the decay type
	  * \return     The decay type
	  *
	  * \remarks This function is deprecated. Use property DecayType.Get() instead.
	  *
	  */
	K_DEPRECATED EDecayType GetDecayType() const;

	/** Sets the distance at which the light's intensity will decay.
	  * \param pDist    The distance
	  *
	  * \remarks This function is deprecated. Use property DecayStart.Set(pDist) instead.
	  *
	  */
	K_DEPRECATED void SetDecayStart( double pDist );

	/** Gets the distance at which the light instensity will decay
	  * \return     The distance
	  *
	  * \remarks This function is deprecated. Use property DecayStart.Get() instead.
	  *
	  */
	K_DEPRECATED double GetDecayStart() const;

    /** Enable gobo ground projection.
	  * \param pEnable     Set to \c true to have the gobo project on the ground/floor.
	  *
	  * \remarks This function is deprecated. Use property DrawGroundProjection.Set(pEnable) instead.
	  *
      */
    K_DEPRECATED void SetGroundProjection (bool pEnable);

    /** Get gobo ground projection flag.
	  * \return     \c true if gobo ground projection is enabled.
	  *
	  * \remarks This function is deprecated. Use property DrawGroundProjection.Get() instead.
	  *
	  */
    K_DEPRECATED bool GetGroundProjection() const;

    /** Enable gobo volumetric light projection.
	  * \param pEnable     Set to \c true to enable volumetric lighting projection.
	  *
	  * \remarks This function is deprecated. Use property DrawVolumetricLight.Set(pEnable) instead.
	  *
	  */
    K_DEPRECATED void SetVolumetricProjection(bool pEnable);

    /** Get gobo volumetric light projection flag.
	  * \return     \c true if gobo volumetric light projection is enabled.
	  *
	  * \remarks This function is deprecated. Use property DrawVolumetricLight.Get() instead.
	  *
	  */
    K_DEPRECATED bool GetVolumetricProjection() const;

    /** Enable gobo front volumetric projection.
	  * \param pEnable     Set to \c true to enable front volumetric lighting projection.
	  * \remarks           This option is not supported in MotionBuilder.
	  *
	  * \remarks This function is deprecated. Use property DrawFrontFacingVolumetricLight.Set(pEnable) instead.
	  *
	  */
    K_DEPRECATED void SetFrontVolumetricProjection(bool pEnable);

    /** Get gobo front volumetric light projection flag.
	  * \return     \c true if gobo front volumetric light projection is enabled.
	  * \remarks    This option is not supported in MotionBuilder.
	  *
	  * \remarks This function is deprecated. Use property DrawFrontFacingVolumetricLight.Get() instead.
	  *
	  */
    K_DEPRECATED bool GetFrontVolumetricProjection() const;

	//@}

	/**
	  * \name Default Animation Values
	  * This set of functions provide direct access to default animation values specific to a light. The default animation 
	  * values are found in the default take node of the associated node. These functions only work if the light has been associated
	  * with a node.
	  */
	//@{

	/** Set default color.
	  * \param pColor     The color of the light. 
	  * \remarks          The default value is white.
	  *
	  * \remarks This function is deprecated. Use property Color.Set(pColor) instead.
	  *
	  */
	K_DEPRECATED void SetDefaultColor(KFbxColor& pColor);

	/** Get default color.
	  * \param pColor     The color of the light.
	  * \return           Input parameter filled with appropriate data.
	  * \remarks          The default value is white.
	  *
	  * \remarks This function is deprecated. Use property Color.Get() instead.
	  *
	  */
	K_DEPRECATED KFbxColor& GetDefaultColor(KFbxColor& pColor) const;

	/** Set default intensity.
	  * \param pIntensity     The intensity value of the light.
	  * \remarks              The intensity range is from 0 to 200, where 200 is full intensity. The default value is 100.
	  *
	  * \remarks This function is deprecated. Use property Intensity.Set(pIntensity) instead.
	  *
	  */
	K_DEPRECATED void SetDefaultIntensity(double pIntensity);

	/** Get default intensity.
	  * \return      The intensity value of the light.
	  * \remarks     The intensity range is from 0 to 200, where 200 is full intensity. The default value is 100.
	  *
	  * \remarks This function is deprecated. Use property Intensity.Get() instead.
	  *
	  */
	K_DEPRECATED double GetDefaultIntensity() const;

	/** Set default cone angle in degrees. The cone angle is the outer cone. The inner cone is set using the HotSpot property.
	  * \param pConeAngle     The cone angle value of the light.
	  * \remarks              The cone angle has range is from 0 to 160 degrees. The default value is 45 degrees. This function has no effect 
	  *                       if the light type is not set to eSPOT.
	  *
	  * \remarks This function is deprecated. Use property ConeAngle.Set(pConeAngle) instead.
	  *
	  */
	K_DEPRECATED void SetDefaultConeAngle(double pConeAngle);

	/** Get default cone angle in degrees.
	  * \return      The cone angle value of the light.
	  * \remarks     The cone angle has range is from 0 to 160 degrees. The default value is 45 degrees. This function has no effect 
	  *              if the light type is not set to eSPOT.
	  *
	  * \remarks This function is deprecated. Use property ConeAngle.Get() instead.
	  *
	  */
	K_DEPRECATED double GetDefaultConeAngle() const;

	/** Set default fog.
	  * \param pFog     The fog value of the light.
	  * \remarks        This fog range is from 0 to 200, where 200 is full fog opacity. The default value is 50. This function has no effect 
	  *                 if the light type is not set to eSPOT.
	  *
	  * \remarks This function is deprecated. Use property Fog.Set(pFog) instead.
	  *
	  */
	K_DEPRECATED void SetDefaultFog(double pFog);

	/** Get default fog.
	  * \return      The fog value of the light.
	  * \remarks     This fog range is from 0 to 200, where 200 is full fog opacity. The default value is 50. This function has no effect 
	  *              if the light type is not set to eSPOT.
	  *
	  * \remarks This function is deprecated. Use property Fog.Get() instead.
	  *
	  */
	K_DEPRECATED double GetDefaultFog() const;

	//@}

	/**
	  * \name Light Property Names
	  */
	//@{	
	static const char*			sLightType;
	static const char*			sCastLight;
	static const char*			sDrawVolumetricLight;
	static const char*			sDrawGroundProjection;
	static const char*			sDrawFrontFacingVolumetricLight;
	static const char*			sColor;
	static const char*			sIntensity;
	static const char*          sHotSpot; // inner cone
	static const char*			sConeAngle; // outer cone
	static const char*			sFog;
	static const char*			sDecayType;
	static const char*			sDecayStart;
	static const char*			sFileName;
	static const char*			sEnableNearAttenuation;
	static const char*			sNearAttenuationStart;
	static const char*			sNearAttenuationEnd;
	static const char*			sEnableFarAttenuation;
	static const char*			sFarAttenuationStart;
	static const char*			sFarAttenuationEnd;
	static const char*			sCastShadows;
	static const char*			sShadowColor;
	//@}

	/**
	  * \name Light Property Default Values
	  */
	//@{	
	static const ELightType		sDefaultLightType;
	static const fbxBool1		sDefaultCastLight;
	static const fbxBool1		sDefaultDrawVolumetricLight;
	static const fbxBool1		sDefaultDrawGroundProjection;
	static const fbxBool1		sDefaultDrawFrontFacingVolumetricLight;
	static const fbxDouble3		sDefaultColor;
	static const fbxDouble1		sDefaultIntensity;
	static const fbxDouble1		sDefaultHotSpot;
	static const fbxDouble1		sDefaultConeAngle;
	static const fbxDouble1		sDefaultFog;
	static const EDecayType		sDefaultDecayType;
	static const fbxDouble1		sDefaultDecayStart;
	static const fbxString		sDefaultFileName;
	static const fbxBool1		sDefaultEnableNearAttenuation;
	static const fbxDouble1		sDefaultNearAttenuationStart;
	static const fbxDouble1		sDefaultNearAttenuationEnd;
	static const fbxBool1		sDefaultEnableFarAttenuation;
	static const fbxDouble1		sDefaultFarAttenuationStart;
	static const fbxDouble1		sDefaultFarAttenuationEnd;
	static const fbxBool1		sDefaultCastShadows;
	static const fbxDouble3		sDefaultShadowColor;
	//@}

	//////////////////////////////////////////////////////////////////////////
	//
	// Properties
	//
	//////////////////////////////////////////////////////////////////////////

	/**
	* \name Properties
	*/
	//@{	
	
	/** This property handles the light type.
	  *
      * To access this property do: LightType.Get().
      * To set this property do: LightType.Set(ELightType).
      *
	  * Default value is ePOINT
	  */
	KFbxTypedProperty<ELightType>		LightType;

	/** This property handles the cast light on object flag.
	  *
      * To access this property do: CastLight.Get().
      * To set this property do: CastLight.Set(fbxBool1).
      *
	  * Default value is true
	  */
	KFbxTypedProperty<fbxBool1>			CastLight;

	/** This property handles the draw volumetric ligtht flag.
	  *
      * To access this property do: DrawVolumetricLight.Get().
      * To set this property do: DrawVolumetricLight.Set(fbxBool1).
      *
	  * Default value is true
	  */
	KFbxTypedProperty<fbxBool1>			DrawVolumetricLight;

	/** This property handles the draw ground projection flag.
	  *
      * To access this property do: DrawGroundProjection.Get().
      * To set this property do: DrawGroundProjection.Set(fbxBool1).
      *
	  * Default value is true
	  */
	KFbxTypedProperty<fbxBool1>			DrawGroundProjection;

	/** This property handles the draw facing volumetric projection flag.
	  *
      * To access this property do: DrawFrontFacingVolumetricLight.Get().
      * To set this property do: DrawFrontFacingVolumetricLight.Set(fbxBool1).
      *
	  * Default value is false
	  */
	KFbxTypedProperty<fbxBool1>			DrawFrontFacingVolumetricLight;

	/** This property handles the light color.
	  *
      * To access this property do: Color.Get().
      * To set this property do: Color.Set(fbxDouble3).
      *
	  * Default value is (1.0, 1.0, 1.0)
	  */
	KFbxTypedProperty<fbxDouble3>		Color;

	/** This property handles the light intensity.
	  *
      * To access this property do: Intensity.Get().
      * To set this property do: Intensity.Set(fbxDouble1).
      *
	  * Default value is 100.0
	  */
	KFbxTypedProperty<fbxDouble1>		Intensity;

	/** This property handles the light inner cone angle (in degrees). Also know as the HotSpot!
	  *
      * To access this property do: HotSpot.Get().
      * To set this property do: HotSpot.Set(fbxDouble1).
      *
	  * Default value is 45.0
	  */
	KFbxTypedProperty<fbxDouble1>		HotSpot;

	/** This property handles the light outer cone angle (in degrees). Also known as the Falloff
	  *
      * To access this property do: ConeAngle.Get().
      * To set this property do: ConeAngle.Set(fbxDouble1).
      *
	  * Default value is 45.0
	  */
	KFbxTypedProperty<fbxDouble1>		ConeAngle;

	/** This property handles the light fog intensity
	  *
      * To access this property do: Fog.Get().
      * To set this property do: Fog.Set(fbxDouble1).
      *
	  * Default value is 50.0
	  */
	KFbxTypedProperty<fbxDouble1>		Fog;

	/** This property handles the decay type 
	  *
      * To access this property do: DecayType.Get().
      * To set this property do: DecayType.Set(EDecayType).
      *
	  * Default value is eNONE
	  */
	KFbxTypedProperty<EDecayType>		DecayType;

	/** This property handles the decay start distance
	  *
      * To access this property do: DecayStart.Get().
      * To set this property do: DecayStart.Set(fbxDouble1).
      *
	  * Default value is 0.0
	  */
	KFbxTypedProperty<fbxDouble1>		DecayStart;

	/** This property handles the gobo file name
	  *
      * To access this property do: FileName.Get().
      * To set this property do: FileName.Set(fbxString).
      *
	  * Default value is ""
	  */
	KFbxTypedProperty<fbxString>		FileName;

	/** This property handles the enable near attenuation flag
	  *
      * To access this property do: EnableNearAttenuation.Get().
      * To set this property do: EnableNearAttenuation.Set(fbxBool1).
      *
	  * Default value is false
	  */
	KFbxTypedProperty<fbxBool1>			EnableNearAttenuation;

	/** This property handles the near attenuation start distance
	  *
      * To access this property do: NearAttenuationStart.Get().
      * To set this property do: NearAttenuationStart.Set(fbxDouble1).
      *
	  * Default value is 0.0
	  */
	KFbxTypedProperty<fbxDouble1>		NearAttenuationStart;

	/** This property handles the near end attenuation 
	  *
      * To access this property do: NearAttenuationEnd.Get().
      * To set this property do: NearAttenuationEnd.Set(fbxDouble1).
      *
	  * Default value is 0.0
	  */
	KFbxTypedProperty<fbxDouble1>		NearAttenuationEnd;

	/** This property handles the enable far attenuation flag
	  *
      * To access this property do: EnableFarAttenuation.Get().
      * To set this property do: EnableFarAttenuation.Set(fbxBool1).
      *
	  * Default value is false
	  */
	KFbxTypedProperty<fbxBool1>			EnableFarAttenuation;

	/** This property handles the far attenuation start distance
	  *
      * To access this property do: FarAttenuationStart.Get().
      * To set this property do: FarAttenuationStart.Set(fbxDouble1).
      *
	  * Default value is 0.0
	  */
	KFbxTypedProperty<fbxDouble1>		FarAttenuationStart;

	/** This property handles the attenuation end distance
	  *
      * To access this property do: FarAttenuationEnd.Get().
      * To set this property do: FarAttenuationEnd.Set(fbxDouble1).
      *
	  * Default value is 0.0
	  */
	KFbxTypedProperty<fbxDouble1>		FarAttenuationEnd;

	/** This property handles the cast shadow flag
	  *
      * To access this property do: CastShadows.Get().
      * To set this property do: CastShadows.Set(fbxBool1).
      *
	  * Default value is false
	  */
	KFbxTypedProperty<fbxBool1>			CastShadows;

	/** This property handles the shadow color
	  *
      * To access this property do: ShadowColor.Get().
      * To set this property do: ShadowColor.Set(fbxDouble3).
      *
	  * Default value is (0.0, 0.0, 0.0)
	  */
	KFbxTypedProperty<fbxDouble3>		ShadowColor;
	
	//@}

///////////////////////////////////////////////////////////////////////////////
//
//  WARNING!
//
//	Anything beyond these lines may not be documented accurately and is 
// 	subject to change without notice.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef DOXYGEN_SHOULD_SKIP_THIS

public:

	// Clone
	virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;

protected:

	KFbxLight(KFbxSdkManager& pManager, char const* pName);
	virtual ~KFbxLight();

	//! Assignment operator.
    KFbxLight& operator= (KFbxLight const& pLight);

	virtual bool ConstructProperties(bool pForceSet);
	virtual void Destruct(bool pRecursive, bool pDependents);

	/**
	  *	Initialize the properties
	  */
	virtual void Init();

	virtual KString		GetTypeName() const;
	virtual KStringList	GetTypeFlags() const;

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

};

typedef KFbxLight* HKFbxLight;

inline EFbxType FbxTypeOf( KFbxLight::ELightType const &pItem )			{ return eENUM; }
inline EFbxType FbxTypeOf( KFbxLight::EDecayType const &pItem )			{ return eENUM; }

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_LIGHT_H_



