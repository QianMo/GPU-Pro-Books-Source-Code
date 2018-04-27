/*!  \file kfbxiosettings.h
 */

#ifndef _FBXSDK_IO_SETTINGS_H_
#define _FBXSDK_IO_SETTINGS_H_

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

// libxml is not included on CFM.
#ifndef KARCH_ENV_MACOSX_CFM


#include <klib/karrayul.h>
#include <klib/kstring.h>

#ifndef MB_FBXSDK
#include <kbaselib_nsuse.h>
#endif


#include <kfbxplugins/kfbxobject.h>

#include <libxml.h>
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <libxml/globals.h>


#include <kfbxplugins/kfbxtypes.h>

#include <kbaselib_forward.h>
#include <fbxfilesdk_nsbegin.h>

#include "kfbxiosettingspath.h"

#define IOSREF  KFbxIOSettings::IOSettingsRef()

#define IOSVisible    true
#define IOSHidden     false

#define IOSSavable    true
#define IOSNotSavable false

#define IOSEnabled    true
#define IOSDisabled   false


#define IOSBinary     0
#define IOSASCII      1

// class to handle old StreamOptions information
class KsoInfo
{

public: enum EIMPEXP
{
    eIMPORT = 0,
    eEXPORT
};

public: 
    KsoInfo();

    //! Set values to default 
    void Reset(EIMPEXP pImpExp);
    
	//! If pTimeMode is set to KTime::eDEFAULT_MODE, pCustomFrameRate is used instead.
    void SetTimeMode(KTime::ETimeMode pTimeMode, double pCustomFrameRate = 0.0);

	//! A time mode set to KTime::eDEFAULT_MODE means a custom frame rate is set.
    KTime::ETimeMode GetTimeMode(){ return mTimeMode; }

	//! Get frame period associated with time mode or a custom frame period if time mode is set to KTime::eDEFAULT_MODE.
    KTime GetFramePeriod();
	
    void SetASFScene(KFbxObject* pASFScene, bool pASFSceneOwned = false);
    KFbxObject* GetASFScene(){ return mASFScene; }

    //-----------------
    // data

private: 
        KTime::ETimeMode mTimeMode;
        KFbxObject * mASFScene;

        EIMPEXP mImpExp;
};

class KFbxSdkManager;


// http://www.microsoft.com/globaldev/reference/winxp/langtla.mspx

//  LOCALE                            TLA  LCID

//  English - United States           ENU  0409
//  German - Germany                  DEU  0407
//  French - France                   FRA  040c
//  Japanese - Japan                  JPN  0411
//  Korean(Extended Wansung) - Korea  KOR  0412
//  Chinese - PRC                     CHS  0804

enum FBXUILANGUAGE
{
    eENU = 0,			        // MAX 409 English - United States
	eDEU,           	        // MAX 407 German - Germany
	eFRA,            	        // MAX 40c French - France
	eJPN,           	        // MAX 411 Japanese - Japan
    eKOR,            	        // MAX 412 Korean(Extended Wansung) - Korea 
    eCHS,	                    // MAX 804 Chinese - PRC
    eFBXUILANGUAGE_COUNT
};

static const char* XML_LAGUAGE_LABELS[] = 
{
	"lbENU",       // = eENU English - United States
	"lbDEU",       // = eDEU German - Germany                  
	"lbFRA",       // = eFRA French - France
	"lbJPN",       // = eJPN Japanese - Japan
	"lbKOR",       // = eKOR Korean(Extended Wansung) - Korea
	"lbCHS"        // = eCHS Chinese - PRC
};

static const char* LAGUAGE_SYMBOLS[] = 
{
	"ENU",       // = eENU English - United States
	"DEU",       // = eDEU German - Germany                  
	"FRA",       // = eFRA French - France
	"JPN",       // = eJPN Japanese - Japan
	"KOR",       // = eKOR Korean(Extended Wansung) - Korea
	"CHS"        // = eCHS Chinese - PRC
};

// this class is located in the UserDataPtr of each KFbxIOSettings properties
class PropInfo
{
public:
	PropInfo();
    ~PropInfo();

	void            * UIWidget;            // UI widget for showing the property
	void            * cbValueChanged;      // call back when value changed
    void            * cbDirty;             // call back when value changed
    KStringList       labels;              // list of labels in many languages
};



/** KFbxIOSettings
*	Class used as container for IO settings.
*   All settings are organised in a propertie hierarchie.
*   A node of settings may contain a bunch of information related to an operation (ex: import / export)
*   it may also contain informations related to a user dialog offering user options
*/
class KFBX_DLL KFbxIOSettings : public KFbxObject
{

KFBXOBJECT_DECLARE(KFbxIOSettings,KFbxObject);

		//! Destructor
public:	virtual ~KFbxIOSettings();

		//! Global allocator
public: static void AllocateIOSettings(KFbxSdkManager& pManager);

public: static bool IsIOSettingsAllocated();

		//! Release memory
public:	void FreeIOSettings();

		//! Ref accessor
public:	static KFbxIOSettings& IOSettingsRef();

		//! Empty the settings hierarchies
public:	virtual void Clear();

        //! Add a property group under the root prop
public: KFbxProperty AddPropertyGroup(char const* pName, KFbxDataType const &pDataType=KFbxDataType(), char const* pLabel="");

		//! Add a property group under another parent prop
public: KFbxProperty AddPropertyGroup(	KFbxProperty const &pParentProperty, 
			                          	char const*		    pName, 
									  	KFbxDataType const &pDataType      = KFbxDataType(), 
									  	char const*         pLabel         = "",
									  	bool				pVisible	   = true,
									  	bool				pSavable	   = true,
									  	bool				pEnabled	   = true
									 );

		//! Add a property under another parent prop with a value to set
public: KFbxProperty AddProperty(    KFbxProperty const     &pParentProperty, 
			                         char const*            pName, 
								     KFbxDataType const     &pDataType      = KFbxDataType(), 
								     char const*            pLabel          = "",
									 void const*            pValue          = NULL,
								     EFbxType               pValueType      = eUNIDENTIFIED,
									 bool					pVisible		= true,
									 bool					pSavable		= true,
									 bool					pEnabled		= true
								   );

		//! Add a property under another parent prop with a value to set and a min max values
public: KFbxProperty AddPropertyMinMax(    KFbxProperty const     &pParentProperty, 
			                         char const*            pName, 
								     KFbxDataType const     &pDataType      = KFbxDataType(), 
								     char const*            pLabel          = "",
									 void const*            pValue          = NULL,
									 double const*			pMinValue		= NULL,
									 double const*			pMaxValue		= NULL,
								     EFbxType               pValueType      = eUNIDENTIFIED,									 
									 bool					pVisible		= true,
									 bool					pSavable		= true,
									 bool					pEnabled		= true
								   );


		//! get a prop by description ex: "Export|Animation|Bake"
public: KFbxProperty GetProperty(char const* pName);

		//! get prop by description from a parent property
public: KFbxProperty GetProperty(KFbxProperty const &pParentProperty, char const* pName);

        //! get/set a bool prop by prop path
public: bool GetBoolProp(char const* pName, bool pDefValue);
public: void SetBoolProp(char const* pName, bool pValue);

        //! get/set a double prop by prop path
public: double GetDoubleProp(char const* pName, double pDefValue);
public: void   SetDoubleProp(char const* pName, double pValue);

        //! get/set a int prop by prop path
public: int    GetIntProp(char const* pName, int pDefValue);
public: void   SetIntProp(char const* pName, int pValue);

        //! get/set a KTime prop by prop path
public: KTime  GetKTimeProp(char const* pName, KTime pDefValue);
public: void   SetKTimeProp(char const* pName, KTime pValue);

        //! get/set an enum prop by prop path
public: KString GetEnumProp(char const* pName, KString pDefValue);
public: int     GetEnumProp(char const* pName, int pDefValue);
public: int     GetEnumIndex(char const* pName, KString pValue);

public: void    SetEnumProp(char const* pName, KString pValue);
public: void    SetEnumProp(char const* pName, int pValue);

public: void    RemoveEnumPropValue(char const* pName, KString pValue);
public: void    EmptyEnumProp(char const* pName);

		// set a specific flag value on a specific property
public: bool    SetFlag(char const* pName, FbxPropertyFlags::eFbxPropertyFlags propFlag, bool pValue);

        //! get/set a string prop by prop path
public: KString GetStringProp(char const* pName, KString pDefValue);
public: void    SetStringProp(char const* pName, KString pValue);


		//! Load the settings hierarchie from an XML file
public:	virtual bool ReadXMLFile(KString& path);

		//! Write the settings hierarchie to an XML file
public:	virtual bool WriteXMLFile(KString& path);

public: bool WriteXmlPropToFile(KString& pFullPath, KString& propPath);

        //! Write a property branch to the registry - can be used to save a "Preset"
public: bool WriteXmlPropToRegistry(KString& regKeyName, KString& regValueName, KString& propPath, KString& regPath );
        
        //! Read a property branch from the registry - can be used to read a "Preset"
public: bool ReadXmlPropFromRegistry(KString& regKeyName, KString& regValueName, KString& regPath);

		//! Read an XML file from MyDocument dir
public: bool ReadXmlPropFromMyDocument(KString& subDir, KString& filename);

		//! Write property branch to an XML file in MyDocument dir
public: bool WriteXmlPropToMyDocument(KString& subDir, KString& filename, KString& propPath);

public: static KString GetUserMyDocumentDir();

		

public:	enum ELoadMode         { eMERGE, eEXCLUSIVE_MERGE, eCREATE };
public: enum EQuaternionMode   { eAS_QUATERNION, eAS_EULER, eRESAMPLE };
public: enum EObjectDerivation { eBY_LAYER, eBY_ENTITY, eBY_BLOCK }; 

public:	enum ESysUnits
	{
		kUNITS_USER,
		kUNITS_INCHES, 
		kUNITS_FEET,
		kUNITS_YARDS,
		kUNITS_MILES,
		kUNITS_MILLIMETERS,
		kUNITS_CENTIMETERS,
		kUNITS_METERS,
		kUNITS_KILOMETERS
	};

public:	enum ESysFrameRate  
	{
		kFRAMERATE_USER,
		kFRAMERATE_HOURS,
		kFRAMERATE_MINUTES,
		kFRAMERATE_SECONDS,
		kFRAMERATE_MILLISECONDS,
		kFRAMERATE_GAMES_15,
		kFRAMERATE_FILM_24,
		kFRAMERATE_PAL_25,
		kFRAMERATE_NTSC_30,
		kFRAMERATE_SHOWSCAN_48,
		kFRAMERATE_PALFIELD_50,
		kFRAMERATE_NTSCFIELD_60
	};
    
// Max
public:	enum EEnveloppeSystem
	{
		eSKIN_MODIFIER,
		ePHYSIQUE,
		eBONESPRO,
		eENVELOPPE_SYSTEM_COUNT
	};

// Max
public: enum EGeometryType
	{
		eTRIANGLE,
		eSIMPLIFIED_POLY,
		ePOLY,
		eNURB,
		ePATCH,
		eGEOMETRY_TYPE_COUNT
	};


//------------------------------------------------------------------------
			// Constructor
		    // Use AllocateIOSettings() to construct a KFbxIOSettings
protected:	KFbxIOSettings(KFbxSdkManager& pManager,char const* pName);

			// add recurively a xml node equivalent for a KFbxProperty and all child KFbxProperties
private:	void AddXMLProp(KFbxProperty& pProp, xmlNodePtr pParent, xmlDocPtr pXmlDoc);

			// create a new xmlNode with all KFbxProperty info
private:	xmlNodePtr GetXMLNode(KFbxProperty& pProp);

public:     KsoInfo impInfo;
public:     KsoInfo expInfo;

public:     FBXUILANGUAGE UILanguage;

private:    void AddNewPropInfo(KFbxProperty &pProp);
private:    void DeletePropInfo(KFbxProperty &pProp);
private:    void DeleteAllPropInfo(KFbxProperty &pProp);

public:     PropInfo * GetPropInfo(KFbxProperty &pProp);
public:     KString GetLanguageLabel(KFbxProperty &pProp);
public:     void SetLanguageLabel(KFbxProperty &pProp, KString &pLabel);
public:     FBXUILANGUAGE Get_Max_Runtime_Language(KString regLocation);

private: xmlNode* GetChildXmlNode(xmlNode* parent, KString* nodeName);
private: xmlNode* AddChildXmlNode(xmlNode* parent, KString* nodeName, KString* nodeValue = NULL);
private: xmlAttr* AddAttributeToXmlNode(xmlNode* node, KString* attName, KString* attValue = NULL);

// the caller must delete the KString returned
private: KString* GetAttributeFromXmlNode(xmlNode* node, KString* attName);

// the caller must delete the KString returned
private: KString* GetInnerTextXmlNode(xmlNode* node);


public: bool IsEnumExist(KFbxProperty& pProp, KString& enumString);
public: int  GetEnumIndex(KFbxProperty& pProp, KString& enumString, bool pNoCase = false);

private: bool WriteXmlDocToRegistry(KString* regPath, KString* regKeyName, KString* regValueName, xmlDocPtr XmlDoc);

// the caller must delete the xmlDocPtr returned
private: xmlDocPtr ReadXmlDocFromRegistry(KString* regPath, KString* regKeyName, KString* regValueName );

private: void CreatePropRecursive(xmlNode* pNode, KFbxProperty& pPropParent);

public: virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pType) const;

};

#include <fbxfilesdk_nsend.h>

#endif // KARCH_ENV_MACOSX_CFM

#endif // #define _FBXSDK_IO_SETTINGS_H_

