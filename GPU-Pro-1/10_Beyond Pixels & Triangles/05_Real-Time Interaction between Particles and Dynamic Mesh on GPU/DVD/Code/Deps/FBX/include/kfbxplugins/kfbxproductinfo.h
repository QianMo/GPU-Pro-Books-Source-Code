/*!  \file kfbxproductinfo.h
 */

#ifndef _KFbxProductInfo_h
#define _KFbxProductInfo_h

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

#include <klib/karrayul.h>
#include <klib/kstring.h>
#include <klib/kstringlist.h>

#include <kfbxplugins/kfbxutilities.h>

#ifndef MB_FBXSDK
#include <kbaselib_nsuse.h>
#endif

#include <kbaselib_forward.h>

#include <fbxfilesdk_nsbegin.h>

#define	PRODUCTINFO_FBXINFO_STRUCTURE			"fbxinfo"
#define	PRODUCTINFO_OS_STRUCTURE				"os"
#define	PRODUCTINFO_OS_NAME_PROPERTY			"name"
#define	PRODUCTINFO_FBXSDK_STRUCTURE			"fbxsdk"
#define	PRODUCTINFO_FBXPLUGIN_STRUCTURE			"fbxplugin"
#define	PRODUCTINFO_FBXPLUGIN_NAME_PROPERTY		"name"
#define	PRODUCTINFO_FBXPLUGIN_SHOW_PROPERTY		"show"
#define	PRODUCTINFO_PACKAGE_NAME_STRUCTURE		"package_name"
#define	PRODUCTINFO_PACKAGE_VERSION_STRUCTURE	"package_version"
#define	PRODUCTINFO_VERSION_NUMBER_STRUCTURE	"version_number"
#define	PRODUCTINFO_BUILD_NUMBER_STRUCTURE		"build_number"
#define	PRODUCTINFO_URL_STRUCTURE				"url"
#define	PRODUCTINFO_MESSAGE_STRUCTURE			"message"
#define	PRODUCTINFO_MESSAGE_LANG_PROPERTY		"lang"

/**FBX SDK product information class
  * \nosubgrouping
  */
class KFBX_DLL KFbxProductInfo
{
public:
	/**
	  * \name Constructors and Destructor
	  */
	//@{

	/** Constructor.
	  * \param pProduct                 Product name.
	  * \param pPackageVersion          Package version.
	  * \param pOS                      Operating System.
	  * \param pVersion                 Product version.
	  * \param pBuildNumber             Build number.
	  * \param pURL                     Product URL.
	  * \param pLang                    Product language
	  * \param pMessage                 Product message               
	  */
	KFbxProductInfo(KString pProduct, KString pPackageVersion, KString pOS, KString pVersion,
					KString pBuildNumber, KString pURL, KString pLang = "", KString pMessage = "");

	/** Constructor.
	  * \param pProduct                 Product name.
	  * \param pPackageVersion          Package version.
	  * \param pOS                      Operating System.
	  * \param pVersion                 Product version.
	  * \param pBuildNumber             Build number.
	  * \param pURL                     Product URL.
	  * \param pMessageList             Product message list
	  * \param pShow                    \c True if product can show.
	  */
	KFbxProductInfo(KString pProduct, KString pPackageVersion, KString pOS, KString pVersion,
					KString pBuildNumber, KString pURL, KStringList *pMessageList = NULL, bool pShow = true);

	//!Destructor.
	virtual ~KFbxProductInfo ();
	//@}

	/**
	  * \name Access.
	  */
	//@{

	/** Retrieve the product name.
	  *\return                 Product name.
	  */
	KString		GetProduct()		{ return mProduct; }

	/** Get whether product can show.
	  * \return                \c True if product can show, \c false otherwise.
	  */
	bool		Show()				{ return mShow; }

    /** Retrieve product package version.
	  *\return                 Product package version.
	  */
	KString		GetPackageVersion() { return mPackageVersion; }

	/** Retrieve the OS.
	  *\return                 OS.
	  */
	KString		GetOS()				{ return mOS; }

    /** Retrieve product version .
	  *\return                 Product version.
	  */
	KString		GetVersion()		{ return mVersion; }

	/** Retrieve product build number.
	  *\return                 Product build number.
	  */
	KString		GetBuildNumber()	{ return mBuildNumber; }

    /** Retrieve product URL.
	  *\return                 Product URL.
	  */
	KString		GetURL()			{ return mURL; }

    /** Retrieve product language count.
	  * \return                Product language count.
	  */
	int			GetLangCount()		{ return mMessageList->GetCount(); }

    /** Retrieve a product language specified by the index i.
	  * \param i               The index of product language.
	  * \return                Product language.
	  */
	KString		GetLang(int i)		{ return mMessageList->GetStringAt(i); }

   /** Retrieve a product message specified by the index i.
	  * \param i               The index of product message.
	  * \return                Product message.
	  */
	KString		GetMessageStr(int i){ return *(KString*)mMessageList->GetReferenceAt(i); }
    //@}
private:
	KString		mProduct;
	bool		mShow;
	KString		mPackageVersion;
	KString		mOS;
	KString		mVersion;
	KString		mBuildNumber;
	KString		mURL;
	KStringList	*mMessageList;

};


class KFbxProductInfo;
typedef class KFBX_DLL KArrayTemplate<KFbxProductInfo *> KFbxArrayProductInfo;

/** FBX SDK product information builder class
  * \nosubgrouping
  */
class KFBX_DLL KFbxProductInfoBuilder
{
public:
    
	/**
	  * \name Constructor and Destructor
	  */
	//@{

	//! Constructor.
	KFbxProductInfoBuilder();

	//! Destructor.
	virtual ~KFbxProductInfoBuilder ();

    //@}

	/** Load file product info;
	  * Then fill an array of KFbxProductInfo for this pProduct (e.g. Maya, Max...),
	  * pPackageVersion (7.0, 7.5...), pOS (Linux, Windows, Windows64, ...), 
	  * (function pFCompareProducts returns 0 for these values)
	  * and that are more recent than the given pVersion and pBuildNumber
	  * (function pFCompareVersions returns 1 for these values).
	  * \param pUrl                         Product URL.
	  * \param pArrayProductInfo            Result array of KFbxProductInfos after comparison. 
	  * \param pRefProductInfo              The product to compare with.
	  * \param pFCompareProducts            The fuction to compare product, package version and OS.
	  * \param pFCompareVersions            The fuction to compare version and buildnumber.
	  * \return                             \c True if OK,\c false otherwise.
	  */
	bool FillArray(
			KString pUrl,
			KFbxArrayProductInfo &pArrayProductInfo,
			KFbxProductInfo *pRefProductInfo,
			int (*pFCompareProducts)(const void*, const void*),
			int (*pFCompareVersions)(const void*, const void*)
	);

private:
	// Array of all KFbxProductInfo found in mXmlDoc
	KFbxArrayProductInfo	mArrayProductInfo;
};

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _KFbxProductInfo


