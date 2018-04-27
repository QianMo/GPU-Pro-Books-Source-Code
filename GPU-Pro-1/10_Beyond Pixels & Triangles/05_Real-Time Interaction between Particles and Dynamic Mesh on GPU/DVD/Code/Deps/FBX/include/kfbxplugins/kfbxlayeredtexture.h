/*!  \file kfbxcache.h
 */

#ifndef _FBXSDK_LAYERED_TEXTURE_H_ 
#define _FBXSDK_LAYERED_TEXTURE_H_

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
#include <kfbxplugins/kfbxgroupname.h>
#include <kfbxplugins/kfbxtexture.h>

#include <fbxfilesdk_nsbegin.h>

/**FBX SDK layered texture class
  * \nosubgrouping
  */
class KFBX_DLL KFbxLayeredTexture : public KFbxTexture
{
public:

	KFBXOBJECT_DECLARE(KFbxLayeredTexture,KFbxTexture);

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


	//KFbxTypedProperty<EBlendMode> BlendMode;

	/** Equality operator
	  * \param pOther                      The object to compare to.
	  * \return                            \c true if pOther is equivalent to this object,\c false otherwise.
	  */
	bool operator==( const KFbxLayeredTexture& pOther ) const;

    /** Set the blending mode for a texture
      * \param pIndex                      The texture index.
      * \param pMode                       The blend mode to set.
      * \return                            \c true on success, \c false otherwise.
      */
    bool SetTextureBlendMode( int pIndex, EBlendMode pMode ); 

    /** Get the blending mode for a texture
      * \param pIndex                      The texture index.
      * \param pMode                       The blend mode is returned here.
      * \return                            \c true on success,\c false otherwise.
      */
    bool GetTextureBlendMode( int pIndex, EBlendMode& pMode ) const;

///////////////////////////////////////////////////////////////////////////////
//
//  WARNING!
//
//	Anything beyond these lines may not be documented accurately and is 
// 	subject to change without notice.
//
///////////////////////////////////////////////////////////////////////////////
#ifndef DOXYGEN_SHOULD_SKIP_THIS

protected:

    struct InputData
    {
        EBlendMode mBlendMode;
    };

    KArrayTemplate<InputData> mInputData;

    KFbxLayeredTexture(KFbxSdkManager& pManager, char const* pName);  
	virtual ~KFbxLayeredTexture();

	virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;
	virtual bool ConstructProperties(bool pForceSet);
	virtual KFbxLayeredTexture& operator = (KFbxLayeredTexture const& pSrc);

    virtual bool ConnecNotify (KFbxConnectEvent const &pEvent);

    bool RemoveInputData( int pIndex );

    friend class KFbxWriterFbx6;
    friend struct KFbxWriterFbx7Impl;

    friend class KFbxReaderFbx6;
    friend struct KFbxReaderFbx7Impl;
private:

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS
};

inline EFbxType FbxTypeOf( KFbxLayeredTexture::EBlendMode const &pItem )				{ return eENUM; }

#include <fbxfilesdk_nsend.h>

#endif //_FBXSDK_LAYERED_TEXTURE_H_ 

