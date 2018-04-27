/*!  \file kfbxvertexcachedeformer.h
 */

#ifndef _FBXSDK_VERTEX_CACHE_DEFORMER_H_ 
#define _FBXSDK_VERTEX_CACHE_DEFORMER_H_

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
#include <kfbxplugins/kfbxdeformer.h>
#include <kfbxplugins/kfbxcache.h>

#include <fbxfilesdk_nsbegin.h>

/** \brief This class deforms control points of a geometry using control point positions
 * stored in the associated cache object.
 * \nosubgrouping
 */
class KFBX_DLL KFbxVertexCacheDeformer : public KFbxDeformer
{
	KFBXOBJECT_DECLARE(KFbxVertexCacheDeformer,KFbxDeformer);

public:
	
	/** Assign a cache object to be used by this deformer.
	  * \param pCache     The cache object.
	  */ 
	void SetCache( KFbxCache* pCache );

	/** Get the cache object used by this deformer.
	  * \return     A pointer to the cache object used by this deformer, or \c NULL if no cache object is assigned.
	  */
	KFbxCache* GetCache() const;

	/** Select the cache channel by name.
	  * \param pName     The name of channel to use within the cache object.
	  */
	void SetCacheChannel( const char* pName );

	/** Get the name of the selected channel.
      * \return     The name of the selected channel within the cache object.
	  */
	KString GetCacheChannel() const;

	/** Activate the deformer.
	  * \param pValue     Set to \c true to enable the deformer.
	  */
	void SetActive( bool pValue );

	/** Indicate if the deformer is active or not.
	  * \return     The current state of the deformer.
	  */
	bool IsActive() const;

	/** Get the deformer type.
	  * \return     Deformer type identifier.
	  */
	virtual EDeformerType GetDeformerType() const { return KFbxDeformer::eVERTEX_CACHE; }

	//! Assigment operator.
	KFbxVertexCacheDeformer& operator = (KFbxVertexCacheDeformer const& pDeformer);

protected:
	/**
	 * \name Properties
	 */
	//@{
		KFbxTypedProperty<fbxBool1>		Active;
		KFbxTypedProperty<fbxString>	Channel;
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
	static const char* ChannelPropertyName;
	static const char* ActivePropertyName;


	// Clone
	virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;

protected:

	KFbxVertexCacheDeformer(KFbxSdkManager& pManager, char const* pName);
	virtual ~KFbxVertexCacheDeformer();

	virtual bool ConstructProperties(bool pForceSet);
	virtual void Destruct(bool pRecursive, bool pDependents);

	virtual KString		GetTypeName() const;
	virtual KStringList	GetTypeFlags() const;

	friend class KFbxReaderFbx;
	friend class KFbxReaderFbx6;
	friend class KFbxWriterFbx;
	friend class KFbxWriterFbx6;

private:
	void ClearCacheConnections();

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

};

#include <fbxfilesdk_nsend.h>

#endif // _FBXSDK_VERTEX_CACHE_DEFORMER_H_ 
