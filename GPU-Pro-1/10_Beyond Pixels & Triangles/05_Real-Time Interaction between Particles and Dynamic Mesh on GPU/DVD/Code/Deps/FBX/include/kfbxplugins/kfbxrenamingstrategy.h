/*!  \file kfbxrenamingstrategy.h
 */
#ifndef _FBXSDK_RENAMING_STRATEGY_H_
#define _FBXSDK_RENAMING_STRATEGY_H_

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

#include <klib/kstring.h>
#include <klib/kstringlist.h>
#include <klib/krenamingstrategy.h>
#include <klib/kcharptrset.h>

#ifndef MB_FBXSDK
#include <kbaselib_nsuse.h>
#endif

#include <fbxfilesdk_nsbegin.h>
class KFbxNode;
class KFbxTexture;
class KFbxVideo;
class KFbxScene;
class KFbxNodeAttribute;


/** \brief This class contains the description of the FBX renaming strategy.
  * \nosubgrouping
  * The KFbxRenamingStrategy object can be setup to rename all the objects in a scene.
  * It can remove nameclashing, remove illegal characters, manage namespaces, and manage backward compatibility.
  *
  */

class KFBX_DLL KFbxRenamingStrategy : public KRenamingStrategy
{
public:

 	/** \enum EMode
	  * - \e eTO_FBX
	  * - \e eFROM_FBX
	  * - \e eMODE_COUNT
	  */
	enum EMode
	{
		eTO_FBX,
		eFROM_FBX,
		eMODE_COUNT
	};

 	/** \enum EClashType
	  * - \e eNAMECLASH_AUTO
	  * - \e eNAMECLASH_TYPE1
	  * - \e eNAMECLASH_TYPE2
	  */
	enum EClashType
	{
		eNAMECLASH_AUTO,
		eNAMECLASH_TYPE1,
		eNAMECLASH_TYPE2
	};


	//! Constructor.
	KFbxRenamingStrategy(EMode pMod, bool pOnCreationRun = false);

	//! Destructor.
	virtual ~KFbxRenamingStrategy ();

	//! Setup the strategy to perform this algorithm
	void SetClashSoverType(EClashType pType);
	
	/** Rename.
	* \param pName     New name.
	* \return          \c true if successful, \c false otherwise.
	*/
	virtual bool Rename(KName& pName);

	//! Empty all memories about given names
	virtual void Clear();
	
	/** Spawn mechanism.  
	 * Create a dynamic renaming strategy instance of the same type.
	 * \return     new KRenamingStrategy
	 */
	virtual KRenamingStrategy* Clone();

	/** Returns a name with its prefix removed.
	 * \param pName    A name containning a prefix.
	 * \return         The part of pName following the "::"
	 */
	static char* NoPrefixName (char const* pName);
	static char* NoPrefixName (KString& pName);

	/** Get the namespace of the last renamed object.
	 * \return     Char pointer to the namespace.
	 */
	virtual char* GetNameSpace() { return mNameSpace.Buffer(); } 

	/** Sets the current scene namespace symbol.
	 * \param pNameSpaceSymbol     namespace symbol.
	 */
	virtual void SetInNameSpaceSymbol(KString pNameSpaceSymbol){mInNameSpaceSymbol = pNameSpaceSymbol;}
	
	/** Sets the wanted scene namespace symbol.
	 * \param pNameSpaceSymbol     namespace symbol.
	 */
	virtual void SetOutNameSpaceSymbol(KString pNameSpaceSymbol){mOutNameSpaceSymbol = pNameSpaceSymbol;}
	
	/** Sets case sensitivity for nameclashing.
	 * \param pIsCaseSensitive     Set to \c true to make the nameclashing case sensitive.
	 */
	virtual void SetCaseSensibility(bool pIsCaseSensitive){mCaseSensitive = pIsCaseSensitive ;}

	/** Sets the flag for character acceptance during renaming.
	 * \param pReplaceNonAlphaNum     Set to \c true to replace illegal characters with an underscore ("_").  
	 */
	virtual void SetReplaceNonAlphaNum(bool pReplaceNonAlphaNum){mReplaceNonAlphaNum = pReplaceNonAlphaNum;}

	/** Sets the flag for first character acceptance during renaming.
	 * \param pFirstNotNum     Set to \c true to add an underscore to the name if the first character is a number.
	 */
	virtual void SetFirstNotNum(bool pFirstNotNum){mFirstNotNum = pFirstNotNum;}

	/** Recusively renames all the unparented namespaced objects (Prefix mode) starting from this node.
	 * \param pNode       Parent node.
	 * \param pIsRoot     The root node.
     * \remarks           This function adds "_NSclash" when it encounters an unparented namespaced object.
	 */
	virtual bool RenameUnparentNameSpace(KFbxNode* pNode, bool pIsRoot = false);

	/** Recusively removes all the unparented namespaced "key" starting from this node.
	 * \param pNode     Parent node.
     * \remarks         This function removes "_NSclash" when encountered. This is the opposite from RenameUnparentNameSpace.
	 */
	virtual bool RemoveImportNameSpaceClash(KFbxNode* pNode);

#ifndef DOXYGEN_SHOULD_SKIP_THIS

	virtual void GetParentsNameSpaceList(KFbxNode* pNode, KArrayTemplate<KString*> &pNameSpaceList);
	virtual bool PropagateNameSpaceChange(KFbxNode* pNode, KString OldNS, KString NewNS);

protected:

	virtual bool RenameToFBX(KName& pName);
	virtual bool RenameFromFBX(KName& pName);
	virtual KString& ReplaceNonAlphaNum(KString& pName,	char* pReplace, bool pIgnoreNameSpace);

	EMode mMode;
	EClashType mType;

	struct NameCell
	{
		NameCell(char const* pName) :
			mName(pName),
			mInstanceCount(0)
		{
		}
			
		KString mName;
		int mInstanceCount;		
	};


	KCharPtrSet					mStringNameArray;
	KArrayTemplate<NameCell*>	mExistingNsList;
	bool						mOnCreationRun;
	bool						mCaseSensitive;
	bool						mReplaceNonAlphaNum;
	bool						mFirstNotNum;
	KString						mNameSpace;
	KString						mInNameSpaceSymbol; //symbol identifying a name space
	KString						mOutNameSpaceSymbol; 

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS 

};

/** \brief This class contains the description of the FBX scene renamer.
  * \nosubgrouping
  * The KFbxSceneRenamer provides a way to easily rename objects in a scene without 
  * using the KFbxRenamingStrategy class. KFbxSceneRenamer removes nameclashing, illegal characters, and manages namespaces.
  * 
  *
  */

class KFBX_DLL KFbxSceneRenamer
{

public:

	/** Create an object of type KFbxSceneRenamer.
	  * \param pScene     Scene to be renamed.
	  */
	KFbxSceneRenamer(KFbxScene* pScene){mScene = pScene;};

	/** Deletes this object. 
	  * All the references will become invalid.
	  */
	virtual ~KFbxSceneRenamer(){};

 	/** \enum ERenamingMode
	  * - \e eNONE
	  * - \e eMAYA_TO_FBX5
	  * - \e eMAYA_TO_FBX_MB75
	  * - \e eMAYA_TO_FBX_MB70
	  * - \e eFBXMB75_TO_FBXMB70
	  * - \e eFBX_TO_FBX
	  * - \e eMAYA_TO_FBX
	  * - \e eFBX_TO_MAYA
	  * - \e eLW_TO_FBX
	  * - \e eFBX_TO_LW
	  * - \e eXSI_TO_FBX
	  * - \e eFBX_TO_XSI
	  * - \e eMAX_TO_FBX
	  * - \e eFBX_TO_MAX
	  * - \e eMB_TO_FBX
	  * - \e eFBX_TO_MB
	  * - \e eDAE_TO_FBX
	  * - \e eFBX_TO_DAE
	  */
	enum ERenamingMode
	{ 
		eNONE,
		eMAYA_TO_FBX5,
		eMAYA_TO_FBX_MB75,
		eMAYA_TO_FBX_MB70,
		eFBXMB75_TO_FBXMB70,
		eFBX_TO_FBX,
		eMAYA_TO_FBX,
		eFBX_TO_MAYA,
		eLW_TO_FBX,
		eFBX_TO_LW,
		eXSI_TO_FBX,
		eFBX_TO_XSI,
		eMAX_TO_FBX,
		eFBX_TO_MAX,
		eMB_TO_FBX,
		eFBX_TO_MB,
		eDAE_TO_FBX,
		eFBX_TO_DAE
	};

	void RenameFor(ERenamingMode pMode);

	/** Rename all object to remove name clashing.
	 * \param pFromFbx                  Set to \c true to enable this flag.
	 * \param pIgnoreNS                 Set to \c true to enable this flag.
	 * \param pIsCaseSensitive          Set to \c true to enable case sensitive renaming.
	 * \param pReplaceNonAlphaNum       Set to \c true to replace non-alphanumeric characters with underscores ("_").
	 * \param pFirstNotNum              Set to \c true toadd a leading _ if first char is a number (for xs:NCName).
	 * \param pInNameSpaceSymbol        Identifier of a namespace.
	 * \param pOutNameSpaceSymbol       Identifier of a namespace.
	 * \param pNoUnparentNS             Set to \c true to not not allow unparent namespace.
	 * \param pRemoveNameSpaceClash     Set to \c true to remove NameSpaceClash token.
	  * \return void.
	 */
	void ResolveNameClashing(	bool pFromFbx, bool pIgnoreNS, bool pIsCaseSensitive,
								bool pReplaceNonAlphaNum, bool pFirstNotNum,
								KString pInNameSpaceSymbol, KString pOutNameSpaceSymbol,
								bool pNoUnparentNS/*for MB < 7.5*/, bool pRemoveNameSpaceClash);
private:

	KRenamingStrategy* mNodeRenamingStrategy;
	KFbxScene* mScene;
};


#include <fbxfilesdk_nsend.h>

#endif // #define _FBXSDK_RENAMING_STRATEGY_H_

