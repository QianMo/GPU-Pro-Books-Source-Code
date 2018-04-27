/*!  \file kfbxplug.h
 */

#ifndef _FBXSDK_PLUG_H_
#define _FBXSDK_PLUG_H_
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

#include <kfbxevents/kfbxemitter.h>

#include <kbaselib_nsbegin.h> // namespace 

#include <kbaselib_nsend.h> // namespace 
#ifndef MB_FBXSDK
	#include <kbaselib_nsuse.h> // namespace 
#endif

#include <fbxfilesdk_nsbegin.h>

class KFbxPlug;
class KFbxSdkManager;
class KFbxClassIdInfo;
class KFbxObject;
class KFbxContainer;
class KFbxPropertyHandle;

typedef KFbxPlug* (*kFbxPlugConstructor)(KFbxSdkManager& pManager, const char* pName, const KFbxPlug* pFrom, const char* pFBXType, const char* pFBXSubType);

/** Base To define the ClassId of an object
  * \remarks This class Id helps the fbxsdk identify the class hierarchy of plugs and objects. Each Plug is identified by a class id and a parent classId
  * \nosubgrouping
  */

class KFBX_DLL kFbxClassId
{
public:
	/** 
	  * \name Constructor and Destructor.
	  */
	//@{
 
	//!Constructor.
	kFbxClassId();

	/** Constructor.
	  * \param pClassName         Class name.
	  * \param pParentClassId     Parent class id.
	  * \param pConstructor       Pointer to a function which constructs KFbxPlug.
	  * \param pFBXType           Fbx file type name.
	  * \param pFBXSubType        Fbx file subtype name.
	  */
	kFbxClassId(const char* pClassName, const kFbxClassId &pParentClassId, kFbxPlugConstructor pConstructor=0, const char* pFBXType=NULL, const char* pFBXSubType=NULL);

	//! Delete this class id.
	void Destroy();
	//@}

	/** Retrieve the class name. 
	  * \return              Class name.
	  */
	const char* GetName() const;

    /**Retrieve the parent class id.
	  *\return      Parent class id.
	  */
	kFbxClassId GetParent() const;

    /** Creat a KFbxPlug from the specified KFbxPlug
	  * \param pManager              The object manager
	  * \param pName                 KFbxPlug name
	  * \param pFrom                 The specified KFbxPlug
	  * \return                      KFbxPlug
	  */
	KFbxPlug* Create(KFbxSdkManager& pManager, const char* pName, const KFbxPlug* pFrom);

	/** Override the KFbxPlug constructor.
	  * \param pConstructor         New KFbxPlug constructor.
	  */
	bool Override(kFbxPlugConstructor pConstructor);

	/** Test if this class is a hierarchical children of the specified class type  
	  * \param pId                   Representing the class type  
	  * \return                      \c true if the object is a hierarchical children of the type specified, \c false otherwise.               
	  */
	bool Is(kFbxClassId const pId) const;

	/** Equivalence operator.
	  * \param pClassId             Another class id to be compared with this class id.
	  * \return                     \c true if equal, \c false otherwise.
	  */
	bool operator == (kFbxClassId const& pClassId) const;

	/** Retrieve the information of this class id.
	  * \return                     the class information.
	  */
	inline KFbxClassIdInfo* GetClassIdInfo() { return mClassInfo; }

   /** Retrieve the information of this class id.
	 * \return                      the class information.
	 */
    inline const KFbxClassIdInfo* GetClassIdInfo() const  { return mClassInfo; }

private:
	friend class KFbxSdkManager;

	//!Set the fbx file type name. 
	bool SetFbxFileTypeName(const char* pName);

	//!Set the fbx file subtype name.
	bool SetFbxFileSubTypeName(const char* pName);
    
public:
	/** Retrieve the fbx file type name.
	  * \param pAskParent   a flag on whether to ask the parent for file type name.
	  * \return             the fbx file type name.
	  */
	const char* GetFbxFileTypeName(bool pAskParent=false) const;

    //!Retrieve the fbx file subtype name.
	const char* GetFbxFileSubTypeName() const;


	/** Get whether this class type is valid.
	  * \return             \c ture if valid, \c false otherwise.
	  */
	inline bool IsValid() const { return mClassInfo ? true : false; }
    
	//! Set object type prefix.
	void SetObjectTypePrefix(const char* pObjectTypePrefix);

	//! Get object type prefix.
	const char* GetObjectTypePrefix();
   
    //!Get the default property handle of root class.
	KFbxPropertyHandle* GetRootClassDefaultPropertyHandle();

	/**Increase the instance reference count of this class type.
	  * \return             the instance reference of this type after increase.
	  */
	int ClassInstanceIncRef();

    /**Decrease the instance reference count of this class type.
	  * \return             the instance reference of this type after decrease.
	  */
	int ClassInstanceDecRef();
	 
	/**Retrieve the instance reference count of this class type.
	  * \return             the instance reference of this type.
	  */
	int GetInstanceRef();

private:
	//!Copy constructor.
	kFbxClassId(KFbxClassIdInfo* mClassInfo);
	KFbxClassIdInfo* mClassInfo;
};

#define KFBXPLUG_DECLARE(Class)											\
	public:																\
    static kFbxClassId ClassId;											\
	static Class* Create(KFbxSdkManager *pManager, const char *pName);	\
	static Class* SdkManagerCreate(KFbxSdkManager *pManager, const char *pName, Class* pFrom)	\
	{																	\
		Class* lClass = new Class(*pManager,pName);						\
		lClass->Construct(pFrom);											\
		return lClass;													\
	}																	\
	virtual kFbxClassId	GetClassId() const { return ClassId; }	\
																		\
	friend class FBXFILESDK_NAMESPACE::KFbxSdkManager;

#define KFBXPLUG_DECLARE_ABSTRACT(Class)								\
	public:																\
	static kFbxClassId	ClassId;										\
	static Class* Create(KFbxSdkManager *pManager, const char *pName);	\
	static kFbxPlugConstructor SdkManagerCreate;						\
	virtual kFbxClassId	GetClassId() const { return ClassId; }	\
																		\
	friend class FBXFILESDK_NAMESPACE::KFbxSdkManager;

#define KFBXPLUG_IMPLEMENT(Class)										\
	kFbxClassId	Class::ClassId;											\
	Class* Class::Create(KFbxSdkManager *pManager, const char *pName)	\
	{																	\
		Class* ClassPtr=0;												\
		return (Class *)pManager->CreateClass( Class::ClassId,pName,NULL );	\
	}																	
	
#define KFBXPLUG_IMPLEMENT_ABSTRACT(Class)								\
	kFbxClassId	Class::ClassId;											\
	kFbxPlugConstructor Class::SdkManagerCreate = 0;					\
	Class* Class::Create(KFbxSdkManager *pManager, const char *pName)	\
	{																	\
		Class* ClassPtr=0;												\
		return (Class *)pManager->CreateClass( Class::ClassId,pName, NULL );	\
	}																	\
	
/** Base class to handle plug connections.
  * \remarks This class is for the FBX SDK internal use only.
  * \nosubgrouping
  */
class KFBX_DLL KFbxPlug : public kfbxevents::KFbxEmitter
{
	KFBXPLUG_DECLARE(KFbxPlug);

	/**
	  * \name Constructor and Destructors.
	  */
	//@{
public:
	/** Delete the object and Unregister from the FbxSdkManager
	  */
	virtual void Destroy(bool pRecursive=false, bool pDependents=false);
	//@}

	/**
	  * \name Object ownership and type management.
	  */
	//@{
public:
	/** Get the KFbxSdkManager that created this object 
	  * \return Pointer to the KFbxSdkManager
	  */
	virtual KFbxSdkManager*	GetFbxSdkManager() const { return 0; }
	/** Test if the class is a hierarchical children of the specified class type  
	  * \param pClassId ClassId representing the class type 
	  * \return Returns true if the object is of the type specified
	  */
	virtual bool					Is(kFbxClassId pClassId) const			{ return GetClassId().Is(pClassId);	}
	template < class T >inline bool	Is(T *pFBX_TYPE) const					{ return Is(T::ClassId); }
	virtual bool					IsRuntime(kFbxClassId pClassId) const	{ return GetRuntimeClassId().Is(pClassId);	}
	virtual bool					SetRuntimeClassId(kFbxClassId pClassId);
	virtual kFbxClassId				GetRuntimeClassId() const;
	virtual bool					IsRuntimePlug() const					{ return !( GetRuntimeClassId() == GetClassId() ); }
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
protected:
	inline KFbxPlug() {}
	inline KFbxPlug(KFbxSdkManager& pManager, const char* pName) {}
	virtual ~KFbxPlug() {}

	virtual void Construct(const KFbxPlug* pFrom);
	virtual void Destruct(bool pRecursive, bool pDependents);
	friend class KFbxProperty;
	friend class KFbxObject;
#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS
};

#define FBX_TYPE(class) ((class const*)0)
#define FBX_CLASSID(class) (class::ClassId)

template < class T > inline T* KFbxCast(KFbxPlug *pPlug)
{
	return pPlug && pPlug->Is(FBX_CLASSID(T)) ? (T *)pPlug : 0;
}

template < class T > inline T const* KFbxCast(KFbxPlug const*pPlug)
{
	return pPlug && pPlug->Is(FBX_CLASSID(T)) ? (T const*)pPlug : 0;
}

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_PLUG_H_

