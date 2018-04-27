/*!  \file kfbxnull.h
 */

#ifndef _FBXSDK_NULL_H_
#define _FBXSDK_NULL_H_

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

#include <kfbxplugins/kfbxnodeattribute.h>

#include <fbxfilesdk_nsbegin.h>

class KFbxSdkManager;

/** \brief This node attribute contains the properties of a null node.
  * \nosubgrouping
  */
class KFBX_DLL KFbxNull : public KFbxNodeAttribute
{
    KFBXOBJECT_DECLARE(KFbxNull,KFbxNodeAttribute);

public:
    //! Return the type of node attribute which is EAttributeType::eNULL.
    virtual EAttributeType GetAttributeType() const;

    //! Reset to default values.
    void Reset();

    /**
      * \name Null Properties
      */
    //@{

    /** \enum ELook Null look types.
      * - \e eNONE
      * - \e eCROSS
      */
    typedef enum {
        eNONE,
        eCROSS,
    } ELook;

    /** Get a flag used to verify that the size has been set.
      * \return      \c true if SetSize() was called for this object.
      * \remarks     When an attribute is not set, the application can choose to ignore the attribute or use the default value.
      *              When Reset() is called, this flag is set to \c false.
      * \remarks     This function is OBSOLETE, DO NOT USE.  It will always return false.  It will be removed on in the next release.
	  * \remarks     This function is deprecated. Use property Size instead.
      */
    K_DEPRECATED bool GetSizeIsSet() const;

    /** Get the default value for the size.
      * \return     Default size of this object (100).
      */
    double GetSizeDefaultValue() const;

    //@}

    /**
      * \name Property Names
      */
    //@{
    static const char*          sSize;
    static const char*          sLook;
    //@}

    /**
      * \name Property Default Values
      */
    //@{
    static const fbxDouble1     sDefaultSize;
    static const ELook      sDefaultLook;
    //@}


    //////////////////////////////////////////////////////////////////////////
    //
    // Properties
    //
    //////////////////////////////////////////////////////////////////////////

    /** This property handles size.
      *
      * To access this property do: Size.Get().
      * To set this property do: Size.Set(fbxDouble1).
      *
      * Default value is 100.
      */
    KFbxTypedProperty<fbxDouble1>       Size;

    /** This property handles the look of the null.
      *
      * To access this property do: Look.Get().
      * To set this property do: Look.Set(ELook).
      *
      * Default value is true
      */
    KFbxTypedProperty<ELook>            Look;


///////////////////////////////////////////////////////////////////////////////
//
//  WARNING!
//
//  Anything beyond these lines may not be documented accurately and is
//  subject to change without notice.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef DOXYGEN_SHOULD_SKIP_THIS

public:
    // Clone
    virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType = eDEEP_CLONE) const;

protected:

    KFbxNull(KFbxSdkManager& pManager, char const* pName);
    ~KFbxNull();

    virtual void Construct(const KFbxNull* pFrom);
    virtual bool ConstructProperties(bool pForceSet);
    virtual void Destruct(bool pRecursive, bool pDependents);

    //! Assignment operator.
    KFbxNull& operator=(KFbxNull const& pNull);

    bool mSizeIsSet;

    virtual KString     GetTypeName() const;
    virtual KStringList GetTypeFlags() const;

    friend class KFbxReaderFbx;
    friend class KFbxWriterFbx;
    friend class KFbxWriterFbx6;
    friend struct KFbxWriterFbx7Impl;

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

};

typedef KFbxNull* HKFbxNull;

inline EFbxType FbxTypeOf( KFbxNull::ELook const &pItem )           { return eENUM; }

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_NULL_H_


