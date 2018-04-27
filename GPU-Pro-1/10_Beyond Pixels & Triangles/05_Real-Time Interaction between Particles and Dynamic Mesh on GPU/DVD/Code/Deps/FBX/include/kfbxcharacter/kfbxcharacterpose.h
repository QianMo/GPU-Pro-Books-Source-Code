/*!  \file kfbxcharacterpose.h
 */

#ifndef _FBXSDK_CHARACTER_POSE_H_
#define _FBXSDK_CHARACTER_POSE_H_

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

#include <kfbxplugins/kfbxobject.h>

#include <klib/kstring.h>

#include <kfbxmath/kfbxxmatrix.h>

#include <kfbxplugins/kfbxnode.h>

#include <kfbxcharacter/kfbxcharacter.h>

#include <fbxfilesdk_nsbegin.h>

class KFbxScene;
class KFbxSdkManager;

class KFbxCharacterPose_internal;

/** \class KFbxCharacterPose
  * \nosubgrouping
  * \brief A character pose is a character and an associated hierarchy of nodes.
  *
  * Only the default position of the nodes is considered, the animation data is ignored.
  *
  * You can access the content of a character pose, using the functions KFbxCharacterPose::GetOffset(),
  * KFbxCharacterPose::GetLocalPosition(), and KFbxCharacterPose::GetGlobalPosition().
  * Their source code is provided inline as examples on how to access the character pose data.
  *
  * To create a character pose, You must first create a hierarchy of nodes under the root
  * node provided by function KFbxCharacterPose::GetRootNode(). Then, feed this hierarchy
  * of nodes into the character returned by function KFbxCharacterPose::GetCharacter().
  * Offsets are set in the character links. Local positions are set using
  * KFbxNode::SetDefaultT(), KFbxNode::SetDefaultR(), and KFbxNode::SetDefaultS().
  *
  * To set local positions from global positions:
  *     -# Declare lCharacterPose as a valid pointer to a KFbxCharacterPose;
  *     -# Call lCharacterPose->GetRootNode()->SetLocalStateId(0, true);
  *     -# Call lCharacterPose->GetRootNode()->SetGlobalStateId(1, true);
  *     -# Call KFbxNode::SetGlobalState() for all nodes found in the hierarchy under lCharacterPose->GetRootNode();
  *     -# Call lCharacterPose->GetRootNode()->ComputeLocalState(1, true);
  *     -# Call lCharacterPose->GetRootNode()->SetCurrentTakeFromLocalState(KTIME_ZERO, true).
  */
class KFBX_DLL KFbxCharacterPose : public KFbxObject
{
    KFBXOBJECT_DECLARE(KFbxCharacterPose,KFbxObject);

public:
    //! Reset to an empty character pose.
    void Reset();

    /** Get the root node.
      * \return     Pointer to the root node.
      */
    KFbxNode* GetRootNode() const;

    /** Get the character.
      * \return     Pointer to the character.
      */
    KFbxCharacter* GetCharacter() const;

    /** Get offset matrix for a given character node.
      * \param pCharacterNodeId     Character Node ID.
      * \param pOffset              Receives offset matrix.
      * \return                     \c true if successful, \c false otherwise.
      */
    bool GetOffset(ECharacterNodeId pCharacterNodeId, KFbxXMatrix& pOffset)
    {
        KFbxCharacterLink lCharacterLink;

        if (!GetCharacter()) return false;

        if (!GetCharacter()->GetCharacterLink(pCharacterNodeId, &lCharacterLink)) return false;

        pOffset.SetTRS(lCharacterLink.mOffsetT, lCharacterLink.mOffsetR, lCharacterLink.mOffsetS);

        return true;
    }

    /** Get local position for a given character node.
      * \param pCharacterNodeId     Character Node ID.
      * \param pLocalT              Receives local translation vector.
      * \param pLocalR              Receives local rotation vector.
      * \param pLocalS              Receives local scaling vector.
      * \return                     \c true if successful, \c false otherwise.
      */
    bool GetLocalPosition(ECharacterNodeId pCharacterNodeId, KFbxVector4& pLocalT, KFbxVector4& pLocalR, KFbxVector4& pLocalS)
    {
        KFbxCharacterLink lCharacterLink;

        if (!GetCharacter()) return false;

        if (!GetCharacter()->GetCharacterLink(pCharacterNodeId, &lCharacterLink)) return false;

        pLocalT = lCharacterLink.mNode->GetLocalTFromDefaultTake();
        pLocalR = lCharacterLink.mNode->GetLocalRFromDefaultTake();
        pLocalS = lCharacterLink.mNode->GetLocalSFromDefaultTake();

        return true;
    }

    /** Get global position for a given character node.
      * \param pCharacterNodeId     Character Node ID.
      * \param pGlobalPosition      Receives global position.
      * \return                     \c true if successful, \c false otherwise.
      */
    bool GetGlobalPosition(ECharacterNodeId pCharacterNodeId, KFbxXMatrix& pGlobalPosition)
    {
        KFbxCharacterLink lCharacterLink;

        if (!GetCharacter()) return false;

        if (!GetCharacter()->GetCharacterLink(pCharacterNodeId, &lCharacterLink)) return false;

        pGlobalPosition = lCharacterLink.mNode->GetGlobalFromDefaultTake();

        return true;
    }

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
    virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;

protected:
    KFbxCharacterPose& operator=(KFbxCharacterPose const& pCharacterPose);

private:
    virtual KString         GetTypeName() const;

    KFbxCharacterPose(KFbxSdkManager& pManager, char const* pName);
    ~KFbxCharacterPose();

    virtual void Destruct(bool pRecursive, bool pDependents);

    void CopyNode(KFbxNode* pSrcNode, KFbxNode* pDstNode);
    void CopyCharacter(KFbxCharacter* pSrcCharacter, KFbxCharacter* pDstCharacter, KFbxNode* pDstRootNode);

    KFbxCharacterPose_internal* mPH;

    KFbxScene* mScene;

    friend class KFbxCharacterPose_internal;
    friend class KFbxScene;
    friend class KFbxReaderFbx;
    friend class KFbxReaderFbx6;
    friend class KFbxWriterFbx;
    friend class KFbxWriterFbx6;

    friend class KFbxReaderFbx7;
    friend class KFbxWriterFbx7;

    friend struct KFbxReaderFbx7Impl;
    friend struct KFbxWriterFbx7Impl;

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

};

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_CHARACTER_POSE_H_


