/*!  \file kfbxpose.h
 */

#ifndef _FBXSDK_POSE_H_
#define _FBXSDK_POSE_H_

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

#include <kfbxplugins/kfbxobject.h>
#include <kfbxplugins/kfbxgroupname.h>

#include <klib/kstring.h>
#include <klib/karrayul.h>
#include <klib/kname.h>
#include <kfbxmath/kfbxmatrix.h>

#include <fbxfilesdk_nsbegin.h>

/** This structure contains the description of a named pose.
  *
  */
typedef struct
{
    KFbxMatrix  mMatrix;        //! Transform matrix
    bool        mMatrixIsLocal; //! If true, the transform matrix above is defined in local coordinates.
    KFbxNode*   mNode;          //! Affected node (to replace the identifier).

} KFbxPoseInfo;

class KFbxScene;
class KFbxUserNotification;

/** This class contains the description of a Pose manager.
  * \nosubgrouping
  * The KFbxPose object can be setup to hold "Bind Pose" data or "Rest Pose" data.
  *
  * The Bind Pose holds the transformation (translation, rotation and scaling)
  * matrix of all the nodes implied in a link deformation. This includes the geometry
  * being deformed, the links deforming the geometry, and recursively all the
  * ancestors nodes of the link. The Bind Pose gives you the transformation of the nodes
  * at the moment of the binding operation when no deformation occurs.
  *
  * The Rest Pose is a snapshot of a node transformation. A Rest Pose can be used
  * to store the position of every node of a character at a certain point in
  * time. This pose can then be used as a reference position for animation tasks,
  * like editing walk cycles.
  *
  * One difference between the two modes is in the validation performed before
  * adding an item and the kind of matrix stored.
  *
  * In "Bind Pose" mode, the matrix is assumed to be defined in the global space,
  * while in "Rest Pose" the type of the matrix may be specified by the caller. So
  * local system matrices can be used. Actually, because there is one such flag for
  * each entry (KFbxPoseInfo), it is possible to have mixed types in a KFbxPose elements.
  * It is therefore the responsability of the caller to check for the type of the retrieved
  * matrix and to do the appropriate conversions if required.
  *
  * The validation of the data to be added consists of the following steps:
  *
  *     \li If this KFbxPose object stores "Bind Poses", then
  *        add a KFbxPoseInfo only if the node is not already
  *        associated to another "Bind Pose". This check is done
  *        by visiting ALL the KFbxPose objects in the system.
  *
  *        The above test is only performed for the "Bind Pose" type. While
  *        the next one is always performed, no matter what kind of poses this
  *        KFbxPose object is setup to hold.
  *
  *     \li If a node is already inserted in the KFbxPose internal list,
  *        then the passed matrix MUST be equal to the one already stored.
  *        If this is not the case, the Add method will return -1, indicating
  *        that no new KFbxPoseInfo has been created.
  *
  * If the Add method succeeds, it will return the index of the KFbxPoseInfo
  * structure that as been created and held by the KFbxPose object.
  *
  * To ensure data integrity, the stored information can only be
  * accessed using the provided methods (read-only). If an entry needs to be
  * modified, the caller has to remove the KFbxPoseInfo item by calling Remove(i)
  * and then Add a new one.
  *
  * The internal list is not ordered and the search inside this is list is linear
  * (from the first element to ... the first match or the end of the list).
  *
  */
class KFBX_DLL KFbxPose : public KFbxObject
{
    KFBXOBJECT_DECLARE(KFbxPose,KFbxObject);
public:
        /** Set the type of pose.
           * \param pIsBindPose If true, type will be bind pose, else rest pose.
          */
        void SetIsBindPose(bool pIsBindPose);

        /** Pose identifier flag.
          * \return \c true if this object holds BindPose data.
          */
          bool IsBindPose() const     { return mType == 'b'; }

        /** Pose identifier flag.
          * \return \c true if this object holds RestPose data.
          */
          bool IsRestPose()     { return mType == 'r'; }

         /** Get number of stored items.
           * \return The number of items stored.
           */
          int GetCount() const { return mPoseInfo.GetCount(); }

         /** Stores the pose transformation for the given node.
           * \param pNode pointer to the node for which the pose is stored.
           * \param pMatrix Pose transform of the node.
           * \param pLocalMatrix Flag to indicate if pMatrix is defined in Local or Global space.
		   * \param pMultipleBindPose
           * \return -1 if the function failed or the index of the stored item.
           */
      	  int Add(KFbxNode* pNode, KFbxMatrix& pMatrix, bool pLocalMatrix = false, bool pMultipleBindPose = true);

         /** Remove the pIndexth item from the Pose object.
           * \param pIndex Index of the item to be removed.
           */
          void Remove(int pIndex);

        /** Get the node name.
          * \param pIndex Index of the queried item.
          * \return The node intial and current names.
          * \remarks If the index is invalid ann empty KName is returned.
          */
          KName GetNodeName(int pIndex) const;

        /** Get the node.
          * \param pIndex Index of the queried item.
          * \return A pointer to the node referenced.
          * \remarks If the index is invalid or no pointer to a node is set, returns NULL.
          *  The returned pointer will become undefined if the KFbxPose object is destroyed.
          */
          KFbxNode* GetNode(int pIndex) const;

        /** Get the transform matrix.
          * \param pIndex Index of the queried item.
          * \return A reference to the pose matrix.
          * \remarks If the index is invalid a reference to an identiy matrix is returned.
          *  The reference will become undefined if the KFbxPose object is destroyed.
          */
          const KFbxMatrix& GetMatrix(int pIndex)       const;

        /** Get the type of the matrix.
          * \param pIndex Index of the queried item.
          * \return \c true if the matrix is defined in the Local coordinate space and false otherwise.
          * \remarks If the KFbxPose object is configured to hold BindPose data, this method will always return \c false.
          */
          bool IsLocalMatrix(int pIndex);

        
        /**
          * \name Search Section
          */
        //@{
          enum KNameComponent {
              INITIALNAME_COMPONENT = 1,
              CURRENTNAME_COMPONENT = 2,
              ALL_NAME_COMPONENTS   = 3
          };

        /** Look in the KFbxPose object for the given node name.
          * \param pNodeName Name of the node we are looking for.
          * \param pCompareWhat Bitwise or of the following flags: INTIALNAME_COMPONENT, CURRENTNAME_COMPONENT
          * \return -1 if the node is not in the list. Otherwise, the index of the
          * corresponding KFbxPoseInfo element.
          */
          int Find(KName& pNodeName, char pCompareWhat = ALL_NAME_COMPONENTS);

        /** Look in the KFbxPose object for the given node.
          * \param pNode the node we are looking for.
          * \return -1 if the node is not in the list. Otherwise, the index of the
          * corresponding KFbxPoseInfo element.
          */
          int Find(KFbxNode* pNode);


        //@}
        /**
          * \name Utility Section
          */
        //@{

        /** Get the list of Poses objects that contain the node with name pNodeName.
          * This method will look in all the poses of all the scenes.
          * \param pManager    The manager owning the poses and scenes.
          * \param pNode       The node being explored.
          * \param pPoseList   List of BindPoses/RestPoses that have the node.
          * \param pIndex      List of indices of the nodes in the corresponding poses lists.
          * \return \c true if the node belongs to at least one Pose (either a BindPose or a RestPose).
          * \remarks The pPoseList and pIndex are filled by this method.
          *  The elements of the returned list must not be deleted since they still belong to the scene.
          */
        static bool GetPosesContaining(KFbxSdkManager& pManager, KFbxNode* pNode,
                                       KArrayTemplate<KFbxPose*>& pPoseList,
                                       KArrayTemplate<int>& pIndex);

        /** Get the list of Poses objects that contain the node with name pNodeName.
          * \param pScene     Scene owning the poses.
          * \param pNode      The node being explored.
          * \param pPoseList  List of BindPoses/RestPoses that have the node.
          * \param pIndex     List of indices of the nodes in the corresponding poses lists.
          * \return \c true if the node belongs to at least one Pose (either a BindPose or a RestPose).
          * \remarks The pPoseList and pIndex are filled by this method.
          *  The elements of the returned list must not be deleted since they still belong to the scene.
          */
        static bool GetPosesContaining(KFbxScene* pScene, KFbxNode* pNode,
                                       KArrayTemplate<KFbxPose*>& pPoseList,
                                       KArrayTemplate<int>& pIndex);

        /** Get the list of BindPose objects that contain the node with name pNodeName.
          * This method will look in all the bind poses of all the scenes.
          * \param pManager     The manager owning the poses.
          * \param pNode        The node being explored.
          * \param pPoseList    List of BindPoses that have the node.
          * \param pIndex       List of indices of the nodes in the corresponding bind poses lists.
          * \return \c true if the node belongs to at least one BindPose.
          * \remarks The pPoseList and pIndex are filled by this method.
          *  The elements of the returned list must not be deleted since they still belong to the scene.
          */
        static bool GetBindPoseContaining(KFbxSdkManager& pManager, KFbxNode* pNode,
                                          KArrayTemplate<KFbxPose*>& pPoseList,
                                          KArrayTemplate<int>& pIndex);

        /** Get the list of BindPose objects that contain the node with name pNodeName.
          * \param pScene       The scene owning the poses.
          * \param pNode        The node being explored.
          * \param pPoseList    List of BindPoses that have the node.
          * \param pIndex       List of indices of the nodes in the corresponding bind poses lists.
          * \return \c true if the node belongs to at least one BindPose.
          * \remarks The pPoseList and pIndex are filled by this method.
          *  The elements of the returned list must not be deleted since they still belong to the scene.
          */
        static bool GetBindPoseContaining(KFbxScene* pScene, KFbxNode* pNode,
                                          KArrayTemplate<KFbxPose*>& pPoseList,
                                          KArrayTemplate<int>& pIndex);

        /** Get the list of RestPose objects that contain the node with name pNodeName.
          * This method will look in all the bind poses of all the scenes.
          * \param pManager     The manager owning the poses.
          * \param pNode        The node being explored.
          * \param pPoseList    List of RestPoses that have the node.
          * \param pIndex       List of indices of the nodes in the corresponding rest poses lists.
          * \return \c true if the node belongs to at least one RestPose.
          * \remarks The pPoseList and pIndex are filled by this method.
          *  The elements of the returned list must not be deleted since they still belong to the scene.
          */
        static bool GetRestPoseContaining(KFbxSdkManager& pManager, KFbxNode* pNode,
                                          KArrayTemplate<KFbxPose*>& pPoseList,
                                          KArrayTemplate<int>& pIndex);

        /** Get the list of RestPose objects that contain the node with name pNodeName.
          * \param pScene       The scene owning the poses.
          * \param pNode        The node being explored.
          * \param pPoseList    List of RestPoses that have the node.
          * \param pIndex       List of indices of the nodes in the corresponding rest poses lists.
          * \return \c true if the node belongs to at least one RestPose.
          * \remarks The pPoseList and pIndex are filled by this method.
          *  The elements of the returned list must not be deleted since they still belong to the scene.
          */
        static bool GetRestPoseContaining(KFbxScene* pScene, KFbxNode* pNode,
                                          KArrayTemplate<KFbxPose*>& pPoseList,
                                          KArrayTemplate<int>& pIndex);

        /** Check this bindpose and report an error if all the conditions to a valid bind pose are not
          * met. The conditions are:
          *
          * a) We are a BindPose.
          * b) For every node in the bind pose, all their parent node are part of the bind pose.
          * c) All the deforming nodes are part of the bind pose.
          * d) All the parents of the deforming nodes are part of the bind pose.
          * e) Each deformer relative matrix correspond to the deformer Inv(bindMatrix) * deformed Geometry bindMatrix.
          *
          * \param pRoot This node is used as the stop point when visiting the parents (cannot be NULL).
          * \param pMatrixCmpTolerance Tolerance value when comparing the matrices.
          * \return true if all the above conditions are met and false otherwise.
          * \remarks If the returned value is false, querying for the error will return the reason of the failure.
          *  As soon as one of the above conditions is not met, this method return ignoring any subsequent errors.
          * Run the IsBindPoseVerbose if more details are needed.
          */
        bool IsValidBindPose(KFbxNode* pRoot, double pMatrixCmpTolerance=0.0001);

        /** Same as IsValidBindPose but slower because it will not stop as soon as a failure occurs. Instead,
          * keeps running to accumulate the faulty nodes (stored in the appropriate array). It is then up to the
          * caller to fill the UserNotification if desired.
          *
          * \param pRoot This node is used as the stop point when visiting the parents (cannot be NULL).
          * \param pMissingAncestors Each ancestor missing from the BindPose is added to this list.
          * \param pMissingDeformers Each deformer missing from the BindPose is added to this list.
          * \param pMissingDeformersAncestors Each deformer ancestors missing from the BindPose is added to this list.
          * \param pWrongMatrices Nodes that yeld to a wrong matric comparisons are added to this list.
          * \param pMatrixCmpTolerance Tolerance value when comparing the matrices.
          */
        bool IsValidBindPoseVerbose(KFbxNode* pRoot,
                                    KArrayTemplate<KFbxNode*>& pMissingAncestors,
                                    KArrayTemplate<KFbxNode*>& pMissingDeformers,
                                    KArrayTemplate<KFbxNode*>& pMissingDeformersAncestors,
                                    KArrayTemplate<KFbxNode*>& pWrongMatrices,
                                    double pMatrixCmpTolerance=0.0001);

        /** Same as IsValidBindPose but slower because it will not stop as soon as a failure occurs. Instead,
          * keeps running to accumulate the faulty nodes and send them directly to the UserNotification.
          *
          * \param pRoot This node is used as the stop point when visiting the parents (cannot be NULL).
          * \param pUserNotification Pointer to the user notification where the messages will be accumulated.
          * \param pMatrixCmpTolerance Tolerance value when comparing the matrices.
          * \remarks If the pUserNotification parameter is NULL, this method will call IsValidBindPose.
          */
        bool IsValidBindPoseVerbose(KFbxNode* pRoot,
                                    KFbxUserNotification* pUserNotification,
                                    double pMatrixCmpTolerance=0.0001);

        /**
          * \name Error Management
          */
        //@{

        /** Retrieve error object.
          * \return Reference to error object.
          */
        KError& GetError();

        /** \enum EError Error identifiers.
          * - \e eERROR
          * - \e eERROR_COUNT
          */
        typedef enum
        {
            eERROR,
            eERROR_VALIDBINDPOSE_FAILURE_INVALIDOBJECT,
            eERROR_VALIDBINDPOSE_FAILURE_INVALIDROOT,
            eERROR_VALIDBINDPOSE_FAILURE_NOTALLANCESTORS_NODES,
            eERROR_VALIDBINDPOSE_FAILURE_NOTALLDEFORMING_NODES,
            eERROR_VALIDBINDPOSE_FAILURE_NOTALLANCESTORS_DEFNODES,
            eERROR_VALIDBINDPOSE_FAILURE_RELATIVEMATRIX,
            eERROR_COUNT
        } EError;

        /** Get last error code.
          * \return Last error code.
          */
        EError GetLastErrorID() const;

        /** Get last error string.
          * \return Textual description of the last error.
          */
        const char* GetLastErrorString() const;

        //@}



///////////////////////////////////////////////////////////////////////////////
//
//  WARNING!
//
//  Anything beyond these lines may not be documented accurately and is
//  subject to change without notice.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef DOXYGEN_SHOULD_SKIP_THIS

    // Clone
    virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;

protected:

    /** Create an object of type KFbxPose.
     * \param pManager (need to exist for extended validation).
     * \param pName Name of this KFbxPose.
     */
    KFbxPose(KFbxSdkManager& pManager, char const* pName);

    /** Deletes this object.
     * All the references to NodeName and Matrix will become invalid.
     */
    ~KFbxPose();

    virtual void Destruct(bool pRecursive, bool pDependents);

    // From KFbxObject
    virtual KString     GetTypeName() const;
    virtual KStringList GetTypeFlags() const;

    KError mError;

    //! Assignment operator.
    KFbxPose& operator=(KFbxPose const& pPose);

    // returns false if pNode is already inserted in the list and the current matrix
    // is different from the stored one. Also, if this pose is a rest pose, check if
    // pNode belongs to other BindPoses (accessed through the scene pointer).
    // pos will contains the index of the KFbxPoseInfo if the parameters are already
    // stored in this object.
    //
    bool ValidateParams(KFbxNode* pNode, KFbxMatrix& pMatrix, int& pPos);

    // Check only on this object's list.
    bool LocalValidateParams(KFbxNode* pNode, KFbxMatrix& pMatrix, int& pPos);

    static bool GetSpecificPoseContaining(
        int poseType,
        KFbxScene* pScene, KFbxNode* pNode,
        KArrayTemplate<KFbxPose*>& pPoseList,
        KArrayTemplate<int>& pIndex);

    friend class KFbxReaderFbx;
    friend class KFbxWriterFbx6;

private:
    // don't give public access to the info otherwise we will loose the ability
    // to maintain data integrity.
    KFbxPoseInfo* GetItem(int pIndex);

private:
    char        mType;

    KArrayTemplate<KFbxPoseInfo*> mPoseInfo;

    bool IsValidBindPoseCommon(KFbxNode* pRoot,
                                KArrayTemplate<KFbxNode*>* pMissingAncestors,
                                KArrayTemplate<KFbxNode*>* pMissingDeformers,
                                KArrayTemplate<KFbxNode*>* pMissingDeformersAncestors,
                                KArrayTemplate<KFbxNode*>* pWrongMatrices,
                                double pMatrixCmpTolerance=0.0001);

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

};

typedef KFbxPose* HKFbxPose;

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_POSE_H_


