/*!  \file kfbxscene.h
 */
#ifndef _FBXSDK_SCENE_H_
#define _FBXSDK_SCENE_H_

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

#include <fbxcore/fbxcollection/kfbxdocument.h>

#include <kfbxplugins/kfbxdeformer.h>
#include <kfbxplugins/kfbxsubdeformer.h>
#include <kfbxplugins/kfbxlayer.h>
#include <kfbxplugins/kfbxnodeattribute.h>
#include <kfbxplugins/kfbxglobalsettings.h>

#include <klib/karrayul.h>
#include <klib/kerror.h>
#include <klib/kstring.h>
#include <klib/kcharptrset.h>

#include <fbxfilesdk_nsbegin.h>

class KFbxSdkManager;
class KFbxObject;
class KFbxNode;
class KFbxVideo;
class KFbxTexture;
class KFbxSurfaceMaterial;
class KFbxMarkerSet;
class KFbxActor;
class KFbxCharacter;
class KFbxControlSetPlug;
class KFbxCharacterPose;
class KFbxPose;
class KFbxDocumentInfo;
class KFbxTakeInfo;
class KFbxGlobalLightSettings;
class KFbxGlobalCameraSettings;
class KFbxGlobalTimeSettings;
class KFbxGlobalSettings;
class KFbxGenericNode;
class KFbxConstraint;
class KFbxGeometry;
class KFbxTakeNodeContainer;
class KFbxLayerContainer;

/** This class contains the description of a complete 3D scene.
  * \nosubgrouping
  * The FBX SDK requires unique names for nodes, textures, materials, generic nodes, videos and gobos contained
  * in a given scene. To build a valid node tree, new nodes must be inserted under the
  * scene's root node.
  *
  * This class also provides access to global settings and take information.
  *
  */
class KFBX_DLL KFbxScene : public KFbxDocument
{
    KFBXOBJECT_DECLARE(KFbxScene,KFbxDocument);

public:
    //! Delete the node tree below the root node and restore default settings.
    void Clear();

    /**
      * \name Node Tree Access
      */
    //@{

    /** Get the root node.
      * \return Pointer to the root node.
      * \remarks This node is not saved. Do not use it to apply a global transformation
      * to the node hierarchy. If a global transformation must be applied, insert a
      * new node below this one.
      */
    KFbxNode* GetRootNode() const;


    /** Fill a node array will all existing node with a given node attribute type.
     *  \param pNodeArray An array of pointers to KFbxNode objects.
     *  \param pAttributeType Node attribute type searched.
     *  \param pDepthFirst If \c true, the node tree is scanned depth first.
     *  \remarks Scans the node hierarchy to find all existing node with a given node attribute type.
     *  The array of pointers is cleared before scanning the node hierarchy.
     */
    void FillNodeArray(KArrayTemplate<KFbxNode*>& pNodeArray,
                       KFbxNodeAttribute::EAttributeType pAttributeType,
                       bool pDepthFirst = false);

    //@}

    /**
      * \name Texture Material and Video Access
      */
    //@{

    /** Clear then fill a texture array with all existing textures included in the scene.
      * \param pTextureArray An array of texture pointers.
      */
    void FillTextureArray(KArrayTemplate<KFbxTexture*>& pTextureArray);

    /** Clear then fill a material array with all existing materials included in the scene.
      * \param pMaterialArray An array of material pointers.
      */
    void FillMaterialArray(KArrayTemplate<KFbxSurfaceMaterial*>& pMaterialArray);

    //@}

    /**
      * \name Generic Node Access
      */
    //@{

    /** Get number of generic nodes.
      * \return Number of Generic Nodes in this scene.
      */
    int             GetGenericNodeCount();

    /** Get generic node at given index.
      * \param pIndex Position in the list of the generic nodes.
      * \return Pointer to the generic node or \c NULL if the index is out of bounds.
      */
    KFbxGenericNode* GetGenericNode(int pIndex);

    /** Access a generic node from its name.
    *   \param pName
    *   \return found generic node
    */
    KFbxGenericNode* GetGenericNode(char* pName);

    /** Add the generic node to this scene.
      * \param pGenericNode Pointer to the generic node to be added.
      * \return If the passed parameter is \c NULL, this method will return \c false otherwise \c true.
      */
    bool             AddGenericNode(KFbxGenericNode* pGenericNode);

    /** Remove the generic node from this scene.
      * \param pGenericNode Pointer to the generic node to be removed.
      * \return If the passed parameter is \c NULL, this method will return \c false otherwise \c true.
      * \remarks The pointed object is not referenced by the scene anymore but is not deleted from the system.
      */
    bool             RemoveGenericNode(KFbxGenericNode* pGenericNode);

    //@}


    /**
      * \name Character Management
      */
    //@{

    /** Get number of characters.
      * \return Number of characters in this scene.
      */
    int GetCharacterCount();

    /** Get character at given index.
      * \param pIndex Position in the list of the characters.
      * \return Pointer to the character or \c NULL if index is out of bounds.
      */
    KFbxCharacter* GetCharacter(int pIndex);

    /** Create a new character.
      * \param pName Name given to character.
      * \return Index of the created character.
      */
    int CreateCharacter(char* pName);

    /** Destroy character.
      * \param pIndex Specify which character to destroy.
      * \remarks After the destruction  of the requested element the list is resized.
      */
    void DestroyCharacter(int pIndex);

    //@}

    /**
      * \name ControlSetPlug Management
      */
    //@{

    /** Get number of ControlSetPlugs.
      * \return Number of ControlSet plugs in this scene.
      */
    int GetControlSetPlugCount();

    /** Get ControlSetPlug at given index.
      * \param pIndex Position in the list of the ControlSetPlug
      * \return Pointer to ControlSetPlug or \c NULL if index is out of bounds.
      */
    KFbxControlSetPlug* GetControlSetPlug(int pIndex);

    /** Create a new ControlSetPlug.
      * \param pName Name given to ControlSetPlug.
      * \return Index of created ControlSetPlug.
      */
    int CreateControlSetPlug(char* pName);

    /** Destroy ControlSetPlug.
      * \param pIndex Specify which ControlSetPlug to destroy.
      * \remarks After the destruction of the requested element the list is resized.
      */
    void DestroyControlSetPlug(int pIndex);

    //@}

    /**
      * \name Character Pose Management
      */
    //@{

    /** Get number of character poses.
      * \return Number of character poses in this scene.
      * \remarks Character Poses and Poses are two distinct entities having their own lists.
      */
    int GetCharacterPoseCount();

    /** Get character pose at given index.
      * \param pIndex Position in the list of character poses.
      * \return Pointer to the character pose or \c NULL if index is out of bounds.
      */
    KFbxCharacterPose* GetCharacterPose(int pIndex);

    /** Create a new character pose.
      * \param pName Name given to character pose.
      * \return Index of created character pose.
      */
    int CreateCharacterPose(char* pName);

    /** Destroy character pose.
      * \param pIndex Specify which character pose to destroy.
      * \remarks After the destruction of the requested element the list is resized.
      */
    void DestroyCharacterPose(int pIndex);

    //@}

    /**
      * \name Pose Management
      */
    //@{

    /** Get number of poses.
      * \return Number of poses in the scene.
      * \remarks Poses and Character Poses are two distinct entities having their own lists.
      */
    int GetPoseCount();

    /** Get pose at given index.
      * \param pIndex Position in the list of poses.
      * \return Pointer to the pose or \c NULL if index is out of bounds.
      */
    KFbxPose* GetPose(int pIndex);

    /** Add a pose to this scene.
      * \param pPose The pose to be added to the scene.
      * \return If the pose is correctly added to the scene, return \c true otherwise, if the pose is
      * already in the scene, returns \c false.
      */
    bool AddPose(KFbxPose* pPose);

    /** Remove the specified pose from the scene.
      * \param pPose The pose to be removed from the scene.
      * \return If the pose was successfully removed from the scene, return \c true otherwise, if the
      * pose could not be found returns \c false.
      */
    bool RemovePose(KFbxPose* pPose);

    /** Remove the pose at the given index from the scene.
      * \param pIndex The zero-based index of the pose to be removed.
      * \return If the pose was successfully removed from the scene, return \c true otherwise, if the
      * pose could not be found returns \c false.
      */
    bool RemovePose(int pIndex);


    //@}
    /**
      * \name Scene information
      */
    //@{

    /** Get the scene information.
      * \return Pointer to the scene information object.
      */
    inline KFbxDocumentInfo* GetSceneInfo() { return GetDocumentInfo(); }

    /** Set the scene information.
      * \param pSceneInfo Pointer to the scene information object.
      */
    inline void SetSceneInfo(KFbxDocumentInfo* pSceneInfo) { SetDocumentInfo(pSceneInfo); }

    //@}

    /**
      * \name Global Settings
      */
    //@{

    /** Access global light settings.
      * \return Reference to the Global Light Settings.
      */
    KFbxGlobalLightSettings& GetGlobalLightSettings();

    /** Access global camera settings.
      * \return Reference to the Global Camera Settings.
      */
    KFbxGlobalCameraSettings& GetGlobalCameraSettings();

    /** Access global time settings.
      * \return Reference to the Global Time Settings.
      */
    KFbxGlobalTimeSettings& GetGlobalTimeSettings();

    /** Access global settings.
      * \return Reference to the Global Settings.
      */
    KFbxGlobalSettings& GetGlobalSettings();

    /** Const access to global settings.
      * \return Const reference to the Global Settings.
      */
    const KFbxGlobalSettings& GetGlobalSettings() const;

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

    void ConnectMaterials();

    void BuildMaterialLayersDirectArray();
    void ReindexMaterialConnections(); // called to make sure that eINDEX is remapped to eINDEX_TO_DIRECT

    /** Clear then fill a node array with all existing nodes included in the scene.
      * \param pNodeArray An array of node pointers.
      */
    void FillNodeArray(KArrayTemplate<KFbxNode*>& pNodeArray);

    /** Clear then fill a pose array with all existing pose included in the scene.
      * \param pPoseArray An array of video pointers.
      */
    void FillPoseArray(KArrayTemplate<KFbxPose*>& pPoseArray);

    // Clone
    virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;


    /**
      * \name Material Access
      */
    //@{

    /** Get number of materials.
      * \return Number of materials in this scene.
      */
    int                     GetMaterialCount            ();

    /** Get the material at the given index.
      * \param pIndex Position in the list of materials.
      * \return Pointer to the material or \c NULL if the index is out of bounds.
      * \remarks pIndex must be between 0 and GetMaterialCount().
      */
    KFbxSurfaceMaterial*            GetMaterial                 (int pIndex);

    /** Get the material by its name.
      * \param pName Name of the material.
      * \return Pointer to the material or \c NULL if not found.
      */
    KFbxSurfaceMaterial*            GetMaterial                 (char* pName);

    /** Add the material to this scene.
      * \param pMaterial Pointer to the material to be added.
      * \return true on successful addition.
      */
    bool                    AddMaterial                 (KFbxSurfaceMaterial* pMaterial);

    /** Remove the material from this scene.
      * \param pMaterial Pointer to the material to be removed.
      * \return true on successful removal.
      */
    bool                    RemoveMaterial              (KFbxSurfaceMaterial* pMaterial);

    //@}

    /**
      * \name Texture Access
      */
    //@{

    /** Get number of textures.
      * \return Number of textures in this scene.
      */
    int                     GetTextureCount             ();

    /** Get the texture at the given index.
      * \param pIndex Position in the list of textures.
      * \return Pointer to the texture or \c NULL if the index is out of bounds.
      * \remarks pIndex must be between 0 and GetTextureCount().
      */
    KFbxTexture*            GetTexture                  (int pIndex);

    /** Get the texture by its name.
      * \param pName Name of the texture.
      * \return Pointer to the texture or \c NULL if not found.
      */
    KFbxTexture*            GetTexture                  (char* pName);

    /** Add the texture to this scene.
      * \param pTexture Pointer to the texture to be added.
      * \return true on successful addition.
      */
    bool                    AddTexture                  (KFbxTexture* pTexture);

    /** Remove the texture from this scene.
      * \param pTexture Pointer to the texture to be removed.
      * \return true on successful removal.
      */
    bool                    RemoveTexture               (KFbxTexture* pTexture);

    //@}

    /**
      * \name Node Access
      */
    //@{

    /** Get number of nodes.
      * \return Number of nodes in this scene.
      */
    int                     GetNodeCount                ();

    /** Get the node at the given index.
      * \param pIndex Position in the list of nodes.
      * \return Pointer to the node or \c NULL if the index is out of bounds.
      * \remarks pIndex must be between 0 and GetNodeCount().
      */
    KFbxNode*               GetNode                     (int pIndex);

    /** Add the node to this scene.
      * \param pNode Pointer to the node to be added.
      * \return true on successful addition.
      */
    bool                    AddNode                     (KFbxNode* pNode);

    /** Remove the node from this scene.
      * \param pNode Pointer to the node to be removed.
      * \return true on successful removal.
      */
    bool                    RemoveNode                  (KFbxNode* pNode);

    /** Helper method for determining the number of nodes that have
      * curves on surface attributes in the scene. Since the curve-on-surface
      * nodes are connected to nurbs geometry and not any KFbxNodes in the
      * scene, they won't normally be picked up in a graph traversal.
      * \return The number of curve-on-surface nodes in the scene
      */
    int                     GetCurveOnSurfaceCount      ();

    KFbxNode*               FindNodeByName              ( const KString& pName );

    //@}

    /**
      * \name Geometry Access
      */
    //@{

    /** Get number of geometries.
      * \return Number of geometries in this scene.
      */
    int                     GetGeometryCount            ();

    /** Get the geometry at the given index.
      * \param pIndex Position in the list of geometries.
      * \return Pointer to the geometry or \c NULL if the index is out of bounds.
      * \remarks pIndex must be between 0 and GetGeometryCount().
      */
    KFbxGeometry*           GetGeometry                 (int pIndex);

    /** Add the geometry to this scene.
      * \param pGeometry Pointer to the geometry to be added.
      * \return true on successful addition.
      */
    bool                    AddGeometry                 (KFbxGeometry* pGeometry);

    /** Remove the geometry from this scene.
      * \param pGeometry Pointer to the geometry to be removed.
      * \return true on successful removal.
      */
    bool                    RemoveGeometry              (KFbxGeometry* pGeometry);

    //@}

    /**
      * \name Video Access
      */
    //@{

    /** Get number of videos.
      * \return Number of videos in this scene.
      */
    int                     GetVideoCount               ();

    /** Get the video at the given index.
      * \param pIndex Position in the list of videos.
      * \return Pointer to the video or \c NULL if the index is out of bounds.
      * \remarks pIndex must be between 0 and GetVideoCount().
      */
    KFbxVideo*              GetVideo                    (int pIndex);

    /** Add the video to this scene.
      * \param pVideo Pointer to the video to be added.
      * \return true on successful addition.
      */
    bool                    AddVideo                    (KFbxVideo* pVideo);

    /** Remove the video from this scene.
      * \param pVideo Pointer to the video to be removed.
      * \return true on successful removal.
      */
    bool                    RemoveVideo                 (KFbxVideo* pVideo);

    //@}

    int                     GetTakeNodeContainerCount   ();
    KFbxTakeNodeContainer*  GetTakeNodeContainer        (int pIndex);
    bool                    AddTakeNodeContainer        (KFbxTakeNodeContainer* pTakeNodeContainer);
    bool                    RemoveTakeNodeContainer     (KFbxTakeNodeContainer* pTakeNodeContainer);

    KSet* AddTakeTimeWarpSet(char *pTakeName);
    KSet* GetTakeTimeWarpSet(char *pTakeName);

private:
    KFbxScene(KFbxSdkManager& pManager, char const* pName);
    ~KFbxScene();

    virtual void Construct(const KFbxScene* pFrom);
    virtual void Destruct(bool pRecursive, bool pDependents);

    KFbxScene& operator=(KFbxScene const& pScene);

    void FillNodeArrayRecursive(KArrayTemplate<KFbxNode*>& pNodeArray,
                                KFbxNodeAttribute::EAttributeType pAttributeType,
                                KFbxNode* pNode);

    void ConnectTextures();
    void ConnectTextureLayerElement(KFbxLayerContainer* pLayerContainer,
                                    KFbxLayerElement::ELayerElementType pLayerType,
                                    KFbxNode* pParentNode);

    void BuildTextureLayersDirectArray();
    void BuildTextureLayersDirectArrayForLayerType(KFbxLayerContainer* pLayerContainer,
                                                   KFbxLayerElement::ELayerElementType pLayerType);

    void FixInheritType(KFbxNode *pNode);

public:
    void ConvertNurbsSurfaceToNurb();
    void ConvertMeshNormals();
    void ConvertNurbCurvesToNulls();

private:
    KFbxNode*                           mRootNode;

    KFbxGlobalLightSettings*            mGlobalLightSettings;
    KFbxGlobalCameraSettings*           mGlobalCameraSettings;
    KFbxGlobalTimeSettings*             mGlobalTimeSettings;

    KCharPtrSet                         mTakeTimeWarpSet;

    friend class KFbxLayerContainer;
    friend class KFbxNodeFinderDuplicateName;

    friend class KFbxWriterFbx;
    friend class KFbxWriterFbx6;
    friend class KFbxWriterFbx7;
    friend class KFbxReaderFbx;
    friend class KFbxReaderFbx6;
    friend class KFbxReaderFbx7;

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

};

typedef  class KFBX_DLL KArrayTemplate<KFbxTakeInfo *>          KArrayKFbxTakeInfo;
typedef KFbxScene* HKFbxScene;

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_SCENE_H_


