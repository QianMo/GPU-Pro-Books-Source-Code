/*
 * OgreMax Sample Viewer and Scene Loader - Ogre3D-based viewer and code for loading and displaying .scene files
 * Copyright 2010 AND Entertainment
 *
 * This code is available under the OgreMax Free License:
 *   -You may use this code for any purpose, commercial or non-commercial.
 *   -If distributing derived works (that use this source code) in binary or source code form, 
 *    you must give the following credit in your work's end-user documentation: 
 *        "Portions of this work provided by OgreMax (www.ogremax.com)"
 *
 * AND Entertainment assumes no responsibility for any harm caused by using this code.
 * 
 * The OgreMax Sample Viewer and Scene Loader were released at www.ogremax.com 
 */


#ifndef OgreMax_OgreMaxModel_INCLUDED
#define OgreMax_OgreMaxModel_INCLUDED


//Includes---------------------------------------------------------------------
#include <OgreResourceGroupManager.h>
#include <OgreEntity.h>
#include <OgreCamera.h>
#include <OgreLight.h>
#include <OgreParticleSystem.h>
#include <OgreMeshManager.h>
#include <OgreBillboardSet.h>
#include "OgreMaxPlatform.hpp"
#include "OgreMaxTypes.hpp"


//Classes----------------------------------------------------------------------
namespace OgreMax
{
    class OgreMaxModelInstanceCallback;
    class OgreMaxScene;

    /**
     * An OgreMax model describes a node hierarchy, including all of the contained
     * objects within those nodes. There is one one root node in this hierarchy
     *
     * Instances of a model can be created within the scene
     */
    class _OgreMaxExport OgreMaxModel
    {
    public:
        OgreMaxModel();
        virtual ~OgreMaxModel();

        /**
         * Loads the model description from the specified file
         * @param fileName [in] - The file name. This file must be within Ogre's
         * configured file system
         * @param resourceGroupName [in] - The resource group that the file is located in
         */
        void Load
            (
            const Ogre::String& fileName,
            const Ogre::String& resourceGroupName = Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME
            );

        /**
         * Options that may be passed to CreateInstance()
         */
        typedef int InstanceOptions;
        enum
        {
            NO_OPTIONS = 0,

            /** Indicates animation states should not be created for animation tracks */
            NO_ANIMATION_STATES = 0x1,

            /**
             * Indicates that the model's defined position/orientation/scale should not be used.
             * This is used in the scene loading code since scene instances have their own transformation
             */
            NO_INITIAL_TRANSFORMATION = 0x2
        };

        /**
         * Creates an instance of the model description within the scene
         * @param sceneManager [in] - The scene manager that will contain the new objects
         * @param baseName [in] - The base name that will be used when generating new object names
         * @param callback [in] - Object-creation callback
         * @param options [in] - Options that control the behavior of the object creation
         * @param parentNode [in] - Optional parent node to which the model becomes a child of.
         * If null, the model is created as a child of the scene's root node
         * @param defaultResourceGroupName [in] - Name of the default resource group to use when loading/creating meshes
         * @param node [in] - The node into which the instance is created
         * @param scene [in] - The scene that owns all the models
         * @return The instance's root node is returned. This is the 'node' parameter
         */
        Ogre::SceneNode* CreateInstance
            (
            Ogre::SceneManager* sceneManager,
            const Ogre::String& baseName = Ogre::StringUtil::BLANK,
            OgreMaxModelInstanceCallback* callback = 0,
            InstanceOptions options = NO_OPTIONS,
            Ogre::SceneNode* parentNode = 0,
            const Ogre::String& defaultResourceGroupName = Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
            Ogre::SceneNode* node = 0,
            OgreMaxScene* scene = 0
            ) const;

    private:
        Ogre::SceneNode* CreateInstance
            (
            Ogre::SceneManager* sceneManager,
            const Ogre::String& baseName,
            OgreMaxModelInstanceCallback* callback,
            InstanceOptions options,
            const Types::NodeParameters& nodeParams,
            Ogre::SceneNode* parentNode,
            const Ogre::String& defaultResourceGroupName,
            Ogre::SceneNode* node = 0,
            OgreMaxScene* scene = 0
            ) const;

        void CreateMovableObject
            (
            Ogre::SceneManager* sceneManager,
            const Ogre::String& baseName,
            Types::ObjectParameters* object,
            const Types::MovableObjectOwner& owner,
            OgreMaxModelInstanceCallback* callback,
            const Ogre::String& defaultResourceGroupName
            ) const;

        void CreateEntity
            (
            Ogre::SceneManager* sceneManager,
            const Ogre::String& baseName,
            const Ogre::String& objectName,
            Types::EntityParameters* entityParams,
            const Types::MovableObjectOwner& owner,
            Types::ObjectExtraDataPtr objectExtraData,
            OgreMaxModelInstanceCallback* callback,
            const Ogre::String& defaultResourceGroupName
            ) const;

        void CreateLight
            (
            Ogre::SceneManager* sceneManager,
            const Ogre::String& objectName,
            Types::LightParameters* lightParams,
            const Types::MovableObjectOwner& owner,
            Types::ObjectExtraDataPtr objectExtraData,
            OgreMaxModelInstanceCallback* callback
            ) const;

        void CreateCamera
            (
            Ogre::SceneManager* sceneManager,
            const Ogre::String& objectName,
            Types::CameraParameters* cameraParams,
            const Types::MovableObjectOwner& owner,
            Types::ObjectExtraDataPtr objectExtraData,
            OgreMaxModelInstanceCallback* callback
            ) const;

        void CreateParticleSystem
            (
            Ogre::SceneManager* sceneManager,
            const Ogre::String& objectName,
            Types::ParticleSystemParameters* particleSystemParams,
            const Types::MovableObjectOwner& owner,
            Types::ObjectExtraDataPtr objectExtraData,
            OgreMaxModelInstanceCallback* callback
            ) const;

        void CreateBillboardSet
            (
            Ogre::SceneManager* sceneManager,
            const Ogre::String& objectName,
            Types::BillboardSetParameters* billboardSetParams,
            const Types::MovableObjectOwner& owner,
            Types::ObjectExtraDataPtr objectExtraData,
            OgreMaxModelInstanceCallback* callback
            ) const;

        void CreatePlane
            (
            Ogre::SceneManager* sceneManager,
            const Ogre::String& baseName,
            const Ogre::String& objectName,
            Types::PlaneParameters* planeParams,
            const Types::MovableObjectOwner& owner,
            Types::ObjectExtraDataPtr objectExtraData,
            OgreMaxModelInstanceCallback* callback,
            const Ogre::String& defaultResourceGroupName
            ) const;

        void LoadNode(const TiXmlElement* objectElement, Types::NodeParameters& node);
        Types::EntityParameters* LoadEntity(const TiXmlElement* objectElement);
        void LoadBoneAttachments(const TiXmlElement* objectElement, Types::EntityParameters& entity);
        void LoadBoneAttachment(const TiXmlElement* objectElement, Types::EntityParameters::BoneAttachment& boneAttachment);
        Types::LightParameters* LoadLight(const TiXmlElement* objectElement);
        Types::CameraParameters* LoadCamera(const TiXmlElement* objectElement);
        Types::ParticleSystemParameters* LoadParticleSystem(const TiXmlElement* objectElement);
        Types::BillboardSetParameters* LoadBillboardSet(const TiXmlElement* objectElement);
        Types::PlaneParameters* LoadPlane(const TiXmlElement* objectElement);
        void LoadBillboard(const TiXmlElement* objectElement, Types::BillboardSetParameters::Billboard& billboard);
        void LoadLightRange(const TiXmlElement* objectElement, Types::LightParameters& light);
        void LoadLightAttenuation(const TiXmlElement* objectElement, Types::LightParameters& light);
        void LoadNodeAnimations(const TiXmlElement* objectElement, Types::NodeParameters& node);
        void LoadNodeAnimation(const TiXmlElement* objectElement, Types::NodeAnimationParameters& animation);
        void LoadNodeAnimationKeyFrame(const TiXmlElement* objectElement, Types::NodeAnimationParameters::KeyFrame& keyframe);

        void HandleNewObjectExtraData(OgreMaxModelInstanceCallback* callback, Types::ObjectExtraDataPtr objectExtraData) const;

    protected:
        Types::NodeParameters rootNode;
    };

    class _OgreMaxExport OgreMaxModelInstanceCallback
    {
    public:
        //MovableObject-creation callbacks, called after an object has been created
        //These are called after the object was attached to its parent node, if there is a parent node
        virtual void CreatedLight(const OgreMaxModel* model, Ogre::Light* light) {CreatedMovableObject(model, light);}
        virtual void CreatedCamera(const OgreMaxModel* model, Ogre::Camera* camera) {CreatedMovableObject(model, camera);}
        virtual void CreatedMesh(const OgreMaxModel* model, Ogre::Mesh* mesh) {}
        virtual void CreatedEntity(const OgreMaxModel* model, Ogre::Entity* entity) {CreatedMovableObject(model, entity);}
        virtual void CreatedParticleSystem(const OgreMaxModel* model, Ogre::ParticleSystem* particleSystem) {CreatedMovableObject(model, particleSystem);}
        virtual void CreatedBillboardSet(const OgreMaxModel* model, Ogre::BillboardSet* billboardSet) {CreatedMovableObject(model, billboardSet);}
        virtual void CreatedPlane(const OgreMaxModel* model, const Ogre::Plane& plane, Ogre::Entity* entity) {CreatedMovableObject(model, entity);}
        virtual void CreatedMovableObject(const OgreMaxModel* model, Ogre::MovableObject* object) {}

        virtual void CreatedNodeAnimationTrack(const OgreMaxModel* model, Ogre::SceneNode* node, Ogre::AnimationTrack* animationTrack, bool enabled, bool looping) {}
        virtual void CreatedNodeAnimationState(const OgreMaxModel* model, Ogre::SceneNode* node, Ogre::AnimationState* animationState) {}

        /** Called after a node is created, but before its entities or any child nodes have been created */
        virtual void StartedCreatingNode(const OgreMaxModel* model, Ogre::SceneNode* node) {}

        /** Called after the node and all its entities and child nodes have been created */
        virtual void FinishedCreatingNode(const OgreMaxModel* model, Ogre::SceneNode* node) {}

        virtual void HandleObjectExtraData(Types::ObjectExtraDataPtr objectExtraData) {}
    };

}


#endif
