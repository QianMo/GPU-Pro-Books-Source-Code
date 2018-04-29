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


#ifndef OgreMax_OgreMaxScene_INCLUDED
#define OgreMax_OgreMaxScene_INCLUDED


//Includes---------------------------------------------------------------------
#include <OgreRenderTargetListener.h>
#include <OgreInstancedGeometry.h>
#include <OgreStaticGeometry.h>
#include "OgreMaxPlatform.hpp"
#include "ProgressCalculator.hpp"
#include "Version.hpp"
#include "OgreMaxModel.hpp"
#include "OgreMaxRenderWindowIterator.hpp"


//Classes----------------------------------------------------------------------
namespace OgreMax
{
    class OgreMaxSceneCallback;

    /**
     * Manages the loading of a OgreMax .scene file.
     * This class is very lightwight in the sense that doesn't maintain a lot of
     * internal state. Instead, wherever possible, most of the data is put into the
     * Ogre scene manager. As such, when deleting an OgreMaxScene, keep in mind that
     * the scene manager and all of resources that were loaded as a result of loading
     * the scene are NOT destroyed. You will need to destroy those manually.
     */
    class _OgreMaxExport OgreMaxScene : public Ogre::RenderTargetListener
    {
    public:
        OgreMaxScene();
        virtual ~OgreMaxScene();

        static Ogre::Real DEFAULT_MAX_DISTANCE;

        /**
         * Options that may be passed to Load()
         */
        typedef int LoadOptions;
        enum
        {
            NO_OPTIONS = 0,

            /** Skips the environment settings in the file. Skipping the environment also skips shadows */
            SKIP_ENVIRONMENT = 0x1,

            /** Skips the shadow settings in the file */
            SKIP_SHADOWS = 0x2,

            /** Skips the sky settings in the file */
            SKIP_SKY = 0x4,

            /** Skips the nodes in the file */
            SKIP_NODES = 0x8,

            /** Skips the externals in the file */
            SKIP_EXTERNALS = 0x10,

            /** Skips the terrain in the file */
            SKIP_TERRAIN = 0x20,

            /** Skips scene level query flags in the file */
            SKIP_QUERY_FLAG_ALIASES = 0x200,

            /** Skips scene level visibility flags in the file */
            SKIP_VISIBILITY_FLAG_ALIASES = 0x400,

            /** Skips scene level resource locations in the file */
            SKIP_RESOURCE_LOCATIONS = 0x800,

            /** Indicates animation states should not be created for node animation tracks */
            NO_ANIMATION_STATES = 0x1000,

            /**
             * Indicates scene 'externals' should not be stored. They will, however, still
             * be loaded and the CreatedExternal() scene callback will be called
             */
            NO_EXTERNALS = 0x2000,

            /**
             * By default, the OgreMaxScene::Load() checks to see if the file that was passed
             * in exists in the file system (outside of the configured resource locations).
             * This flag disables that logic
             */
            NO_FILE_SYSTEM_CHECK = 0x4000,

            /**
             * Indicates that a light should be created if the loaded scene doesn't contain a light
             * Any lights created as a result of setting the default lighting are not passed to
             * the scene callback
             */
            SET_DEFAULT_LIGHTING = 0x8000,

            /**
             * Indicates that the 'fileNameOrContent' parameter in Load() should be treated as the
             * scene XML. In other words, no file is loaded from Ogre's file system.
             */
            FILE_NAME_CONTAINS_CONTENT = 0x10000
        };

        /**
         * Loads a scene from the specified file.
         * @param fileNameOrContent [in] - The name of the file to load. This file name can contain a directory path
         * or be a base name that exists within Ogre's resource manager. If the file contains a directory path,
         * the directory will be added to Ogre's resource manager as a resource location and the directory will be
         * set as a base resource location. If the file doesn't contain a directory path, and assuming
         * If the FILE_NAME_CONTAINS_CONTENT flag is specified as a load option, this parameter is treated
         * as though it is the scene file itself. That is, the string contains the scene file XML.
         * SetBaseResourceLocation() hasn't already been called elsewhere, no resource location configuration
         * will occur during the load
         * @param renderWindow [in] - The render window in use
         * @param loadOptions [in] - Options used to specify the loading behavior. Optional.
         * @param sceneManager [in] - The scene manager into which the scene is loaded. Optional. If not specified,
         * the scene manager specified in the scene file will be used to create a new scene manager
         * @param rootNode [in] - Root node under which all loaded scene nodes will be created. If specified,
         * this should be a scene node in the specified scene manager
         * @param callback [in] - Pointer to a callback object. Optional.
         * @param defaultResourceGroupName [in] - The resource group name that is used by default when
         * creating new resources. Optional.
         */
        void Load
            (
            const Ogre::String& fileNameOrContent,
            Ogre::RenderWindow* renderWindow,
            LoadOptions loadOptions = NO_OPTIONS,
            Ogre::SceneManager* sceneManager = 0,
            Ogre::SceneNode* rootNode = 0,
            OgreMaxSceneCallback* callback = 0,
            const Ogre::String& defaultResourceGroupName = Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME
            );

        void Load
            (
            const Ogre::String& fileName,
            OgreMaxRenderWindowIterator& renderWindows,
            LoadOptions loadOptions = NO_OPTIONS,
            Ogre::SceneManager* sceneManager = 0,
            Ogre::SceneNode* rootNode = 0,
            OgreMaxSceneCallback* callback = 0,
            const Ogre::String& defaultResourceGroupName = Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME
            );

        /**
         * Destroys the internal OgreMaxScene objects
         * Note that this does not affect the scene manager or any resources.
         * Those must be destroyed manually
         */
        void Destroy();

        /** Flags that define which name prefixes to set */
        typedef int WhichNamePrefix;
        enum
        {
            OBJECT_NAME_PREFIX = 0x1,
            NODE_NAME_PREFIX = 0x2,
            NODE_ANIMATION_NAME_PREFIX = 0x4,
            ALL_NAME_PREFIXES = OBJECT_NAME_PREFIX | NODE_NAME_PREFIX | NODE_ANIMATION_NAME_PREFIX
        };

        /**
         * Sets one or more name prefixes. This should be called before Load() is called
         * @param name [in] - The prefix. This may be null
         * @param prefixes [in] - Flags indicating which prefixes to set
         */
        void SetNamePrefix(const Ogre::String& name, WhichNamePrefix prefixes = ALL_NAME_PREFIXES);

        void Update(Ogre::Real elapsedTime);

        /** Gets the base resource location */
        const Ogre::String& GetBaseResourceLocation() const;

        /**
         * Sets the base resource location, a directory, which all other resource locations
         * are considered to be relative to.
         * @param directory [in] - The base directory. If this is an empty string, resource
         * locations loaded from the scene file are not added to the Ogre resource group manager
         */
        void SetBaseResourceLocation(const Ogre::String& directory);

        //Various getters
        Ogre::SceneManager* GetSceneManager();
        Ogre::SceneNode* GetRootNode();
        Ogre::SceneNode* GetSkyNode();

        /**
         * Gets the scene node with the specified name, optionally prepending the scene node name prefix
         * @param name [in] - The name of the scene node to get
         * @param nameIncludesPrefix [in] - Indicates whether the name already includes the prefix. If false,
         * the scene node name prefix is used with the node name.
         * @return The scene node with the specified name is returned
         */
        Ogre::SceneNode* GetSceneNode(const Ogre::String& name, bool nameIncludesPrefix);

        /**
         * Gets the entity with the specified name, optionally prepending the object name prefix
         * @param name [in] - The name of the entity to get
         * @param nameIncludesPrefix [in] - Indicates whether the name already includes the prefix. If false,
         * the object name prefix is used with the name.
         * @return The entity with the specified name is returned
         */
        Ogre::Entity* GetEntity(const Ogre::String& name, bool nameIncludesPrefix);

        /** Same as GetEntity(), but for cameras. */
        Ogre::Camera* GetCamera(const Ogre::String& name, bool nameIncludesPrefix);

        /** Gets the up axis type */
        Types::UpAxis GetUpAxis() const;

        /** Gets the up axis vector */
        const Ogre::Vector3& GetUpVector() const;

        /**
         * Gets the scene scaling - in units per meter.
         * This will be zero if the setting was not defined in the scene file
         */
        Ogre::Real GetUnitsPerMeter();

        /**
         * Gets the scene's unit type (meters, feet, and so on).
         * This is a descriptive string that may contain an additional divide factor
         */
        const Ogre::String GetUnitType();

        /** Gets the 'author' that created the scene. */
        const Ogre::String& GetAuthor() const;

        /** Gets the name of the application that generated the scene. */
        const Ogre::String& GetApplication() const;

        /** Gets the extra data associated with the scene */
        const Types::ObjectExtraData& GetSceneExtraData() const;

        /** Gets the extra data associated with the terrain */
        const Types::ObjectExtraData& GetTerrainExtraData() const;

        /** Gets the extra data associated with the sky box */
        const Types::ObjectExtraData& GetSkyBoxExtraData() const;

        /** Gets the extra data associated with the sky dome */
        const Types::ObjectExtraData& GetSkyDomeExtraData() const;

        /** Gets the extra data associated with the sky plane */
        const Types::ObjectExtraData& GetSkyPlaneExtraData() const;

        typedef std::vector<Types::ExternalItem> ExternalItemVector;
        typedef ExternalItemVector ExternalItemList; //Users of previous versions may need this

        /** Gets the list of external items */
        const ExternalItemVector& GetExternalItems() const;

        typedef std::vector<Types::ExternalUserData> ExternalUserDataVector;
        const ExternalUserDataVector& GetExternalUserData() const;

        /** Gets the background color */
        const Ogre::ColourValue& GetBackgroundColor() const;

        /** Gets the environment near range */
        Ogre::Real GetEnvironmentNear() const;

        /** Gets the environment far range */
        Ogre::Real GetEnvironmentFar() const;

        /** Gets the loaded fog settings */
        const Types::FogParameters& GetFogParameters() const;

        /** Gets plane object used as the current shadow optimal plane */
        Ogre::MovablePlane* GetShadowOptimalPlane();

        /** Gets the plane object with the specified name */
        Ogre::MovablePlane* GetMovablePlane(const Ogre::String& name);

        typedef std::map<const Ogre::String, Ogre::AnimationState*> AnimationStates;

        /** Gets all the animation states created by the scene */
        AnimationStates& GetAnimationStates();

        /** Gets the animation state created by the scene with the specified name */
        Ogre::AnimationState* GetAnimationState(const Ogre::String& animationName);

        typedef std::map<const Ogre::MovableObject*, Types::ObjectExtraDataPtr> ObjectExtraDataMap;

        /** Gets the collection of extra data for all Ogre::MovableObjects */
        ObjectExtraDataMap& GetAllObjectExtraData();

        typedef std::map<const Ogre::Node*, Types::ObjectExtraDataPtr> NodeObjectExtraDataMap;

        /** Gets the collection of extra data for all Ogre::Nodes */
        NodeObjectExtraDataMap& GetAllNodeObjectExtraData();

        typedef std::map<const Ogre::String, OgreMaxModel*> ModelMap;

        /** Gets the collection of all model templates, each of which is keyed by file name */
        const ModelMap& GetModelMap() const;

        /** Gets the model template with the specified file name, loading it if it isn't already loaded */
        OgreMaxModel* GetModel(const Ogre::String& fileName);

        /** Gets the collection of scene query flag aliases */
        const Types::FlagAliases& GetQueryFlagAliases() const;

        /** Gets the collection of scene visibility flag aliases */
        const Types::FlagAliases& GetVisibilityFlagAliases() const;

        /** A single resource location */
        class ResourceLocation
        {
        public:
            ResourceLocation()
            {
                this->recursive = false;
                this->initialized = false;
            }

            ResourceLocation
                (
                const Ogre::String& name,
                const Ogre::String& type,
                bool recursive = false
                )
            {
                this->name = name;
                this->type = type;
                this->recursive = recursive;
                this->initialized = false;
            }

            bool operator < (const ResourceLocation& other) const
            {
                return this->name < other.name;
            }

            Ogre::String name;
            Ogre::String type;
            bool recursive;

        private:
            friend class OgreMaxScene;

            bool initialized;
        };

        /**
         * Adds a new resource location.
         * The new resource location isn't visible to Ogre until CommitResourceLocations() is called
         */
        bool AddResourceLocation(const ResourceLocation& resourceLocation);

        /** Adds new resource locations to Ogre3D's resource group manager */
        void CommitResourceLocations();

        typedef std::set<ResourceLocation> ResourceLocations;

        /** Gets the resource locations that were configured through the scene */
        const ResourceLocations& GetResourceLocations() const;

        Ogre::ResourceLoadingListener& GetResourceLoadingListener();

        /** Gets the extra data, if any, for the specified object */
        Types::ObjectExtraData* GetObjectExtraData(const Ogre::MovableObject* object) const;

        /**
         * Gets the extra data, if any, for the specified node
         * This only works for nodes that were exported as 'empty'
         */
        Types::ObjectExtraData* GetObjectExtraData(const Ogre::Node* node) const;

        /** For internal use only */
        void HandleNewObjectExtraData(Types::ObjectExtraDataPtr objectExtraData);

        /** This is to be called if you manually destroy a scene node */
        void DestroyedSceneNode(const Ogre::SceneNode* node);

        /** This is to be called if you manually destroy a movable object (entity, light, camera, etc.)*/
        void DestroyedObject(const Ogre::MovableObject* object);

    protected:
        /**
         * Loads a scene from the specified XML element
         * @param objectElement [in] - The 'scene' XML element
         */
        void Load
            (
            TiXmlElement* objectElement,
            OgreMaxRenderWindowIterator& renderWindows,
            LoadOptions loadOptions = NO_OPTIONS,
            Ogre::SceneManager* sceneManager = 0,
            Ogre::SceneNode* rootNode = 0,
            OgreMaxSceneCallback* callback = 0,
            const Ogre::String& defaultResourceGroupName = Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME
            );

        //Ogre::RenderTargetListener methods
        void preRenderTargetUpdate(const Ogre::RenderTargetEvent& e);
        void postRenderTargetUpdate(const Ogre::RenderTargetEvent& e);

        /**
         * Determines the name of an object stored in an XML element. The name is
         * presumed to be for a new object, so if the name is not unique an exception is thrown
         * @param objectElement [in] - The XML element that contains the object
         * @param node [in] - The scene node that will contain the object once it's
         * parsed from the XML element.
         * @return If the XML element contains the 'name' attribute, that attribute's value is returned.
         * Otherwise the parent node's name is returned
         */
        Ogre::String GetNewObjectName(const TiXmlElement* objectElement, Ogre::SceneNode* node);

        /** Updates the progress for the specified progress calculator. */
        void UpdateLoadProgress(ProgressCalculator* calculator, Ogre::Real amount);

        void LoadScene(const TiXmlElement* objectElement);
        void FinishLoadingLookAndTrackTargets();

        bool LoadResourceLocations(const TiXmlElement* objectElement);
        void LoadInstancedGeometries(const TiXmlElement* objectElement);
        void LoadInstancedGeometry(const TiXmlElement* objectElement);
        void LoadInstancedGeometryEntity(const TiXmlElement* objectElement, Ogre::InstancedGeometry* instancedGeometry);
        void LoadStaticGeometries(const TiXmlElement* objectElement);
        void LoadStaticGeometry(const TiXmlElement* objectElement);
        void LoadStaticGeometryEntity(const TiXmlElement* objectElement, Ogre::StaticGeometry* staticGeometry);
        void LoadPortalConnectedZones(const TiXmlElement* objectElement);

        void LoadNodes(const TiXmlElement* objectElement);
        void LoadQueryFlagAliases(const TiXmlElement* objectElement);
        void LoadVisibilityFlagAliases(const TiXmlElement* objectElement);
        void LoadExternals(const TiXmlElement* objectElement);
        void LoadExternalUserDatas(const TiXmlElement* objectElement);
        void LoadEnvironment(const TiXmlElement* objectElement);
        void LoadRenderTextures(const TiXmlElement* objectElement);
        void FinishLoadingRenderTextures();
        void GetRenderTextureObjects(Types::LoadedRenderTexture* loadedRenderTexture);
        void LoadTerrain(const TiXmlElement* objectElement);
        
        void LoadNode(const TiXmlElement* objectElement, Ogre::SceneNode* parentNode);

        void LoadFog(const TiXmlElement* objectElement);
        void LoadSkyBox(const TiXmlElement* objectElement);
        void LoadSkyDome(const TiXmlElement* objectElement);
        void LoadSkyPlane(const TiXmlElement* objectElement);
        void LoadClipping(const TiXmlElement* objectElement, Ogre::Real& nearClip, Ogre::Real& farClip);
        void LoadShadows(const TiXmlElement* objectElement);
        void LoadExternalItem(const TiXmlElement* objectElement, Types::ExternalItem& item);
        void LoadExternalUserData(const TiXmlElement* objectElement, Types::ExternalUserData& userData);

        void LoadEntity(const TiXmlElement* objectElement, const Types::MovableObjectOwner& owner);
        void LoadLight(const TiXmlElement* objectElement, const Types::MovableObjectOwner& owner);
        void LoadCamera(const TiXmlElement* objectElement, const Types::MovableObjectOwner& owner);
        void LoadParticleSystem(const TiXmlElement* objectElement, const Types::MovableObjectOwner& owner);
        void LoadBillboardSet(const TiXmlElement* objectElement, const Types::MovableObjectOwner& owner);
        void LoadPlane(const TiXmlElement* objectElement, const Types::MovableObjectOwner& owner);

        void LoadBoneAttachments(const TiXmlElement* objectElement, Ogre::Entity* entity);
        void LoadBoneAttachment(const TiXmlElement* objectElement, Ogre::Entity* entity);
        void LoadLookTarget(const TiXmlElement* objectElement, Ogre::SceneNode* node, Ogre::Camera* camera);
        void LoadTrackTarget(const TiXmlElement* objectElement, Ogre::SceneNode* node, Ogre::Camera* camera);

        void LoadBillboard(const TiXmlElement* objectElement, Ogre::BillboardSet* billboardSet);
        void LoadLightRange(const TiXmlElement* objectElement, Ogre::Light* light);
        void LoadLightAttenuation(const TiXmlElement* objectElement, Ogre::Light* light);

        void LoadNodeAnimations(const TiXmlElement* objectElement, Ogre::SceneNode* node);
        void LoadNodeAnimation(const TiXmlElement* objectElement, Ogre::SceneNode* node);
        void LoadNodeAnimationKeyFrame(const TiXmlElement* objectElement, Ogre::NodeAnimationTrack* animationTrack);

        void LoadObjectNames(const TiXmlElement* objectElement, const Ogre::String& elementName, std::vector<Ogre::String>& names);

        void LoadRenderTextureMaterials
            (
            const TiXmlElement* childElement,
            std::vector<Types::RenderTextureParameters::Material>& materials
            );

        Ogre::ShadowCameraSetup* ParseShadowCameraSetup(const Ogre::String& type, Ogre::Plane optimalPlane);

    protected:
        LoadOptions loadOptions;
        OgreMaxRenderWindowIterator* renderWindows;
        Ogre::SceneManager* sceneManager;
        Ogre::SceneNode* rootNode;
        Ogre::SceneNode* skyNode;
        OgreMaxSceneCallback* callback;
        Ogre::String defaultResourceGroupName;

        Ogre::String baseResourceLocation;
        bool loadedFromFileSystem;

        Types::ObjectExtraData sceneExtraData;
        Version formatVersion;
        Version minOgreVersion;
        Version ogreMaxVersion;
        Ogre::String author;
        Ogre::String application;
        Types::UpAxis upAxis;
        Ogre::Real unitsPerMeter;
        Ogre::String unitType;

        Types::ObjectExtraData terrainExtraData;
        Types::ObjectExtraData skyBoxExtraData;
        Types::ObjectExtraData skyDomeExtraData;
        Types::ObjectExtraData skyPlaneExtraData;

        ExternalItemVector externalItems;
        ExternalUserDataVector externalUserDatas;

        Ogre::ColourValue backgroundColor;

        Ogre::Real environmentNear;
        Ogre::Real environmentFar;

        Types::FogParameters fogParameters;

        Ogre::MovablePlane* shadowOptimalPlane;

        /** Movable planes that were created when loading Ogre plane objects */
        typedef std::map<Ogre::String, Ogre::MovablePlane*> MovablePlanesMap;
        MovablePlanesMap movablePlanes;

        /** The objects that were explicitly loaded from the scene file */
        std::map<Ogre::String, Ogre::MovableObject*> loadedObjects;

        /** Tracks scene loading progress. */
        class LoadSceneProgressCalculator : public ProgressCalculator
        {
        public:
            LoadSceneProgressCalculator()
            {
                this->nodes = AddCalculator("nodes");
            }

        public:
            //Child calculators
            ProgressCalculator* nodes;
        };
        LoadSceneProgressCalculator loadProgress;

        /** Look target information, encountered during the loading of a scene */
        typedef std::list<Types::LookTarget> LookTargetList;
        LookTargetList lookTargets;

        /** Track target information, encountered during the loading of a scene */
        typedef std::list<Types::TrackTarget> TrackTargetList;
        TrackTargetList trackTargets;

        /** The created animation states */
        AnimationStates animationStates;

        /**
         * A lookup table for all the ObjectExtraData created for each object
         * This will be destroyed when the OgreMaxScene is destroyed, so if you
         * would like to keep this data, it should be 'stolen' by calling
         * GetAllObjectExtraData(), copying the values, then clearing the values
         * that are contained in the OgreMaxScene
         */
        ObjectExtraDataMap allObjectExtraData;

        /**
         * A lookup table for all the ObjectExtraData created for each object
         */
        NodeObjectExtraDataMap allNodeObjectExtraData;

        /**
         * A lookup table for all the models that have been loaded
         */
        ModelMap modelMap;

        /**
         * The index into the first unused element in loadedRenderTextures
         * Used during the loading process
         */
        size_t currentRenderTextureIndex;

        /** All the render textures */
        std::vector<Types::LoadedRenderTexture*> loadedRenderTextures;

        /**
         * All the render textures, keyed by the corresponding render targets
         * The pointers reference the entries in loadedRenderTextures
         */
        typedef std::map<Ogre::RenderTarget*, Types::LoadedRenderTexture*> RenderTargetMap;
        RenderTargetMap renderTargets;

        /** The query flag aliases */
        Types::FlagAliases queryFlags;

        /** The visibility flag aliases */
        Types::FlagAliases visibilityFlags;

        /** The active scheme, saved before a render texture update and restored after */
        Ogre::String previousActiveScheme;

        //Various name prefixes
        Ogre::String objectNamePrefix;
        Ogre::String nodeNamePrefix;
        Ogre::String nodeAnimationNamePrefix;

        /** Loaded resource locations */
        ResourceLocations resourceLocations;

        /** A resource listener that uses the first encountered resource in the event of a name collision */
        class ResourceLoadingListener : public Ogre::ResourceLoadingListener
        {
        public:
            Ogre::DataStreamPtr resourceLoading(const Ogre::String& name, const Ogre::String& group, Ogre::Resource* resource) {return Ogre::DataStreamPtr();}
            void resourceStreamOpened(const Ogre::String& name, const Ogre::String& group, Ogre::Resource* resource, Ogre::DataStreamPtr& dataStream) {}
 	        bool resourceCollision(Ogre::Resource* resource, Ogre::ResourceManager* resourceManager) {return false;}
        };
        ResourceLoadingListener resourceLoadingListener;
    };

    /** A interface that receives notifications during the loading of the scene */
    class OgreMaxSceneCallback
    {
    public:
        virtual ~OgreMaxSceneCallback() {}

        /** Called before a scene is to be loaded. */
        virtual void StartedLoad(const OgreMaxScene* scene) {}

        virtual void CreatedSceneManager(const OgreMaxScene* scene, Ogre::SceneManager* sceneManager) {}

        virtual void CreatedExternal(const OgreMaxScene* scene, const OgreMax::Types::ExternalItem& item) {}
        virtual void CreatedExternalUserData(const OgreMaxScene* scene, const OgreMax::Types::ExternalUserData& userData) {}

        //Called when scene (not per-object) user data is loaded. Will only be called if there is user data
        virtual void LoadedUserData(const OgreMaxScene* scene, const Ogre::String& userDataReference, const Ogre::String& userData) {}

        //Called when resource locations are loaded. Will only be called if there are resource locations
        virtual void LoadingResourceLocations(const OgreMaxScene* scene) {}
        virtual void LoadedResourceLocations(const OgreMaxScene* scene, const OgreMaxScene::ResourceLocations& resourceLocations) {}

        //Resource-based callbacks, called before a resource is created
        //The main purpose of these callbacks is to get the resource group name for the
        //resource being loaded, though other parameters may be modified as well
        virtual void LoadingRenderTexture(const OgreMaxScene* scene, OgreMax::Types::RenderTextureParameters& renderTextureParameters) {}
        virtual void LoadingSceneFile(const OgreMaxScene* scene, const Ogre::String& fileName, Ogre::String& resourceGroupName) {}
        virtual void LoadingSkyBox(const OgreMaxScene* scene, OgreMax::Types::SkyBoxParameters& skyBoxParameters) {}
        virtual void LoadingSkyDome(const OgreMaxScene* scene, OgreMax::Types::SkyDomeParameters& skyDomeParameters) {}
        virtual void LoadingSkyPlane(const OgreMaxScene* scene, OgreMax::Types::SkyPlaneParameters& skyPlaneParameters) {}
        virtual void LoadingEntity(const OgreMaxScene* scene, OgreMax::Types::EntityParameters& parameters) {}
        virtual void LoadingPlane(const OgreMaxScene* scene, OgreMax::Types::PlaneParameters& parameters) {}

        /**
         * Called right before an Ogre animation is created. The parameters may be modified.
         * In particular modifying the length is useful in cases where there are multiple node animation
         * tracks on the animation, each of which starts and ends at different times.
         */
        virtual void LoadingNodeAnimation(const OgreMaxScene* scene, Types::NodeAnimationParameters& parameters) {}

        //MovableObject-creation callbacks, called after an object has been created
        //These are called after the object was attached to its parent node, if there is a parent node
        virtual void CreatedLight(const OgreMaxScene* scene, Ogre::Light* light) {CreatedMovableObject(scene, light);}
        virtual void CreatedCamera(const OgreMaxScene* scene, Ogre::Camera* camera) {CreatedMovableObject(scene, camera);}
        virtual void CreatedMesh(const OgreMaxScene* scene, Ogre::Mesh* mesh) {}
        virtual void CreatedEntity(const OgreMaxScene* scene, Ogre::Entity* entity) {CreatedMovableObject(scene, entity);}
        virtual void CreatedParticleSystem(const OgreMaxScene* scene, Ogre::ParticleSystem* particleSystem) {CreatedMovableObject(scene, particleSystem);}
        virtual void CreatedBillboardSet(const OgreMaxScene* scene, Ogre::BillboardSet* billboardSet) {CreatedMovableObject(scene, billboardSet);}
        virtual void CreatedPlane(const OgreMaxScene* scene, const Ogre::Plane& plane, Ogre::Entity* entity) {CreatedMovableObject(scene, entity);}
        virtual void CreatedMovableObject(const OgreMaxScene* scene, Ogre::MovableObject* object) {}

        virtual void CreatedNodeAnimation(const OgreMaxScene* scene, Ogre::SceneNode* Node, Ogre::Animation* animation) {}
        virtual void CreatedNodeAnimationTrack(const OgreMaxScene* scene, Ogre::SceneNode* node, Ogre::AnimationTrack* animationTrack, bool enabled, bool looping) {}
        virtual void CreatedNodeAnimationState(const OgreMaxScene* scene, Ogre::SceneNode* node, Ogre::AnimationState* animationState) {}

        //Render texture creation callbacks
        virtual Ogre::Camera* GetRenderTextureCamera(const OgreMaxScene* scene, const OgreMax::Types::RenderTextureParameters& renderTextureParameters) {return 0;}

        virtual void CreatedRenderTexture(const OgreMaxScene* scene, const OgreMax::Types::LoadedRenderTexture* renderTexture) {}

        /** Called after a node is created, but before its entities or any child nodes have been created */
        virtual void StartedCreatingNode(const OgreMaxScene* scene, Ogre::SceneNode* node) {}

        /** Called after the node and all its entities and child nodes have been created */
        virtual void FinishedCreatingNode(const OgreMaxScene* scene, Ogre::SceneNode* node) {}

        virtual void CreatedTerrain(const OgreMaxScene* scene, const Ogre::String& dataFile) {}

        virtual void HandleObjectExtraData(Types::ObjectExtraDataPtr objectExtraData) {}

        //Progress callback
        virtual void UpdatedLoadProgress(const OgreMaxScene* scene, Ogre::Real progress) {}

        /**
         * Called after a scene is loaded.
         * @param scene [in] - The scene that was loaded
         * @param success [in] - Indicates whether the scene was successfully loaded. False
         * indicates an error occurred.
         */
        virtual void FinishedLoad(const OgreMaxScene* scene, bool success) {}

        /**
         * Notifies the caller that shadow textures are about to be created
         * @param scene [in] - The scene that was loaded
         * @param parameters [in/out] - The shadow parameters that are in effect. Any of the texture-related
         * parameters can be modified. The most typical one to be modified is the texture pixel format.
         */
        virtual void CreatingShadowTextures(const OgreMaxScene* scene, Types::ShadowParameters& shadowParameters) {}
    };

}

#endif
