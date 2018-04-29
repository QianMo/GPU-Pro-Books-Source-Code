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


//Includes---------------------------------------------------------------------
#include "OgreMaxScene.hpp"
#include "OgreMaxUtilities.hpp"
#include <OgreHardwarePixelBuffer.h>
#include <OgreRenderTarget.h>
#include <OgreRenderTexture.h>
#include <OgreMaterialManager.h>
#include <OgreRoot.h>
#include <OgreRenderSystem.h>
#include <OgreRenderWindow.h>
#include <OgreSubEntity.h>
#include <OgreBillboard.h>
#include <OgreShadowCameraSetupFocused.h>
#include <OgreShadowCameraSetupLiSPSM.h>
#include <OgreShadowCameraSetupPlaneOptimal.h>

using namespace Ogre;
using namespace OgreMax;
using namespace OgreMax::Types;


//Local classes----------------------------------------------------------------
/**
 * Internal model loading callback used when creating an instance of a model
 * It translates model callbacks to scene callbacks
 */
class SceneModelInstanceCallback : public OgreMaxModelInstanceCallback
{
public:
    SceneModelInstanceCallback(OgreMaxScene* scene, OgreMaxSceneCallback* sceneCallback)
    {
        this->scene = scene;
        this->sceneCallback = sceneCallback;
    }

    virtual void CreatedLight(const OgreMaxModel* model, Light* light)
    {
        if (this->sceneCallback != 0)
            this->sceneCallback->CreatedLight(this->scene, light);

        CreatedMovableObject(model, light);
    }

    virtual void CreatedCamera(const OgreMaxModel* model, Camera* camera)
    {
        if (this->sceneCallback != 0)
            this->sceneCallback->CreatedCamera(this->scene, camera);

        CreatedMovableObject(model, camera);
    }

    virtual void CreatedMesh(const OgreMaxModel* model, Mesh* mesh)
    {
        if (this->sceneCallback != 0)
            this->sceneCallback->CreatedMesh(this->scene, mesh);
    }

    virtual void CreatedEntity(const OgreMaxModel* model, Entity* entity)
    {
        if (this->sceneCallback != 0)
            this->sceneCallback->CreatedEntity(this->scene, entity);

        CreatedMovableObject(model, entity);
    }

    virtual void CreatedParticleSystem(const OgreMaxModel* model, ParticleSystem* particleSystem)
    {
        if (this->sceneCallback != 0)
            this->sceneCallback->CreatedParticleSystem(this->scene, particleSystem);

        CreatedMovableObject(model, particleSystem);
    }

    virtual void CreatedBillboardSet(const OgreMaxModel* model, BillboardSet* billboardSet)
    {
        if (this->sceneCallback != 0)
            this->sceneCallback->CreatedBillboardSet(this->scene, billboardSet);

        CreatedMovableObject(model, billboardSet);
    }

    virtual void CreatedPlane(const OgreMaxModel* model, const Plane& plane, Entity* entity)
    {
        if (this->sceneCallback != 0)
            this->sceneCallback->CreatedPlane(this->scene, plane, entity);

        CreatedMovableObject(model, entity);
    }

    virtual void CreatedMovableObject(const OgreMaxModel* model, MovableObject* object)
    {
        if (this->sceneCallback != 0)
            this->sceneCallback->CreatedMovableObject(this->scene, object);
    }

    virtual void CreatedNodeAnimationTrack(const OgreMaxModel* model, SceneNode* node, AnimationTrack* animationTrack, bool enabled, bool looping)
    {
        if (this->sceneCallback != 0)
            this->sceneCallback->CreatedNodeAnimationTrack(this->scene, node, animationTrack, enabled, looping);
    }

    virtual void CreatedNodeAnimationState(const OgreMaxModel* model, SceneNode* node, AnimationState* animationState)
    {
        if (this->sceneCallback != 0)
            this->sceneCallback->CreatedNodeAnimationState(this->scene, node, animationState);
    }

    virtual void StartedCreatingNode(const OgreMaxModel* model, SceneNode* node)
    {
        node->setFixedYawAxis(false, this->scene->GetUpAxis() == UP_AXIS_Y ? Vector3::UNIT_Y : Vector3::UNIT_Z);

        if (this->sceneCallback != 0)
            this->sceneCallback->StartedCreatingNode(this->scene, node);
    }

    virtual void FinishedCreatingNode(const OgreMaxModel* model, SceneNode* node)
    {
        if (this->sceneCallback != 0)
            this->sceneCallback->FinishedCreatingNode(this->scene, node);
    }

    virtual void HandleObjectExtraData(ObjectExtraDataPtr objectExtraData)
    {
        this->scene->HandleNewObjectExtraData(objectExtraData);
    }

private:
    OgreMaxScene* scene;
    OgreMaxSceneCallback* sceneCallback;
};

/** Sets Ogre's global resource loading listener if there is none */
class SetResourceLoadingListener
{
public:
    SetResourceLoadingListener(Ogre::ResourceLoadingListener& listener)
    {
        //Set listener if none has already been set
        if (Ogre::ResourceGroupManager::getSingleton().getLoadingListener() == 0)
        {
            this->listener = &listener;
            Ogre::ResourceGroupManager::getSingleton().setLoadingListener(this->listener);
        }
        else
            this->listener = 0;
    }

    ~SetResourceLoadingListener()
    {
        //Remove listener
        if (this->listener != 0)
            Ogre::ResourceGroupManager::getSingleton().setLoadingListener(0);
    }

private:
    Ogre::ResourceLoadingListener* listener;
};


//Static initialization--------------------------------------------------------
Real OgreMaxScene::DEFAULT_MAX_DISTANCE = 100000;


//Implementation---------------------------------------------------------------
OgreMaxScene::OgreMaxScene()
{
    this->sceneManager = 0;
    this->rootNode = 0;
    this->skyNode = 0;
    this->callback = 0;

    this->upAxis = UP_AXIS_Y;
    this->unitsPerMeter = 0;

    this->environmentNear = 0;
    this->environmentFar = DEFAULT_MAX_DISTANCE;

    this->shadowOptimalPlane = 0;

    this->backgroundColor = ColourValue::Black;

    this->loadedRenderTextures.reserve(32);

    this->currentRenderTextureIndex = 0;

    this->loadedFromFileSystem = false;
}

OgreMaxScene::~OgreMaxScene()
{
    Destroy();
}

void OgreMaxScene::Load
    (
    const String& fileName,
    RenderWindow* renderWindow,
    LoadOptions loadOptions,
    SceneManager* sceneManager,
    SceneNode* rootNode,
    OgreMaxSceneCallback* callback,
    const String& defaultResourceGroupName
    )
{
    OgreMaxOneRenderWindow oneRenderWindow(renderWindow);

    Load(fileName, oneRenderWindow, loadOptions, sceneManager, rootNode, callback, defaultResourceGroupName);
}

void OgreMaxScene::Load
    (
    const String& fileNameOrContent,
    OgreMaxRenderWindowIterator& renderWindows,
    LoadOptions loadOptions,
    SceneManager* sceneManager,
    SceneNode* rootNode,
    OgreMaxSceneCallback* callback,
    const String& defaultResourceGroupName
    )
{
    bool isFileName = (loadOptions & FILE_NAME_CONTAINS_CONTENT) == 0;

    //Parse the directory and file base name from the input file name, if it's a file name
    String directory, fileBaseName;
    if (isFileName)
    {
        StringUtil::splitFilename(fileNameOrContent, fileBaseName, directory);
        if (!directory.empty())
        {
            //A full path was passed in. Assume the caller wants the directory to be added as
            //a resource location and that the directory be the base all other resource
            //locations are relative to
            SetBaseResourceLocation(directory);
        }
    }

    //Notify callback, possibly getting a new resource group name
    String resourceGroupName = defaultResourceGroupName;
    if (callback != 0)
        callback->LoadingSceneFile(this, fileBaseName, resourceGroupName);

    //Load the XML document
    TiXmlDocument document;
    if (isFileName)
    {
        //Load it from a file
        String absoluteFileName = OgreMaxUtilities::JoinPath(this->baseResourceLocation, fileBaseName);
        if (loadOptions & NO_FILE_SYSTEM_CHECK)
        {
            //Caller wants to load from the Ogre resource system
            OgreMaxUtilities::LoadXmlDocument(fileBaseName, document, resourceGroupName);
        }
        else if (OgreMaxUtilities::IsFileReadable(absoluteFileName))
        {
            //Load from disk
            if (document.LoadFile(absoluteFileName.c_str()))
            {
                //The file was successfully loaded. Enable the use of the resource locations in the file
                this->loadedFromFileSystem = true;
            }
            else
            {
                StringUtil::StrStreamType errorMessage;
                errorMessage << "Unable to load OgreMax scene file: " << absoluteFileName;

                OGRE_EXCEPT
                    (
                    Exception::ERR_INVALID_STATE,
		            errorMessage.str(),
		            "OgreMaxScene::Load"
                    );
            }
        }
        else if (!this->baseResourceLocation.empty())
        {
            //File isn't readable and the caller wanted it to be
            StringUtil::StrStreamType errorMessage;
            errorMessage << "Unable to read OgreMax scene file: " << absoluteFileName;

            OGRE_EXCEPT
                (
                Exception::ERR_FILE_NOT_FOUND,
	            errorMessage.str(),
	            "OgreMaxScene::Load"
                );
        }
        else
        {
            //Everything else failed. Load from the Ogre resource system
            OgreMaxUtilities::LoadXmlDocument(fileBaseName, document, resourceGroupName);
        }
    }
    else
    {
        //Load it from memory
        document.Parse(fileNameOrContent.c_str());
        if (document.Error())
        {
            StringUtil::StrStreamType errorMessage;
            errorMessage << "There was an error parsing the OgreMax scene XML: " << document.ErrorDesc();

            OGRE_EXCEPT
                (
                Exception::ERR_INVALIDPARAMS,
		        errorMessage.str(),
                "OgreMaxScene::Load"
                );
        }
    }

    Load(document.RootElement(), renderWindows, loadOptions, sceneManager, rootNode, callback, resourceGroupName);
}

void OgreMaxScene::Load
    (
    TiXmlElement* objectElement,
    OgreMaxRenderWindowIterator& renderWindows,
    LoadOptions loadOptions,
    SceneManager* sceneManager,
    SceneNode* rootNode,
    OgreMaxSceneCallback* callback,
    const String& defaultResourceGroupName
    )
{
    this->renderWindows = &renderWindows;
    this->loadOptions = loadOptions;
    this->sceneManager = sceneManager;
    this->rootNode = rootNode;
    this->callback = callback;
    this->defaultResourceGroupName = defaultResourceGroupName;

    SetResourceLoadingListener setResourceLoadingListener(this->resourceLoadingListener);

    //Send "start" notification to callback
    if (this->callback != 0)
        this->callback->StartedLoad(this);

    try
    {
        //Perform scene loading
        LoadScene(objectElement);
    }
    catch (...)
    {
        //Send "ended with failure" notification to callback
        if (this->callback != 0)
            this->callback->FinishedLoad(this, false);

        throw;
    }

    //Send "ended with success" notification to callback
    if (this->callback != 0)
        this->callback->FinishedLoad(this, true);
}

void OgreMaxScene::Update(Real elapsedTime)
{
    //Update managed animation states
    for (AnimationStates::iterator stateIterator = this->animationStates.begin();
        stateIterator != this->animationStates.end();
        ++stateIterator)
    {
        AnimationState* animationState = stateIterator->second;
        if (animationState->getEnabled())
            animationState->addTime(elapsedTime);
    }
}

const String& OgreMaxScene::GetBaseResourceLocation() const
{
    return this->baseResourceLocation;
}

void OgreMaxScene::SetBaseResourceLocation(const String& directory)
{
    this->baseResourceLocation = directory;
}

void OgreMaxScene::Destroy()
{
    //Delete planes
    delete this->shadowOptimalPlane;
    this->shadowOptimalPlane = 0;

    for (MovablePlanesMap::iterator plane = this->movablePlanes.begin();
        plane != this->movablePlanes.end();
        ++plane)
    {
        delete plane->second;
    }
    this->movablePlanes.clear();

    //Delete render textures
    //Note that this does not destroy the Ogre resources that the render textures use
    for (size_t renderTextureIndex = 0; renderTextureIndex < this->loadedRenderTextures.size(); renderTextureIndex++)
    {
        LoadedRenderTexture* loadedRenderTexture = this->loadedRenderTextures[renderTextureIndex];

        size_t faceCount = loadedRenderTexture->renderTexture->getNumFaces();
        for (size_t faceIndex = 0; faceIndex < faceCount; faceIndex++)
        {
            RenderTarget* renderTarget = loadedRenderTexture->renderTexture->getBuffer(faceIndex)->getRenderTarget();
            renderTarget->removeListener(this);
            renderTarget->removeAllViewports();
        }

        delete loadedRenderTexture;
    }
    this->loadedRenderTextures.clear();

    //Delete the extra object data
    this->allObjectExtraData.clear();

    //Delete the models
    for (ModelMap::iterator item = this->modelMap.begin();
        item != this->modelMap.end();
        ++item)
    {
        delete item->second;
    }
    this->modelMap.clear();

    //Clear out some lists
    this->externalItems.clear();
    this->loadedObjects.clear();
    this->animationStates.clear();
    this->renderTargets.clear();
    this->queryFlags.clear();
    this->visibilityFlags.clear();
    this->resourceLocations.clear();

    //Reset some variables
    this->currentRenderTextureIndex = 0;
    this->baseResourceLocation = StringUtil::BLANK;
    this->loadedFromFileSystem = false;
}

void OgreMaxScene::SetNamePrefix(const String& name, WhichNamePrefix prefixes)
{
    if (prefixes & OBJECT_NAME_PREFIX)
        this->objectNamePrefix = name;
    if (prefixes & NODE_NAME_PREFIX)
        this->nodeNamePrefix = name;
    if (prefixes & NODE_ANIMATION_NAME_PREFIX)
        this->nodeAnimationNamePrefix = name;
}

SceneManager* OgreMaxScene::GetSceneManager()
{
    return this->sceneManager;
}

SceneNode* OgreMaxScene::GetRootNode()
{
    return this->rootNode;
}

SceneNode* OgreMaxScene::GetSkyNode()
{
    return this->skyNode;
}

SceneNode* OgreMaxScene::GetSceneNode(const String& name, bool nameIncludesPrefix)
{
    SceneNode* node = 0;

    if (nameIncludesPrefix)
    {
        if (this->sceneManager->hasSceneNode(name))
            node = this->sceneManager->getSceneNode(name);
    }
    else
    {
        String prefixedName = this->nodeNamePrefix;
        prefixedName += name;
        if (this->sceneManager->hasSceneNode(prefixedName))
            node = this->sceneManager->getSceneNode(prefixedName);
    }

    return node;
}

Entity* OgreMaxScene::GetEntity(const String& name, bool nameIncludesPrefix)
{
    Entity* entity = 0;

    if (nameIncludesPrefix)
    {
        if (this->sceneManager->hasEntity(name))
            entity = this->sceneManager->getEntity(name);
    }
    else
    {
        String prefixedName = this->objectNamePrefix;
        prefixedName += name;
        if (this->sceneManager->hasEntity(prefixedName))
            entity = this->sceneManager->getEntity(prefixedName);
    }

    return entity;
}

Camera* OgreMaxScene::GetCamera(const String& name, bool nameIncludesPrefix)
{
    Camera* camera = 0;

    if (nameIncludesPrefix)
    {
        if (this->sceneManager->hasCamera(name))
            camera = this->sceneManager->getCamera(name);
    }
    else
    {
        String prefixedName = this->objectNamePrefix;
        prefixedName += name;
        if (this->sceneManager->hasCamera(prefixedName))
            camera = this->sceneManager->getCamera(prefixedName);
    }

    return camera;
}

UpAxis OgreMaxScene::GetUpAxis() const
{
    return this->upAxis;
}

const Vector3& OgreMaxScene::GetUpVector() const
{
    return this->upAxis == UP_AXIS_Y ? Vector3::UNIT_Y : Vector3::UNIT_Z;
}

Real OgreMaxScene::GetUnitsPerMeter()
{
    return this->unitsPerMeter;
}

const String OgreMaxScene::GetUnitType()
{
    return this->unitType;
}

const String& OgreMaxScene::GetAuthor() const
{
    return this->author;
}

const String& OgreMaxScene::GetApplication() const
{
    return this->application;
}

const ObjectExtraData& OgreMaxScene::GetSceneExtraData() const
{
    return this->sceneExtraData;
}

const ObjectExtraData& OgreMaxScene::GetTerrainExtraData() const
{
    return this->terrainExtraData;
}

const ObjectExtraData& OgreMaxScene::GetSkyBoxExtraData() const
{
    return this->skyBoxExtraData;
}

const ObjectExtraData& OgreMaxScene::GetSkyDomeExtraData() const
{
    return this->skyDomeExtraData;
}

const ObjectExtraData& OgreMaxScene::GetSkyPlaneExtraData() const
{
    return this->skyPlaneExtraData;
}

const OgreMaxScene::ExternalItemList& OgreMaxScene::GetExternalItems() const
{
    return this->externalItems;
}

const OgreMaxScene::ExternalUserDataVector& OgreMaxScene::GetExternalUserData() const
{
    return this->externalUserDatas;
}

const ColourValue& OgreMaxScene::GetBackgroundColor() const
{
    return this->backgroundColor;
}

Real OgreMaxScene::GetEnvironmentNear() const
{
    return this->environmentNear;
}

const FogParameters& OgreMaxScene::GetFogParameters() const
{
    return this->fogParameters;
}

Real OgreMaxScene::GetEnvironmentFar() const
{
    return this->environmentFar;
}

MovablePlane* OgreMaxScene::GetShadowOptimalPlane()
{
    return this->shadowOptimalPlane;
}

MovablePlane* OgreMaxScene::GetMovablePlane(const String& name)
{
    MovablePlane* plane = 0;

    String movablePlaneName;
    OgreMaxUtilities::CreateMovablePlaneName(movablePlaneName, name);
    MovablePlanesMap::iterator planeIterator = this->movablePlanes.find(movablePlaneName);
    if (planeIterator != this->movablePlanes.end())
        plane = planeIterator->second;

    return plane;
}

OgreMaxScene::AnimationStates& OgreMaxScene::GetAnimationStates()
{
    return this->animationStates;
}

AnimationState* OgreMaxScene::GetAnimationState(const String& animationName)
{
    AnimationStates::iterator stateIterator = this->animationStates.find(animationName);
    if (stateIterator != this->animationStates.end())
        return stateIterator->second;
    else
        return 0;
}

OgreMaxScene::ObjectExtraDataMap& OgreMaxScene::GetAllObjectExtraData()
{
    return this->allObjectExtraData;
}

OgreMaxScene::NodeObjectExtraDataMap& OgreMaxScene::GetAllNodeObjectExtraData()
{
    return this->allNodeObjectExtraData;
}

const OgreMaxScene::ModelMap& OgreMaxScene::GetModelMap() const
{
    return this->modelMap;
}

OgreMaxModel* OgreMaxScene::GetModel(const String& fileName)
{
    //Look up item and return it if found
    ModelMap::iterator item = this->modelMap.find(fileName);
    if (item != this->modelMap.end())
        return item->second;

    //Model couldn't be found, so load it
    OgreMaxModel* model = new OgreMaxModel;
    try
    {
        model->Load(fileName, this->defaultResourceGroupName);
    }
    catch (...)
    {
        delete model;
        throw;
    }

    //If we're here, the model was loaded. Add it to the lookup
    this->modelMap[fileName] = model;

    return model;
}

const FlagAliases& OgreMaxScene::GetQueryFlagAliases() const
{
    return this->queryFlags;
}

const FlagAliases& OgreMaxScene::GetVisibilityFlagAliases() const
{
    return this->visibilityFlags;
}

ObjectExtraData* OgreMaxScene::GetObjectExtraData(const MovableObject* object) const
{
    ObjectExtraDataMap::const_iterator item = this->allObjectExtraData.find(object);
    if (item != this->allObjectExtraData.end())
        return item->second.getPointer();
    else
        return 0;
}

Types::ObjectExtraData* OgreMaxScene::GetObjectExtraData(const Ogre::Node* node) const
{
    NodeObjectExtraDataMap::const_iterator item = this->allNodeObjectExtraData.find(node);
    if (item != this->allNodeObjectExtraData.end())
        return item->second.getPointer();
    else
        return 0;
}

const OgreMaxScene::ResourceLocations& OgreMaxScene::GetResourceLocations() const
{
    return this->resourceLocations;
}

Ogre::ResourceLoadingListener& OgreMaxScene::GetResourceLoadingListener()
{
    return this->resourceLoadingListener;
}

void OgreMaxScene::HandleNewObjectExtraData(ObjectExtraDataPtr objectExtraData)
{
    bool hasOwner =
        objectExtraData->object != 0 ||
        objectExtraData->node != 0;

    if (hasOwner && this->callback != 0)
        this->callback->HandleObjectExtraData(objectExtraData);

    //Store the object extra data if there is an object associated with it
    if (objectExtraData->object != 0)
        this->allObjectExtraData[objectExtraData->object] = objectExtraData;
    else if (objectExtraData->node != 0)
        this->allNodeObjectExtraData[objectExtraData->node] = objectExtraData;
}

void OgreMaxScene::DestroyedSceneNode(const SceneNode* node)
{
    NodeObjectExtraDataMap::iterator extraDataIterator = this->allNodeObjectExtraData.find(node);
    if (extraDataIterator != this->allNodeObjectExtraData.end())
        this->allNodeObjectExtraData.erase(extraDataIterator);
}

void OgreMaxScene::DestroyedObject(const MovableObject* object)
{
    ObjectExtraDataMap::iterator extraDataIterator = this->allObjectExtraData.find(object);
    if (extraDataIterator != this->allObjectExtraData.end())
        this->allObjectExtraData.erase(extraDataIterator);
}

void OgreMaxScene::preRenderTargetUpdate(const RenderTargetEvent& e)
{
    RenderTargetMap::iterator renderTarget = this->renderTargets.find(e.source);
    if (renderTarget != this->renderTargets.end())
    {
        LoadedRenderTexture& loadedRenderTexture = *renderTarget->second;
        RenderTextureParameters& params = loadedRenderTexture.parameters;

        //Set scheme
        this->previousActiveScheme = MaterialManager::getSingleton().getActiveScheme();
        if (params.scheme.empty())
            MaterialManager::getSingleton().setActiveScheme(MaterialManager::DEFAULT_SCHEME_NAME);
        else
            MaterialManager::getSingleton().setActiveScheme(params.scheme);

        //Hide all the hidden objects
        if (loadedRenderTexture.renderObjectNode != 0 && params.hideRenderObject)
            loadedRenderTexture.renderObjectNode->setVisible(false, false);
        loadedRenderTexture.hiddenObjects.Hide();

        //Show all the exclusive objects
        loadedRenderTexture.exclusiveObjects.Show();

        //If render target is for a cube map texture, update cube face camera positions
        if (params.textureType == TEX_TYPE_CUBE_MAP)
        {
            //Get the position from which the cube map will be rendered
            Vector3 position;
            loadedRenderTexture.GetReferencePosition(position);

            //Set position into cube face cameras
            size_t faceCount = loadedRenderTexture.renderTexture->getNumFaces();
            for (size_t faceIndex = 0; faceIndex < faceCount; faceIndex++)
                loadedRenderTexture.cubeFaceCameras[faceIndex]->setPosition(position);
        }
    }
}

void OgreMaxScene::postRenderTargetUpdate(const RenderTargetEvent& e)
{
    RenderTargetMap::iterator renderTarget = this->renderTargets.find(e.source);
    if (renderTarget != this->renderTargets.end())
    {
        LoadedRenderTexture& loadedRenderTexture = *renderTarget->second;
        RenderTextureParameters& params = loadedRenderTexture.parameters;

        //Restore scheme
        MaterialManager::getSingleton().setActiveScheme(this->previousActiveScheme);

        //Show all the hidden objects
        if (loadedRenderTexture.renderObjectNode != 0 && params.hideRenderObject)
            loadedRenderTexture.renderObjectNode->setVisible(true, false);
        loadedRenderTexture.hiddenObjects.Show();

        //Hide all the exclusive objects
        loadedRenderTexture.exclusiveObjects.Hide();
    }
}

String OgreMaxScene::GetNewObjectName(const TiXmlElement* objectElement, SceneNode* node)
{
    //Get the name from either the "name" attribute or the node
    String name = OgreMaxUtilities::GetStringAttribute(objectElement, "name");
    if (name.empty() && node != 0)
        name = node->getName();

    String prefixedName = this->objectNamePrefix;
    prefixedName += name;

    //Make sure the name is unique
    if (this->loadedObjects.find(prefixedName) != this->loadedObjects.end())
    {
        StringUtil::StrStreamType errorMessage;
        errorMessage << "Duplicate object name: " << prefixedName;

        OGRE_EXCEPT
            (
            Exception::ERR_DUPLICATE_ITEM,
		    errorMessage.str(),
		    "OgreMaxScene::GetNewObjectName"
            );
    }

    return prefixedName;
}

void OgreMaxScene::UpdateLoadProgress(ProgressCalculator* calculator, Real amount)
{
    //Update the specified calculator
    calculator->Update(amount);

    //Send progress notification to callback
    if (this->callback != 0)
        this->callback->UpdatedLoadProgress(this, this->loadProgress.GetProgress());
}

void OgreMaxScene::LoadScene(const TiXmlElement* objectElement)
{
    static const Version CURRENT_OGRE_VERSION(OGRE_VERSION_MAJOR, OGRE_VERSION_MINOR, OGRE_VERSION_PATCH);
    static const String CURRENT_OGRE_VERSION_STRING = CURRENT_OGRE_VERSION.ToString();

    this->sceneExtraData.id = OgreMaxUtilities::GetStringAttribute(objectElement, "id");
    this->formatVersion = Version(OgreMaxUtilities::GetStringAttribute(objectElement, "formatVersion"));
    this->minOgreVersion = Version(OgreMaxUtilities::GetStringAttribute(objectElement, "minOgreVersion"));
    this->ogreMaxVersion = Version(OgreMaxUtilities::GetStringAttribute(objectElement, "ogreMaxVersion"));
    this->author = OgreMaxUtilities::GetStringAttribute(objectElement, "author");
    this->application = OgreMaxUtilities::GetStringAttribute(objectElement, "application");
    String upAxisText = OgreMaxUtilities::GetStringAttribute(objectElement, "upAxis");
    if (!upAxisText.empty())
        this->upAxis = OgreMaxUtilities::ParseUpAxis(upAxisText);
    this->unitsPerMeter = OgreMaxUtilities::GetRealAttribute(objectElement, "unitsPerMeter", 0);
    this->unitType = OgreMaxUtilities::GetStringAttribute(objectElement, "unitType");

    //Make sure the Ogre version used to compile this viewer supports the Ogre version
    //required by the scene file
    if (this->minOgreVersion > CURRENT_OGRE_VERSION)
    {
        StringUtil::StrStreamType errorMessage;
        errorMessage
            << "The scene file's required Ogre version (" << this->minOgreVersion.ToString()
            << ") is not supported by the Ogre version this viewer uses ( " << CURRENT_OGRE_VERSION_STRING << ")";

        OGRE_EXCEPT
            (
            Exception::ERR_NOT_IMPLEMENTED,
		    errorMessage.str(),
		    "OgreMaxScene::LoadScene"
            );
    }

    //Create scene manager
    if (this->sceneManager == 0)
    {
        //Create the scene manager
        String sceneManager = OgreMaxUtilities::GetStringAttribute(objectElement, "sceneManager", "generic");
        SceneType sceneType;
        if (OgreMaxUtilities::ParseSceneManager(sceneManager, sceneType))
            this->sceneManager = Root::getSingleton().createSceneManager(sceneType, this->sceneExtraData.id);
        else
            this->sceneManager = Root::getSingleton().createSceneManager(sceneManager, this->sceneExtraData.id);

        //Notify callback
        if (this->callback != 0)
            this->callback->CreatedSceneManager(this, this->sceneManager);
    }

    //Get the root node
    if (this->rootNode == 0)
        this->rootNode = this->sceneManager->getRootSceneNode();

    //Load scene user data
    const TiXmlElement* userDataReferenceElement = objectElement->FirstChildElement("userDataReference");
    const TiXmlElement* userDataElement = objectElement->FirstChildElement("userData");
    if (userDataReferenceElement != 0 || userDataElement != 0)
    {
        if (userDataReferenceElement != 0)
            OgreMaxUtilities::LoadUserDataReference(userDataReferenceElement, this->sceneExtraData.userDataReference);
        if (userDataElement != 0)
            OgreMaxUtilities::GetChildText(userDataElement, this->sceneExtraData.userData);

        if (this->callback != 0)
            this->callback->LoadedUserData(this, this->sceneExtraData.userDataReference, this->sceneExtraData.userData);
    }

    //Load resource locations
    if ((this->loadOptions & SKIP_RESOURCE_LOCATIONS) == 0)
    {
        const TiXmlElement* resourceLocationsElement = objectElement->FirstChildElement("resourceLocations");
        if (resourceLocationsElement != 0 || this->loadedFromFileSystem)
        {
            //Notify callback
            if (this->callback != 0)
                this->callback->LoadingResourceLocations(this);

            LoadResourceLocations(resourceLocationsElement);

            //If necessary, add the base resource directory
            if (this->loadedFromFileSystem)
            {
                String baseResourceLocation = !this->baseResourceLocation.empty() ? this->baseResourceLocation : "./";

                ResourceLocation resourceLocation(baseResourceLocation, "FileSystem");
                AddResourceLocation(resourceLocation);
            }

            //Commit new resource locations
            CommitResourceLocations();

            //Notify callback
            if (this->callback != 0)
                this->callback->LoadedResourceLocations(this, this->resourceLocations);
        }
    }

    //Load environment settings
    const TiXmlElement* environmentElement = objectElement->FirstChildElement("environment");
    if (environmentElement != 0)
        LoadEnvironment(environmentElement);

    //Create render textures
    const TiXmlElement* renderTexturesElement = objectElement->FirstChildElement("renderTextures");
    if (renderTexturesElement != 0)
        LoadRenderTextures(renderTexturesElement);

    //Parse child elements
    const TiXmlElement* instancedGeometriesElement = 0;
    const TiXmlElement* staticGeometriesElement = 0;
    const TiXmlElement* portalConnectedZonesElement = 0;
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "nodes")
            LoadNodes(childElement);
        else if (elementName == "externals")
            LoadExternals(childElement);
        else if (elementName == "externalUserData")
            LoadExternalUserDatas(childElement);
        else if (elementName == "terrain")
            LoadTerrain(childElement);
        else if (elementName == "light")
            LoadLight(childElement, MovableObjectOwner());
        else if (elementName == "camera")
            LoadCamera(childElement, MovableObjectOwner());
        else if (elementName == "queryFlags")
            LoadQueryFlagAliases(childElement);
        else if (elementName == "visibilityFlags")
            LoadVisibilityFlagAliases(childElement);
        else if (elementName == "instancedGeometries")
            instancedGeometriesElement = childElement;
        else if (elementName == "staticGeometries")
            staticGeometriesElement = childElement;
        else if (elementName == "portalConnectedZones")
            portalConnectedZonesElement = childElement;
    }

    //Set default lighting if necessary
    if (this->loadOptions & SET_DEFAULT_LIGHTING)
        OgreMaxUtilities::SetDefaultLighting(this->sceneManager, this->upAxis);

    //Load instanced geometries
    if (instancedGeometriesElement != 0)
        LoadInstancedGeometries(instancedGeometriesElement);

    //Load static geometries
    if (staticGeometriesElement != 0)
        LoadStaticGeometries(staticGeometriesElement);

    //Load portal connected zones
    if (portalConnectedZonesElement == 0)
        LoadPortalConnectedZones(portalConnectedZonesElement);

    //Perform final steps for look and tracking targets
    FinishLoadingLookAndTrackTargets();

    //Perform final steps for render textures
    FinishLoadingRenderTextures();

    //Set fog parameters into scene manager
    if (environmentElement != 0 && (this->loadOptions & SKIP_ENVIRONMENT) == 0)
    {
        Real linearStart = this->fogParameters.linearStart * this->environmentFar;
        Real linearEnd = this->fogParameters.linearEnd * this->environmentFar;
        this->sceneManager->setFog
            (
            this->fogParameters.mode,
            this->fogParameters.color,
            this->fogParameters.expDensity,
            linearStart,
            linearEnd
            );
    }
}


void OgreMaxScene::FinishLoadingLookAndTrackTargets()
{
    //Resolve look targets
    for (LookTargetList::iterator lookTarget = this->lookTargets.begin();
        lookTarget != this->lookTargets.end();
        ++lookTarget)
    {
        SceneNode* lookTargetNode = GetSceneNode(lookTarget->nodeName, false);

        //Get position
        Vector3 position;
        if (lookTarget->isPositionSet)
            position = lookTarget->position;
        else
        {
            lookTarget->relativeTo = Node::TS_WORLD;
            position = lookTargetNode->_getDerivedPosition();
        }

        //Set look at depending on whether we have a node or camera
        if (lookTarget->sourceNode != 0)
            lookTarget->sourceNode->lookAt(position, lookTarget->relativeTo, lookTarget->localDirection);
        else if (lookTarget->sourceCamera != 0)
            lookTarget->sourceCamera->lookAt(position);
    }
    this->lookTargets.clear();

    //Resolve track targets
    for (TrackTargetList::iterator trackTarget = this->trackTargets.begin();
        trackTarget != this->trackTargets.end();
        ++trackTarget)
    {
        SceneNode* trackTargetNode = GetSceneNode(trackTarget->nodeName, false);

        //Set tracking depending on whether we have a node or camera
        if (trackTarget->sourceNode != 0)
            trackTarget->sourceNode->setAutoTracking(true, trackTargetNode, trackTarget->localDirection, trackTarget->offset);
        else if (trackTarget->sourceCamera != 0)
            trackTarget->sourceCamera->setAutoTracking(true, trackTargetNode, trackTarget->offset);
    }
    this->trackTargets.clear();
}

void OgreMaxScene::LoadInstancedGeometries(const TiXmlElement* objectElement)
{
    //Ensure instancing is supported
    if (!Root::getSingleton().getRenderSystem()->getCapabilities()->hasCapability(RSC_VERTEX_PROGRAM))
	{
		OGRE_EXCEPT
            (
            Exception::ERR_INVALIDPARAMS,
            "Instanced geometry is not supported by the current render system and/or video card",
            "OgreMaxScene::LoadInstancedGeometry"
            );
	}

    //Read all the instanced geometries
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "instancedGeometry")
            LoadInstancedGeometry(childElement);
    }
}

void OgreMaxScene::LoadInstancedGeometry(const TiXmlElement* objectElement)
{
    String name = OgreMaxUtilities::GetStringAttribute(objectElement, "name");
    bool castShadows = OgreMaxUtilities::GetBoolAttribute(objectElement, "castShadows", true);
    ObjectVisibility visibility = OgreMaxUtilities::GetObjectVisibilityAttribute(objectElement, "visible");
    unsigned int batchCount = OgreMaxUtilities::GetUIntAttribute(objectElement, "batchCount", 0);
    String renderQueue = OgreMaxUtilities::GetStringAttribute(objectElement, "renderQueue");
    Real renderingDistance = OgreMaxUtilities::GetRealAttribute(objectElement, "renderingDistance", 0);
    Vector3 origin = Vector3::ZERO;
    Vector3 dimensions(1000000, 1000000, 1000000);
    const TiXmlElement* entitiesElement = 0;

    //Iterate over all the child elements
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "origin")
            origin = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "dimensions")
            dimensions = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "entities")
            entitiesElement = childElement;
    }

    //Create the instanced geometry
    InstancedGeometry* instancedGeometry = this->sceneManager->createInstancedGeometry(name);
    instancedGeometry->setCastShadows(castShadows);
    OgreMaxUtilities::SetObjectVisibility(instancedGeometry, visibility);
    instancedGeometry->setOrigin(origin);
    instancedGeometry->setBatchInstanceDimensions(dimensions);
    if (!renderQueue.empty())
        instancedGeometry->setRenderQueueGroup(OgreMaxUtilities::ParseRenderQueue(renderQueue));
    instancedGeometry->setRenderingDistance(renderingDistance);

    //Add the entities
    childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(entitiesElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "entity")
            LoadInstancedGeometryEntity(childElement, instancedGeometry);
    }

    //Build the instanced geometry
    instancedGeometry->build();

    //Add additional batch instances
    for (unsigned int batchIndex = 0; batchIndex < batchCount; batchIndex++)
        instancedGeometry->addBatchInstance();
}

void OgreMaxScene::LoadInstancedGeometryEntity(const TiXmlElement* objectElement, InstancedGeometry* instancedGeometry)
{
    static const String TEMP_ENTITY_NAME("__instancedGeometryEntity");

    String meshFile = OgreMaxUtilities::GetStringAttribute(objectElement, "meshFile");
    Vector3 position = Vector3::ZERO;
    Quaternion rotation = Quaternion::IDENTITY;
    Vector3 scale = Vector3::UNIT_SCALE;
    std::vector<EntityParameters::Subentity> subentities;

    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "position")
            position = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "rotation")
            rotation = OgreMaxUtilities::LoadRotation(childElement);
        else if (elementName == "scale")
            scale = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "subentities")
            OgreMaxUtilities::LoadSubentities(childElement, subentities);
    }

    if (!meshFile.empty())
    {
        //Create temporary entity
        Entity* entity = OgreMaxUtilities::CreateEntity(this->sceneManager, TEMP_ENTITY_NAME, meshFile, subentities);

        //Add entity to the static geometry
        instancedGeometry->addEntity(entity, position, rotation, scale);

        //Destroy entity
        this->sceneManager->destroyEntity(entity);
    }
}

void OgreMaxScene::LoadStaticGeometries(const TiXmlElement* objectElement)
{
    //Read all the static geometries
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "staticGeometry")
            LoadStaticGeometry(childElement);
    }
}

void OgreMaxScene::LoadStaticGeometry(const TiXmlElement* objectElement)
{
    String name = OgreMaxUtilities::GetStringAttribute(objectElement, "name");
    bool castShadows = OgreMaxUtilities::GetBoolAttribute(objectElement, "castShadows", true);
    ObjectVisibility visibility = OgreMaxUtilities::GetObjectVisibilityAttribute(objectElement, "visible");
    String renderQueue = OgreMaxUtilities::GetStringAttribute(objectElement, "renderQueue");
    Real renderingDistance = OgreMaxUtilities::GetRealAttribute(objectElement, "renderingDistance", 0);
    Vector3 origin = Vector3::ZERO;
    Vector3 dimensions(1000000, 1000000, 1000000);
    const TiXmlElement* entitiesElement = 0;

    //Iterate over all the child elements
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "origin")
            origin = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "dimensions")
            dimensions = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "entities")
            entitiesElement = childElement;
    }

    //Create the instanced geometry
    StaticGeometry* staticGeometry = this->sceneManager->createStaticGeometry(name);
    staticGeometry->setCastShadows(castShadows);
    OgreMaxUtilities::SetObjectVisibility(staticGeometry, visibility);
    staticGeometry->setOrigin(origin);
    staticGeometry->setRegionDimensions(dimensions);
    if (!renderQueue.empty())
        staticGeometry->setRenderQueueGroup(OgreMaxUtilities::ParseRenderQueue(renderQueue));
    staticGeometry->setRenderingDistance(renderingDistance);

    //Add the entities
    childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(entitiesElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "entity")
            LoadStaticGeometryEntity(childElement, staticGeometry);
    }

    //Build the static geometry
    staticGeometry->build();
}

void OgreMaxScene::LoadStaticGeometryEntity(const TiXmlElement* objectElement, StaticGeometry* staticGeometry)
{
    static const String TEMP_ENTITY_NAME("__staticGeometryEntity");

    String meshFile = OgreMaxUtilities::GetStringAttribute(objectElement, "meshFile");
    Vector3 position = Vector3::ZERO;
    Quaternion rotation = Quaternion::IDENTITY;
    Vector3 scale = Vector3::UNIT_SCALE;
    std::vector<EntityParameters::Subentity> subentities;

    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "position")
            position = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "rotation")
            rotation = OgreMaxUtilities::LoadRotation(childElement);
        else if (elementName == "scale")
            scale = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "subentities")
            OgreMaxUtilities::LoadSubentities(childElement, subentities);
    }

    if (!meshFile.empty())
    {
        //Create temporary entity
        Entity* entity = OgreMaxUtilities::CreateEntity(this->sceneManager, TEMP_ENTITY_NAME, meshFile, subentities);
        
        //Add entity to the static geometry
        staticGeometry->addEntity(entity, position, rotation, scale);

        //Destroy entity
        this->sceneManager->destroyEntity(entity);
    }
}

void OgreMaxScene::LoadPortalConnectedZones(const TiXmlElement* objectElement)
{
#if OGRE_VERSION_MAJOR > 1 && OGRE_VERSION_MINOR >= 5
    //TODO: Implement this
#endif
}

bool OgreMaxScene::LoadResourceLocations(const TiXmlElement* objectElement)
{
    //Exit early if resource locations are not allowed
    if (!this->loadedFromFileSystem)
        return false;

    size_t locationCount = 0;

    ResourceLocation resourceLocation;

    //Iterate over all the resource groups
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "resourceLocation")
        {
            //Name
            resourceLocation.name = OgreMaxUtilities::GetStringAttribute(childElement, "name");
            if (!this->baseResourceLocation.empty())
                resourceLocation.name = OgreMaxUtilities::JoinPath(this->baseResourceLocation, resourceLocation.name);

            //Type
            resourceLocation.type = OgreMaxUtilities::GetStringAttribute(childElement, "type");

            //Recursive
            resourceLocation.recursive = OgreMaxUtilities::GetBoolAttribute(childElement, "recursive", false);

            //Add location
            if (AddResourceLocation(resourceLocation))
                locationCount++;
        }
    }

    return locationCount > 0;
}

bool OgreMaxScene::AddResourceLocation(const ResourceLocation& resourceLocation)
{
    //Normalize the resource location by expanding the name to a full path
    ResourceLocation normalizedResourceLocation = resourceLocation;
    OgreMaxUtilities::MakeFullPath(normalizedResourceLocation.name);

    //Determine whether resource location has already been added
    bool isNewResourceLocation =
        this->resourceLocations.find(normalizedResourceLocation) == this->resourceLocations.end();

    //Add resource location if it's new
    if (isNewResourceLocation)
    {
        //Hold onto the resource location for later
        this->resourceLocations.insert(normalizedResourceLocation);
    }

    return isNewResourceLocation;
}

void OgreMaxScene::CommitResourceLocations()
{
    //Count the number of uninitialized locations
    int newResourceLocationCount = 0;
    for (ResourceLocations::iterator resourceLocationIterator = this->resourceLocations.begin();
        resourceLocationIterator != this->resourceLocations.end();
        ++resourceLocationIterator)
    {
        if (!resourceLocationIterator->initialized)
        {
            newResourceLocationCount++;

            //Add resource location to Ogre
            ResourceGroupManager::getSingleton().addResourceLocation
                (
                resourceLocationIterator->name,
                resourceLocationIterator->type,
                this->defaultResourceGroupName,
                resourceLocationIterator->recursive
                );

            //The resource location is no longer uninitialized
            //This seems to fix a compiler error with GCC
            const_cast<ResourceLocation&>(*resourceLocationIterator).initialized = true;
        }
    }

    if (newResourceLocationCount > 0)
        ResourceGroupManager::getSingleton().initialiseAllResourceGroups();
}

void OgreMaxScene::LoadQueryFlagAliases(const TiXmlElement* objectElement)
{
    //Exit early if skip option is set
    if ((this->loadOptions & SKIP_QUERY_FLAG_ALIASES) != 0)
        return;

    //Allocate the query flags
    size_t oldSize = this->queryFlags.size();
    size_t queryFlagCount = OgreMaxUtilities::GetElementCount(objectElement, "queryFlag");
    this->queryFlags.resize(oldSize + queryFlagCount);

    //Iterate over all the query flags
    size_t index = 0;
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        this->queryFlags[oldSize + index].name = OgreMaxUtilities::GetStringAttribute(childElement, "name");
        this->queryFlags[oldSize + index].bit = OgreMaxUtilities::GetIntAttribute(childElement, "bit", 0);
        index++;
    }
}

void OgreMaxScene::LoadVisibilityFlagAliases(const TiXmlElement* objectElement)
{
    //Exit early if skip option is set
    if ((this->loadOptions & SKIP_VISIBILITY_FLAG_ALIASES) != 0)
        return;

    //Allocate the visibility flags
    size_t oldSize = this->queryFlags.size();
    size_t visibilityFlagCount = OgreMaxUtilities::GetElementCount(objectElement, "visibilityFlag");
    this->visibilityFlags.resize(oldSize + visibilityFlagCount);

    //Iterate over all the visibility flags
    size_t index = 0;
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        this->visibilityFlags[oldSize + index].name = OgreMaxUtilities::GetStringAttribute(childElement, "name");
        this->visibilityFlags[oldSize + index].bit = OgreMaxUtilities::GetIntAttribute(childElement, "bit", 0);
        index++;
    }
}

void OgreMaxScene::LoadNodes(const TiXmlElement* objectElement)
{
    //Exit early if skip option is set
    if ((this->loadOptions & SKIP_NODES) != 0)
        return;

    //Initialize progress counters
    size_t nodeCount = OgreMaxUtilities::GetElementCount(objectElement, "node");
    this->loadProgress.nodes->SetRange(nodeCount);

    //Iterate over all the node children
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "node" || elementName == "modelInstance")
            LoadNode(childElement, this->rootNode);
        else if (elementName == "position")
            this->rootNode->setPosition(OgreMaxUtilities::LoadXYZ(childElement));
        else if (elementName == "rotation")
            this->rootNode->setOrientation(OgreMaxUtilities::LoadRotation(childElement));
        else if (elementName == "scale")
            this->rootNode->setScale(OgreMaxUtilities::LoadXYZ(childElement));
    }

    this->rootNode->setInitialState();
}

void OgreMaxScene::LoadExternals(const TiXmlElement* objectElement)
{
    //Exit early if skip option is set
    if ((this->loadOptions & SKIP_EXTERNALS) != 0)
        return;

    //Initialize progress counter
    size_t oldSize = this->externalItems.size();
    if ((this->loadOptions & NO_EXTERNALS) == 0)
    {
        size_t itemCount = OgreMaxUtilities::GetChildElementCount(objectElement, "item");
        this->externalItems.resize(oldSize + itemCount);
    }

    //Iterate over all the node children
    size_t index = 0;
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "item")
        {
            if ((this->loadOptions & NO_EXTERNALS) == 0)
                LoadExternalItem(childElement, this->externalItems[oldSize + index++]);
            else
            {
                ExternalItem item;
                LoadExternalItem(childElement, item);
            }
        }
    }
}

void OgreMaxScene::LoadExternalUserDatas(const TiXmlElement* objectElement)
{
    //Initialize progress counter
    size_t oldSize = this->externalItems.size();
    size_t userDataCount = OgreMaxUtilities::GetChildElementCount(objectElement, "userData");
    this->externalUserDatas.resize(oldSize + userDataCount);

    //Iterate over all the node children
    size_t index = 0;
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "userData")
            LoadExternalUserData(childElement, this->externalUserDatas[oldSize + index++]);
    }
}

void OgreMaxScene::LoadEnvironment(const TiXmlElement* objectElement)
{
    //Exit early if skip option is set
    if ((this->loadOptions & SKIP_ENVIRONMENT) != 0)
        return;

    //Iterate over all the node children
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "fog")
            LoadFog(childElement);
        else if (elementName == "skyBox")
            LoadSkyBox(childElement);
        else if (elementName == "skyDome")
            LoadSkyDome(childElement);
        else if (elementName == "skyPlane")
            LoadSkyPlane(childElement);
        else if (elementName == "clipping")
            OgreMaxUtilities::LoadClipping(childElement, this->environmentNear, this->environmentFar);
        else if (elementName == "colourAmbient")
            this->sceneManager->setAmbientLight(OgreMaxUtilities::LoadColor(childElement));
        else if (elementName == "colourBackground")
        {
            this->backgroundColor = OgreMaxUtilities::LoadColor(childElement);

            //Set all viewports to the specified background color
            if (this->renderWindows->Start())
            {
                do
                {
                    RenderWindow* renderWindow = this->renderWindows->GetCurrent();
                    unsigned short viewportCount = renderWindow->getNumViewports();
                    for (unsigned short viewportIndex = 0; viewportIndex < viewportCount; viewportIndex++)
                        renderWindow->getViewport(viewportIndex)->setBackgroundColour(backgroundColor);
                }while (this->renderWindows->MoveNext());
            }
        }
        else if (elementName == "shadows")
            LoadShadows(childElement);
    }
}

void OgreMaxScene::LoadRenderTextures(const TiXmlElement* objectElement)
{
    //Get default pixel format
    //It's unlikely that this would vary among render windows, but just take the minimum anyway
    unsigned int bestColorDepth = 32;
    if (this->renderWindows->Start())
    {
        do
        {
            bestColorDepth = std::min(this->renderWindows->GetCurrent()->getColourDepth(), bestColorDepth);
        }while (this->renderWindows->MoveNext());
    }
    PixelFormat defaultPixelFormat = (bestColorDepth == 16) ? PF_A4R4G4B4 : PF_A8R8G8B8;

    //Read all the render textures
    const TiXmlElement* renderTextureElement = 0;
    while (renderTextureElement = OgreMaxUtilities::IterateChildElements(objectElement, renderTextureElement))
    {
        LoadedRenderTexture* loadedRenderTexture = new LoadedRenderTexture;
        this->loadedRenderTextures.push_back(loadedRenderTexture);

        RenderTextureParameters& renderTextureParams = loadedRenderTexture->parameters;

        renderTextureParams.name = OgreMaxUtilities::GetStringAttribute(renderTextureElement, "name");
        renderTextureParams.width = OgreMaxUtilities::GetIntAttribute(renderTextureElement, "width", 512);
        renderTextureParams.height = OgreMaxUtilities::GetIntAttribute(renderTextureElement, "height", 512);
        renderTextureParams.pixelFormat = OgreMaxUtilities::GetPixelFormatAttribute(renderTextureElement, "pixelFormat", defaultPixelFormat);
        renderTextureParams.textureType = OgreMaxUtilities::GetTextureTypeAttribute(renderTextureElement, "textureType", renderTextureParams.textureType);
        renderTextureParams.cameraName = OgreMaxUtilities::GetStringAttribute(renderTextureElement, "camera");
        renderTextureParams.scheme = OgreMaxUtilities::GetStringAttribute(renderTextureElement, "scheme");
        renderTextureParams.clearEveryFrame = OgreMaxUtilities::GetBoolAttribute(renderTextureElement, "clearEveryFrame", renderTextureParams.clearEveryFrame);
        renderTextureParams.autoUpdate = OgreMaxUtilities::GetBoolAttribute(renderTextureElement, "autoUpdate", renderTextureParams.autoUpdate);
        renderTextureParams.hideRenderObject = OgreMaxUtilities::GetBoolAttribute(renderTextureElement, "hideRenderObject", renderTextureParams.hideRenderObject);
        renderTextureParams.renderObjectName = OgreMaxUtilities::GetStringAttribute(renderTextureElement, "renderObjectName");
        renderTextureParams.backgroundColor = this->backgroundColor;
        renderTextureParams.resourceGroupName = this->defaultResourceGroupName;

        String elementName;
        const TiXmlElement* childElement = 0;
        while (childElement = OgreMaxUtilities::IterateChildElements(renderTextureElement, childElement))
        {
            elementName = childElement->Value();

            if (elementName == "backgroundColor")
                renderTextureParams.backgroundColor = OgreMaxUtilities::LoadColor(childElement);
            else if (elementName == "materials")
            {
                size_t materialCount = OgreMaxUtilities::GetElementCount(childElement, "material");
                renderTextureParams.materials.resize(materialCount);
                LoadRenderTextureMaterials(childElement, renderTextureParams.materials);
            }
            else if (elementName == "hiddenObjects")
                LoadObjectNames(childElement, "hiddenObject", renderTextureParams.hiddenObjects);
            else if (elementName == "exclusiveObjects")
                LoadObjectNames(childElement, "exclusiveObject", renderTextureParams.exclusiveObjects);
            else if (elementName == "renderPlane")
                renderTextureParams.renderPlane = OgreMaxUtilities::LoadPlane(childElement);
        }

        //Notify callback
        if (this->callback != 0)
            this->callback->LoadingRenderTexture(this, renderTextureParams);

        //Create the render texture
        loadedRenderTexture->renderTexture = TextureManager::getSingleton().createManual
            (
            renderTextureParams.name,
            renderTextureParams.resourceGroupName,
            renderTextureParams.textureType,
            renderTextureParams.width,
            renderTextureParams.height,
            0,
            renderTextureParams.pixelFormat,
            TU_RENDERTARGET
            );

        //Initialize all the texture's render targets
        size_t faceCount = loadedRenderTexture->renderTexture->getNumFaces();
        for (size_t faceIndex = 0; faceIndex < faceCount; faceIndex++)
        {
            RenderTarget* renderTarget = loadedRenderTexture->renderTexture->getBuffer(faceIndex)->getRenderTarget();
            renderTarget->setAutoUpdated(renderTextureParams.autoUpdate);
            renderTarget->addListener(this);
            this->renderTargets[renderTarget] = loadedRenderTexture;
        }
    }
}

void OgreMaxScene::FinishLoadingRenderTextures()
{
    static const Quaternion CUBE_FACE_CAMERA_ORIENTATIONS[] =
        {
        Quaternion(0.707107, 0, -0.707107, 0),
        Quaternion(-0.707107, 0, -0.707107, 0),
        Quaternion(0.707107, 0.707107, 0, 0),
        Quaternion(0.707107, -0.707107, 0, 0),
        Quaternion(1, 0, 0, 0),
        Quaternion(0, 0, -1, 0)
        };
    static int cubeFaceCameraCounter = 0; //Used for generating unique names

    for (; this->currentRenderTextureIndex < this->loadedRenderTextures.size(); this->currentRenderTextureIndex++)
    {
        LoadedRenderTexture* loadedRenderTexture = this->loadedRenderTextures[this->currentRenderTextureIndex];
        const RenderTextureParameters& renderTextureParams = loadedRenderTexture->parameters;

        //Create the scheme
        if (!renderTextureParams.scheme.empty())
            MaterialManager::getSingleton()._getSchemeIndex(renderTextureParams.scheme);

        //Get the camera
        //First try the camera callback
        if (this->callback != 0)
            loadedRenderTexture->camera = this->callback->GetRenderTextureCamera(this, renderTextureParams);

        //If there's no camera yet, get the camera from the scene
        if (loadedRenderTexture->camera == 0 && !renderTextureParams.cameraName.empty())
            loadedRenderTexture->camera = GetCamera(renderTextureParams.cameraName, false);
        
        //Set up viewport and render object (either a reflection plane for 2D, or an object for 3D)
        size_t faceCount = loadedRenderTexture->renderTexture->getNumFaces();
        if (renderTextureParams.textureType == TEX_TYPE_2D)
        {
            Viewport* viewport = 0;
            if (loadedRenderTexture->camera != 0)
            {
                RenderTarget* renderTarget = loadedRenderTexture->renderTexture->getBuffer()->getRenderTarget();

                //Add viewport
                Viewport* viewport = renderTarget->addViewport(loadedRenderTexture->camera);
                viewport->setClearEveryFrame(renderTextureParams.clearEveryFrame);
                viewport->setBackgroundColour(renderTextureParams.backgroundColor);
                viewport->setMaterialScheme(renderTextureParams.scheme);
                loadedRenderTexture->viewports[0] = viewport;

                //Set up render object (reflection plane)
                if (!renderTextureParams.renderObjectName.empty())
                {
                    //Build the plane name
                    String movablePlaneName;
                    OgreMaxUtilities::CreateMovablePlaneName(movablePlaneName, renderTextureParams.renderObjectName);

                    //Get or create the render plane
                    MovablePlanesMap::iterator planeIterator = this->movablePlanes.find(movablePlaneName);
                    if (planeIterator != this->movablePlanes.end())
                    {
                        //Found an existing movable plane
                        loadedRenderTexture->renderPlane = planeIterator->second;
                        loadedRenderTexture->renderObjectNode = (SceneNode*)loadedRenderTexture->renderPlane->getParentNode();
                    }
                    else
                    {
                        //Create a new movable plane
                        loadedRenderTexture->renderPlane = new MovablePlane(movablePlaneName);
                        this->movablePlanes[movablePlaneName] = loadedRenderTexture->renderPlane;
                        loadedRenderTexture->renderPlane->normal = renderTextureParams.renderPlane.normal;
                        loadedRenderTexture->renderPlane->d = renderTextureParams.renderPlane.d;
                    }

                    //Configure reflection
                    for (size_t materialIndex = 0; materialIndex < renderTextureParams.materials.size(); materialIndex++)
                    {
                        const RenderTextureParameters::Material& renderTextureMaterial =
                            renderTextureParams.materials[materialIndex];

                        MaterialPtr material = MaterialManager::getSingleton().getByName(renderTextureMaterial.name);
                        if (!material.isNull())
                        {
                            if (renderTextureMaterial.techniqueIndex < material->getNumTechniques())
                            {
                                Technique* technique = material->getTechnique(renderTextureMaterial.techniqueIndex);
                                if (renderTextureMaterial.passIndex < technique->getNumPasses())
                                {
                                    Pass* pass = technique->getPass(renderTextureMaterial.passIndex);
                                    if (renderTextureMaterial.textureUnitIndex < pass->getNumTextureUnitStates())
                                    {
                                        TextureUnitState* textureUnit =
                                            pass->getTextureUnitState(renderTextureMaterial.textureUnitIndex);
                                        textureUnit->setProjectiveTexturing(true, loadedRenderTexture->camera);
                                    }
                                }
                            }
                        }
                    }
                    loadedRenderTexture->camera->enableReflection(loadedRenderTexture->renderPlane);
                    loadedRenderTexture->camera->enableCustomNearClipPlane(loadedRenderTexture->renderPlane);
                }
            }
        }
        else if (renderTextureParams.textureType == TEX_TYPE_CUBE_MAP)
        {
            //Get the render object, if any
            if (!renderTextureParams.renderObjectName.empty())
            {
                Entity* renderObject = GetEntity(renderTextureParams.renderObjectName, false);
                if (renderObject != 0)
                    loadedRenderTexture->renderObjectNode = (SceneNode*)renderObject->getParentNode();
            }

            //Get the position from which the cube map will be rendered
            Vector3 position;
            loadedRenderTexture->GetReferencePosition(position);

            //Create a camera and viewport for each cube face
            String cameraName;
            for (size_t faceIndex = 0; faceIndex < faceCount; faceIndex++)
            {
                //Build a unique camera name
                if (loadedRenderTexture->camera != 0)
                    cameraName = loadedRenderTexture->camera->getName() + "_CubeFaceCamera";
                else
                    cameraName = "CubeFaceCamera" + StringConverter::toString(cubeFaceCameraCounter++) + "_";
                cameraName += StringConverter::toString(faceIndex);

                //Create camera
                Camera* cubeFaceCamera = this->sceneManager->createCamera(cameraName);
                cubeFaceCamera->setAspectRatio(1);
                cubeFaceCamera->setFOVy(Degree(90));
                cubeFaceCamera->setPosition(position);
                cubeFaceCamera->setOrientation(CUBE_FACE_CAMERA_ORIENTATIONS[faceIndex]);

                //Use the reference camera's clip distances, if possible
                if (loadedRenderTexture->camera != 0)
                {
                    cubeFaceCamera->setNearClipDistance(loadedRenderTexture->camera->getNearClipDistance());
                    cubeFaceCamera->setFarClipDistance(loadedRenderTexture->camera->getFarClipDistance());
                }
                else
                {
                    cubeFaceCamera->setNearClipDistance((Real).1);
                    cubeFaceCamera->setFarClipDistance((Real)10000);
                }

                loadedRenderTexture->cubeFaceCameras[faceIndex] = cubeFaceCamera;

                //Add viewport
                RenderTarget* renderTarget = loadedRenderTexture->renderTexture->getBuffer(faceIndex)->getRenderTarget();
                Viewport* viewport = renderTarget->addViewport(cubeFaceCamera);
                viewport->setClearEveryFrame(renderTextureParams.clearEveryFrame);
                viewport->setBackgroundColour(renderTextureParams.backgroundColor);
                viewport->setMaterialScheme(renderTextureParams.scheme);
                loadedRenderTexture->viewports[faceIndex] = viewport;
            }

            //Configure reflection
            for (size_t materialIndex = 0; materialIndex < renderTextureParams.materials.size(); materialIndex++)
            {
                const RenderTextureParameters::Material& renderTextureMaterial =
                    renderTextureParams.materials[materialIndex];

                MaterialPtr material = MaterialManager::getSingleton().getByName(renderTextureMaterial.name);
                if (!material.isNull())
                {
                    if (renderTextureMaterial.techniqueIndex < material->getNumTechniques())
                    {
                        Technique* technique = material->getTechnique(renderTextureMaterial.techniqueIndex);
                        if (renderTextureMaterial.passIndex < technique->getNumPasses())
                        {
                            Pass* pass = technique->getPass(renderTextureMaterial.passIndex);
                            if (renderTextureMaterial.textureUnitIndex < pass->getNumTextureUnitStates())
                            {
                                TextureUnitState* textureUnit =
                                    pass->getTextureUnitState(renderTextureMaterial.textureUnitIndex);
                                textureUnit->setEnvironmentMap(true, TextureUnitState::ENV_REFLECTION);                                
                            }
                        }
                    }
                }
            }
        }

        GetRenderTextureObjects(loadedRenderTexture);

        //Notify callback
        if (this->callback != 0)
            this->callback->CreatedRenderTexture(this, loadedRenderTexture);
    }
}

void OgreMaxScene::GetRenderTextureObjects(LoadedRenderTexture* loadedRenderTexture)
{
    const RenderTextureParameters& renderTextureParams = loadedRenderTexture->parameters;

    //Get hidden objects
    loadedRenderTexture->hiddenObjects.reserve(renderTextureParams.hiddenObjects.size());
    for (size_t hiddenIndex = 0; hiddenIndex < renderTextureParams.hiddenObjects.size(); hiddenIndex++)
    {
        if (this->sceneManager->hasEntity(renderTextureParams.hiddenObjects[hiddenIndex]))
        {
            Entity* object = GetEntity(renderTextureParams.hiddenObjects[hiddenIndex], false);
            if (object != 0)
                loadedRenderTexture->hiddenObjects.push_back((SceneNode*)object->getParentNode());
        }
    }
    loadedRenderTexture->hiddenObjects.Hide();

    //Get exclusive objects
    loadedRenderTexture->exclusiveObjects.reserve(renderTextureParams.exclusiveObjects.size());
    for (size_t exclusiveIndex = 0; exclusiveIndex < renderTextureParams.exclusiveObjects.size(); exclusiveIndex++)
    {
        if (this->sceneManager->hasEntity(renderTextureParams.exclusiveObjects[exclusiveIndex]))
        {
            Entity* object = GetEntity(renderTextureParams.exclusiveObjects[exclusiveIndex], false);
            if (object != 0)
                loadedRenderTexture->exclusiveObjects.push_back((SceneNode*)object->getParentNode());
        }
    }
    loadedRenderTexture->exclusiveObjects.Hide();
}

void OgreMaxScene::LoadTerrain(const TiXmlElement* objectElement)
{
    //Exit early if skip option is set
    if ((this->loadOptions & SKIP_TERRAIN) != 0)
        return;

    String renderQueue = OgreMaxUtilities::GetStringAttribute(objectElement, "renderQueue");

    String dataFile = OgreMaxUtilities::GetStringAttribute(objectElement, "dataFile");
    if (!dataFile.empty())
    {
        this->sceneManager->setWorldGeometry(dataFile);
        if (!renderQueue.empty())
            this->sceneManager->setWorldGeometryRenderQueue(OgreMaxUtilities::ParseRenderQueue(renderQueue));
    }

    //Parse child elements
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "userDataReference")
            OgreMaxUtilities::LoadUserDataReference(childElement, this->terrainExtraData.userDataReference);
        else if (elementName == "userData")
            OgreMaxUtilities::GetChildText(childElement, this->terrainExtraData.userData);
    }

    if (this->sceneManager->hasSceneNode("Terrain"))
    {
        SceneNode* terrainNode = this->sceneManager->getSceneNode("Terrain");
        terrainNode->setOrientation(OgreMaxUtilities::GetOrientation(this->upAxis));
    }

    //Notify callback
    if (this->callback != 0)
        this->callback->CreatedTerrain(this, dataFile);
}

void OgreMaxScene::LoadNode(const TiXmlElement* objectElement, SceneNode* parentNode)
{
    ObjectExtraData extraData;

    String name = this->nodeNamePrefix;
    name += OgreMaxUtilities::GetStringAttribute(objectElement, "name");

    //Get/load the model
    String modelFileName = OgreMaxUtilities::GetStringAttribute(objectElement, "modelFile");
    if (modelFileName.empty())
        modelFileName = OgreMaxUtilities::GetStringAttribute(objectElement, "modelName");
    OgreMaxModel* model = modelFileName.empty() ? 0 : GetModel(modelFileName);

    extraData.id = OgreMaxUtilities::GetStringAttribute(objectElement, "id");

    //Create the node
    SceneNode* node = name.empty() ? parentNode->createChildSceneNode() : parentNode->createChildSceneNode(name);

    //Notify callback
    if (this->callback != 0)
        this->callback->StartedCreatingNode(this, node);

    //Create the model instance if there is one
    if (model != 0)
    {
        //Determine options
        OgreMaxModel::InstanceOptions instanceOptions = OgreMaxModel::NO_INITIAL_TRANSFORMATION;
        if (this->loadOptions & NO_ANIMATION_STATES)
            instanceOptions |= OgreMaxModel::NO_ANIMATION_STATES;

        //Create the node that will contain the instance
        SceneNode* modelNode = node->createChildSceneNode();

        //Create the instance
        SceneModelInstanceCallback sceneModelInstanceCallback(this, this->callback);
        model->CreateInstance
            (
            this->sceneManager,
            name,
            (this->callback != 0) ? &sceneModelInstanceCallback : 0,
            instanceOptions,
            node,
            this->defaultResourceGroupName,
            modelNode,
            this
            );
    }

    //Iterate over all the node children
    bool isInitialStateSet = false;
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "userDataReference")
            OgreMaxUtilities::LoadUserDataReference(childElement, extraData.userDataReference);
        else if (elementName == "userData")
            OgreMaxUtilities::GetChildText(childElement, extraData.userData);
        else if (elementName == "position")
            node->setPosition(OgreMaxUtilities::LoadXYZ(childElement));
        else if (elementName == "rotation")
            node->setOrientation(OgreMaxUtilities::LoadRotation(childElement));
        else if (elementName == "scale")
            node->setScale(OgreMaxUtilities::LoadXYZ(childElement));
        else if (elementName == "lookTarget")
            LoadLookTarget(childElement, node, 0);
        else if (elementName == "trackTarget")
            LoadTrackTarget(childElement, node, 0);
        else if (elementName == "node" || elementName == "modelInstance")
            LoadNode(childElement, node);
        else if (elementName == "entity")
            LoadEntity(childElement, node);
        else if (elementName == "light")
            LoadLight(childElement, node);
        else if (elementName == "camera")
            LoadCamera(childElement, node);
        else if (elementName == "particleSystem")
            LoadParticleSystem(childElement, node);
        else if (elementName == "billboardSet")
            LoadBillboardSet(childElement, node);
        else if (elementName == "plane")
            LoadPlane(childElement, node);
        else if (elementName == "animations")
        {
            LoadNodeAnimations(childElement, node);

            OgreMaxUtilities::SetIdentityInitialState(node);
            isInitialStateSet = true;
        }
    }

    //Set the initial state if it hasn't already been set
    if (!isInitialStateSet)
        node->setInitialState();

    //Set the node's visibility
    String visibilityText = OgreMaxUtilities::GetStringAttribute(objectElement, "visibility");
    NodeVisibility visibility = OgreMaxUtilities::ParseNodeVisibility(visibilityText);
    OgreMaxUtilities::SetNodeVisibility(node, visibility);

    //Handle node extra data
    if (extraData.HasUserData())
    {
        ObjectExtraDataPtr objectExtraData(new ObjectExtraData(extraData));

        //Set extra data owner node
        objectExtraData->node = node;

        //Process the extra data
        HandleNewObjectExtraData(objectExtraData);
    }

    //Notify callback
    if (this->callback != 0)
        this->callback->FinishedCreatingNode(this, node);

    //Update progress counter
    UpdateLoadProgress(this->loadProgress.nodes, 1);
}

void OgreMaxScene::LoadFog(const TiXmlElement* objectElement)
{
    this->fogParameters.expDensity = OgreMaxUtilities::GetRealAttribute(objectElement, "expDensity", this->fogParameters.expDensity);
    this->fogParameters.linearStart = OgreMaxUtilities::GetRealAttribute(objectElement, "linearStart", this->fogParameters.linearStart);
    this->fogParameters.linearEnd = OgreMaxUtilities::GetRealAttribute(objectElement, "linearEnd", this->fogParameters.linearEnd);

    String fogModeText = OgreMaxUtilities::GetStringAttribute(objectElement, "mode", "none");
    if (!fogModeText.empty())
        this->fogParameters.mode = OgreMaxUtilities::ParseFogMode(fogModeText);

    const TiXmlElement* colorElement = objectElement->FirstChildElement("colourDiffuse");
    if (colorElement != 0)
        this->fogParameters.color = OgreMaxUtilities::LoadColor(colorElement);
}

void OgreMaxScene::LoadSkyBox(const TiXmlElement* objectElement)
{
    //Exit early if skip option is set
    if ((this->loadOptions & SKIP_SKY) != 0)
        return;

    SkyBoxParameters parameters;
    parameters.enabled = OgreMaxUtilities::GetBoolAttribute(objectElement, "enable", true);
    parameters.material = OgreMaxUtilities::GetStringAttribute(objectElement, "material");
    parameters.distance = OgreMaxUtilities::GetRealAttribute(objectElement, "distance", 5000);
    parameters.drawFirst = OgreMaxUtilities::GetBoolAttribute(objectElement, "drawFirst", true);
    parameters.resourceGroupName = this->defaultResourceGroupName;

    //Parse child elements
    String elementName;
    const TiXmlElement* childElement = 0;
    const TiXmlElement* animationsElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "rotation")
            parameters.rotation = OgreMaxUtilities::LoadRotation(childElement);
        else if (elementName == "userDataReference")
            OgreMaxUtilities::LoadUserDataReference(childElement, parameters.extraData.userDataReference);
        else if (elementName == "userData")
            OgreMaxUtilities::GetChildText(childElement, parameters.extraData.userData);
        else if (elementName == "animations")
            animationsElement = childElement;
    }

    //Notify callback
    if (this->callback != 0)
        this->callback->LoadingSkyBox(this, parameters);

    //Hold onto extra data
    this->skyBoxExtraData = parameters.extraData;

    //Create the sky
    this->sceneManager->setSkyBox
        (
        parameters.enabled,
        parameters.material,
        parameters.distance,
        parameters.drawFirst,
        OgreMaxUtilities::GetOrientation(this->upAxis),
        parameters.resourceGroupName
        );
    this->skyNode = this->sceneManager->getSkyBoxNode();
    if (this->skyNode != 0)
    {
        this->skyNode->setOrientation(parameters.rotation);

        //Add animations
        if (animationsElement != 0)
        {
            LoadNodeAnimations(animationsElement, this->skyNode);
            OgreMaxUtilities::SetIdentityInitialState(this->skyNode);
        }
        else
            this->skyNode->setInitialState();
    }
}

void OgreMaxScene::LoadSkyDome(const TiXmlElement* objectElement)
{
    //Exit early if skip option is set
    if ((this->loadOptions & SKIP_SKY) != 0)
        return;

    SkyDomeParameters parameters;
    parameters.enabled = OgreMaxUtilities::GetBoolAttribute(objectElement, "enable", true);
    parameters.material = OgreMaxUtilities::GetStringAttribute(objectElement, "material");
    parameters.curvature = OgreMaxUtilities::GetRealAttribute(objectElement, "curvature", 10);
    parameters.tiling = OgreMaxUtilities::GetRealAttribute(objectElement, "tiling", 8);
    parameters.distance = OgreMaxUtilities::GetRealAttribute(objectElement, "distance", 4000);
    parameters.drawFirst = OgreMaxUtilities::GetBoolAttribute(objectElement, "drawFirst", true);
    parameters.xSegments = OgreMaxUtilities::GetIntAttribute(objectElement, "xSegments", 16);
    parameters.ySegments = OgreMaxUtilities::GetIntAttribute(objectElement, "ySegments", 16);
    parameters.resourceGroupName = this->defaultResourceGroupName;

    //Parse child elements
    String elementName;
    const TiXmlElement* childElement = 0;
    const TiXmlElement* animationsElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "rotation")
            parameters.rotation = OgreMaxUtilities::LoadRotation(childElement);
        else if (elementName == "userDataReference")
            OgreMaxUtilities::LoadUserDataReference(childElement, parameters.extraData.userDataReference);
        else if (elementName == "userData")
            OgreMaxUtilities::GetChildText(childElement, parameters.extraData.userData);
        else if (elementName == "animations")
            animationsElement = childElement;
    }

    //Notify callback
    if (this->callback != 0)
        this->callback->LoadingSkyDome(this, parameters);

    //Hold onto extra data
    this->skyDomeExtraData = parameters.extraData;

    //Create the sky
    this->sceneManager->setSkyDome
        (
        parameters.enabled,
        parameters.material,
        parameters.curvature,
        parameters.tiling,
        parameters.distance,
        parameters.drawFirst,
        OgreMaxUtilities::GetOrientation(this->upAxis),
        parameters.xSegments,
        parameters.ySegments,
        -1,
        parameters.resourceGroupName
        );
    this->skyNode = this->sceneManager->getSkyDomeNode();
    if (this->skyNode != 0)
    {
        this->skyNode->setOrientation(parameters.rotation);

        //Add animations
        if (animationsElement != 0)
        {
            LoadNodeAnimations(animationsElement, this->skyNode);
            OgreMaxUtilities::SetIdentityInitialState(this->skyNode);
        }
        else
            this->skyNode->setInitialState();
    }
}

void OgreMaxScene::LoadSkyPlane(const TiXmlElement* objectElement)
{
    //Exit early if skip option is set
    if ((this->loadOptions & SKIP_SKY) != 0)
        return;

    SkyPlaneParameters parameters;
    parameters.enabled = OgreMaxUtilities::GetBoolAttribute(objectElement, "enable", true);
    parameters.material = OgreMaxUtilities::GetStringAttribute(objectElement, "material");
    parameters.plane = OgreMaxUtilities::GetPlaneAttributes(objectElement, 0, -1, 0, 5000);
    parameters.scale = OgreMaxUtilities::GetRealAttribute(objectElement, "scale", 1000);
    parameters.bow = OgreMaxUtilities::GetRealAttribute(objectElement, "bow", 0);
    parameters.tiling = OgreMaxUtilities::GetRealAttribute(objectElement, "tiling", 10);
    parameters.drawFirst = OgreMaxUtilities::GetBoolAttribute(objectElement, "drawFirst", true);
    parameters.xSegments = OgreMaxUtilities::GetIntAttribute(objectElement, "xSegments", 1);
    parameters.ySegments = OgreMaxUtilities::GetIntAttribute(objectElement, "ySegments", 1);
    parameters.resourceGroupName = this->defaultResourceGroupName;

    //Parse child elements
    String elementName;
    const TiXmlElement* childElement = 0;
    const TiXmlElement* animationsElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "rotation")
            parameters.rotation = OgreMaxUtilities::LoadRotation(childElement);
        else if (elementName == "userDataReference")
            OgreMaxUtilities::LoadUserDataReference(childElement, parameters.extraData.userDataReference);
        else if (elementName == "userData")
            OgreMaxUtilities::GetChildText(childElement, parameters.extraData.userData);
        else if (elementName == "animations")
            animationsElement = childElement;
    }

    //Notify callback
    if (this->callback != 0)
        this->callback->LoadingSkyPlane(this, parameters);

    //Hold onto extra data
    this->skyPlaneExtraData = parameters.extraData;

    //Create the sky
    this->sceneManager->setSkyPlane
        (
        parameters.enabled,
        parameters.plane,
        parameters.material,
        parameters.scale,
        parameters.tiling,
        parameters.drawFirst,
        parameters.bow,
        parameters.xSegments,
        parameters.ySegments,
        parameters.resourceGroupName
        );
    this->skyNode = this->sceneManager->getSkyPlaneNode();
    if (this->skyNode != 0)
    {
        this->skyNode->setOrientation(parameters.rotation);

        //Add animations
        if (animationsElement != 0)
        {
            LoadNodeAnimations(animationsElement, this->skyNode);
            OgreMaxUtilities::SetIdentityInitialState(this->skyNode);
        }
        else
            this->skyNode->setInitialState();
    }
}

void OgreMaxScene::LoadShadows(const TiXmlElement* objectElement)
{
    //Exit early if skip option is set
    if ((this->loadOptions & SKIP_SHADOWS) != 0)
        return;

    ShadowParameters params;
    String techniqueText = OgreMaxUtilities::GetStringAttribute(objectElement, "technique", "none");
    if (!techniqueText.empty())
        params.shadowTechnique = OgreMaxUtilities::ParseShadowTechnique(techniqueText);
    params.selfShadow = OgreMaxUtilities::GetBoolAttribute(objectElement, "selfShadow", params.selfShadow);
    params.farDistance = OgreMaxUtilities::GetRealAttribute(objectElement, "farDistance", params.farDistance);

    //Parse child elements
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "shadowTextures")
        {
            params.textureSize = OgreMaxUtilities::GetIntAttribute(childElement, "size", params.textureSize);
            params.textureCount = OgreMaxUtilities::GetIntAttribute(childElement, "count", params.textureCount);
            params.textureOffset = OgreMaxUtilities::GetRealAttribute(childElement, "offset", params.textureOffset);
            params.textureFadeStart = OgreMaxUtilities::GetRealAttribute(childElement, "fadeStart", params.textureFadeStart);
            params.textureFadeEnd = OgreMaxUtilities::GetRealAttribute(childElement, "fadeEnd", params.textureFadeEnd);
            params.textureShadowCasterMaterial = OgreMaxUtilities::GetStringAttribute(childElement, "shadowCasterMaterial");
            params.textureShadowReceiverMaterial = OgreMaxUtilities::GetStringAttribute(childElement, "shadowReceiverMaterial");
        }
        else if (elementName == "colourShadow")
            params.shadowColor = OgreMaxUtilities::LoadColor(childElement);
        else if (elementName == "shadowCameraSetup")
        {
            params.cameraSetup = OgreMaxUtilities::GetStringAttribute(childElement, "type", "lispsm");

            const Vector3& upVector = GetUpVector();
            params.optimalPlane = OgreMaxUtilities::GetPlaneAttributes(objectElement, upVector.x, upVector.y, upVector.z, 0);
        }
    }

    //Set the shadow parameters
    if (params.shadowTechnique == SHADOWTYPE_NONE)
    {
        //Turn off shadows
        this->sceneManager->setShadowTechnique(SHADOWTYPE_NONE);
    }
    else
    {
        //Turn on shadows

        this->sceneManager->setShadowTechnique(params.shadowTechnique);
        this->sceneManager->setShadowTextureSelfShadow(params.selfShadow);
        this->sceneManager->setShadowColour(params.shadowColor);
        if (params.farDistance > 0)
            this->sceneManager->setShadowFarDistance(params.farDistance);

        //Set shadow texture parameters if necessary
        if (this->sceneManager->isShadowTechniqueTextureBased())
        {
            RenderSystem* renderSystem = Root::getSingleton().getRenderSystem();

            //Determine texture size
            if (!renderSystem->getCapabilities()->hasCapability(RSC_HWRENDER_TO_TEXTURE))
            {
                //Render to texture not supported, so ensure the shadow texture
                //size doesn't exceed the window size

                //Take minimum render window dimension as window size
                int windowSize = 4096;
                if (this->renderWindows->Start())
                {
                    do
                    {
                        RenderWindow* renderWindow = this->renderWindows->GetCurrent();
                        windowSize = (int)std::min(renderWindow->getWidth(), renderWindow->getHeight());
                    }while (this->renderWindows->MoveNext());
                }

                //Use the lesser of the texture and window sizes
                params.textureSize = std::min(params.textureSize, windowSize);
            }

            //If necessary, make sure the texture size is a power of two
            if (!OgreMaxUtilities::IsPowerOfTwo(params.textureSize) &&
                !renderSystem->getCapabilities()->hasCapability(RSC_NON_POWER_OF_2_TEXTURES))
            {
                params.textureSize = OgreMaxUtilities::NextSmallestPowerOfTwo(params.textureSize);
            }

            //Determine texture pixel format
            if (this->callback != 0)
                this->callback->CreatingShadowTextures(this, params);
            if (params.pixelFormat == PF_UNKNOWN)
            {
                //Choose a default format
                if (renderSystem->getName().find("GL") != String::npos)
                {
	                //OpenGL performs better with a half-float format
	                params.pixelFormat = PF_FLOAT16_R;
                }
                else
                {
	                //D3D is the opposite - if you ask for PF_FLOAT16_R you
	                //get an integer format instead. You can ask for PF_FLOAT16_GR
	                //but the precision doesn't work well
                    params.pixelFormat = PF_FLOAT32_R;
                }
            }

            //Set texture size, count, pixel format
            this->sceneManager->setShadowTextureSettings(params.textureSize, params.textureCount, params.pixelFormat);

            //Set other texture settings
            this->sceneManager->setShadowDirLightTextureOffset(params.textureOffset);
            this->sceneManager->setShadowTextureFadeStart(params.textureFadeStart);
            this->sceneManager->setShadowTextureFadeEnd(params.textureFadeEnd);
            this->sceneManager->setShadowTextureCasterMaterial(params.textureShadowCasterMaterial);
            this->sceneManager->setShadowTextureReceiverMaterial(params.textureShadowReceiverMaterial);
        }

        //Set shadow camera setup
        ShadowCameraSetupPtr shadowCameraSetupPtr;
        if (!params.cameraSetup.empty())
        {
            //Use the specified setup
            shadowCameraSetupPtr = ShadowCameraSetupPtr(ParseShadowCameraSetup(params.cameraSetup, params.optimalPlane));
        }
        else
        {
            //Create the appropriate default setup
            shadowCameraSetupPtr = ShadowCameraSetupPtr(new DefaultShadowCameraSetup());
        }
	    this->sceneManager->setShadowCameraSetup(shadowCameraSetupPtr);
    }
}

void OgreMaxScene::LoadExternalItem(const TiXmlElement* objectElement, ExternalItem& item)
{
    item.name = OgreMaxUtilities::GetStringAttribute(objectElement, "name");
    item.type = OgreMaxUtilities::GetStringAttribute(objectElement, "type");

    //Parse child elements
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "file")
            item.file = OgreMaxUtilities::GetStringAttribute(childElement, "name");
        else if (elementName == "position")
            item.position = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "rotation")
            item.rotation = OgreMaxUtilities::LoadRotation(childElement);
        else if (elementName == "scale")
            item.scale = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "boundingVolume")
            OgreMaxUtilities::LoadBoundingVolume(childElement, item.boundingVolume);
        else if (elementName == "userDataReference")
            OgreMaxUtilities::LoadUserDataReference(childElement, item.userDataReference);
        else if (elementName == "userData")
            OgreMaxUtilities::GetChildText(childElement, item.userData);
        else if (elementName == "noteTracks")
            OgreMaxUtilities::LoadNoteTracks(childElement, item.noteTracks);
    }

    //Notify callback
    if (this->callback != 0)
        this->callback->CreatedExternal(this, item);
}

void OgreMaxScene::LoadExternalUserData(const TiXmlElement* objectElement, ExternalUserData& userData)
{
    userData.type = OgreMaxUtilities::GetStringAttribute(objectElement, "type");
    userData.name = OgreMaxUtilities::GetStringAttribute(objectElement, "name");
    userData.id = OgreMaxUtilities::GetStringAttribute(objectElement, "id");
    userData.userDataReference = OgreMaxUtilities::GetStringAttribute(objectElement, "userDataReference");
    OgreMaxUtilities::GetChildText(objectElement, userData.userData);

    //Notify callback
    if (this->callback != 0)
        this->callback->CreatedExternalUserData(this, userData);
}

void OgreMaxScene::LoadEntity(const TiXmlElement* objectElement, const MovableObjectOwner& owner)
{
    ObjectExtraDataPtr objectExtraData(new ObjectExtraData);

    EntityParameters parameters;
    parameters.name = GetNewObjectName(objectElement, owner.node);
    parameters.queryFlags = OgreMaxUtilities::GetUIntAttribute(objectElement, "queryFlags", 0);
    parameters.visibilityFlags = OgreMaxUtilities::GetUIntAttribute(objectElement, "visibilityFlags", 0);
    parameters.visibility = OgreMaxUtilities::GetObjectVisibilityAttribute(objectElement, "visible");
    parameters.meshFile = OgreMaxUtilities::GetStringAttribute(objectElement, "meshFile");
    parameters.materialFile = OgreMaxUtilities::GetStringAttribute(objectElement, "materialFile");
    parameters.castShadows = OgreMaxUtilities::GetBoolAttribute(objectElement, "castShadows", true);

    String renderQueue = OgreMaxUtilities::GetStringAttribute(objectElement, "renderQueue");
    parameters.renderQueue = OgreMaxUtilities::ParseRenderQueue(renderQueue);

    parameters.renderingDistance = OgreMaxUtilities::GetRealAttribute(objectElement, "renderingDistance", 0);

    parameters.resourceGroupName = this->defaultResourceGroupName;

    objectExtraData->id = OgreMaxUtilities::GetStringAttribute(objectElement, "id");
    objectExtraData->receiveShadows = OgreMaxUtilities::GetBoolAttribute(objectElement, "receiveShadows", true);

    //Parse child elements
    const TiXmlElement* boneAttachmentsElement = 0;
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "vertexBuffer")
            OgreMaxUtilities::LoadBufferUsage(childElement, parameters.vertexBufferUsage, parameters.vertexBufferShadowed);
        else if (elementName == "indexBuffer")
            OgreMaxUtilities::LoadBufferUsage(childElement, parameters.indexBufferUsage, parameters.indexBufferShadowed);
        else if (elementName == "userDataReference")
            OgreMaxUtilities::LoadUserDataReference(childElement, objectExtraData->userDataReference);
        else if (elementName == "userData")
            OgreMaxUtilities::GetChildText(childElement, objectExtraData->userData);
        else if (elementName == "noteTracks")
        {
            objectExtraData->noteTracks = SharedPtr<NoteTracks>(new NoteTracks);
            OgreMaxUtilities::LoadNoteTracks(childElement, *objectExtraData->noteTracks.get());
        }
        else if (elementName == "customParameters")
            OgreMaxUtilities::LoadCustomParameters(childElement, parameters.customParameters);
        else if (elementName == "subentities")
            OgreMaxUtilities::LoadSubentities(childElement, parameters.subentities);
        else if (elementName == "boneAttachments")
            boneAttachmentsElement = childElement;
    }

    parameters.extraData = objectExtraData;

    //Notify callback
    if (this->callback != 0)
        this->callback->LoadingEntity(this, parameters);

    //Load the mesh
    bool isNewMesh = !MeshManager::getSingleton().resourceExists(parameters.meshFile);
    MeshPtr mesh = MeshManager::getSingleton().load
        (
        parameters.meshFile,
        parameters.resourceGroupName,
        parameters.vertexBufferUsage, parameters.indexBufferUsage,
        parameters.vertexBufferShadowed, parameters.indexBufferShadowed
        );

    //Notify callback if the mesh was just loaded
    if (isNewMesh && this->callback != 0)
        this->callback->CreatedMesh(this, mesh.getPointer());

    //Create entity
    Entity* entity = this->sceneManager->createEntity(parameters.name, parameters.meshFile);
    if (parameters.queryFlags != 0)
        entity->setQueryFlags(parameters.queryFlags);
    if (parameters.visibilityFlags != 0)
        entity->setVisibilityFlags(parameters.visibilityFlags);
    OgreMaxUtilities::SetObjectVisibility(entity, parameters.visibility);
    entity->setCastShadows(parameters.castShadows);
    entity->setRenderQueueGroup(parameters.renderQueue);
    entity->setRenderingDistance(parameters.renderingDistance);
    OgreMaxUtilities::SetCustomParameters(entity, parameters.customParameters);
    if (!parameters.materialFile.empty())
        entity->setMaterialName(parameters.materialFile);
    objectExtraData->object = entity;

    //Set subentity materials
    size_t subentityCount = std::min(parameters.subentities.size(), (size_t)entity->getNumSubEntities());
    for (size_t subentityIndex = 0; subentityIndex < subentityCount; subentityIndex++)
    {
        SubEntity* subentity = entity->getSubEntity((unsigned int)subentityIndex);
        if (!parameters.subentities[subentityIndex].materialName.empty())
            subentity->setMaterialName(parameters.subentities[subentityIndex].materialName);
    }

    //Attach entity to the owner
    owner.Attach(entity);

    //Load bone attachments
    if (boneAttachmentsElement != 0)
        LoadBoneAttachments(boneAttachmentsElement, entity);

    //Add to loaded objects map
    this->loadedObjects[parameters.name] = entity;

    //Process the extra data
    HandleNewObjectExtraData(objectExtraData);

    //Notify callback
    if (this->callback != 0)
        this->callback->CreatedEntity(this, entity);
}

void OgreMaxScene::LoadLight(const TiXmlElement* objectElement, const MovableObjectOwner& owner)
{
    ObjectExtraDataPtr objectExtraData(new ObjectExtraData);

    String name = GetNewObjectName(objectElement, owner.node);
    objectExtraData->id = OgreMaxUtilities::GetStringAttribute(objectElement, "id");
    uint32 queryFlags = OgreMaxUtilities::GetUIntAttribute(objectElement, "queryFlags", 0);
    uint32 visibilityFlags = OgreMaxUtilities::GetUIntAttribute(objectElement, "visibilityFlags", 0);
    ObjectVisibility visibility = OgreMaxUtilities::GetObjectVisibilityAttribute(objectElement, "visible");
    String type = OgreMaxUtilities::GetStringAttribute(objectElement, "type", "point");
    bool castShadows = OgreMaxUtilities::GetBoolAttribute(objectElement, "castShadows", true);
    float power = OgreMaxUtilities::GetRealAttribute(objectElement, "power", 1);

    //Create the light
    Light* light = this->sceneManager->createLight(name);
    if (queryFlags != 0)
        light->setQueryFlags(queryFlags);
    if (visibilityFlags != 0)
        light->setVisibilityFlags(visibilityFlags);
    OgreMaxUtilities::SetObjectVisibility(light, visibility);
    light->setType(OgreMaxUtilities::ParseLightType(type));
    light->setCastShadows(castShadows);
    light->setPowerScale(power);

    //Parse child elements
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "colourDiffuse")
            light->setDiffuseColour(OgreMaxUtilities::LoadColor(childElement));
        else if (elementName == "colourSpecular")
            light->setSpecularColour(OgreMaxUtilities::LoadColor(childElement));
        else if (elementName == "lightRange")
            LoadLightRange(childElement, light);
        else if (elementName == "lightAttenuation")
            LoadLightAttenuation(childElement, light);
        else if (elementName == "position")
            light->setPosition(OgreMaxUtilities::LoadXYZ(childElement));
        else if (elementName == "normal")
            light->setDirection(OgreMaxUtilities::LoadXYZ(childElement));
        else if (elementName == "userDataReference")
            OgreMaxUtilities::LoadUserDataReference(childElement, objectExtraData->userDataReference);
        else if (elementName == "userData")
            OgreMaxUtilities::GetChildText(childElement, objectExtraData->userData);
        else if (elementName == "noteTracks")
        {
            objectExtraData->noteTracks = SharedPtr<NoteTracks>(new NoteTracks);
            OgreMaxUtilities::LoadNoteTracks(childElement, *objectExtraData->noteTracks.get());
        }
    }

    //Set extra data owner object
    objectExtraData->object = light;

    //Attach light to the node
    owner.Attach(light);

    //Add to loaded objects map
    this->loadedObjects[name] = light;

    //Process the extra data
    HandleNewObjectExtraData(objectExtraData);

    //Notify callback
    if (this->callback != 0)
        this->callback->CreatedLight(this, light);

    //A light was loaded, so there's no need for the default lighting flag to be set
    this->loadOptions = this->loadOptions & ~SET_DEFAULT_LIGHTING;
}

void OgreMaxScene::LoadCamera(const TiXmlElement* objectElement, const MovableObjectOwner& owner)
{
    ObjectExtraDataPtr objectExtraData(new ObjectExtraData);

    String name = GetNewObjectName(objectElement, owner.node);
    objectExtraData->id = OgreMaxUtilities::GetStringAttribute(objectElement, "id");
    uint32 queryFlags = OgreMaxUtilities::GetUIntAttribute(objectElement, "queryFlags", 0);
    uint32 visibilityFlags = OgreMaxUtilities::GetUIntAttribute(objectElement, "visibilityFlags", 0);
    ObjectVisibility visibility = OgreMaxUtilities::GetObjectVisibilityAttribute(objectElement, "visible");
    Real fov = OgreMaxUtilities::GetRealAttribute(objectElement, "fov", Math::PI/2);
    Real aspectRatio = OgreMaxUtilities::GetRealAttribute(objectElement, "aspectRatio", (Real)1.33);
    String projectionType = OgreMaxUtilities::GetStringAttribute(objectElement, "type", "perspective");

    //Create the camera
    Camera* camera = this->sceneManager->createCamera(name);
    if (queryFlags != 0)
        camera->setQueryFlags(queryFlags);
    if (visibilityFlags != 0)
        camera->setVisibilityFlags(visibilityFlags);
    OgreMaxUtilities::SetObjectVisibility(camera, visibility);
    camera->setFOVy(Radian(fov));
    camera->setAspectRatio(aspectRatio);
    camera->setProjectionType(OgreMaxUtilities::ParseProjectionType(projectionType));

    //Parse child elements
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "clipping")
        {
            Real nearClip, farClip;
            OgreMaxUtilities::LoadClipping(childElement, nearClip, farClip);
            camera->setNearClipDistance(nearClip);
            camera->setFarClipDistance(farClip);
        }
        else if (elementName == "position")
            camera->setPosition(OgreMaxUtilities::LoadXYZ(childElement));
        else if (elementName == "rotation")
            camera->setOrientation(OgreMaxUtilities::LoadRotation(childElement));
        else if (elementName == "normal")
            camera->setDirection(OgreMaxUtilities::LoadXYZ(childElement));
        else if (elementName == "lookTarget")
            LoadLookTarget(childElement, 0, camera);
        else if (elementName == "trackTarget")
            LoadTrackTarget(childElement, 0, camera);
        else if (elementName == "userDataReference")
            OgreMaxUtilities::LoadUserDataReference(childElement, objectExtraData->userDataReference);
        else if (elementName == "userData")
            OgreMaxUtilities::GetChildText(childElement, objectExtraData->userData);
        else if (elementName == "noteTracks")
        {
            objectExtraData->noteTracks = SharedPtr<NoteTracks>(new NoteTracks);
            OgreMaxUtilities::LoadNoteTracks(childElement, *objectExtraData->noteTracks.get());
        }
    }

    //Set extra data owner object
    objectExtraData->object = camera;

    //Attach camera to the node
    owner.Attach(camera);

    //Add to loaded objects map
    this->loadedObjects[name] = camera;

    //Process the extra data
    HandleNewObjectExtraData(objectExtraData);

    //Notify callback
    if (this->callback != 0)
        this->callback->CreatedCamera(this, camera);
}

void OgreMaxScene::LoadParticleSystem(const TiXmlElement* objectElement, const MovableObjectOwner& owner)
{
    ObjectExtraDataPtr objectExtraData(new ObjectExtraData);

    String name = GetNewObjectName(objectElement, owner.node);
    objectExtraData->id = OgreMaxUtilities::GetStringAttribute(objectElement, "id");
    uint32 queryFlags = OgreMaxUtilities::GetUIntAttribute(objectElement, "queryFlags", 0);
    uint32 visibilityFlags = OgreMaxUtilities::GetUIntAttribute(objectElement, "visibilityFlags", 0);
    ObjectVisibility visibility = OgreMaxUtilities::GetObjectVisibilityAttribute(objectElement, "visible");
    String renderQueue = OgreMaxUtilities::GetStringAttribute(objectElement, "renderQueue");
    Real renderingDistance = OgreMaxUtilities::GetRealAttribute(objectElement, "renderingDistance", 0);
    String file = OgreMaxUtilities::GetStringAttribute(objectElement, "file");

    //Create the particle system
    ParticleSystem* particleSystem = this->sceneManager->createParticleSystem(name, file);
    if (queryFlags != 0)
        particleSystem->setQueryFlags(queryFlags);
    if (visibilityFlags != 0)
        particleSystem->setVisibilityFlags(visibilityFlags);
    OgreMaxUtilities::SetObjectVisibility(particleSystem, visibility);
    particleSystem->setRenderQueueGroup(OgreMaxUtilities::ParseRenderQueue(renderQueue));
    particleSystem->setRenderingDistance(renderingDistance);

    //Parse child elements
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "userDataReference")
            OgreMaxUtilities::LoadUserDataReference(childElement, objectExtraData->userDataReference);
        else if (elementName == "userData")
            OgreMaxUtilities::GetChildText(childElement, objectExtraData->userData);
        else if (elementName == "noteTracks")
        {
            objectExtraData->noteTracks = SharedPtr<NoteTracks>(new NoteTracks);
            OgreMaxUtilities::LoadNoteTracks(childElement, *objectExtraData->noteTracks.get());
        }
    }

    //Set extra data owner object
    objectExtraData->object = particleSystem;

    //Attach particle system to the node
    owner.Attach(particleSystem);

    //Add to loaded objects map
    this->loadedObjects[name] = particleSystem;

    //Process the extra data
    HandleNewObjectExtraData(objectExtraData);

    //Notify callback
    if (this->callback != 0)
        this->callback->CreatedParticleSystem(this, particleSystem);
}

void OgreMaxScene::LoadBillboardSet(const TiXmlElement* objectElement, const MovableObjectOwner& owner)
{
    ObjectExtraDataPtr objectExtraData(new ObjectExtraData);

    String name = GetNewObjectName(objectElement, owner.node);
    objectExtraData->id = OgreMaxUtilities::GetStringAttribute(objectElement, "id");
    uint32 queryFlags = OgreMaxUtilities::GetUIntAttribute(objectElement, "queryFlags", 0);
    uint32 visibilityFlags = OgreMaxUtilities::GetUIntAttribute(objectElement, "visibilityFlags", 0);
    ObjectVisibility visibility = OgreMaxUtilities::GetObjectVisibilityAttribute(objectElement, "visible");
    String material = OgreMaxUtilities::GetStringAttribute(objectElement, "material");
    Real width = OgreMaxUtilities::GetRealAttribute(objectElement, "width", 10);
    Real height = OgreMaxUtilities::GetRealAttribute(objectElement, "height", 10);
    String type = OgreMaxUtilities::GetStringAttribute(objectElement, "type", "point");
    String origin = OgreMaxUtilities::GetStringAttribute(objectElement, "origin", "center");
    String rotationType = OgreMaxUtilities::GetStringAttribute(objectElement, "rotationType", "vertex");
    uint32 poolSize = OgreMaxUtilities::GetUIntAttribute(objectElement, "poolSize", 0);
    bool autoExtendPool = OgreMaxUtilities::GetBoolAttribute(objectElement, "autoExtendPool", true);
    bool cullIndividual = OgreMaxUtilities::GetBoolAttribute(objectElement, "cullIndividual", false);
    bool sort = OgreMaxUtilities::GetBoolAttribute(objectElement, "sort", false);
    bool accurateFacing = OgreMaxUtilities::GetBoolAttribute(objectElement, "accurateFacing", false);
    String renderQueue = OgreMaxUtilities::GetStringAttribute(objectElement, "renderQueue");
    Real renderingDistance = OgreMaxUtilities::GetRealAttribute(objectElement, "renderingDistance", 0);
    std::vector<CustomParameter> customParameters;

    //Create the particle system
    BillboardSet* billboardSet = this->sceneManager->createBillboardSet(name);
    if (queryFlags != 0)
        billboardSet->setQueryFlags(queryFlags);
    if (visibilityFlags != 0)
        billboardSet->setVisibilityFlags(visibilityFlags);
    OgreMaxUtilities::SetObjectVisibility(billboardSet, visibility);
    billboardSet->setRenderQueueGroup(OgreMaxUtilities::ParseRenderQueue(renderQueue));
    billboardSet->setRenderingDistance(renderingDistance);
    if (!material.empty())
        billboardSet->setMaterialName(material);
    billboardSet->setDefaultWidth(width);
    billboardSet->setDefaultHeight(height);
    billboardSet->setBillboardType(OgreMaxUtilities::ParseBillboardType(type));
    billboardSet->setBillboardOrigin(OgreMaxUtilities::ParseBillboardOrigin(origin));
    billboardSet->setBillboardRotationType(OgreMaxUtilities::ParseBillboardRotationType(rotationType));
    if (poolSize > 0)
        billboardSet->setPoolSize(poolSize);
    billboardSet->setAutoextend(autoExtendPool);
    billboardSet->setCullIndividually(cullIndividual);
    billboardSet->setSortingEnabled(sort);
    billboardSet->setUseAccurateFacing(accurateFacing);

    //Parse child elements
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "billboard")
            LoadBillboard(childElement, billboardSet);
        else if (elementName == "commonDirection")
        {
            Vector3 commonDirection = OgreMaxUtilities::LoadXYZ(childElement);
            billboardSet->setCommonDirection(commonDirection);
        }
        else if (elementName == "commonUpVector")
        {
            Vector3 commonUpVector = OgreMaxUtilities::LoadXYZ(childElement);
            billboardSet->setCommonUpVector(commonUpVector);
        }
        else if (elementName == "userDataReference")
            OgreMaxUtilities::LoadUserDataReference(childElement, objectExtraData->userDataReference);
        else if (elementName == "userData")
            OgreMaxUtilities::GetChildText(childElement, objectExtraData->userData);
        else if (elementName == "noteTracks")
        {
            objectExtraData->noteTracks = SharedPtr<NoteTracks>(new NoteTracks);
            OgreMaxUtilities::LoadNoteTracks(childElement, *objectExtraData->noteTracks.get());
        }
        else if (elementName == "customParameters")
            OgreMaxUtilities::LoadCustomParameters(childElement, customParameters);
    }

    OgreMaxUtilities::SetCustomParameters(billboardSet, customParameters);

    //Set extra data owner object
    objectExtraData->object = billboardSet;

    //Attach billboard set to the node
    owner.Attach(billboardSet);

    //Add to loaded objects map
    this->loadedObjects[name] = billboardSet;

    //Process the extra data
    HandleNewObjectExtraData(objectExtraData);

    //Notify callback
    if (this->callback != 0)
        this->callback->CreatedBillboardSet(this, billboardSet);
}

void OgreMaxScene::LoadPlane(const TiXmlElement* objectElement, const MovableObjectOwner& owner)
{
    ObjectExtraDataPtr objectExtraData(new ObjectExtraData);

    PlaneParameters parameters;
    parameters.name = GetNewObjectName(objectElement, owner.node);
    parameters.queryFlags = OgreMaxUtilities::GetUIntAttribute(objectElement, "queryFlags", 0);
    parameters.visibilityFlags = OgreMaxUtilities::GetUIntAttribute(objectElement, "visibilityFlags", 0);
    parameters.visibility = OgreMaxUtilities::GetObjectVisibilityAttribute(objectElement, "visible");
    parameters.planeName = parameters.name;
    parameters.distance = OgreMaxUtilities::GetRealAttribute(objectElement, "distance", 0);
    parameters.width = OgreMaxUtilities::GetRealAttribute(objectElement, "width", 10);
    parameters.height = OgreMaxUtilities::GetRealAttribute(objectElement, "height", 10);
    parameters.xSegments = OgreMaxUtilities::GetIntAttribute(objectElement, "xSegments", 1);
    parameters.ySegments = OgreMaxUtilities::GetIntAttribute(objectElement, "ySegments", 1);
    parameters.numTexCoordSets = OgreMaxUtilities::GetIntAttribute(objectElement, "numTexCoordSets", 1);
    parameters.uTile = OgreMaxUtilities::GetRealAttribute(objectElement, "uTile", 1);
    parameters.vTile = OgreMaxUtilities::GetRealAttribute(objectElement, "vTile", 1);
    parameters.material = OgreMaxUtilities::GetStringAttribute(objectElement, "material");
    parameters.normals = OgreMaxUtilities::GetBoolAttribute(objectElement, "normals", true);
    parameters.createMovablePlane = OgreMaxUtilities::GetBoolAttribute(objectElement, "movablePlane", true);
    parameters.castShadows = OgreMaxUtilities::GetBoolAttribute(objectElement, "castShadows", true);
    parameters.resourceGroupName = this->defaultResourceGroupName;

    String renderQueue = OgreMaxUtilities::GetStringAttribute(objectElement, "renderQueue");
    parameters.renderQueue = OgreMaxUtilities::ParseRenderQueue(renderQueue);

    parameters.renderingDistance = OgreMaxUtilities::GetRealAttribute(objectElement, "renderingDistance", 0);

    objectExtraData->id = OgreMaxUtilities::GetStringAttribute(objectElement, "id");
    objectExtraData->receiveShadows = OgreMaxUtilities::GetBoolAttribute(objectElement, "receiveShadows", true);

    //Parse child elements
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "normal")
            parameters.normal = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "upVector")
            parameters.upVector = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "vertexBuffer")
            OgreMaxUtilities::LoadBufferUsage(childElement, parameters.vertexBufferUsage, parameters.vertexBufferShadowed);
        else if (elementName == "indexBuffer")
            OgreMaxUtilities::LoadBufferUsage(childElement, parameters.indexBufferUsage, parameters.indexBufferShadowed);
        else if (elementName == "userDataReference")
            OgreMaxUtilities::LoadUserDataReference(childElement, objectExtraData->userDataReference);
        else if (elementName == "userData")
            OgreMaxUtilities::GetChildText(childElement, objectExtraData->userData);
        else if (elementName == "noteTracks")
        {
            objectExtraData->noteTracks = SharedPtr<NoteTracks>(new NoteTracks);
            OgreMaxUtilities::LoadNoteTracks(childElement, *objectExtraData->noteTracks.get());
        }
        else if (elementName == "customParameters")
            OgreMaxUtilities::LoadCustomParameters(childElement, parameters.customParameters);
    }

    parameters.extraData = objectExtraData;

    //Notify callback
    if (this->callback != 0)
        this->callback->LoadingPlane(this, parameters);

    //Create movable plane if the name hasn't already been used
    MovablePlane* movablePlane = 0;
    if (parameters.createMovablePlane)
    {
        String movablePlaneName;
        OgreMaxUtilities::CreateMovablePlaneName(movablePlaneName, parameters.planeName);
        if (this->movablePlanes.find(movablePlaneName) == this->movablePlanes.end())
        {
            movablePlane = new MovablePlane(movablePlaneName);
            this->movablePlanes[movablePlaneName] = movablePlane;
            movablePlane->normal = parameters.normal;
            movablePlane->d = parameters.distance;
        }
    }

    //Create plane mesh
    Plane plane(parameters.normal, parameters.distance);
	MeshManager::getSingleton().createPlane
        (
        parameters.planeName,
        parameters.resourceGroupName,
        plane,
        parameters.width, parameters.height,
        parameters.xSegments, parameters.ySegments,
        parameters.normals, parameters.numTexCoordSets,
        parameters.uTile, parameters.vTile,
        parameters.upVector,
        parameters.vertexBufferUsage, parameters.indexBufferUsage,
        parameters.vertexBufferShadowed, parameters.indexBufferShadowed
        );

    //Create plane entity
    Entity* entity = this->sceneManager->createEntity(parameters.name, parameters.planeName);
    if (parameters.queryFlags != 0)
        entity->setQueryFlags(parameters.queryFlags);
    if (parameters.visibilityFlags != 0)
        entity->setVisibilityFlags(parameters.visibilityFlags);
    OgreMaxUtilities::SetObjectVisibility(entity, parameters.visibility);
    entity->setCastShadows(parameters.castShadows);
    entity->setRenderQueueGroup(parameters.renderQueue);
    entity->setRenderingDistance(parameters.renderingDistance);
    OgreMaxUtilities::SetCustomParameters(entity, parameters.customParameters);
    if (!parameters.material.empty())
        entity->setMaterialName(parameters.material);
    objectExtraData->object = entity;

    //Attach plane entity and movable object to the node
    owner.Attach(entity);
    if (movablePlane != 0)
        owner.Attach(movablePlane);

    //Add to loaded objects map
    this->loadedObjects[parameters.name] = entity;

    //Process the extra data
    HandleNewObjectExtraData(objectExtraData);

    //Notify callback
    if (this->callback != 0)
        this->callback->CreatedPlane(this, plane, entity);
}

void OgreMaxScene::LoadBoneAttachments(const TiXmlElement* objectElement, Entity* entity)
{
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
        LoadBoneAttachment(childElement, entity);
}

void OgreMaxScene::LoadBoneAttachment(const TiXmlElement* objectElement, Entity* entity)
{
    bool attachedSomething = false;
    MovableObjectOwner owner(entity);
    String name = OgreMaxUtilities::GetStringAttribute(objectElement, "name");
    owner.boneName = OgreMaxUtilities::GetStringAttribute(objectElement, "bone");

    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();
        if (elementName == "position")
            owner.attachPosition = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "rotation")
            owner.attachRotation = OgreMaxUtilities::LoadRotation(childElement);
        else if (elementName == "scale")
            owner.attachScale = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "entity")
        {
            LoadEntity(childElement, owner);
            attachedSomething = true;
        }
        else if (elementName == "light")
        {
            LoadLight(childElement, owner);
            attachedSomething = true;
        }
        else if (elementName == "camera")
        {
            LoadCamera(childElement, owner);
            attachedSomething = true;
        }
        else if (elementName == "particleSystem")
        {
            LoadParticleSystem(childElement, owner);
            attachedSomething = true;
        }
        else if (elementName == "billboardSet")
        {
            LoadBillboardSet(childElement, owner);
            attachedSomething = true;
        }
        else if (elementName == "plane")
        {
            LoadPlane(childElement, owner);
            attachedSomething = true;
        }
    }

    if (!attachedSomething)
        owner.AttachEmpty(name);
}

void OgreMaxScene::LoadLookTarget(const TiXmlElement* objectElement, SceneNode* node, Camera* camera)
{
    LookTarget lookTarget(node, camera);

    lookTarget.nodeName = OgreMaxUtilities::GetStringAttribute(objectElement, "nodeName");

    String relativeTo = OgreMaxUtilities::GetStringAttribute(objectElement, "relativeTo");
    if (!relativeTo.empty())
        lookTarget.relativeTo = OgreMaxUtilities::ParseTransformSpace(relativeTo);

    //Parse child elements
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "position")
        {
            lookTarget.position = OgreMaxUtilities::LoadXYZ(childElement);
            lookTarget.isPositionSet = true;
        }
        else if (elementName == "localDirection")
            lookTarget.localDirection = OgreMaxUtilities::LoadXYZ(childElement);
    }

    //Store look target information for later
    this->lookTargets.push_back(lookTarget);
}

void OgreMaxScene::LoadTrackTarget(const TiXmlElement* objectElement, SceneNode* node, Camera* camera)
{
    TrackTarget trackTarget(node, camera);

    trackTarget.nodeName = OgreMaxUtilities::GetStringAttribute(objectElement, "nodeName");

    //Parse child elements
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "offset")
            trackTarget.offset = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "localDirection")
            trackTarget.localDirection = OgreMaxUtilities::LoadXYZ(childElement);
    }

    //Store track target information for later
    this->trackTargets.push_back(trackTarget);
}

void OgreMaxScene::LoadBillboard(const TiXmlElement* objectElement, BillboardSet* billboardSet)
{
    Real width = OgreMaxUtilities::GetRealAttribute(objectElement, "width", 0);
    Real height = OgreMaxUtilities::GetRealAttribute(objectElement, "height", 0);
    Radian rotationAngle = Radian(OgreMaxUtilities::GetRealAttribute(objectElement, "rotation", 0));

    Vector3 position = Vector3::ZERO;
    ColourValue color = ColourValue::White;
    FloatRect texCoordRectangle(0, 0, 0, 0);

    //Parse child elements
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "position")
            position = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "rotation")
        {
            Quaternion rotation = OgreMaxUtilities::LoadRotation(childElement);
            Vector3 rotationAxis;
            rotation.ToAngleAxis(rotationAngle, rotationAxis);
        }
        else if (elementName == "colourDiffuse")
            color = OgreMaxUtilities::LoadColor(childElement);
        else if (elementName == "texCoordRectangle")
            texCoordRectangle = OgreMaxUtilities::LoadFloatRectangle(childElement);
    }

    //Create the billboard
    Billboard* billboard = billboardSet->createBillboard(position, color);

    //Set rotation angle
    if (rotationAngle.valueRadians() != 0)
        billboard->setRotation(rotationAngle);

    //Set dimensions
    if (width != 0 && height != 0)
        billboard->setDimensions(width, height);

    //Set texture coordinate rectangle
    if (texCoordRectangle.width() != 0 && texCoordRectangle.height() != 0)
        billboard->setTexcoordRect(texCoordRectangle);
}

void OgreMaxScene::LoadLightRange(const TiXmlElement* objectElement, Light* light)
{
    if (light->getType() == Light::LT_SPOTLIGHT)
    {
        String value;

        value = OgreMaxUtilities::GetStringAttribute(objectElement, "inner");
        if (!value.empty())
            light->setSpotlightInnerAngle(Radian(StringConverter::parseReal(value)));

        value = OgreMaxUtilities::GetStringAttribute(objectElement, "outer");
        if (!value.empty())
            light->setSpotlightOuterAngle(Radian(StringConverter::parseReal(value)));

        value = OgreMaxUtilities::GetStringAttribute(objectElement, "falloff");
        if (!value.empty())
            light->setSpotlightFalloff(StringConverter::parseReal(value));
    }
}

void OgreMaxScene::LoadLightAttenuation(const TiXmlElement* objectElement, Light* light)
{
    String value;

    value = OgreMaxUtilities::GetStringAttribute(objectElement, "range");
    Real range = value.empty() ? light->getAttenuationRange() : StringConverter::parseReal(value);

    value = OgreMaxUtilities::GetStringAttribute(objectElement, "constant");
    Real constant = value.empty() ? light->getAttenuationConstant() : StringConverter::parseReal(value);

    value = OgreMaxUtilities::GetStringAttribute(objectElement, "linear");
    Real linear = value.empty() ? light->getAttenuationLinear() : StringConverter::parseReal(value);

    value = OgreMaxUtilities::GetStringAttribute(objectElement, "quadric");
    Real quadric = value.empty() ? light->getAttenuationQuadric() : StringConverter::parseReal(value);

    light->setAttenuation(range, constant, linear, quadric);
}

void OgreMaxScene::LoadNodeAnimations(const TiXmlElement* objectElement, SceneNode* node)
{
    //Parse child elements
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "animation")
            LoadNodeAnimation(childElement, node);
    }
}

void OgreMaxScene::LoadNodeAnimation(const TiXmlElement* objectElement, SceneNode* node)
{
    Types::NodeAnimationParameters params;

    //Get enabled and looping states
    params.enable = OgreMaxUtilities::GetBoolAttribute(objectElement, "enable", params.enable);
    params.looping = OgreMaxUtilities::GetBoolAttribute(objectElement, "loop", params.looping);

    //Animation name
    params.name = this->nodeAnimationNamePrefix;
    params.name += OgreMaxUtilities::GetStringAttribute(objectElement, "name");

    //Get existing animation or create new one
    Animation* animation;
    if (this->sceneManager->hasAnimation(params.name))
        animation = this->sceneManager->getAnimation(params.name);
    else
    {
        //Length
        params.length = OgreMaxUtilities::GetRealAttribute(objectElement, "length", 0);

        //Interpolation mode
        String interpolationModeText = OgreMaxUtilities::GetStringAttribute(objectElement, "interpolationMode");
        if (!interpolationModeText.empty())
            params.interpolationMode = OgreMaxUtilities::ParseAnimationInterpolationMode(interpolationModeText);

        //Rotation interpolation mode
        String rotationInterpolationModeText = OgreMaxUtilities::GetStringAttribute(objectElement, "rotationInterpolationMode");
        if (!rotationInterpolationModeText.empty())
            params.rotationInterpolationMode = OgreMaxUtilities::ParseAnimationRotationInterpolationMode(rotationInterpolationModeText);

        //Notify the callback
        if (this->callback != 0)
            this->callback->LoadingNodeAnimation(this, params);

        //Create animation
        animation = this->sceneManager->createAnimation(params.name, params.length);
        animation->setInterpolationMode(params.interpolationMode);
        animation->setRotationInterpolationMode(params.rotationInterpolationMode);

        //Notify the callback
        if (this->callback != 0)
            this->callback->CreatedNodeAnimation(this, node, animation);
    }

    //Create animation track for node
    NodeAnimationTrack* animationTrack = animation->createNodeTrack(animation->getNumNodeTracks() + 1, node);

    //Load animation keyframes
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "keyframe")
            LoadNodeAnimationKeyFrame(childElement, animationTrack);
    }

    //Notify callback
    if (this->callback != 0)
        this->callback->CreatedNodeAnimationTrack(this, node, animationTrack, params.enable, params.looping);

    if ((this->loadOptions & NO_ANIMATION_STATES) == 0)
    {
        //Create a new animation state to track the animation
        if (GetAnimationState(params.name) == 0)
        {
            //No animation state has been created for the animation yet
            AnimationState* animationState = this->sceneManager->createAnimationState(params.name);
            this->animationStates[params.name] = animationState;
            animationState->setEnabled(params.enable);
            animationState->setLoop(params.looping);

            //Notify callback
            if (this->callback != 0)
                this->callback->CreatedNodeAnimationState(this, node, animationState);
        }
    }
}

void OgreMaxScene::LoadNodeAnimationKeyFrame(const TiXmlElement* objectElement, NodeAnimationTrack* animationTrack)
{
    //Key time
    Real keyTime = OgreMaxUtilities::GetRealAttribute(objectElement, "time", 0);

    //Create the key frame
    TransformKeyFrame* keyFrame = animationTrack->createNodeKeyFrame(keyTime);

    //Parse child elements
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "translation")
            keyFrame->setTranslate(OgreMaxUtilities::LoadXYZ(childElement));
        else if (elementName == "rotation")
            keyFrame->setRotation(OgreMaxUtilities::LoadRotation(childElement));
        else if (elementName == "scale")
            keyFrame->setScale(OgreMaxUtilities::LoadXYZ(childElement));
    }
}

void OgreMaxScene::LoadObjectNames(const TiXmlElement* objectElement, const String& elementName, std::vector<String>& names)
{
    size_t objectCount = OgreMaxUtilities::GetElementCount(objectElement, elementName);
    names.resize(objectCount);

    size_t index = 0;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
        names[index++] = childElement->Attribute("name");
}

void OgreMaxScene::LoadRenderTextureMaterials
    (
    const TiXmlElement* objectElement,
    std::vector<RenderTextureParameters::Material>& materials
    )
{
    size_t index = 0;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        RenderTextureParameters::Material& material = materials[index++];
        material.name = childElement->Attribute("name");
        material.techniqueIndex = OgreMaxUtilities::GetIntAttribute(childElement, "technique", 0);
        material.passIndex = OgreMaxUtilities::GetIntAttribute(childElement, "pass", 0);
        material.textureUnitIndex = OgreMaxUtilities::GetIntAttribute(childElement, "textureUnit", 0);
    }
}

ShadowCameraSetup* OgreMaxScene::ParseShadowCameraSetup(const String& type, Plane optimalPlane)
{
    String typeLower = type;
    StringUtil::toLowerCase(typeLower);

    if (typeLower == "uniform")
        return new DefaultShadowCameraSetup;
    else if (typeLower == "uniformfocused")
        return new FocusedShadowCameraSetup;
    else if (typeLower == "lispsm")
        return new LiSPSMShadowCameraSetup;
    else if (typeLower == "planeoptimal")
    {
        //The plane optimal setup requires a plane.
        //Create one if it hasn't already been created
        if (this->shadowOptimalPlane == 0)
            this->shadowOptimalPlane = new MovablePlane("_shadowOptimalPlane");

        //Update plane
        *(Plane*)this->shadowOptimalPlane = optimalPlane;

        return new PlaneOptimalShadowCameraSetup(this->shadowOptimalPlane);
    }

    StringUtil::StrStreamType errorMessage;
    errorMessage << "Invalid shadow camera setup specified: " << type;

    OGRE_EXCEPT
        (
        Exception::ERR_INVALIDPARAMS,
	    errorMessage.str(),
	    "OgreMaxScene::ParseShadowCameraSetup"
        );
}
