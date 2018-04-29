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
#include "OgreMaxModel.hpp"
#include "OgreMaxUtilities.hpp"
#include "OgreMaxScene.hpp"
#include <OgreSubEntity.h>
#include <OgreBillboard.h>

using namespace Ogre;
using namespace OgreMax;
using namespace OgreMax::Types;


//Implementation---------------------------------------------------------------
OgreMaxModel::OgreMaxModel()
{
}

OgreMaxModel::~OgreMaxModel()
{
}

void OgreMaxModel::Load(const String& fileName, const String& resourceGroupName)
{
    //Load from the XML document
    TiXmlDocument document;
    OgreMaxUtilities::LoadXmlDocument(fileName, document, resourceGroupName);
    LoadNode(document.RootElement(), this->rootNode);
}

void OgreMaxModel::LoadNode(const TiXmlElement* objectElement, NodeParameters& node)
{
    ObjectExtraData extraData;
    node.name = OgreMaxUtilities::GetStringAttribute(objectElement, "name");
    node.modelFile = OgreMaxUtilities::GetStringAttribute(objectElement, "modelFile");
    if (node.modelFile.empty())
        node.modelFile = OgreMaxUtilities::GetStringAttribute(objectElement, "modelName");
    node.visibility = OgreMaxUtilities::GetNodeVisibilityAttribute(objectElement, "visibility");
    node.childNodes.resize(OgreMaxUtilities::GetChildElementCount(objectElement, "node"));
    extraData.id = OgreMaxUtilities::GetStringAttribute(objectElement, "id");

    //Iterate over all the node children
    size_t childNodeIndex = 0;
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
            node.position = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "rotation")
            node.orientation = OgreMaxUtilities::LoadRotation(childElement);
        else if (elementName == "scale")
            node.scale = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "node")
            LoadNode(childElement, node.childNodes[childNodeIndex++]);
        else if (elementName == "entity")
            node.objects.push_back(LoadEntity(childElement));
        else if (elementName == "light")
            node.objects.push_back(LoadLight(childElement));
        else if (elementName == "camera")
            node.objects.push_back(LoadCamera(childElement));
        else if (elementName == "particleSystem")
            node.objects.push_back(LoadParticleSystem(childElement));
        else if (elementName == "billboardSet")
            node.objects.push_back(LoadBillboardSet(childElement));
        else if (elementName == "plane")
            node.objects.push_back(LoadPlane(childElement));
        else if (elementName == "animations")
            LoadNodeAnimations(childElement, node);
    }

    //Handle node extra data
    if (extraData.HasUserData())
    {
        ObjectExtraDataPtr objectExtraData(new ObjectExtraData(extraData));
        node.extraData = objectExtraData;
    }
}

EntityParameters* OgreMaxModel::LoadEntity(const TiXmlElement* objectElement)
{
    EntityParameters* parameters = new EntityParameters;
    parameters->name = OgreMaxUtilities::GetStringAttribute(objectElement, "name");
    parameters->queryFlags = OgreMaxUtilities::GetUIntAttribute(objectElement, "queryFlags", 0);
    parameters->visibilityFlags = OgreMaxUtilities::GetUIntAttribute(objectElement, "visibilityFlags", 0);
    parameters->visibility = OgreMaxUtilities::GetObjectVisibilityAttribute(objectElement, "visible");
    parameters->meshFile = OgreMaxUtilities::GetStringAttribute(objectElement, "meshFile");
    parameters->materialFile = OgreMaxUtilities::GetStringAttribute(objectElement, "materialFile");
    parameters->castShadows = OgreMaxUtilities::GetBoolAttribute(objectElement, "castShadows", true);

    String renderQueue = OgreMaxUtilities::GetStringAttribute(objectElement, "renderQueue");
    parameters->renderQueue = OgreMaxUtilities::ParseRenderQueue(renderQueue);

    parameters->renderingDistance = OgreMaxUtilities::GetRealAttribute(objectElement, "renderingDistance", 0);

    parameters->extraData = ObjectExtraDataPtr(new ObjectExtraData);
    parameters->extraData->id = OgreMaxUtilities::GetStringAttribute(objectElement, "id");
    parameters->extraData->receiveShadows = OgreMaxUtilities::GetBoolAttribute(objectElement, "receiveShadows", parameters->extraData->receiveShadows);

    //Parse child elements
    const TiXmlElement* boneAttachmentsElement = 0;
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "vertexBuffer")
            OgreMaxUtilities::LoadBufferUsage(childElement, parameters->vertexBufferUsage, parameters->vertexBufferShadowed);
        else if (elementName == "indexBuffer")
            OgreMaxUtilities::LoadBufferUsage(childElement, parameters->indexBufferUsage, parameters->indexBufferShadowed);
        else if (elementName == "userDataReference")
            OgreMaxUtilities::LoadUserDataReference(childElement, parameters->extraData->userDataReference);
        else if (elementName == "userData")
            OgreMaxUtilities::GetChildText(childElement, parameters->extraData->userData);
        else if (elementName == "noteTracks")
        {
            parameters->extraData->noteTracks = NoteTracksPtr(new NoteTracks);
            OgreMaxUtilities::LoadNoteTracks(childElement, *parameters->extraData->noteTracks.get());
        }
        else if (elementName == "customParameters")
            OgreMaxUtilities::LoadCustomParameters(childElement, parameters->customParameters);
        else if (elementName == "subentities")
            OgreMaxUtilities::LoadSubentities(childElement, parameters->subentities);
        else if (elementName == "boneAttachments")
            boneAttachmentsElement = childElement;
    }

    //Load bone attachments
    if (boneAttachmentsElement != 0)
        LoadBoneAttachments(boneAttachmentsElement, *parameters);

    return parameters;
}

void OgreMaxModel::LoadBoneAttachments(const TiXmlElement* objectElement, EntityParameters& entity)
{
    entity.boneAttachments.resize(OgreMaxUtilities::GetChildElementCount(objectElement, "boneAttachment"));

    size_t index = 0;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
        LoadBoneAttachment(childElement, entity.boneAttachments[index++]);
}

void OgreMaxModel::LoadBoneAttachment(const TiXmlElement* objectElement, EntityParameters::BoneAttachment& boneAttachment)
{
    boneAttachment.name = OgreMaxUtilities::GetStringAttribute(objectElement, "name");
    boneAttachment.boneName = OgreMaxUtilities::GetStringAttribute(objectElement, "bone");

    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();
        if (elementName == "position")
            boneAttachment.attachPosition = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "rotation")
            boneAttachment.attachRotation = OgreMaxUtilities::LoadRotation(childElement);
        else if (elementName == "scale")
            boneAttachment.attachScale = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "entity")
            boneAttachment.object = LoadEntity(childElement);
        else if (elementName == "light")
            boneAttachment.object = LoadLight(childElement);
        else if (elementName == "camera")
            boneAttachment.object = LoadCamera(childElement);
        else if (elementName == "particleSystem")
            boneAttachment.object = LoadParticleSystem(childElement);
        else if (elementName == "billboardSet")
            boneAttachment.object = LoadBillboardSet(childElement);
        else if (elementName == "plane")
            boneAttachment.object = LoadPlane(childElement);
    }
}

LightParameters* OgreMaxModel::LoadLight(const TiXmlElement* objectElement)
{
    LightParameters* parameters = new LightParameters;
    parameters->name = OgreMaxUtilities::GetStringAttribute(objectElement, "name");
    parameters->queryFlags = OgreMaxUtilities::GetUIntAttribute(objectElement, "queryFlags", 0);
    parameters->visibilityFlags = OgreMaxUtilities::GetUIntAttribute(objectElement, "visibilityFlags", 0);
    parameters->visibility = OgreMaxUtilities::GetObjectVisibilityAttribute(objectElement, "visible");

    String type = OgreMaxUtilities::GetStringAttribute(objectElement, "type", "point");
    parameters->lightType = OgreMaxUtilities::ParseLightType(type);
    parameters->castShadows = OgreMaxUtilities::GetBoolAttribute(objectElement, "castShadows", parameters->castShadows);
    parameters->power = OgreMaxUtilities::GetRealAttribute(objectElement, "power", parameters->power);

    parameters->extraData = ObjectExtraDataPtr(new ObjectExtraData);
    parameters->extraData->id = OgreMaxUtilities::GetStringAttribute(objectElement, "id");

    //Parse child elements
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "colourDiffuse")
            parameters->diffuseColor = OgreMaxUtilities::LoadColor(childElement);
        else if (elementName == "colourSpecular")
            parameters->specularColor = OgreMaxUtilities::LoadColor(childElement);
        else if (elementName == "lightRange")
            LoadLightRange(childElement, *parameters);
        else if (elementName == "lightAttenuation")
            LoadLightAttenuation(childElement, *parameters);
        else if (elementName == "position")
            parameters->position = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "normal")
            parameters->direction = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "userDataReference")
            OgreMaxUtilities::LoadUserDataReference(childElement, parameters->extraData->userDataReference);
        else if (elementName == "userData")
            OgreMaxUtilities::GetChildText(childElement, parameters->extraData->userData);
        else if (elementName == "noteTracks")
        {
            parameters->extraData->noteTracks = NoteTracksPtr(new NoteTracks);
            OgreMaxUtilities::LoadNoteTracks(childElement, *parameters->extraData->noteTracks.get());
        }
    }

    return parameters;
}

CameraParameters* OgreMaxModel::LoadCamera(const TiXmlElement* objectElement)
{
    CameraParameters* parameters = new CameraParameters;
    parameters->name = OgreMaxUtilities::GetStringAttribute(objectElement, "name");
    parameters->queryFlags = OgreMaxUtilities::GetUIntAttribute(objectElement, "queryFlags", 0);
    parameters->visibilityFlags = OgreMaxUtilities::GetUIntAttribute(objectElement, "visibilityFlags", 0);
    parameters->visibility = OgreMaxUtilities::GetObjectVisibilityAttribute(objectElement, "visible");

    parameters->fov = Radian(OgreMaxUtilities::GetRealAttribute(objectElement, "fov", Math::PI/2));
    parameters->aspectRatio = OgreMaxUtilities::GetRealAttribute(objectElement, "aspectRatio", (Real)1.33);

    String projectionType = OgreMaxUtilities::GetStringAttribute(objectElement, "type", "perspective");
    parameters->projectionType = OgreMaxUtilities::ParseProjectionType(projectionType);

    parameters->extraData = ObjectExtraDataPtr(new ObjectExtraData);
    parameters->extraData->id = OgreMaxUtilities::GetStringAttribute(objectElement, "id");

    //Parse child elements
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "clipping")
            OgreMaxUtilities::LoadClipping(childElement, parameters->nearClip, parameters->farClip);
        else if (elementName == "position")
            parameters->position = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "rotation")
            parameters->orientation = OgreMaxUtilities::LoadRotation(childElement);
        else if (elementName == "normal")
            parameters->direction = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "userDataReference")
            OgreMaxUtilities::LoadUserDataReference(childElement, parameters->extraData->userDataReference);
        else if (elementName == "userData")
            OgreMaxUtilities::GetChildText(childElement, parameters->extraData->userData);
        else if (elementName == "noteTracks")
        {
            parameters->extraData->noteTracks = NoteTracksPtr(new NoteTracks);
            OgreMaxUtilities::LoadNoteTracks(childElement, *parameters->extraData->noteTracks.get());
        }
    }

    return parameters;
}

ParticleSystemParameters* OgreMaxModel::LoadParticleSystem(const TiXmlElement* objectElement)
{
    ParticleSystemParameters* parameters = new ParticleSystemParameters;
    parameters->name = OgreMaxUtilities::GetStringAttribute(objectElement, "name");
    parameters->queryFlags = OgreMaxUtilities::GetUIntAttribute(objectElement, "queryFlags", 0);
    parameters->visibilityFlags = OgreMaxUtilities::GetUIntAttribute(objectElement, "visibilityFlags", 0);
    parameters->visibility = OgreMaxUtilities::GetObjectVisibilityAttribute(objectElement, "visible");
    parameters->file = OgreMaxUtilities::GetStringAttribute(objectElement, "file");

    String renderQueue = OgreMaxUtilities::GetStringAttribute(objectElement, "renderQueue");
    parameters->renderQueue = OgreMaxUtilities::ParseRenderQueue(renderQueue);

    parameters->renderingDistance = OgreMaxUtilities::GetRealAttribute(objectElement, "renderingDistance", 0);

    parameters->extraData = ObjectExtraDataPtr(new ObjectExtraData);
    parameters->extraData->id = OgreMaxUtilities::GetStringAttribute(objectElement, "id");
    parameters->extraData->receiveShadows = OgreMaxUtilities::GetBoolAttribute(objectElement, "receiveShadows", parameters->extraData->receiveShadows);

    //Parse child elements
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "userDataReference")
            OgreMaxUtilities::LoadUserDataReference(childElement, parameters->extraData->userDataReference);
        else if (elementName == "userData")
            OgreMaxUtilities::GetChildText(childElement, parameters->extraData->userData);
        else if (elementName == "customParameters")
            OgreMaxUtilities::LoadCustomParameters(childElement, parameters->customParameters);
        else if (elementName == "noteTracks")
        {
            parameters->extraData->noteTracks = NoteTracksPtr(new NoteTracks);
            OgreMaxUtilities::LoadNoteTracks(childElement, *parameters->extraData->noteTracks.get());
        }
    }

    return parameters;
}

BillboardSetParameters* OgreMaxModel::LoadBillboardSet(const TiXmlElement* objectElement)
{
    BillboardSetParameters* parameters = new BillboardSetParameters;
    parameters->name = OgreMaxUtilities::GetStringAttribute(objectElement, "name");
    parameters->queryFlags = OgreMaxUtilities::GetUIntAttribute(objectElement, "queryFlags", 0);
    parameters->visibilityFlags = OgreMaxUtilities::GetUIntAttribute(objectElement, "visibilityFlags", 0);
    parameters->visibility = OgreMaxUtilities::GetObjectVisibilityAttribute(objectElement, "visible");

    parameters->material = OgreMaxUtilities::GetStringAttribute(objectElement, "material");
    parameters->width = OgreMaxUtilities::GetRealAttribute(objectElement, "width", 10);
    parameters->height = OgreMaxUtilities::GetRealAttribute(objectElement, "height", 10);

    String type = OgreMaxUtilities::GetStringAttribute(objectElement, "type", "point");
    parameters->billboardType = OgreMaxUtilities::ParseBillboardType(type);

    String origin = OgreMaxUtilities::GetStringAttribute(objectElement, "origin", "center");
    parameters->origin = OgreMaxUtilities::ParseBillboardOrigin(origin);

    String rotationType = OgreMaxUtilities::GetStringAttribute(objectElement, "rotationType", "vertex");
    parameters->rotationType = OgreMaxUtilities::ParseBillboardRotationType(rotationType);

    parameters->poolSize = OgreMaxUtilities::GetUIntAttribute(objectElement, "poolSize", parameters->poolSize);
    parameters->autoExtendPool = OgreMaxUtilities::GetBoolAttribute(objectElement, "autoExtendPool", parameters->autoExtendPool);
    parameters->cullIndividual = OgreMaxUtilities::GetBoolAttribute(objectElement, "cullIndividual", parameters->cullIndividual);
    parameters->sort = OgreMaxUtilities::GetBoolAttribute(objectElement, "sort", parameters->sort);
    parameters->accurateFacing = OgreMaxUtilities::GetBoolAttribute(objectElement, "accurateFacing", parameters->accurateFacing);

    String renderQueue = OgreMaxUtilities::GetStringAttribute(objectElement, "renderQueue");
    parameters->renderQueue = OgreMaxUtilities::ParseRenderQueue(renderQueue);

    parameters->renderingDistance = OgreMaxUtilities::GetRealAttribute(objectElement, "renderingDistance", 0);

    parameters->extraData = ObjectExtraDataPtr(new ObjectExtraData);
    parameters->extraData->id = OgreMaxUtilities::GetStringAttribute(objectElement, "id");
    parameters->extraData->receiveShadows = OgreMaxUtilities::GetBoolAttribute(objectElement, "receiveShadows", parameters->extraData->receiveShadows);

    //Parse child elements
    parameters->billboards.resize(OgreMaxUtilities::GetChildElementCount(objectElement, "billboard"));

    size_t billboardIndex = 0;
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "billboard")
            LoadBillboard(childElement, parameters->billboards[billboardIndex++]);
        else if (elementName == "commonDirection")
            parameters->commonDirection = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "commonUpVector")
            parameters->commonUpVector = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "userDataReference")
            OgreMaxUtilities::LoadUserDataReference(childElement, parameters->extraData->userDataReference);
        else if (elementName == "userData")
            OgreMaxUtilities::GetChildText(childElement, parameters->extraData->userData);
        else if (elementName == "noteTracks")
        {
            parameters->extraData->noteTracks = NoteTracksPtr(new NoteTracks);
            OgreMaxUtilities::LoadNoteTracks(childElement, *parameters->extraData->noteTracks.get());
        }
        else if (elementName == "customParameters")
            OgreMaxUtilities::LoadCustomParameters(childElement, parameters->customParameters);
    }

    return parameters;
}

PlaneParameters* OgreMaxModel::LoadPlane(const TiXmlElement* objectElement)
{
    PlaneParameters* parameters = new PlaneParameters;
    parameters->name = OgreMaxUtilities::GetStringAttribute(objectElement, "name");
    parameters->queryFlags = OgreMaxUtilities::GetUIntAttribute(objectElement, "queryFlags", 0);
    parameters->visibilityFlags = OgreMaxUtilities::GetUIntAttribute(objectElement, "visibilityFlags", 0);
    parameters->visibility = OgreMaxUtilities::GetObjectVisibilityAttribute(objectElement, "visible");

    parameters->planeName = parameters->name;
    parameters->distance = OgreMaxUtilities::GetRealAttribute(objectElement, "distance", 0);
    parameters->width = OgreMaxUtilities::GetRealAttribute(objectElement, "width", 10);
    parameters->height = OgreMaxUtilities::GetRealAttribute(objectElement, "height", 10);
    parameters->xSegments = OgreMaxUtilities::GetIntAttribute(objectElement, "xSegments", 1);
    parameters->ySegments = OgreMaxUtilities::GetIntAttribute(objectElement, "ySegments", 1);
    parameters->numTexCoordSets = OgreMaxUtilities::GetIntAttribute(objectElement, "numTexCoordSets", 1);
    parameters->uTile = OgreMaxUtilities::GetRealAttribute(objectElement, "uTile", 1);
    parameters->vTile = OgreMaxUtilities::GetRealAttribute(objectElement, "vTile", 1);
    parameters->material = OgreMaxUtilities::GetStringAttribute(objectElement, "material");
    parameters->normals = OgreMaxUtilities::GetBoolAttribute(objectElement, "normals", true);
    parameters->createMovablePlane = OgreMaxUtilities::GetBoolAttribute(objectElement, "movablePlane", true);
    parameters->castShadows = OgreMaxUtilities::GetBoolAttribute(objectElement, "castShadows", true);

    String renderQueue = OgreMaxUtilities::GetStringAttribute(objectElement, "renderQueue");
    parameters->renderQueue = OgreMaxUtilities::ParseRenderQueue(renderQueue);

    parameters->renderingDistance = OgreMaxUtilities::GetRealAttribute(objectElement, "renderingDistance", 0);

    parameters->extraData = ObjectExtraDataPtr(new ObjectExtraData);
    parameters->extraData->id = OgreMaxUtilities::GetStringAttribute(objectElement, "id");
    parameters->extraData->receiveShadows = OgreMaxUtilities::GetBoolAttribute(objectElement, "receiveShadows", parameters->extraData->receiveShadows);

    //Parse child elements
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "normal")
            parameters->normal = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "upVector")
            parameters->upVector = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "vertexBuffer")
            OgreMaxUtilities::LoadBufferUsage(childElement, parameters->vertexBufferUsage, parameters->vertexBufferShadowed);
        else if (elementName == "indexBuffer")
            OgreMaxUtilities::LoadBufferUsage(childElement, parameters->indexBufferUsage, parameters->indexBufferShadowed);
        else if (elementName == "userDataReference")
            OgreMaxUtilities::LoadUserDataReference(childElement, parameters->extraData->userDataReference);
        else if (elementName == "userData")
            OgreMaxUtilities::GetChildText(childElement, parameters->extraData->userData);
        else if (elementName == "noteTracks")
        {
            parameters->extraData->noteTracks = NoteTracksPtr(new NoteTracks);
            OgreMaxUtilities::LoadNoteTracks(childElement, *parameters->extraData->noteTracks.get());
        }
        else if (elementName == "customParameters")
            OgreMaxUtilities::LoadCustomParameters(childElement, parameters->customParameters);
    }

    return parameters;
}

void OgreMaxModel::LoadBillboard(const TiXmlElement* objectElement, BillboardSetParameters::Billboard& billboard)
{
    billboard.width = OgreMaxUtilities::GetRealAttribute(objectElement, "width", 0);
    billboard.height = OgreMaxUtilities::GetRealAttribute(objectElement, "height", 0);
    billboard.rotationAngle = Radian(OgreMaxUtilities::GetRealAttribute(objectElement, "rotation", 0));

    //Parse child elements
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "position")
            billboard.position = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "rotation")
        {
            Quaternion rotation = OgreMaxUtilities::LoadRotation(childElement);
            Vector3 rotationAxis;
            rotation.ToAngleAxis(billboard.rotationAngle, rotationAxis);
        }
        else if (elementName == "colourDiffuse")
            billboard.color = OgreMaxUtilities::LoadColor(childElement);
        else if (elementName == "texCoordRectangle")
            billboard.texCoordRectangle = OgreMaxUtilities::LoadFloatRectangle(childElement);
    }
}

void OgreMaxModel::LoadLightRange(const TiXmlElement* objectElement, LightParameters& light)
{
    if (light.lightType == Light::LT_SPOTLIGHT)
    {
        String value;

        value = OgreMaxUtilities::GetStringAttribute(objectElement, "inner");
        if (!value.empty())
            light.spotlightInnerAngle = Radian(StringConverter::parseReal(value));

        value = OgreMaxUtilities::GetStringAttribute(objectElement, "outer");
        if (!value.empty())
            light.spotlightOuterAngle = Radian(StringConverter::parseReal(value));

        value = OgreMaxUtilities::GetStringAttribute(objectElement, "falloff");
        if (!value.empty())
            light.spotlightFalloff = StringConverter::parseReal(value);
    }
}

void OgreMaxModel::LoadLightAttenuation(const TiXmlElement* objectElement, LightParameters& light)
{
    String value;

    value = OgreMaxUtilities::GetStringAttribute(objectElement, "range");
    if (!value.empty())
        light.attenuationRange = StringConverter::parseReal(value);

    value = OgreMaxUtilities::GetStringAttribute(objectElement, "constant");
    if (!value.empty())
        light.attenuationConstant = StringConverter::parseReal(value);

    value = OgreMaxUtilities::GetStringAttribute(objectElement, "linear");
    if (!value.empty())
        light.attenuationLinear = StringConverter::parseReal(value);

    value = OgreMaxUtilities::GetStringAttribute(objectElement, "quadric");
    if (!value.empty())
        light.attenuationQuadric = StringConverter::parseReal(value);
}

void OgreMaxModel::LoadNodeAnimations(const TiXmlElement* objectElement, NodeParameters& node)
{
    node.animations.resize(OgreMaxUtilities::GetChildElementCount(objectElement, "animation"));

    //Parse child elements
    int animationIndex = 0;
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "animation")
            LoadNodeAnimation(childElement, node.animations[animationIndex++]);
    }
}

void OgreMaxModel::LoadNodeAnimation(const TiXmlElement* objectElement, NodeAnimationParameters& animation)
{
    //Animation name
    animation.name = OgreMaxUtilities::GetStringAttribute(objectElement, "name");

    //Length
    animation.length = OgreMaxUtilities::GetRealAttribute(objectElement, "length", 0);

    //Interpolation mode
    String interpolationModeText = OgreMaxUtilities::GetStringAttribute(objectElement, "interpolationMode");
    if (!interpolationModeText.empty())
        animation.interpolationMode = OgreMaxUtilities::ParseAnimationInterpolationMode(interpolationModeText);

    //Rotation interpolation mode
    String rotationInterpolationModeText = OgreMaxUtilities::GetStringAttribute(objectElement, "rotationInterpolationMode");
    if (!rotationInterpolationModeText.empty())
        animation.rotationInterpolationMode = OgreMaxUtilities::ParseAnimationRotationInterpolationMode(rotationInterpolationModeText);

    //Get enabled and looping states
    animation.enable = OgreMaxUtilities::GetBoolAttribute(objectElement, "enable", true);
    animation.looping = OgreMaxUtilities::GetBoolAttribute(objectElement, "loop", true);

    //Load animation keyframes
    animation.keyframes.resize(OgreMaxUtilities::GetChildElementCount(objectElement, "keyframe"));

    int keyframeIndex = 0;
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "keyframe")
            LoadNodeAnimationKeyFrame(childElement, animation.keyframes[keyframeIndex++]);
    }
}

void OgreMaxModel::LoadNodeAnimationKeyFrame(const TiXmlElement* objectElement, NodeAnimationParameters::KeyFrame& keyframe)
{
    //Key time
    keyframe.time = OgreMaxUtilities::GetRealAttribute(objectElement, "time", 0);

    //Parse child elements
    String elementName;
    const TiXmlElement* childElement = 0;
    while (childElement = OgreMaxUtilities::IterateChildElements(objectElement, childElement))
    {
        elementName = childElement->Value();

        if (elementName == "translation")
            keyframe.translation = OgreMaxUtilities::LoadXYZ(childElement);
        else if (elementName == "rotation")
            keyframe.rotation = OgreMaxUtilities::LoadRotation(childElement);
        else if (elementName == "scale")
            keyframe.scale = OgreMaxUtilities::LoadXYZ(childElement);
    }
}

SceneNode* OgreMaxModel::CreateInstance
    (
    SceneManager* sceneManager,
    const String& baseName,
    OgreMaxModelInstanceCallback* callback,
    InstanceOptions options,
    SceneNode* parentNode,
    const String& defaultResourceGroupName,
    SceneNode* node,
    OgreMaxScene* scene
    ) const
{
    if (parentNode == 0)
        parentNode = sceneManager->getRootSceneNode();

    return CreateInstance
        (
        sceneManager,
        baseName,
        callback,
        options,
        this->rootNode,
        parentNode,
        defaultResourceGroupName,
        node,
        scene
        );
}

SceneNode* OgreMaxModel::CreateInstance
    (
    SceneManager* sceneManager,
    const String& baseName,
    OgreMaxModelInstanceCallback* callback,
    InstanceOptions options,
    const NodeParameters& nodeParams,
    Ogre::SceneNode* parentNode,
    const String& defaultResourceGroupName,
    SceneNode* node,
    OgreMaxScene* scene
    ) const
{
    //Create the node if necessary
    bool newNode = false;
    if (node == 0)
    {
        newNode = true;

        String nodeName = baseName + nodeParams.name;
        node = parentNode->createChildSceneNode(nodeName);
    }

    //Notify callback
    if (newNode && callback != 0)
        callback->StartedCreatingNode(this, node);

    //Set initial transformation
    if ((options & NO_INITIAL_TRANSFORMATION) == 0)
    {
        node->setPosition(nodeParams.position);
        node->setOrientation(nodeParams.orientation);
        node->setScale(nodeParams.scale);
    }

    //Create the model instance (an instance within an instance) if there is one
    if (!nodeParams.modelFile.empty() && scene != 0)
    {
        OgreMaxModel* model = scene->GetModel(nodeParams.modelFile);
        if (model != 0)
        {
            model->CreateInstance
                (
                sceneManager,
                node->getName(),
                callback,
                options | NO_INITIAL_TRANSFORMATION,
                parentNode,
                defaultResourceGroupName,
                node,
                scene
                );
        }
    }

    //Create the objects
    MovableObjectOwner nodeOwner(node);
    for (NodeParameters::Objects::const_iterator objectIterator = nodeParams.objects.begin();
        objectIterator != nodeParams.objects.end();
        ++objectIterator)
    {
        CreateMovableObject(sceneManager, baseName, *objectIterator, nodeOwner, callback, defaultResourceGroupName);
    }

    //Handle node animations
    bool isInitialStateSet = false;
    if (!nodeParams.animations.empty())
    {
        for (size_t animationIndex = 0; animationIndex < nodeParams.animations.size(); animationIndex++)
        {
            const NodeAnimationParameters& animationParams = nodeParams.animations[animationIndex];
            String animationName = baseName + animationParams.name;

            //Create new animation if it doesn't already exist. It shouldn't
            if (!sceneManager->hasAnimation(animationName))
            {
                //Create animation
                Animation* animation = sceneManager->createAnimation(animationName, animationParams.length);
                animation->setInterpolationMode(animationParams.interpolationMode);
                animation->setRotationInterpolationMode(animationParams.rotationInterpolationMode);

                //Create animation track for node
                NodeAnimationTrack* animationTrack = animation->createNodeTrack(animation->getNumNodeTracks() + 1, node);

                //Load animation keyframes
                for (size_t keyframeIndex = 0; keyframeIndex < animationParams.keyframes.size(); keyframeIndex++)
                {
                    const NodeAnimationParameters::KeyFrame& keyframeParams = animationParams.keyframes[keyframeIndex];
                    TransformKeyFrame* keyFrame = animationTrack->createNodeKeyFrame(keyframeParams.time);

                    keyFrame->setTranslate(keyframeParams.translation);
                    keyFrame->setRotation(keyframeParams.rotation);
                    keyFrame->setScale(keyframeParams.scale);
                }

                //Notify callback
                if (callback != 0)
                    callback->CreatedNodeAnimationTrack(this, node, animationTrack, animationParams.enable, animationParams.looping);

                if ((options & NO_ANIMATION_STATES) == 0)
                {
                    //Create a new animation state to track the animation
                    AnimationState* animationState = sceneManager->createAnimationState(animationName);
                    animationState->setEnabled(animationParams.enable);
                    animationState->setLoop(animationParams.looping);

                    //Notify callback
                    if (callback != 0)
                        callback->CreatedNodeAnimationState(this, node, animationState);
                }
            }
        }

        OgreMaxUtilities::SetIdentityInitialState(node);
        isInitialStateSet = true;
    }

    //Iterate over all the node children
    for (size_t childNodeIndex = 0; childNodeIndex < nodeParams.childNodes.size(); childNodeIndex++)
    {
        CreateInstance
            (
            sceneManager,
            baseName,
            callback,
            options & ~NO_INITIAL_TRANSFORMATION,
            nodeParams.childNodes[childNodeIndex],
            node,
            defaultResourceGroupName,
            0,
            scene
            );
    }

    //Set the initial state if it hasn't already been set
    if (!isInitialStateSet)
        node->setInitialState();

    //Set the node's visibility
    OgreMaxUtilities::SetNodeVisibility(node, nodeParams.visibility);

    //Handle node extra data
    if (!nodeParams.extraData.isNull() && nodeParams.extraData->HasUserData())
    {
        ObjectExtraDataPtr objectExtraData(new ObjectExtraData(*nodeParams.extraData.getPointer()));

        //Set extra data owner node
        objectExtraData->node = node;

        //Process the extra data
        HandleNewObjectExtraData(callback, objectExtraData);
    }

    //Notify callback
    if (newNode && callback != 0)
        callback->FinishedCreatingNode(this, node);

    return node;
}

void OgreMaxModel::CreateMovableObject
    (
    SceneManager* sceneManager,
    const String& baseName,
    ObjectParameters* object,
    const MovableObjectOwner& owner,
    OgreMaxModelInstanceCallback* callback,
    const String& defaultResourceGroupName
    ) const
{
    String objectName = baseName + object->name;
    ObjectExtraDataPtr objectExtraData(new ObjectExtraData(*object->extraData.getPointer()));

    switch (object->objectType)
    {
        case ObjectParameters::ENTITY:
        {
            CreateEntity(sceneManager, baseName, objectName, (EntityParameters*)object, owner, objectExtraData, callback, defaultResourceGroupName);
            break;
        }
        case ObjectParameters::LIGHT:
        {
            CreateLight(sceneManager, objectName, (LightParameters*)object, owner, objectExtraData, callback);
            break;
        }
        case ObjectParameters::CAMERA:
        {
            CreateCamera(sceneManager, objectName, (CameraParameters*)object, owner, objectExtraData, callback);
            break;
        }
        case ObjectParameters::PARTICLE_SYSTEM:
        {
            CreateParticleSystem(sceneManager, objectName, (ParticleSystemParameters*)object, owner, objectExtraData, callback);
            break;
        }
        case ObjectParameters::BILLBOARD_SET:
        {
            CreateBillboardSet(sceneManager, objectName, (BillboardSetParameters*)object, owner, objectExtraData, callback);
            break;
        }
        case ObjectParameters::PLANE:
        {
            CreatePlane(sceneManager, baseName, objectName, (PlaneParameters*)object, owner, objectExtraData, callback, defaultResourceGroupName);
            break;
        }
    }
}

void OgreMaxModel::CreateEntity
    (
    SceneManager* sceneManager,
    const String& baseName,
    const String& objectName,
    EntityParameters* entityParams,
    const MovableObjectOwner& owner,
    ObjectExtraDataPtr objectExtraData,
    OgreMaxModelInstanceCallback* callback,
    const String& defaultResourceGroupName
    ) const
{
    //Load the mesh
    bool isNewMesh = !MeshManager::getSingleton().resourceExists(entityParams->meshFile);
    MeshPtr mesh = MeshManager::getSingleton().load
        (
        entityParams->meshFile,
        !entityParams->resourceGroupName.empty() ? entityParams->resourceGroupName : defaultResourceGroupName,
        entityParams->vertexBufferUsage, entityParams->indexBufferUsage,
        entityParams->vertexBufferShadowed, entityParams->indexBufferShadowed
        );

    //Notify callback if the mesh was just loaded
    if (isNewMesh && callback != 0)
        callback->CreatedMesh(this, mesh.getPointer());

    //Create entity
    Entity* entity = sceneManager->createEntity(objectName, entityParams->meshFile);
    if (entityParams->queryFlags != 0)
        entity->setQueryFlags(entityParams->queryFlags);
    if (entityParams->visibilityFlags != 0)
        entity->setVisibilityFlags(entityParams->visibilityFlags);
    OgreMaxUtilities::SetObjectVisibility(entity, entityParams->visibility);
    entity->setCastShadows(entityParams->castShadows);
    entity->setRenderQueueGroup(entityParams->renderQueue);
    entity->setRenderingDistance(entityParams->renderingDistance);
    OgreMaxUtilities::SetCustomParameters(entity, entityParams->customParameters);
    if (!entityParams->materialFile.empty())
        entity->setMaterialName(entityParams->materialFile);

    //Set subentity materials
    size_t subentityCount = std::min(entityParams->subentities.size(), (size_t)entity->getNumSubEntities());
    for (size_t subentityIndex = 0; subentityIndex < subentityCount; subentityIndex++)
    {
        SubEntity* subentity = entity->getSubEntity((unsigned int)subentityIndex);
        if (!entityParams->subentities[subentityIndex].materialName.empty())
            subentity->setMaterialName(entityParams->subentities[subentityIndex].materialName);
    }

    //Create bone attachments
    if (!entityParams->boneAttachments.empty())
    {
        MovableObjectOwner entityOwner(entity);

        for (size_t boneAttachmentIndex = 0; boneAttachmentIndex < entityParams->boneAttachments.size(); boneAttachmentIndex++)
        {
            EntityParameters::BoneAttachment& boneAttachment = entityParams->boneAttachments[boneAttachmentIndex];
            entityOwner.boneName = boneAttachment.boneName;
            entityOwner.attachPosition = boneAttachment.attachPosition;
            entityOwner.attachScale = boneAttachment.attachScale;
            entityOwner.attachRotation = boneAttachment.attachRotation;
            if (boneAttachment.object != 0)
            {
                CreateMovableObject
                    (
                    sceneManager,
                    baseName,
                    boneAttachment.object,
                    entityOwner,
                    callback,
                    defaultResourceGroupName
                    );
            }
            else
                entityOwner.AttachEmpty(boneAttachment.name);
        }
    }

    //Set extra data owner object
    objectExtraData->object = entity;

    //Attach entity to the owner
    owner.Attach(entity);

    //Process the extra data
    HandleNewObjectExtraData(callback, objectExtraData);

    //Notify callback
    if (callback != 0)
        callback->CreatedEntity(this, entity);
}

void OgreMaxModel::CreateLight
    (
    SceneManager* sceneManager,
    const String& objectName,
    LightParameters* lightParams,
    const MovableObjectOwner& owner,
    ObjectExtraDataPtr objectExtraData,
    OgreMaxModelInstanceCallback* callback
    ) const
{
    //Create the light
    Light* light = sceneManager->createLight(objectName);
    if (lightParams->queryFlags != 0)
        light->setQueryFlags(lightParams->queryFlags);
    if (lightParams->visibilityFlags != 0)
        light->setVisibilityFlags(lightParams->visibilityFlags);
    OgreMaxUtilities::SetObjectVisibility(light, lightParams->visibility);
    light->setType(lightParams->lightType);
    light->setCastShadows(lightParams->castShadows);
    light->setPowerScale(lightParams->power);
    light->setDiffuseColour(lightParams->diffuseColor);
    light->setSpecularColour(lightParams->specularColor);
    light->setSpotlightInnerAngle(Radian(lightParams->spotlightInnerAngle));
    light->setSpotlightOuterAngle(Radian(lightParams->spotlightOuterAngle));
    light->setSpotlightFalloff(lightParams->spotlightFalloff);
    light->setAttenuation(lightParams->attenuationRange, lightParams->attenuationConstant, lightParams->attenuationLinear, lightParams->attenuationQuadric);
    light->setPosition(lightParams->position);
    light->setDirection(lightParams->direction);

    //Set extra data owner object
    objectExtraData->object = light;

    //Attach light to the owner
    owner.Attach(light);

    //Process the extra data
    HandleNewObjectExtraData(callback, objectExtraData);

    //Notify callback
    if (callback != 0)
        callback->CreatedLight(this, light);
}

void OgreMaxModel::CreateCamera
    (
    SceneManager* sceneManager,
    const String& objectName,
    CameraParameters* cameraParams,
    const MovableObjectOwner& owner,
    ObjectExtraDataPtr objectExtraData,
    OgreMaxModelInstanceCallback* callback
    ) const
{
    //Create the camera
    Camera* camera = sceneManager->createCamera(objectName);
    if (cameraParams->queryFlags != 0)
        camera->setQueryFlags(cameraParams->queryFlags);
    if (cameraParams->visibilityFlags != 0)
        camera->setVisibilityFlags(cameraParams->visibilityFlags);
    OgreMaxUtilities::SetObjectVisibility(camera, cameraParams->visibility);
    camera->setFOVy(Radian(cameraParams->fov));
    camera->setAspectRatio(cameraParams->aspectRatio);
    camera->setProjectionType(cameraParams->projectionType);
    camera->setNearClipDistance(cameraParams->nearClip);
    camera->setFarClipDistance(cameraParams->farClip);
    camera->setPosition(cameraParams->position);
    camera->setOrientation(cameraParams->orientation);
    camera->setDirection(cameraParams->direction);

    //Set extra data owner object
    objectExtraData->object = camera;

    //Attach camera to the owner
    owner.Attach(camera);

    //Process the extra data
    HandleNewObjectExtraData(callback, objectExtraData);

    //Notify callback
    if (callback != 0)
        callback->CreatedCamera(this, camera);
}

void OgreMaxModel::CreateParticleSystem
    (
    SceneManager* sceneManager,
    const String& objectName,
    ParticleSystemParameters* particleSystemParams,
    const MovableObjectOwner& owner,
    ObjectExtraDataPtr objectExtraData,
    OgreMaxModelInstanceCallback* callback
    ) const
{
    //Create the particle system
    ParticleSystem* particleSystem = sceneManager->createParticleSystem(objectName, particleSystemParams->file);
    if (particleSystemParams->queryFlags != 0)
        particleSystem->setQueryFlags(particleSystemParams->queryFlags);
    if (particleSystemParams->visibilityFlags != 0)
        particleSystem->setVisibilityFlags(particleSystemParams->visibilityFlags);
    OgreMaxUtilities::SetObjectVisibility(particleSystem, particleSystemParams->visibility);
    particleSystem->setRenderQueueGroup(particleSystemParams->renderQueue);
    particleSystem->setRenderingDistance(particleSystemParams->renderingDistance);

    //Set extra data owner object
    objectExtraData->object = particleSystem;

    //Attach particle system to the owner
    owner.Attach(particleSystem);

    //Process the extra data
    HandleNewObjectExtraData(callback, objectExtraData);

    //Notify callback
    if (callback != 0)
        callback->CreatedParticleSystem(this, particleSystem);
}

void OgreMaxModel::CreateBillboardSet
    (
    SceneManager* sceneManager,
    const String& objectName,
    BillboardSetParameters* billboardSetParams,
    const MovableObjectOwner& owner,
    ObjectExtraDataPtr objectExtraData,
    OgreMaxModelInstanceCallback* callback
    ) const
{
    //Create the billboard set
    BillboardSet* billboardSet = sceneManager->createBillboardSet(objectName);
    if (billboardSetParams->queryFlags != 0)
        billboardSet->setQueryFlags(billboardSetParams->queryFlags);
    if (billboardSetParams->visibilityFlags != 0)
        billboardSet->setVisibilityFlags(billboardSetParams->visibilityFlags);
    OgreMaxUtilities::SetObjectVisibility(billboardSet, billboardSetParams->visibility);
    billboardSet->setRenderQueueGroup(billboardSetParams->renderQueue);
    billboardSet->setRenderingDistance(billboardSetParams->renderingDistance);
    OgreMaxUtilities::SetCustomParameters(billboardSet, billboardSetParams->customParameters);
    if (!billboardSetParams->material.empty())
        billboardSet->setMaterialName(billboardSetParams->material);
    billboardSet->setDefaultWidth(billboardSetParams->width);
    billboardSet->setDefaultHeight(billboardSetParams->height);
    billboardSet->setBillboardType(billboardSetParams->billboardType);
    billboardSet->setBillboardOrigin(billboardSetParams->origin);
    billboardSet->setBillboardRotationType(billboardSetParams->rotationType);
    billboardSet->setCommonDirection(billboardSetParams->commonDirection);
    billboardSet->setCommonUpVector(billboardSetParams->commonUpVector);
    if (billboardSetParams->poolSize > 0)
        billboardSet->setPoolSize(billboardSetParams->poolSize);
    billboardSet->setAutoextend(billboardSetParams->autoExtendPool);
    billboardSet->setCullIndividually(billboardSetParams->cullIndividual);
    billboardSet->setSortingEnabled(billboardSetParams->sort);
    billboardSet->setUseAccurateFacing(billboardSetParams->accurateFacing);

    //Load billboards
    for (size_t billboardIndex = 0; billboardIndex < billboardSetParams->billboards.size(); billboardIndex++)
    {
        BillboardSetParameters::Billboard& billboardParams = billboardSetParams->billboards[billboardIndex];

        Billboard* billboard = billboardSet->createBillboard(billboardParams.position, billboardParams.color);
        if (billboardParams.rotationAngle.valueRadians() != 0)
            billboard->setRotation(billboardParams.rotationAngle);
        if (billboardParams.width != 0 && billboardParams.height != 0)
            billboard->setDimensions(billboardParams.width, billboardParams.height);
        if (billboardParams.texCoordRectangle.width() != 0 && billboardParams.texCoordRectangle.height() != 0)
            billboard->setTexcoordRect(billboardParams.texCoordRectangle);
    }

    //Set extra data owner object
    objectExtraData->object = billboardSet;

    //Attach billboard set to the owner
    owner.Attach(billboardSet);

    //Process the extra data
    HandleNewObjectExtraData(callback, objectExtraData);

    //Notify callback
    if (callback != 0)
        callback->CreatedBillboardSet(this, billboardSet);
}

void OgreMaxModel::CreatePlane
    (
    SceneManager* sceneManager,
    const String& baseName,
    const String& objectName,
    PlaneParameters* planeParameters,
    const MovableObjectOwner& owner,
    ObjectExtraDataPtr objectExtraData,
    OgreMaxModelInstanceCallback* callback,
    const String& defaultResourceGroupName
    ) const
{
    //Create plane mesh
    String planeName = baseName + planeParameters->planeName;
    Plane plane(planeParameters->normal, planeParameters->distance);
    MeshManager::getSingleton().createPlane
        (
        planeName,
        !planeParameters->resourceGroupName.empty() ? planeParameters->resourceGroupName : defaultResourceGroupName,
        plane,
        planeParameters->width, planeParameters->height,
        planeParameters->xSegments, planeParameters->ySegments,
        planeParameters->normals, planeParameters->numTexCoordSets,
        planeParameters->uTile, planeParameters->vTile,
        planeParameters->upVector,
        planeParameters->vertexBufferUsage, planeParameters->indexBufferUsage,
        planeParameters->vertexBufferShadowed, planeParameters->indexBufferShadowed
        );

    //Create plane entity
    Entity* entity = sceneManager->createEntity(objectName, planeName);
    if (planeParameters->queryFlags != 0)
        entity->setQueryFlags(planeParameters->queryFlags);
    if (planeParameters->visibilityFlags != 0)
        entity->setVisibilityFlags(planeParameters->visibilityFlags);
    OgreMaxUtilities::SetObjectVisibility(entity, planeParameters->visibility);
    entity->setCastShadows(planeParameters->castShadows);
    entity->setRenderQueueGroup(planeParameters->renderQueue);
    entity->setRenderingDistance(planeParameters->renderingDistance);
    OgreMaxUtilities::SetCustomParameters(entity, planeParameters->customParameters);
    if (!planeParameters->material.empty())
        entity->setMaterialName(planeParameters->material);

    //Set extra data owner object
    objectExtraData->object = entity;

    //Attach plane entity to the owner
    owner.Attach(entity);

    //Process the extra data
    HandleNewObjectExtraData(callback, objectExtraData);

    //Notify callback
    if (callback != 0)
        callback->CreatedPlane(this, plane, entity);
}

void OgreMaxModel::HandleNewObjectExtraData(OgreMaxModelInstanceCallback* callback, ObjectExtraDataPtr objectExtraData) const
{
    bool hasOwner =
        objectExtraData->object != 0 ||
        objectExtraData->node != 0;

    if (hasOwner && callback != 0)
        callback->HandleObjectExtraData(objectExtraData);
}
