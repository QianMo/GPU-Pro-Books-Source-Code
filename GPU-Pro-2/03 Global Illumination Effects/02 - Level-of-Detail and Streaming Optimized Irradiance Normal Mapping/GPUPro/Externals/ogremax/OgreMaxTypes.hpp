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


#ifndef OgreMax_OgreMaxTypes_INCLUDED
#define OgreMax_OgreMaxTypes_INCLUDED


//Includes---------------------------------------------------------------------
#include "tinyxml/tinyxml.h"
#include <OgreString.h>
#include <OgreStringConverter.h>
#include <OgreMaterial.h>
#include <OgreVector3.h>
#include <OgreVector4.h>
#include <OgrePlane.h>
#include <OgreQuaternion.h>
#include <OgreHardwareBuffer.h>
#include <OgreRenderQueue.h>
#include <OgreLight.h>
#include <OgreCamera.h>
#include <OgreBillboardSet.h>
#include <OgrePixelFormat.h>
#include <OgreAnimation.h>
#include <OgreAnimationState.h>
#include <OgreTexture.h>
#include <OgreSceneNode.h>
#include <OgreEntity.h>
#include <OgreTagPoint.h>
#include <OgreSkeletonInstance.h>


//Classes----------------------------------------------------------------------
namespace OgreMax
{
    namespace Types
    {
        enum UpAxis
        {
            UP_AXIS_Y,
            UP_AXIS_Z
        };

        enum NodeVisibility
        {
            NODE_VISIBILITY_DEFAULT,
            NODE_VISIBLE,
            NODE_HIDDEN,
            NODE_TREE_VISIBLE,
            NODE_TREE_HIDDEN
        };

        enum ObjectVisibility
        {
            OBJECT_VISIBILITY_DEFAULT,
            OBJECT_VISIBLE,
            OBJECT_HIDDEN
        };

        /** A custom parameter for renderables */
        struct CustomParameter
        {
            size_t id;
            Ogre::Vector4 value;
        };

        /** A simple bounding volume, centered around the owner's origin */
        struct BoundingVolume
        {
            BoundingVolume()
            {
                this->type = NONE;
                this->radius = 0;
                this->size = Ogre::Vector3::ZERO;
            }

            enum Type
            {
                NONE,
                SPHERE,
                BOX,
                CYLINDER,
                CAPSULE,
                MESH
            };

            /** The bounding volume type */
            Type type;

            /** The bounding radius. Used when 'type' is SPHERE, CYLINDER, or CAPSULE */
            float radius;

            /** 
             * The size. All elements are used when 'type' is BOX, 'x' is used when CYLINDER or CAPSULE.
             * Note that when type is CAPSULE, the size given is the distance along the noncurved part of the capsule.
             */
            Ogre::Vector3 size;

            /** A single face */
            struct Face
            {
                Ogre::Vector3 vertex[3];
            };

            /** Faces of the mesh bounding volume. Used when 'type' is MESH */
            std::vector<Face> meshFaces;
        };

        /** A single note in a note track */
        struct Note
        {
            Ogre::Real time;
            Ogre::String text;
        };

        /** A collection of notes */
        struct NoteTrack
        {
            Ogre::String name;
            std::vector<Note> notes;
        };

        /** A collection of note tracks */
        typedef std::vector<NoteTrack> NoteTracks;

        typedef Ogre::SharedPtr<NoteTracks> NoteTracksPtr;

        /** A single external item. */
        struct ExternalItem
        {
            ExternalItem()
            {
                this->position = Ogre::Vector3::ZERO;
                this->scale = Ogre::Vector3::UNIT_SCALE;
            }

            Ogre::String name;
            Ogre::String type;
            Ogre::String file;
            Ogre::String userDataReference;
            Ogre::String userData;
            Ogre::Vector3 position;
            Ogre::Quaternion rotation;
            Ogre::Vector3 scale;
            BoundingVolume boundingVolume;
            NoteTracks noteTracks;
        };

        struct ExternalUserData
        {
            Ogre::String type;
            Ogre::String name;
            Ogre::String id;
            Ogre::String userDataReference;
            Ogre::String userData;
        };

        /**
         * Extra data placed for a various object types.
         * Depending on the context and type of object, some of the fields in this structure might not be used
         */
        struct ObjectExtraData
        {
            ObjectExtraData()
            {
                this->node = 0;
                this->object = 0;
                this->receiveShadows = false;
            }

            /**
             * Initializes the data from another data object
             * Note that the object/node fields aren't copied
             */
            ObjectExtraData(ObjectExtraData& other)
            {
                this->node = 0;
                this->object = 0;
                this->id = other.id;
                this->userDataReference = other.userDataReference;
                this->userData = other.userData;
                this->receiveShadows = other.receiveShadows;
                this->receiveShadows = other.receiveShadows;
                this->noteTracks = other.noteTracks;
            }

            /** Determines if there is any user data */
            bool HasUserData() const
            {
                return !this->id.empty() || !this->userDataReference.empty() || !this->userData.empty();
            }

            /** If set, the extra data belongs to an Ogre::Node */
            Ogre::Node* node;

            /** If set, the extra data belongs to an Ogre::MovableObject */
            Ogre::MovableObject* object;

            Ogre::String id;
            Ogre::String userDataReference;
            Ogre::String userData;
            bool receiveShadows;
            NoteTracksPtr noteTracks;
        };

        typedef Ogre::SharedPtr<ObjectExtraData> ObjectExtraDataPtr;

        struct FogParameters
        {
            FogParameters()
            {
                this->mode = Ogre::FOG_NONE;

                this->expDensity = .001;
                this->linearStart = 0;
                this->linearEnd = 1;

                this->color = Ogre::ColourValue::White;
            }

            Ogre::FogMode mode;

            Ogre::Real expDensity;
            Ogre::Real linearStart;
            Ogre::Real linearEnd;

            Ogre::ColourValue color;
        };

        struct NodeAnimationParameters
        {
            NodeAnimationParameters()
            {
                this->length = 0;
                this->interpolationMode = Ogre::Animation::IM_SPLINE;
                this->rotationInterpolationMode = Ogre::Animation::RIM_LINEAR;
                this->enable = true;
                this->looping = true;
            }

            Ogre::String name;
            Ogre::Real length;
            Ogre::Animation::InterpolationMode interpolationMode;
            Ogre::Animation::RotationInterpolationMode rotationInterpolationMode;
            bool enable;
            bool looping;

            struct KeyFrame
            {
                KeyFrame()
                {
                    this->time = 0;
                    this->translation = Ogre::Vector3::ZERO;
                    this->rotation = Ogre::Quaternion::IDENTITY;
                    this->scale = Ogre::Vector3::UNIT_SCALE;
                }

                Ogre::Real time;
                Ogre::Vector3 translation;
                Ogre::Quaternion rotation;
                Ogre::Vector3 scale;
            };

            std::vector<KeyFrame> keyframes;
        };

        struct SkyBoxParameters
        {
            SkyBoxParameters()
            {
                this->enabled = true;
                this->distance = 0;
                this->drawFirst = true;
            }

            bool enabled;
            Ogre::String material;
            Ogre::Real distance;
            bool drawFirst;
            Ogre::Quaternion rotation;
            Ogre::String resourceGroupName;
            ObjectExtraData extraData;
        };

        struct SkyDomeParameters
        {
            SkyDomeParameters()
            {
                this->enabled = true;
                this->curvature = 0;
                this->tiling = 0;
                this->distance = 0;
                this->drawFirst = true;
                this->xSegments = 0;
                this->ySegments = 0;
            }

            bool enabled;
            Ogre::String material;
            Ogre::Real curvature;
            Ogre::Real tiling;
            Ogre::Real distance;
            bool drawFirst;
            int xSegments;
            int ySegments;
            Ogre::Quaternion rotation;
            Ogre::String resourceGroupName;
            ObjectExtraData extraData;
        };

        struct SkyPlaneParameters
        {
            SkyPlaneParameters()
            {
                this->enabled = true;
                this->scale = 1;
                this->bow = 0;
                this->tiling = 10;
                this->drawFirst = true;
                this->xSegments = 1;
                this->ySegments = 1;
            }

            bool enabled;
            Ogre::String material;
            Ogre::Plane plane;
            Ogre::Real scale;
            Ogre::Real bow;
            Ogre::Real tiling;
            bool drawFirst;
            int xSegments;
            int ySegments;
            Ogre::Quaternion rotation;
            Ogre::String resourceGroupName;
            ObjectExtraData extraData;
        };

        struct ObjectParameters
        {
            enum ObjectType
            {
                NONE,
                ENTITY,
                LIGHT,
                CAMERA,
                PARTICLE_SYSTEM,
                BILLBOARD_SET,
                PLANE
            };

            ObjectParameters()
            {
                this->objectType = NONE;
                this->renderQueue = Ogre::RENDER_QUEUE_MAIN;
                this->renderingDistance = 0;
                this->queryFlags = 0;
                this->visibilityFlags = 0;
                this->visibility = OBJECT_VISIBILITY_DEFAULT;
            }

            virtual ~ObjectParameters()
            {
            }

            /** Name of the object */
            Ogre::String name;

            /**
             * The object type.
             * This can be used to determine which ObjectParameters subclass can be used
             */
            ObjectType objectType;

            /** The object's extra data */
            ObjectExtraDataPtr extraData;

            /** Object query flags */
            Ogre::uint32 queryFlags;

            /** Object visibility flags */
            Ogre::uint32 visibilityFlags;

            /** Indicates whether object is visible */
            ObjectVisibility visibility;

            /** Rendering queue. Not used by all types */
            Ogre::uint8 renderQueue;

            /** Rendering distance. Not used by all types */
            Ogre::Real renderingDistance;

            typedef std::vector<CustomParameter> CustomParameters;

            /** Custom values. Not used by all types */
            CustomParameters customParameters;
        };

        struct EntityParameters : ObjectParameters
        {
            EntityParameters()
            {
                this->objectType = ENTITY;

                this->castShadows = true;

                this->vertexBufferUsage = Ogre::HardwareBuffer::HBU_STATIC_WRITE_ONLY;
	            this->indexBufferUsage = Ogre::HardwareBuffer::HBU_STATIC_WRITE_ONLY;
	            this->vertexBufferShadowed = true;
                this->indexBufferShadowed = true;
            }

            Ogre::String meshFile;
            Ogre::String materialFile;
            bool castShadows;

            Ogre::HardwareBuffer::Usage vertexBufferUsage;
	        Ogre::HardwareBuffer::Usage indexBufferUsage;
	        bool vertexBufferShadowed;
            bool indexBufferShadowed;

            Ogre::String resourceGroupName;

            struct Subentity
            {
                Ogre::String materialName;
            };
            std::vector<Subentity> subentities;

            struct BoneAttachment
            {
                BoneAttachment()
                {
                    this->object = 0;
                    this->attachPosition = Ogre::Vector3::ZERO;
                    this->attachScale = Ogre::Vector3::UNIT_SCALE;
                    this->attachRotation = Ogre::Quaternion::IDENTITY;
                }

                ~BoneAttachment()
                {
                    delete this->object;
                }

                /** Gets the name of the attachment itself */
                const Ogre::String& GetName() const
                {
                    return this->object != 0 ? this->object->name : this->name;
                }

                Ogre::String name; //Used if object is null
                Ogre::String boneName;
                ObjectParameters* object;
                Ogre::Vector3 attachPosition;
                Ogre::Vector3 attachScale;
                Ogre::Quaternion attachRotation;
            };
            std::vector<BoneAttachment> boneAttachments;
        };

        struct LightParameters : ObjectParameters
        {
            LightParameters()
            {
                this->objectType = LIGHT;

                this->lightType = Ogre::Light::LT_POINT;
                this->castShadows = false;
                this->power = 1;

                this->diffuseColor = Ogre::ColourValue::White;
                this->specularColor = Ogre::ColourValue::Black;

                this->spotlightInnerAngle = Ogre::Degree((Ogre::Real)40);
                this->spotlightOuterAngle = Ogre::Degree((Ogre::Real)30);
                this->spotlightFalloff = 1;

                this->attenuationRange = 100000;
                this->attenuationConstant = 1;
                this->attenuationLinear = 0;
                this->attenuationQuadric = 0;

                this->position = Ogre::Vector3::ZERO;
                this->direction = Ogre::Vector3::UNIT_Z;
            }

            Ogre::Light::LightTypes lightType;
            bool castShadows;
            Ogre::Real power;

            Ogre::ColourValue diffuseColor;
            Ogre::ColourValue specularColor;

            Ogre::Radian spotlightInnerAngle;
            Ogre::Radian spotlightOuterAngle;
            Ogre::Real spotlightFalloff;

            Ogre::Real attenuationRange;
            Ogre::Real attenuationConstant;
            Ogre::Real attenuationLinear;
            Ogre::Real attenuationQuadric;

            Ogre::Vector3 position;
            Ogre::Vector3 direction;
        };

        struct CameraParameters : ObjectParameters
        {
            CameraParameters()
            {
                this->objectType = CAMERA;

                this->fov = Ogre::Radian(Ogre::Math::PI/2);
                this->aspectRatio = (Ogre::Real)1.33;
                this->projectionType = Ogre::PT_PERSPECTIVE;

                this->nearClip = 100;
                this->farClip = 100000;

                this->position = Ogre::Vector3::ZERO;
                this->direction = Ogre::Vector3::NEGATIVE_UNIT_Z;
            }

            Ogre::Radian fov;
            Ogre::Real aspectRatio;
            Ogre::ProjectionType projectionType;

            Ogre::Real nearClip;
            Ogre::Real farClip;

            Ogre::Vector3 position;
            Ogre::Quaternion orientation;
            Ogre::Vector3 direction;
        };

        struct ParticleSystemParameters : ObjectParameters
        {
            ParticleSystemParameters()
            {
                this->objectType = PARTICLE_SYSTEM;
            }

            Ogre::String file;
        };

        struct BillboardSetParameters : ObjectParameters
        {
            BillboardSetParameters()
            {
                this->objectType = BILLBOARD_SET;
                this->commonDirection = Ogre::Vector3::UNIT_Z;
                this->commonUpVector = Ogre::Vector3::UNIT_Y;
                this->billboardType = Ogre::BBT_POINT;
                this->origin = Ogre::BBO_CENTER;
                this->rotationType = Ogre::BBR_TEXCOORD;
                this->poolSize = 0;
                this->autoExtendPool = true;
                this->cullIndividual = false;
                this->sort = false;
                this->accurateFacing = false;
            }

            Ogre::String material;
            Ogre::Real width;
            Ogre::Real height;
            Ogre::BillboardType billboardType;
            Ogre::BillboardOrigin origin;
            Ogre::BillboardRotationType rotationType;
            Ogre::Vector3 commonDirection;
            Ogre::Vector3 commonUpVector;
            Ogre::uint32 poolSize;
            bool autoExtendPool;
            bool cullIndividual;
            bool sort;
            bool accurateFacing;

            struct Billboard
            {
                Billboard() : texCoordRectangle(0, 0, 0, 0)
                {
                    this->width = 0;
                    this->height = 0;
                    this->rotationAngle = Ogre::Radian(0);

                    this->position = Ogre::Vector3::ZERO;
                    this->color = Ogre::ColourValue::White;
                }

                Ogre::Real width;
                Ogre::Real height;
                Ogre::FloatRect texCoordRectangle;
                Ogre::Radian rotationAngle;

                Ogre::Vector3 position;
                Ogre::ColourValue color;
            };
            std::vector<Billboard> billboards;
        };

        struct PlaneParameters : ObjectParameters
        {
            PlaneParameters()
            {
                this->objectType = PLANE;

                this->xSegments = 0;
                this->ySegments = 0;
                this->numTexCoordSets = 0;
                this->uTile = 0;
                this->vTile = 0;
                this->normals = true;
                this->createMovablePlane = true;
                this->castShadows = true;
                this->normal = Ogre::Vector3::ZERO;
                this->upVector = Ogre::Vector3::UNIT_Z;

                this->vertexBufferUsage = Ogre::HardwareBuffer::HBU_STATIC_WRITE_ONLY;
                this->indexBufferUsage = Ogre::HardwareBuffer::HBU_STATIC_WRITE_ONLY;
                this->vertexBufferShadowed = true;
                this->indexBufferShadowed = true;
            }

            Ogre::String planeName;
            Ogre::Real distance;
            Ogre::Real width;
            Ogre::Real height;
            int xSegments;
            int ySegments;
            int numTexCoordSets;
            Ogre::Real uTile;
            Ogre::Real vTile;
            Ogre::String material;
            bool normals;
            bool createMovablePlane;
            bool castShadows;

            Ogre::Vector3 normal;
            Ogre::Vector3 upVector;

            Ogre::HardwareBuffer::Usage vertexBufferUsage;
            Ogre::HardwareBuffer::Usage indexBufferUsage;
            bool vertexBufferShadowed;
            bool indexBufferShadowed;

            Ogre::String resourceGroupName;
        };

        struct NodeParameters
        {
            NodeParameters()
            {
                this->visibility = NODE_VISIBILITY_DEFAULT;
                this->position = Ogre::Vector3::ZERO;
                this->scale = Ogre::Vector3::UNIT_SCALE;
            }

            ~NodeParameters()
            {
                for (Objects::iterator objectIterator = this->objects.begin();
                    objectIterator != this->objects.end();
                    ++objectIterator)
                {
                    delete *objectIterator;
                }
            }

            /** The nodes's extra data */
            ObjectExtraDataPtr extraData;

            Ogre::String name;
            Ogre::String modelFile;

            NodeVisibility visibility;

            Ogre::Vector3 position;
            Ogre::Quaternion orientation;
            Ogre::Vector3 scale;

            std::vector<NodeParameters> childNodes;

            std::vector<NodeAnimationParameters> animations;

            typedef std::list<ObjectParameters*> Objects;
            Objects objects;
        };

        struct RenderTextureParameters
        {
            RenderTextureParameters()
            {
                this->width = this->height = 512;
                this->pixelFormat = Ogre::PF_A8R8G8B8;
                this->textureType = Ogre::TEX_TYPE_2D;
                this->clearEveryFrame = true;
                this->autoUpdate = true;
                this->hideRenderObject = true;
            }

            Ogre::String name;
            int width;
            int height;
            Ogre::PixelFormat pixelFormat;
            Ogre::TextureType textureType;
            Ogre::String cameraName;
            Ogre::String scheme;
            Ogre::ColourValue backgroundColor;
            bool clearEveryFrame;
            bool autoUpdate;
            bool hideRenderObject;
            Ogre::String renderObjectName;
            Ogre::Plane renderPlane;
            std::vector<Ogre::String> hiddenObjects;
            std::vector<Ogre::String> exclusiveObjects;
            Ogre::String resourceGroupName;

            struct Material
            {
                Ogre::String name;
                unsigned short techniqueIndex;
                unsigned short passIndex;
                unsigned short textureUnitIndex;
            };
            std::vector<Material> materials;
        };

        struct ShadowParameters
        {
            ShadowParameters()
            {
                this->shadowTechnique = Ogre::SHADOWTYPE_NONE;
                this->selfShadow = true;
                this->farDistance = 0;
                this->textureSize = 512;
                this->textureCount = 2;
                this->textureOffset = (Ogre::Real).6;
                this->textureFadeStart = (Ogre::Real).7;
                this->textureFadeEnd = (Ogre::Real).9;
                this->pixelFormat = Ogre::PF_UNKNOWN;
                this->shadowColor = Ogre::ColourValue::Black;
            }

            Ogre::ShadowTechnique shadowTechnique;
            bool selfShadow;
            Ogre::Real farDistance;
            int textureSize;
            int textureCount;
            Ogre::Real textureOffset;
            Ogre::Real textureFadeStart;
            Ogre::Real textureFadeEnd;
            Ogre::String textureShadowCasterMaterial;
            Ogre::String textureShadowReceiverMaterial;
            Ogre::PixelFormat pixelFormat;
            Ogre::ColourValue shadowColor;
            Ogre::String cameraSetup;
            Ogre::Plane optimalPlane;
        };

        struct LookTarget
        {
            /**
             * Initializes the LookTarget for a scene node or camera.
             * Either sourceNode or sourceCamera must be non-null
             */
            LookTarget(Ogre::SceneNode* sourceNode, Ogre::Camera* sourceCamera)
            {
                this->sourceNode = sourceNode;
                this->sourceCamera = sourceCamera;
                this->relativeTo = Ogre::Node::TS_LOCAL;
                this->isPositionSet = false;
                this->position = Ogre::Vector3::ZERO;
                this->localDirection = Ogre::Vector3::NEGATIVE_UNIT_Z;
            }

            Ogre::SceneNode* sourceNode;
            Ogre::Camera* sourceCamera;
            Ogre::String nodeName;
            Ogre::Node::TransformSpace relativeTo;
            bool isPositionSet;
            Ogre::Vector3 position;
            Ogre::Vector3 localDirection;
        };

        struct TrackTarget
        {
            /**
             * Initializes the TrackTarget for a scene node or camera.
             * Either sourceNode or sourceCamera must be non-null
             */
            TrackTarget(Ogre::SceneNode* sourceNode, Ogre::Camera* sourceCamera)
            {
                this->sourceNode = sourceNode;
                this->sourceCamera = sourceCamera;
                this->offset = Ogre::Vector3::ZERO;
                this->localDirection = Ogre::Vector3::NEGATIVE_UNIT_Z;
            }

            Ogre::SceneNode* sourceNode;
            Ogre::Camera* sourceCamera;
            Ogre::String nodeName;
            Ogre::Vector3 offset;
            Ogre::Vector3 localDirection;
        };

        struct SceneNodeArray : std::vector<Ogre::SceneNode*>
        {
            void Show()
            {
                for (size_t i = 0; i < size(); i++)
                    (*this)[i]->setVisible(true, false);
            }

            void Hide()
            {
                for (size_t i = 0; i < size(); i++)
                    (*this)[i]->setVisible(false, false);
            }
        };

        /** A loaded render texture */
        struct LoadedRenderTexture
        {
            enum {CUBE_FACE_COUNT = 6};

            LoadedRenderTexture()
            {
                this->renderObjectNode = 0;
                this->renderPlane = 0;

                this->camera = 0;

                for (int index = 0; index < CUBE_FACE_COUNT; index++)
                {
                    this->cubeFaceCameras[index] = 0;
                    this->viewports[index] = 0;
                }
            }

            /** Sets the position of all cube face cameras */
            void SetCubeFaceCameraPosition(const Ogre::Vector3& position)
            {
                for (int index = 0; index < CUBE_FACE_COUNT; index++)
                    this->cubeFaceCameras[index]->setPosition(position);
            }

            /**
             * Gets the 'reference' position, which is used when updating a cube map render texture.
             * The preferred position is that of the render object. If there is no render object, the reference
             * camera is used. If there is no camera, the zero vector is used.
             * @param position [out] - The position. If there is no reference object, this is set to zero.
             * @return Returns true if there was a reference object to use, false otherwise.
             */
            bool GetReferencePosition(Ogre::Vector3& position) const
            {
                bool result = true;

                if (this->renderObjectNode != 0)
                    position = this->renderObjectNode->_getDerivedPosition();
                else if (this->camera != 0)
                    position = this->camera->getDerivedPosition();
                else
                {
                    position = Ogre::Vector3::ZERO;
                    result = false;
                }

                return result;
            }

            RenderTextureParameters parameters;
            Ogre::TexturePtr renderTexture;
            Ogre::SceneNode* renderObjectNode;
            Ogre::MovablePlane* renderPlane;

            Ogre::Camera* camera;
            Ogre::Camera* cubeFaceCameras[CUBE_FACE_COUNT];
            Ogre::Viewport* viewports[CUBE_FACE_COUNT];

            SceneNodeArray hiddenObjects;
            SceneNodeArray exclusiveObjects;
        };

        /** Maps a query flag bit to a name */
        struct FlagAlias
        {
            FlagAlias()
            {
                this->bit = 0;
            }

            Ogre::String name;
            int bit;
        };

        /** A collection of flag aliases */
        struct FlagAliases : std::vector<FlagAlias>
        {
            /**
             * Gets the name that corresponds to the specified bit
             * @param bit [in] - The index of the bit to look up
             * @param name [out] - The name of the bit
             * @return Returns true if the bit's name was found, false otherwise
             */
            bool GetBitName(int bit, Ogre::String& name)
            {
                for (size_t index = 0; index < size(); index++)
                {
                    if ((*this)[index].bit == bit)
                    {
                        name = (*this)[index].name;
                        return true;
                    }
                }

                return false;
            }
        };

        /** Used for attaching MovableObject instances to an owner */
        struct MovableObjectOwner
        {
            /** No owner */
            MovableObjectOwner()
            {
                this->node = 0;
                this->entity = 0;
                this->attachPosition = Ogre::Vector3::ZERO;
                this->attachScale = Ogre::Vector3::UNIT_SCALE;
                this->attachRotation = Ogre::Quaternion::IDENTITY;
            }

            /** The owner is a scene node */
            MovableObjectOwner(Ogre::SceneNode* node)
            {
                this->node = node;
                this->entity = 0;
                this->attachPosition = Ogre::Vector3::ZERO;
                this->attachScale = Ogre::Vector3::UNIT_SCALE;
                this->attachRotation = Ogre::Quaternion::IDENTITY;
            }

            /** The owner is a bone within an entity's skeleton */
            MovableObjectOwner
                (
                Ogre::Entity* entity,
                const Ogre::String& boneName = Ogre::StringUtil::BLANK,
                const Ogre::Vector3& attachPosition = Ogre::Vector3::ZERO,
                const Ogre::Vector3& attachScale = Ogre::Vector3::UNIT_SCALE,
                const Ogre::Quaternion& attachRotation = Ogre::Quaternion::IDENTITY
                )
            {
                this->node = 0;
                this->entity = entity;
                this->boneName = boneName;
                this->attachPosition = attachPosition;
                this->attachScale = attachScale;
                this->attachRotation = attachRotation;
            }

            /** Attaches the movable object to the owner */
            void Attach(Ogre::MovableObject* object) const
            {
                if (this->node != 0)
                    this->node->attachObject(object);
                else if (this->entity != 0 && !this->boneName.empty())
                {
                    //TODO: Modify Ogre to accept object->getName() when creating TagPoint
                    Ogre::TagPoint* tagPoint = this->entity->attachObjectToBone(this->boneName, object);
                    tagPoint->setPosition(this->attachPosition);
                    tagPoint->setScale(this->attachScale);
                    tagPoint->setOrientation(this->attachRotation);
                }
            }

            /**
             * Attaches an empty object to the owner. This has no effect if the owner is a node since
             * there's no notion of an "empty" object for nodes. For entities, an "empty" object corresponds
             * to a tag point that has no attachment
             */
            void AttachEmpty(const Ogre::String& name = Ogre::StringUtil::BLANK) const
            {
                if (this->entity != 0 && !this->boneName.empty())
                {
                    Ogre::SkeletonInstance* skeleton = this->entity->getSkeleton();
                    Ogre::Bone* bone = skeleton->getBone(this->boneName);
                    //TODO: Modify Ogre to accept name when creating TagPoint
                    Ogre::TagPoint* tagPoint = skeleton->createTagPointOnBone(bone);
                    tagPoint->setPosition(this->attachPosition);
                    tagPoint->setScale(this->attachScale);
                    tagPoint->setOrientation(this->attachRotation);
                }
            }

            Ogre::SceneNode* node;
            Ogre::Entity* entity;
            Ogre::String boneName;
            Ogre::Vector3 attachPosition;
            Ogre::Vector3 attachScale;
            Ogre::Quaternion attachRotation;
        };

    }

}

#endif
