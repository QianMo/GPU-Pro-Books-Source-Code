#ifndef SCENEDATASTRUCTS_H
#define SCENEDATASTRUCTS_H

#include <string>
#include <vector>

#include "glm/glm.hpp" // for vec3 etc data types

using namespace std;



/// Defines a camera pose.
struct Pose
{
   glm::vec3 userPosition;
   glm::vec3 viewDirection;
   glm::vec3 upVector;
   float angleX;
   float angleY;
   float angleZ;
};

/// Defines the properties of the scene camera.

struct CameraData
{
   vector<Pose> poses;
   int currentPoseIndex;

   float fovh;
   float aspect; 
   float zNear;
   float zFar;
};



/// Defines the properties of a point light.
struct PointLightData
{
   glm::vec3 I;
   glm::vec3 position;
   float constantAttenuation;
   float quadraticAttenuation;
};

/// Defines the properties of a spot light.
struct SpotLightData : PointLightData
{
   // angles defining spot direction 
   // (default: (0, 0, -1), 
   // (rotate by these angles to get the spot direction)
   float angleX;
   float angleY;
   float angleZ;

   float cutoffAngle;
   float spotExponent;
};


/// This data is used by both static and dynamic elements.
struct CommonElementData
{
   string pathModel;
   string pathAtlas;
   string name;

   int atlasWidth;
   int atlasHeight;

   // obj-loader settings: 
   // always compute facet normals
   bool computedVertexNormals;
   float vertexNormalsAngle;
   bool vertexNormalsSmoothingGroups;
   bool unitized;
   bool centered;
   float fixedScaleFactor;
   unsigned int defaultDrawMode;
};

/// Defines a static (geometry may not move or deform) scene element.
struct StaticElementData : CommonElementData
{
   glm::vec3 position;
   glm::vec3 rotation;
   float scaleFactor;
};

/// Defines the transformation and animation properties of a dynamic element instance.
struct DynamicInstanceData
{
   glm::vec3 position;
   glm::vec3 rotation;
   float scaleFactor;

   bool isUserMovable;     ///< May the user move and rotate this instance?

   bool looping; ///< is the obj-animation a loop?
   bool forwards; ///< is the obj-animation (initially) played forwards or backwards
   int stepInterval; ///< ms (for obj-animation)
   int startAtFrame;
};

/// Defines a dynamic (animated and/or user movable) scene element.
struct DynamicElementData : CommonElementData
{
   int animFileStartIndex; ///< Number of first obj-File to load.
   int animFileEndIndex;   ///< Number of last  obj-File to load.
   string pathSequence; ///< (Empty if no animation)
   int sequenceReadInMethod; 

   /// The model is reused for rendering it several times 
   /// but with other transformation and animation properties.
   vector<DynamicInstanceData> instances; 
};


/// An instance of this struct holds all information of the XML scene description.
struct SceneData
{
   int windowWidth;
   int windowHeight;
   string name;
   string parameterFilename;
   CameraData cameraData;
   vector<SpotLightData> spotLights;
   vector<StaticElementData> staticElements;
   vector<DynamicElementData> dynamicElements;
   bool automaticRotation; 

};


#endif
