#ifndef COMMON_H
#define COMMON_H


#ifndef WORLD_XAXIS
#define WORLD_XAXIS glm::vec3(1.0, 0.0, 0.0)
#endif

#ifndef WORLD_YAXIS
#define WORLD_YAXIS glm::vec3(0.0, 1.0, 0.0)
#endif

#ifndef WORLD_ZAXIS
#define WORLD_ZAXIS glm::vec3(0.0, 0.0, 1.0)
#endif

#ifndef SETTINGS
#define SETTINGS Settings::Instance()
#endif

#ifndef SCENE
#define SCENE Scene::Instance()
#endif

#ifndef MAX_RAYS
#define MAX_RAYS 128
#endif

#ifndef MAX_ATLAS_RESOLUTION
#define MAX_ATLAS_RESOLUTION 2048
#endif

#define RAND(a,b) (a + float(rand()) / RAND_MAX * (b-(a)))

#ifndef D_PI
#define D_PI 3.1415926535897932384626433832795
#endif

#ifndef F_PI
#define F_PI 3.1415926535897932384626433832795f
#endif

/// Enumeration used by drawing methods
enum GeometryType { ALL, STATIC, DYNAMIC };
/// Defines the GBuffer content
enum Buffer { POSITION, NORMAL, MATERIAL, DIRECTLIGHT };
/// Defines Spot Map content 
enum SpotBuffer { MAP_POSITION, MAP_NORMAL, MAP_MATERIAL, MAP_DIRECTLIGHT };
/// Defines the resolution of the indirect light buffer
enum BufferSize {FULL, HALF, QUARTER, EIGHTH, SIXTEENTH};

#endif // COMMON_H