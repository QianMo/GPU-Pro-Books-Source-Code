/******************************************************************/
/* Scene.h                                                        */
/* -----------------------                                        */
/*                                                                */
/* The file defines a scene class that encapsulates all the       */
/*     information necessary to render an image with a ray tracer */
/* Also note that this class includes the TraceRay() method,      */
/*     which actually traces a ray through the scene.             */
/* Scenes should be setup similar to the example(s) in the file   */
/*     SceneSetup.cpp.                                            */
/*                                                                */
/* Chris Wyman (10/26/2006)                                       */
/******************************************************************/

#ifndef SCENE_H
#define SCENE_H

#include "DataTypes/Color.h"
#include "DataTypes/MathDefs.h"
#include "DataTypes/Array1D.h"
#include "DataTypes/Matrix4x4.h"
#include "Objects/Group.h"
#include "Objects/Object.h"
#include "Scene/Camera.h"
#include "Scene/GLLight.h"
#include "Scene/AreaLight.h"
#include "Materials/Material.h"
#include "Utils/ProgramPathLists.h"
#include "Utils/Trackball.h"
#include "Utils/glslProgram.h"
#include "Interface/UIVars/UIVariable.h"
#include "Interface/UIVars/UIInt.h"
#include "Interface/UIVars/UIBool.h"
#include "Interface/UIVars/UIFloat.h"

class FrameBuffer;

class Scene {
private:
	Camera *camera;           // Scene camera.  There is only one (for now).
	Group *geometry;          // Scene geometry.  There is only one, as it should be a container object.
	Array1D<GLLight *> light; // Scene lights.  A list of all lights in the scene.
	Array1D<AreaLight *>areas; // Scene area lights.

	// These are used for loading the scene into memory from a file
	Array1D<Material *   > fileMaterials;  // a list of materials, so they can be named and reused
	Array1D<GLTexture *  > fileTextures;   // a list of textures, so they can be reused
	Array1D<GLSLProgram *> fileShaders;    // a list of shaders, so they can be reloaded when needed

	// Some textures (aka, video textures) may change every frame!
	Array1D<GLTexture *  > dynamicTextures; // a list of video textures

	// The following variables are declared in the scene file.  These variables have an associated
	//    data type and a character string name.  A search through the identifier list gives the
	//    array index of the identifier, which is the same index as the corresponding data in the
	//    data array.
	Array1D<UIVariable *>	uiVars;
	bool sceneFileDataAccessed;

	// Used for loading objects.  Some may have names associated, in which case we may 
	//     need the names later in the file to reference them (e.g., for instances)
	Array1D<Object *> namedObjects;
	Array1D<char *>   objectNames;

	// Size of resulting image
	int screenWidth, screenHeight;

	// Number of lights enabled.  This is updated (only) everytime 
	//    SetupEnabledLightsWithCurrentModelview() is called.
	int lightsEnabled;

	// Describes verbosity of output.  Not all output to the console is
	//    surpressed with this flag.
	bool verbose;

	// Used if our scene has any spheres or cylinders
	GLUquadricObj *quadObj;

	// If we call CreateShadowMap, we'll store some temporary data here to allow usage
	//   of the shadow map by anyone with access to the scene data.  This could be 
	//   modified to encapsulate the data better, by making the scene object control
	//	 the shadow map.
	GLuint shadowMapTexID[4], causticMapTexID[4], causticDepthMapTexID[4];
	float shadMapTransMatrix[4][16], shadMapMatrix[4][16];

	UIFloat *lightIntensityModifier, *extinctionCoef;
	float sceneAmbient[4];

	bool cullingDisabled;
	double frameTime;

	GLSLProgram *drawOnlyTexture;
public:
	// Constructors & destructors
	Scene();										// Default constructor
	Scene( char *filename, bool verbose=false );    // Reads a scene from the file
	~Scene();

	// Get/set the camera
	inline Camera *GetCamera( void ) { return camera; }
	void SetCamera( Camera *cam );

	// Get the width and height of the image
	inline int GetWidth( void ) const			{ return screenWidth; }
	inline int GetHeight( void ) const			{ return screenHeight; }
	inline void SetWidth( int newWidth )		{ screenWidth = newWidth; }
	inline void SetHeight( int newHeight )		{ screenHeight = newHeight; }
	
	// Reload all the shaders
	bool ReloadShaders( void );

	// Get/set the geometry.  Note, this is usually a group-type object
	inline Group *GetGeometry( void ) const    { return geometry; }
	inline void SetGeometry( Group *obj )      { geometry = obj; }

	// Deal with scene lights
	inline int GetNumLights( void ) const          { return light.Size(); }
	inline int GetNumAreaLights( void ) const	   { return areas.Size(); }
	inline void AddLight( GLLight *lght )          { light.Add( lght ); }
	inline GLLight *GetLight( int i )              { return light[i]; }
	inline AreaLight *GetAreaLight( int i )		   { return areas[i]; }
	inline float *GetSceneAmbient( void )          { return sceneAmbient; }
	inline float GetLightIntensityModifier( void ) { return *lightIntensityModifier; }
	inline int NumLightsEnabled( void ) const      { return lightsEnabled; }  // Might be out of date!!

	inline float GetFogExtinctionCoef( void )	   { return *extinctionCoef; }

	inline bool IsVerbose( void ) const            { return verbose; }

	// Setup trackballs for objects and lights.  These are called by object
	//    and/or light constructors when they find out the user wants to 
	//    associate a trackball with the object/light.  The scene takes care
	//    of setting things up correctly.
	// 'i' is a user spectified trackball ID.  Usually this works in conjunction
	//    with a scene-defined user interface that attaches some manipulation to
	//    this particular trackball ID.  A warning is printed if this is out of 
	//    the valid range.
	// The callee is responsible for creating the trackball and remembering a 
	//    pointer for any further use.
	void SetupLightTrackball( int i, Trackball *ball );
	void SetupObjectTrackball( int i, Trackball *ball );

	// Deal with scene textures & materials
	inline void AddTexture( GLTexture *tex )      { fileTextures.Add( tex ); }
	inline void AddMaterial( Material *mat )      { fileMaterials.Add( mat ); }
	inline void AddShader( GLSLProgram *shader )  { fileShaders.Add( shader ); }

	// Gets pointer to a variable defined in the scene file.
	UIVariable *GetSceneVariable( char *varName, UIVariable *defaultValue );


	// Setup light positions (there might be multiple of these, depending on
	//    if you want the Scene to setup the lights using the *current* modelview
	//    matrix or the *correct* modelview as the scene sees it)
	void SetupEnabledLightsWithCurrentModelview( void );

	// OpenGL scene setup
	inline void PerspectiveMatrix( void )          { gluPerspective( camera->GetFovy(), ((float)screenWidth)/screenHeight, camera->GetNear(), camera->GetFar() ); }
	inline Matrix4x4 GetPerspectiveMatrix( void )  { return Matrix4x4::Perspective( camera->GetFovy(), ((float)screenWidth)/screenHeight, camera->GetNear(), camera->GetFar() ); }
	inline void LookAtMatrix( void )               { camera->LookAtMatrix(); }
	inline void LookAtInverseMatrix( void )		   { camera->InverseLookAtMatrix(); }

	void LightPerspectiveMatrix( int i, float aspect );
	void LightPerspectiveInverseMatrix( int i, float aspect );
	void LightLookAtMatrix( int i );
	void LightLookAtInverseMatrix( int i );
	Matrix4x4 GetLightLookAtMatrix( int i );
	Matrix4x4 GetLightPerspectiveMatrix( int i, float aspect );

	// Draw Geometry
	inline void Draw( unsigned int matlFlags=MATL_FLAGS_NONE, unsigned int optionFlags=OBJECT_OPTION_NONE )							    
		{ geometry->Draw( this,matlFlags,optionFlags ); }
	inline void DrawOnly( unsigned int propertyFlags, unsigned int matlFlags=MATL_FLAGS_NONE, unsigned int optionFlags=OBJECT_OPTION_NONE ) 
		{ geometry->DrawOnly( this,propertyFlags,matlFlags,optionFlags); }

	// Preprocess the scene
	void Preprocess( void );
	void SetupDefaultScenePaths( void );

	// Functions that are really only valid during scene preprocessing. 
	//    Calling at other times may return NULL values, invalid values, or
	//    otherwise corrupted data (i.e., undefined behavior)
	inline GLUquadricObj *GetQuadric( void )   { return quadObj; }

	// Process the scene between frames
	inline void IncrementFrameTime( double increment ) { frameTime += increment; }
	inline double GetFrameTime( void )                 { return frameTime; }
	void PerFrameUpdate( float currentTime );
	void PerFrameUpdate( void );

	// Used for loading the scene...  You may need to add more of these, or you 
	//    may be able to handle them using class constructors.
	Material *ExistingMaterialFromFile( char *name );
	GLTexture *ExistingTextureFromFile( char *filename );
	GLTexture *GetNamedTexture( char *name );
	Object *ExistingObjectFromFile( char *name );
	GLLight *ExistingLightFromFile( char *name );
	Material *LoadMaterial( char *typeString, FILE *file );
	Object *LoadObject( char *typeString, FILE *file );
	Camera *LoadCamera( char *typeString, FILE *file );
	Material *GetDefaultMaterial( void ) { return fileMaterials[0]; }

	// Some unhandled keywords are bracketed by an end.  They cannot simply
	//    be ignored -- all lines until an "end" must be ignored.  This 
	//    function ignores all such lines and prints a warning message.
	void UnhandledKeyword( FILE *f, char *typeString=0, char *keyword=0 );

	// To truly deal with unhandled keywords, we must deal with unhandled
	//    keywords with nested keyword blocks (e.g., a material defined 
	//    inside an object).  In these cases, searching for the first "end"
	//    doesn't work.  This checks if the keyword opens a block, and if
	//    so UnhandledKeyword() can be called recursively.
	bool IsBlockKeyword( char *keyword );


	// Rendering functions
	void CreateShadowMap( FrameBuffer *shadMapBuf, float *shadMapMatrixTranspose, int lightNum, float shadowMapBias=0 );

	// Functions to access internal scene shaders
	inline GLSLProgram *GetSimpleTextureShader( void )      { return drawOnlyTexture; }

	// Get/Set data for various rendered buffers.  Note CreateShadowMap() sets the appropriate
	//   values for the shadow map data...
	inline GLuint GetShadowMapID( int mapNum ) const		{ return shadowMapTexID[mapNum]; }
	inline void SetShadowMapID( int mapNum, GLuint id )		{ shadowMapTexID[mapNum] = id; }
	inline float *GetShadowMapTransposeMatrix( int mapNum )	{ return shadMapTransMatrix[mapNum]; }
	inline float *GetShadowMapMatrix( int mapNum )          { return shadMapMatrix[mapNum]; }
	inline void SetShadowMapTransposeMatrix( int mapNum, float *m ) { memcpy( shadMapTransMatrix[mapNum], m, 16*sizeof(float) ); }
	inline GLuint GetCausticMapID( int mapNum ) const		{ return causticMapTexID[mapNum]; }
	inline void SetCausticMapID( int mapNum, GLuint id ) 	{ causticMapTexID[mapNum] = id; }
	inline GLuint GetCausticDepthID( int mapNum ) const		{ return causticDepthMapTexID[mapNum]; }
	inline void SetCausticDepthID( int mapNum, GLuint id ) 	{ causticDepthMapTexID[mapNum] = id; }

	inline bool HasCulling( void ) { return !cullingDisabled; }

	// Search paths for the scene.  This is already pretty well encapsulated,
	//    so I see no reason not to offer users direct access.
	ProgramSearchPaths *paths;
};



#endif

