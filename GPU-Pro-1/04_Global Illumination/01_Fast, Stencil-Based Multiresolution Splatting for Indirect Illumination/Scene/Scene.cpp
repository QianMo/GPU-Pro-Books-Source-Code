/******************************************************************/
/* Scene.cpp                                                      */
/* -----------------------                                        */
/*                                                                */
/* The file defines a scene class that encapsulates all the       */
/*     information necessary to render an image with a ray tracer */
/* Also not that this class includes the TraceRay() method, which */
/*     actually traces a ray through the scene.                   */
/*                                                                */
/* Chris Wyman (10/26/2006)                                       */
/******************************************************************/

#include "Utils/ImageIO/imageIO.h"
#include "DataTypes/Color.h"
#include "DataTypes/glTexture.h"
#include "DataTypes/glVideoTexture.h"
#include "Scene/Scene.h"
#include "Scene/Camera.h"
#include "Scene/glLight.h"
#include "Scene/areaLight.h"
#include "Objects/Group.h"
#include "Objects/Triangle.h"
#include "Objects/Sphere.h"
#include "Objects/Quad.h"
#include "Objects/Cylinder.h"
#include "Objects/Mesh.h"
#include "Objects/ImportedScene.h"
#include "Materials/Material.h"
#include "Materials/GLMaterial.h"
#include "Materials/GLLambertianMaterial.h"
#include "Materials/GLLambertianTexMaterial.h"
#include "Materials/GLConstantMaterial.h"
#include "Materials/GLSLShaderMaterial.h"
#include "Utils/ProgramPathLists.h"
#include "Interface/SceneFileDefinedInteraction.h"
#include "Utils/Trackball.h"

// This is a big hack...  Sometimes the global scene pointer
//    is needed before the Scene( filename ) constructor returns...
//    So this needs to be set in the constructor.
extern Scene *scene;

SceneDefinedUI *ui=0;
Trackball *eyeBall      = 0;
Trackball *lightBall[4] = {0,0,0,0};
Trackball *objBall[4]   = {0,0,0,0};


void Scene::SetupDefaultScenePaths( void )
{
	paths->AddShaderPath( "shaders/" );
	paths->AddShaderPath( "shaders/utilityShaders/" );
	paths->AddShaderPath( "shaders/normalSurfaceShaders/" );
	paths->AddShaderPath( "shaders/basicRefraction/" );
	paths->AddShaderPath( "shaders/simpleCausticMaps/" );
	paths->AddShaderPath( "shaders/hierarchicalCausticMaps/" );
	paths->AddShaderPath( "shaders/adaptiveCausticMaps/" );
	paths->AddShaderPath( "shaders/deferredLighting/" );
	paths->AddShaderPath( "shaders/subsplatRefinement/" );
	paths->AddShaderPath( "shaders/subSplatSplatting/" );
	paths->AddShaderPath( "shaders/areaLightSampling/" );
	paths->AddTexturePath( "textures/" );
	paths->AddModelPath( "models/" );	
	paths->AddScenePath( "scenes/" );
}

Scene::Scene() : 
	camera(0), geometry(0), frameTime(0),
	screenWidth(256), screenHeight(256), 
	verbose(true), sceneFileDataAccessed(false),cullingDisabled(false)
{
	paths = new ProgramSearchPaths();
	SetupDefaultScenePaths();
	ui = new SceneDefinedUI();
	lightIntensityModifier = new UIFloat( 1.0 );
	extinctionCoef		   = new UIFloat( 0.0 );
	sceneAmbient[0] = sceneAmbient[1] = sceneAmbient[2] = 0.05; sceneAmbient[3] = 1.0;
}

// Free all the allocated memory inside the scene.
Scene::~Scene()
{
	for (unsigned int i=0; i<fileTextures.Size(); i++)
		delete fileTextures[i];

	if (camera) delete camera;
	if (geometry) delete geometry;
}

// Set the camera to a new camera.
void Scene::SetCamera( Camera *cam )
{
	if (camera) delete camera;
	camera = cam;
}

void Scene::PerFrameUpdate( float currentTime ) 
{ 
	if (camera->NeedPerFrameUpdates())
		camera->Update( currentTime );
	if (geometry->NeedPerFrameUpdates()) 
		geometry->Update( currentTime ); 
	for (unsigned int i=0; i<light.Size(); i++)
	{
		if (light[i]->NeedPerFrameUpdates()) 
			light[i]->Update( currentTime );
	}
	for (unsigned int i=0; i<dynamicTextures.Size(); i++)
		dynamicTextures[i]->Update( currentTime );
	for (unsigned int i=0; i<areas.Size(); i++)
		if (areas[i]->NeedPerFrameUpdates())
			areas[i]->Update( currentTime );
}

void Scene::PerFrameUpdate( void ) 
{ 
	if (camera->NeedPerFrameUpdates())
		camera->Update( frameTime );
	if (geometry->NeedPerFrameUpdates()) 
		geometry->Update( frameTime ); 
	for (unsigned int i=0; i<light.Size(); i++)
	{
		if (light[i]->NeedPerFrameUpdates()) 
			light[i]->Update( frameTime );
	}
	for (unsigned int i=0; i<dynamicTextures.Size(); i++)
		dynamicTextures[i]->Update( frameTime );
	for (unsigned int i=0; i<areas.Size(); i++)
		if (areas[i]->NeedPerFrameUpdates())
			areas[i]->Update( frameTime );
}

void Scene::SetupEnabledLightsWithCurrentModelview( void )
{
	int currentlyEnabled = 0;
	for (unsigned int i=0; i<light.Size(); i++)
		if (light[i]->IsEnabled())
		{
			light[i]->SetLightUsingCurrentTransforms();
			currentlyEnabled++;
		}
	lightsEnabled = currentlyEnabled;
}


void Scene::SetupLightTrackball( int i, Trackball *ball )
{
	if (i < 0 || i > 3)
	{
		char buf[255];
		sprintf( buf, "Light trackball ID #%d out of range (0..3)!", i);
		Warning(buf);
		return;
	}
	lightBall[i] = ball;
}

void Scene::SetupObjectTrackball( int i, Trackball *ball )
{
	if (i < 0 || i > 3)
	{
		char buf[255];
		sprintf( buf, "Object trackball ID #%d out of range (0..3)!", i);
		Warning(buf);
		return;
	}
	objBall[i] = ball;
}

// Check if this scene file keyword opens a block (that needs closure
//    with a matching "end" line).  Technically, this need not return
//    true on *all* block keywords, only those that can be nested inside
//    other blocks.
bool Scene::IsBlockKeyword( char *keyword )
{
	if (!strcmp(keyword, "object") || !strcmp(keyword, "material") || !strcmp(keyword, "instance"))
		return true;
	return false;
}

void Scene::UnhandledKeyword( FILE *f, char *typeString, char *keyword )
{
	char buf[ MAXLINELENGTH ];
	if (typeString && keyword)
	{
		sprintf(buf, "Currently unhandled %s: '%s'!", typeString, keyword);
		Warning( buf );
	}

	// Search the scene file.
	char token[256], *ptr;
	while( fgets(buf, MAXLINELENGTH, f) != NULL )  
	{
		ptr = StripLeadingWhiteSpace( buf );
		if (ptr[0] == '#') continue;
		ptr = StripLeadingTokenToBuffer( ptr, token );
		MakeLower( token );
		if (!strcmp(token,"end")) break;
		else if (IsBlockKeyword( token ))
			UnhandledKeyword( f );
	}
}

// A constructor to read a scene from a file
Scene::Scene( char *filename, bool verbose ) : 
	camera(0), geometry(0), screenWidth(256), screenHeight(256), verbose(verbose),
	sceneFileDataAccessed(false), lightIntensityModifier(0), extinctionCoef(0), frameTime(0),cullingDisabled(false)
{
	// HACK!
	scene = this;
	// END HACK!

	paths = new ProgramSearchPaths();
	SetupDefaultScenePaths();
	ui = 0;
	sceneAmbient[0] = sceneAmbient[1] = sceneAmbient[2] = 0.05; sceneAmbient[3] = 1.0;

	printf("    (-) Loading scene from '%s'...\n", filename);
	char buf[ MAXLINELENGTH ], token[256], *ptr;
	FILE *sceneFile = paths->OpenScene( filename, "r" );
	if (!sceneFile) 
		FatalError( "Scene::Scene() unable to open scene file '%s'!", filename );

	// Setup a default material type, in case objects don't define their own...
	Material *defaultMatl = new GLMaterial( "__sceneDefaultMaterial" );
	fileMaterials.Add( defaultMatl );

	// A group for all scene objects
	geometry = new Group();

	// Iterate through the lines of the file....
	int flag=0;
	while( fgets(buf, MAXLINELENGTH, sceneFile) != NULL )  
	{
		// Is this line a comment?
		ptr = StripLeadingWhiteSpace( buf );
		if (ptr == 0 || ptr[0] == '\n' || ptr[0] == '#') continue;

		// Nope.  So find out what the command is...
		ptr = StripLeadingTokenToBuffer( ptr, token );
		MakeLower( token );
	
		// Take different measures, depending on the command.  You will need to add more!
		if (!strcmp(token, "")) continue;
		else if (!strcmp(token, "epsilon"))
			Warning( "Keyword 'epsilon' ignored for OpenGL programs!" );
		else if (!strcmp(token, "keymap") || !strcmp(token, "uimap") || !strcmp(token, "controls"))
		{
			ui = new SceneDefinedUI( sceneFile );
			// UnhandledKeyword( sceneFile, "keyword", token );
		}
		else if (!strcmp(token, "scattering"))
			Warning( "Keyword 'scattering' currently ignored!" );
		else if (!strcmp(token, "float"))
		{
			char name[256];
			ptr = StripLeadingTokenToBuffer( ptr, name );
			MakeLower( name );
			ptr = StripLeadingTokenToBuffer( ptr, token );
			uiVars.Add( new UIFloat( atof( token ), name ) );
			if (sceneFileDataAccessed)
				Error("Cannot use *any* variables in scene file until *all* are defined!  Expect undefined behavior!");
		}
		else if (!strcmp(token, "int"))
		{
			char name[256];
			ptr = StripLeadingTokenToBuffer( ptr, name );
			MakeLower( name );
			ptr = StripLeadingTokenToBuffer( ptr, token );
			uiVars.Add( new UIInt( atoi( token ), name ) );
			if (sceneFileDataAccessed)
				Error("Cannot use *any* variables in scene file until *all* are defined!  Expect undefined behavior!");
		}
		else if (!strcmp(token, "noculling"))
			cullingDisabled = true;
		else if (!strcmp(token, "bool"))
		{
			bool value=false;
			char name[256];
			ptr = StripLeadingTokenToBuffer( ptr, name );
			MakeLower( name );
			ptr = StripLeadingTokenToBuffer( ptr, token );
			MakeLower( token );
			if (!strcmp(token,"t") || !strcmp(token,"true") || token[0]=='y' || token[0]=='1') 
				value = true;
			uiVars.Add( new UIBool( value, name ) );
			if (sceneFileDataAccessed)
				Error("Cannot use *any* variables in scene file until *all* are defined!  Expect undefined behavior!");
		}
		else if (!strcmp(token, "ambienttest"))
			Warning( "Keyword 'ambienttest' ignored for OpenGL programs!" );
		else if (!strcmp(token, "ambient"))
		{
			Color loadedAmbient( ptr );
			sceneAmbient[0] = loadedAmbient.Red();
			sceneAmbient[1] = loadedAmbient.Green();
			sceneAmbient[2] = loadedAmbient.Blue();
			sceneAmbient[3] = 1.0;
		}
		else if (!strcmp(token,"background"))
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			if (!strcmp(token,"color"))
				Warning( "Currently unhandled keyword: 'background'!" );
			else
				UnhandledKeyword( sceneFile, "keyword", "background" );
		}
		else if (!strcmp(token,"material"))
		{
			Material *matl = LoadMaterial( ptr, sceneFile );
			if (matl) fileMaterials.Add( matl );
		}
		else if (!strcmp(token,"object"))
		{
			Object *obj = LoadObject( ptr, sceneFile );
			if (obj) geometry->Add( obj );
		}
		else if ( (flag = !strcmp(token,"texture")) || !strcmp(token,"videotex"))
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			if ( GetNamedTexture( token ) )
				Error("Multiply defined texture, named %s!", token);
			else
			{
				char file[256];
				ptr = StripLeadingTokenToBuffer( ptr, file );
				char *fullPath = paths->GetTexturePath( file );
				GLTexture *tex = 0;
				if (flag)
					tex = new GLTexture( fullPath, TEXTURE_REPEAT_S | TEXTURE_REPEAT_T | TEXTURE_MIN_LINEAR_MIP_LINEAR, true );
				else
					tex = new GLVideoTexture( fullPath, 30.0, TEXTURE_REPEAT_S | TEXTURE_REPEAT_T | TEXTURE_MIN_LINEAR_MIP_LINEAR );
				tex->SetName( token );
				free( fullPath );
				fileTextures.Add( tex );
			}
		}
		else if (!strcmp(token,"directory") || !strcmp(token,"dir") || !strcmp(token,"path"))
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			for (int i=0;ptr[i];i++)
				if (ptr[i] == '\r' || ptr[i] == '\n' || ptr[i] == '\t')	ptr[i] = 0;

			MakeLower( token );
			if( !strcmp(token, "model") || !strcmp(token, "models")) 
				paths->AddModelPath( ptr );
			if (!strcmp(token, "texture") || !strcmp(token, "textures")) 
				paths->AddTexturePath( ptr );
			if (!strcmp(token, "shader") || !strcmp(token, "shaders")) 
				paths->AddShaderPath( ptr );
		}
		else if (!strcmp(token,"light"))
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			MakeLower( token );
			if( !strcmp(token, "point") ) 
				AddLight( new GLLight( sceneFile, this ) );
			else if( !strcmp(token, "area") ) 
				areas.Add( new AreaLight( sceneFile, this ) );
		}
		else if (!strcmp(token,"shadowmap"))
			UnhandledKeyword( sceneFile, "keyword", token );
		else if (!strcmp(token,"camera"))  
			SetCamera( LoadCamera( ptr, sceneFile) );
		else if (!strcmp(token,"frametime"))
		{
			ptr = StripLeadingTokenToBuffer( ptr, token );
			frameTime = atof( token );
		}
		else
			Error( "Unknown scene command '%s' in Scene::Scene()!", token );
	}

	if (!ui)
		ui = new SceneDefinedUI();

	if (!camera)
		FatalError("Scene file has no camera specification!");

	// If there are any trackballs, make sure they're visible to the interface routines.
	eyeBall = camera->GetTrackball();

	// Make sure the trackballs all have the correct size.
	if (eyeBall) eyeBall->ResizeTrackballWindow( screenWidth, screenHeight );
	for (int i=0; i<4; i++)
	{
		if (lightBall[i]) lightBall[i]->ResizeTrackballWindow( screenWidth, screenHeight );
		if (objBall[i])   objBall[i]->ResizeTrackballWindow( screenWidth, screenHeight );
	}

	// Copy variables from the UI into the scene so we can access them!
	if (verbose) printf("    (-) Initializing scene-file defined variables...\n");
	ui->CopyBoundVariables( &uiVars );

	lightIntensityModifier = (UIFloat *)this->GetSceneVariable( "lightintensity", new UIFloat( 1.0 ) );
	extinctionCoef = (UIFloat *)this->GetSceneVariable( "extinctioncoef", new UIFloat( 0.0 ) );

	// Clean up and close the file
	fclose( sceneFile );
	if (verbose) printf("    (-) Finished loading scene.\n");
}



Object *Scene::LoadObject( char *typeString, FILE *file )
{
	Object *objPtr=0;
	char token[256], *ptr;
	ptr = StripLeadingTokenToBuffer( typeString, token );
	MakeLower( token );	
	if (!strcmp(token,"sphere"))
		//UnhandledKeyword( file, "object type", token );
		objPtr = new Sphere( file, this );
	else if (!strcmp(token,"parallelogram"))
		objPtr = new Quad( file, this );
	else if (!strcmp(token,"texparallelogram"))
		objPtr = new Quad( file, this );
	else if (!strcmp(token,"testquad"))
		objPtr = new Quad( file, this );
	else if (!strcmp(token,"testdisplacedquad"))
		objPtr = new Quad( file, this );
	else if (!strcmp(token,"noisyquad"))
		objPtr = new Quad( file, this );
	else if (!strcmp(token,"triangle") || !strcmp(token,"tri"))
		objPtr = new Triangle( file, this );
	else if (!strcmp(token,"textri"))
		objPtr = new Triangle( file, this );
	else if (!strcmp(token,"cyl") || !strcmp(token,"cylinder"))
		objPtr = new Cylinder( file, this );
	else if (!strcmp(token,"mesh"))
	{	
		objPtr = new Mesh( ptr, file, this );
		ptr = StripLeadingTokenToBuffer( ptr, token ); // remove object-type from string (in case mesh is named)
	}
	else if (!strcmp(token,"import"))
	{	
		objPtr = new ImportedScene( ptr, file, this );
		ptr = StripLeadingTokenToBuffer( ptr, token ); // remove object-type from string (in case imported scene is named)
	}
	else if (!strcmp(token,"instance") || !strcmp(token,"group"))
		objPtr = new Group( file, this );
	else if (!strcmp(token,"plane"))
		Error("The InfinitePlane class has been removed.  Use a Parallelogram instead!");
	else
		FatalError("Unknown object type '%s' in LoadObject()!", token);

	// The next string on the line is a name for the current object.  Do you want this?
	ptr = StripLeadingTokenToBuffer( ptr, token );
	if ( token[0] && !IsWhiteSpace( token[0] ) && objPtr )
	{
		if (!objPtr->GetMaterial() || objPtr->GetMaterial() == fileMaterials[0] )
		{
			if (verbose) printf("    (-) Finished reading object '%s'...\n", token );
		}
		else if (verbose) 
			printf("    (-) Finished reading object '%s' with material '%s'...\n", 
					token, objPtr->GetMaterial()->GetName() );
		namedObjects.Add( objPtr );
		objectNames.Add( strdup( token ) );
	}
	return objPtr;
}

// Given the string (read from the file) after the 'camera' keyword, determine what
//    *type* of camera, load the camera, and return it.  
Camera *Scene::LoadCamera( char *typeString, FILE *file )
{
	Camera *cameraPtr;
	char token[256], *ptr;
	ptr = StripLeadingTokenToBuffer( typeString, token );
	MakeLower( token );	
	if (!strcmp(token,"pinhole"))
		cameraPtr = new Camera( file, this );
	else
	{
		Warning("Camera type '%' unhandled by OpenGL programs, trying pinhole camera!", token);
		cameraPtr = new Camera( file, this );
	}
	return cameraPtr;
}


// Given the string (read from the file) after the 'material' keyword, determine what
//    *type* of material it is, load that material, give it (optionally) a name, and
//    return it
Material *Scene::LoadMaterial( char *typeString, FILE *file )
{
	Material *matlPtr=0;
	char token[256], *ptr;
	ptr = StripLeadingTokenToBuffer( typeString, token );
	MakeLower( token );	
	if (!strcmp(token,"lambertian"))
		matlPtr = new GLLambertianMaterial( file, this );
	else if (!strcmp(token,"glmatl"))
		matlPtr = new GLMaterial( file, this );
	else if (!strcmp(token,"texmat") || !strcmp(token,"texturematerial") || !strcmp(token,"lambertiantexture"))
		matlPtr = new GLLambertianTexMaterial( file, this );
	else if (!strcmp(token,"fogtesttex"))
		matlPtr = new GLLambertianTexMaterial( file, this );
	else if (!strcmp(token,"lambertianpathtraced"))
		matlPtr = new GLLambertianMaterial( file, this );
	else if (!strcmp(token,"lambertianexplicitdirect"))
		matlPtr = new GLLambertianMaterial( file, this );
	else if (!strcmp(token,"shader"))
		matlPtr = new GLSLShaderMaterial( file, this );
	else if (!strcmp(token,"reflective"))
		UnhandledKeyword( file, "material type", token );
		//matlPtr = new ReflectiveMaterial( file, this );
	else if (!strcmp(token,"refractive"))
		UnhandledKeyword( file, "material type", token );
		//matlPtr = new RefractiveMaterial( file, this );
	else if (!strcmp(token,"constant"))
		matlPtr = new GLConstantMaterial( file, this );
	else if (!strcmp(token,"simpleemitter") || !strcmp(token,"ashikhminshirley") || 
		     !strcmp(token,"ashikhmin") || !strcmp(token,"brdf") || 
			 !strcmp(token,"fresnelreflective"))
		UnhandledKeyword( file, "material type", token );
	else
		FatalError("Unknown material type '%s'!", token);

	// The next string on the line is a material name.  Get that and set the internal name
	ptr = StripLeadingTokenToBuffer( ptr, token );
	MakeLower( token );	
	if (matlPtr) matlPtr->SetName( token );

	if (matlPtr && verbose) printf("    (-) Successfully read material '%s'\n", matlPtr->GetName() );

	return matlPtr;
}


// Check if a material is already in the list.  If so, return it.  Else return NULL;
Material *Scene::ExistingMaterialFromFile( char *name )
{
	MakeLower( name );	
	for (unsigned int i=0; i<fileMaterials.Size(); i++)
		if (fileMaterials[i]->GetName() && !strcmp(name, fileMaterials[i]->GetName()))
			return fileMaterials[i];
	return 0;
}

GLTexture *Scene::ExistingTextureFromFile( char *filename )
{
	for (unsigned int i=0; i<fileTextures.Size(); i++)
		if (!strcmp(filename, fileTextures[i]->GetFilename()))
			return fileTextures[i];
	return 0;
}

GLTexture *Scene::GetNamedTexture( char *name )
{
	for (unsigned int i=0; i<fileTextures.Size(); i++)
		if (!strcmp(name, fileTextures[i]->GetName()))
			return fileTextures[i];
	return 0;
}

// Check if a light is already in the list.  If so, return it.  Else return NULL;
GLLight *Scene::ExistingLightFromFile( char *name )
{
	MakeLower( name );	
	for (unsigned int i=0; i<light.Size(); i++)
		if (light[i]->GetName() && !strcmp(name, light[i]->GetName()))
			return light[i];
	return 0;
}

Object *Scene::ExistingObjectFromFile( char *name )
{
	for (unsigned int i=0; i<objectNames.Size(); i++)
		if (!strcmp(name, objectNames[i]))
			return namedObjects[i];
	return 0;	
}

UIVariable *Scene::GetSceneVariable( char *varName, UIVariable *defaultValue )
{
	UIVariable *retVal = defaultValue;
	sceneFileDataAccessed = true;
	for (unsigned int i=0; i<uiVars.Size(); i++)
		if (!strcmp(varName,uiVars[i]->GetName())) 
		{
			if (defaultValue) delete retVal;
			return uiVars[i];
		}
	return retVal;
}


bool Scene::ReloadShaders( void )
{
	bool ok = true;
	for (unsigned int i=0; i<fileShaders.Size(); i++)
		ok &= fileShaders[i]->ReloadShaders();
	return ok;
}


void Scene::Preprocess( void )
{
	// Setup a tesselator if needed by scene objects
	quadObj = gluNewQuadric();
	gluQuadricDrawStyle  ( quadObj, GLU_FILL );
	gluQuadricOrientation( quadObj, GLU_OUTSIDE );
	gluQuadricNormals    ( quadObj, GLU_SMOOTH );
	gluQuadricTexture    ( quadObj, GLU_TRUE );

	printf("(+) Preprocessing scene...\n");
	if (verbose) printf("    (-) Preprocessing scene geometry...\n");
	geometry->Preprocess( this );
	if (verbose) printf("    (-) Setting up scene textures...\n");
	for (unsigned int i=0; i<fileTextures.Size(); i++)
	{
		fileTextures[i]->Preprocess();
		if (fileTextures[i]->NeedsUpdates())
			dynamicTextures.Add( fileTextures[i] );
	}
	if (verbose) printf("    (-) Checking if materials need preprocessing...\n");
	for (unsigned int i=0; i<fileMaterials.Size(); i++)
	{
		if (fileMaterials[i]->NeedsPreprocessing()) 
			fileMaterials[i]->Preprocess( this );
	}

	// A shader that is used for reflective shadow maps.  Outputs just the
	//    texture color to one color channel, the normal to another, and the
	//    linear depth to an alpha component.
	drawOnlyTexture = new GLSLProgram( "onlyTexture.vert.glsl", 
									   NULL, 
									   "onlyTexture.frag.glsl", 
									   true, scene->paths->GetShaderPathList() );

	// It's possible that the scene file defined a frametime != 0, in which case
	//    we have to update the scene to accomodate this time.  This needs to be
	//    the last thing before finishing the preprocess.
	scene->PerFrameUpdate( );

	if (verbose) printf("(+) Done with Scene::Preprocess()!\n");

	gluDeleteQuadric( quadObj );
}





