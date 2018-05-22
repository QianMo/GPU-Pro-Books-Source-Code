#ifndef APPLICATION_H
#define APPLICATION_H

#include <PATH_POINT_LIGHT.h>

#define MOUSE_SPEED 0.5f
#define CAMERA_MOVE_SPEED 0.2f
#define DIRLIGHT_MOVE_SPEED 0.00005f
#define NUM_PATH_POINT_LIGHTS 12

class CAMERA;
class FONT;
class GLOBAL_ILLUM;
class DIRECTIONAL_LIGHT;

// APPLICATION
//   Demo application, that demonstrate the "Rasterized Voxel based Dynamic Global Illumination"
//   technique.
class APPLICATION
{
public:
  APPLICATION()
	{
		quit = false;
		mainCamera = NULL;
    defaultFont = NULL;
		globalIllum = NULL;
		dirLight = NULL;
		pathLightsAnimated = true;
		pathLigthsEnabled = true;
		showHelp = false;
	}

	bool Init();

	void Run();

	void Shutdown();

	void Quit();

private:
	bool OnInit();
	
	void OnRun();
	
	void OnShutdown();  
	
	void HandleInput();

	void DisplayInfo();

	bool quit;
	CAMERA *mainCamera;
	FONT *defaultFont;
	GLOBAL_ILLUM *globalIllum;
	DIRECTIONAL_LIGHT *dirLight;
	PATH_POINT_LIGHT pathPointLights[NUM_PATH_POINT_LIGHTS];
	bool pathLigthsEnabled;
	bool pathLightsAnimated;
	bool showHelp;
	
};

#endif