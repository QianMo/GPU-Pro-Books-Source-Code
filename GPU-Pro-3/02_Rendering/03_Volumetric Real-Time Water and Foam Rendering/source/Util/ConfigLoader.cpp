#include <iostream>
#include <fstream>
#include <string>

#include "../Util/ConfigLoader.h"
#include "../Util/Math.h"

#include "../Main/DemoManager.h"

#include <GL/glut.h>


// -----------------------------------------------------------------------------
// ----------------------- ConfigLoader::ConfigLoader --------------------------
// -----------------------------------------------------------------------------
ConfigLoader::ConfigLoader(void):
		screenWidth(640),
		screenHeight(480),
		bitsPerPixel(32),
		refreshRate(60),
		fullScreen(0),
		sceneIndex(0)
{
}

// -----------------------------------------------------------------------------
// ----------------------- ConfigLoader::~ConfigLoader -------------------------
// -----------------------------------------------------------------------------
ConfigLoader::~ConfigLoader(void)
{

}


// -----------------------------------------------------------------------------
// ----------------------- ConfigLoader::LoadConfigFile ------------------------
// -----------------------------------------------------------------------------
void ConfigLoader::LoadConfigFile(const char* filename)
{
	screenWidth	 = 640;
	screenHeight = 480;
	bitsPerPixel = 32;
	refreshRate  = 60;
	fullScreen   = 0;
	sceneIndex = 0;

	// open config file
	std::ifstream file(filename);
	if(file) {
		// loop until end of file
		while(!file.eof()) {
			char input[128];
			file >> input;

			// check what we found and store
			if (input == string("screenWidth"))
			{
				file >> screenWidth;
			}
			else if (input == string("screenHeight"))
			{
				file >> screenHeight;
			}
			else if (input == string("bitsPerPixel"))
			{
				file >> bitsPerPixel;
			}
			else if (input == string("refreshRate"))
			{
				file >> refreshRate;
			}
			else if (input == string("fullScreen"))
			{
				file >> fullScreen;
			}
			else if (input == string("windowName"))
			{
				file >> windowName;
			}
			else if (input == string("sceneIndex"))
			{
				file >> sceneIndex;
			}
			
		}
		// close config file
		file.close();

		fullScreen = Math::Clamp(fullScreen, 0, 1);

		sceneIndex = Math::Clamp(sceneIndex, 0, DemoManager::NUM_SCENES-1);

		if (fullScreen)
		{
			if (!(((screenWidth == 1600) && (screenHeight == 1200)) ||
				((screenWidth == 1280) && (screenHeight == 1024)) ||
				((screenWidth == 1024) && (screenHeight == 768)) ||
				((screenWidth == 800) && (screenHeight == 600)) ||
				((screenWidth == 640) && (screenHeight == 480))))
			{
				//screenWidth = 640;
				//screenHeight = 480;

				screenWidth = glutGet(GLUT_SCREEN_WIDTH);
				screenHeight = glutGet(GLUT_SCREEN_HEIGHT);
			}
		}
	}
}