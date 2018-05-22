#ifndef DEMO_H
#define DEMO_H

#include <FILE_MANAGER.h>
#include <WINDOW.h>
#include <TIME_MANAGER.h>
#include <INPUT_MANAGER.h>
#include <RESOURCE_MANAGER.h>
#include <DX11_RENDERER.h>

// DEMO
//   Simple demo framework, offering global access of static members.
class DEMO
{
public:
  static bool Create();

	static void Release();

	// manager for file operations
	static FILE_MANAGER *fileManager;

	// application window
	static WINDOW *window;

	// manager for timing
	static TIME_MANAGER *timeManager;

	// manager for input (keyboard/ mouse)
	static INPUT_MANAGER *inputManager;

	// DirectX 11 renderer
	static DX11_RENDERER *renderer;

	// manager for resources
	static RESOURCE_MANAGER *resourceManager;

};

#endif