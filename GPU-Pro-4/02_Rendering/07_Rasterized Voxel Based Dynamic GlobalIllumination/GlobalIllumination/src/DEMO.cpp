#include <stdafx.h>
#include <DEMO.h>

FILE_MANAGER* DEMO::fileManager = NULL;
WINDOW* DEMO::window = NULL;
TIME_MANAGER* DEMO::timeManager = NULL;
INPUT_MANAGER* DEMO::inputManager = NULL;
DX11_RENDERER* DEMO::renderer = NULL;
RESOURCE_MANAGER* DEMO::resourceManager = NULL;

bool DEMO::Create()
{
	fileManager = new FILE_MANAGER;
	if(!fileManager)
		return false;

	window = new WINDOW;
	if(!window)
    return false;

	timeManager = new TIME_MANAGER;
	if(!timeManager)
		return false;

	inputManager = new INPUT_MANAGER;
	if(!inputManager)
		return false;

  renderer = new DX11_RENDERER;
	if(!renderer)
		return false;

  resourceManager = new RESOURCE_MANAGER;
	if(!resourceManager)
		return false;

  return true;
}

void DEMO::Release()
{
	SAFE_DELETE(resourceManager);
	SAFE_DELETE(renderer);
	SAFE_DELETE(inputManager);
  SAFE_DELETE(timeManager);
	SAFE_DELETE(window);
	SAFE_DELETE(fileManager);
}
