#include <stdafx.h>
#include <Demo.h>

LogManager* Demo::logManager = nullptr;
ThreadManager* Demo::threadManager = nullptr;
FileManager* Demo::fileManager = nullptr;
Window* Demo::window = nullptr;
TimeManager* Demo::timeManager = nullptr;
InputManager* Demo::inputManager = nullptr;
DX12_Renderer* Demo::renderer = nullptr;
ResourceManager* Demo::resourceManager = nullptr;
GuiManager* Demo::guiManager = nullptr;

bool Demo::Create()
{
  logManager = new LogManager;
	if(!logManager)
		return false;

  threadManager = new ThreadManager;
	if(!threadManager)
		return false;

  fileManager = new FileManager;
  if(!fileManager)
    return false;

  window = new Window;
  if(!window)
    return false;

  timeManager = new TimeManager;
  if(!timeManager)
    return false;

  inputManager = new InputManager;
  if(!inputManager)
    return false;

  resourceManager = new ResourceManager;
  if(!resourceManager)
    return false;

  renderer = new DX12_Renderer;
  if (!renderer)
    return false;

  guiManager = new GuiManager;
  if(!guiManager)
    return false;

  return true;
}

void Demo::Release()
{
  SAFE_DELETE(guiManager);
  SAFE_DELETE(renderer);
  SAFE_DELETE(resourceManager);
  SAFE_DELETE(inputManager);
  SAFE_DELETE(timeManager);
  SAFE_DELETE(window);
  SAFE_DELETE(fileManager);
  SAFE_DELETE(threadManager);
  SAFE_DELETE(logManager);
}
