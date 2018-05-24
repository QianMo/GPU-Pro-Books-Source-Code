#include <stdafx.h>
#include <Demo.h>

FileManager* Demo::fileManager = NULL;
Window* Demo::window = NULL;
TimeManager* Demo::timeManager = NULL;
InputManager* Demo::inputManager = NULL;
OGL_Renderer* Demo::renderer = NULL;
ResourceManager* Demo::resourceManager = NULL;

bool Demo::Create()
{
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

  renderer = new OGL_Renderer;
  if(!renderer)
    return false;

  resourceManager = new ResourceManager;
  if(!resourceManager)
    return false;

  return true;
}

void Demo::Release()
{
  SAFE_DELETE(resourceManager);
  SAFE_DELETE(renderer);
  SAFE_DELETE(inputManager);
  SAFE_DELETE(timeManager);
  SAFE_DELETE(window);
  SAFE_DELETE(fileManager);
}
