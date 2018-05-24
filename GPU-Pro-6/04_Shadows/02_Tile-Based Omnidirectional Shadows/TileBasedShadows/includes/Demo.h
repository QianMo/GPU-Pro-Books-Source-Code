#ifndef DEMO_H
#define DEMO_H

#include <FileManager.h>
#include <Window.h>
#include <TimeManager.h>
#include <InputManager.h>
#include <ResourceManager.h>
#include <OGL_Renderer.h>

// Demo
//
// Simple demo framework, offering global access of static members.
class Demo
{
public:
  static bool Create();

  static void Release();

  // manager for file operations
  static FileManager *fileManager;

  // application window
  static Window *window;

  // manager for timing
  static TimeManager *timeManager;

  // manager for input (keyboard/ mouse)
  static InputManager *inputManager;

  // OpenGL renderer
  static OGL_Renderer *renderer;

  // manager for resources
  static ResourceManager *resourceManager;

};

#endif