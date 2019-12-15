#ifndef DEMO_H
#define DEMO_H

#include <LogManager.h>
#include <ThreadManager.h>
#include <FileManager.h>
#include <Window.h>
#include <TimeManager.h>
#include <InputManager.h>
#include <DX12_Renderer.h>
#include <ResourceManager.h>
#include <GuiManager.h>

// Demo
//
// Simple demo framework, offering global access of static members.
class Demo
{
public:
  static bool Create();

  static void Release();

  static LogManager *logManager;

	static ThreadManager *threadManager;

  static FileManager *fileManager;

  static Window *window;

  static TimeManager *timeManager;

  static InputManager *inputManager;

  static DX12_Renderer *renderer;

  static ResourceManager *resourceManager;

  static GuiManager *guiManager;

};

#endif