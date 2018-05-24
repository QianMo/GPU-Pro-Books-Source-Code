#ifndef APPLICATION_H
#define APPLICATION_H

#include <PathPointLight.h>

class Camera;
class Font;
class DemoMesh;
class TiledDeferred;

// Application
//
class Application
{
public:
  Application():
    tiledDeferredPP(NULL),
    mainCamera(NULL),
    defaultFont(NULL),
    numActiveLights(MAX_NUM_POINT_LIGHTS/2),
    quit(false),
    pathLightsAnimated(true),
    showHelp(false),
    showSettings(false),
    showProfilingInfo(false)
  {
  }

  bool Init();

  void Run();

  void Shutdown();

  void Quit();

  void SetNumActiveLights(unsigned int numActiveLights);

  unsigned int GetNumActiveLights() const
  {
    return numActiveLights;
  }

  void EnablePathLightAnimation(bool enable);

  bool IsPathLightAnimationEnabled() const
  {
    return pathLightsAnimated;
  }

  void ShowProfilingInfo(bool show)
  {
    showProfilingInfo = show;
  }

  bool IsProfilingInfoShown() const
  {
    return showProfilingInfo;
  }

private:
  bool OnInit();

  void OnRun();

  void HandleInput();

  void DisplayInfo();

  TiledDeferred *tiledDeferredPP;
  Camera *mainCamera;
  POINT prevMousePos;
  Font *defaultFont;
  PathPointLight pathPointLights[MAX_NUM_POINT_LIGHTS];
  unsigned int numActiveLights;
  bool quit:1;
  bool pathLightsAnimated:1;
  bool showHelp:1;
  bool showSettings:1;
  bool showProfilingInfo:1;

};

#endif