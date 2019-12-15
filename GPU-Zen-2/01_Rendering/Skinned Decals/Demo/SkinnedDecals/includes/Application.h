#ifndef APPLICATION_H
#define APPLICATION_H

class Camera;
class Font;
class Shading;
class DirectionalLight;
class DemoModel;

// Application
//
class Application
{
public:
  Application():
    shadingPP(nullptr),
    mainCamera(nullptr),
    defaultFont(nullptr),
    dirLight(nullptr),
    model(nullptr),
    quit(false),
    showHelp(false),
    showSettings(false)
  {
    ResetSettings();
  }

  void ResetSettings();

  bool Init();

  void Run();

  void Shutdown();

  void Quit();

private:
  bool OnInit();

  void OnRun();

  void HandleInput();

  void DisplayInfo();

  Shading *shadingPP;
  Camera *mainCamera;
  POINT prevMousePos;
  Font *defaultFont;
  DirectionalLight *dirLight;
  DemoModel *model;

  UINT decalMaterialIndex;
  float decalWidth;
  float decalHeight;
  float decalAngle;
  float dirLightDirX;
  float dirLightDirY;
  float dirLightDirZ;

  bool quit;
  bool showHelp;
  bool showSettings;
  bool showProfiling;
  bool pauseAnim;
  bool useDecals;
  bool debugDecalMask;

};

#endif