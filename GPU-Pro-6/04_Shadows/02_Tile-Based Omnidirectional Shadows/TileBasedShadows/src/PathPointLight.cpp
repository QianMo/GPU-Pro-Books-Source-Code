#include <stdafx.h>
#include <Demo.h>
#include <TiledDeferred.h>
#include <PathPointLight.h>

#define PATH_POINTLIGHT_MOVE_SPEED 0.1f
#define NUM_COLORS 12

Aabb PathPointLight::boundingBox;

static const Color colors[NUM_COLORS] = 
{ 
  Color(1.0f, 0.0f, 0.0f),
  Color(0.0f, 1.0f, 0.0f),
  Color(0.0f, 0.0, 1.0f),
  Color(1.0f, 1.0f, 0.0f),
  Color(1.0f, 0.5f, 0.0f),
  Color(0.5f, 1.0f, 0.0f),
  Color(0.0f, 1.0f, 1.0f),
  Color(0.0f, 1.0f, 0.5f),
  Color(0.0f, 0.5f, 1.0f),
  Color(1.0f, 0.0f, 1.0f),
  Color(1.0f, 0.0f, 0.5f),
  Color(0.5f, 0.0f, 1.0f)
};

static float GetRandomNumber(float maxValue, bool symmetric)
{
  if(symmetric)
    return ((((rand()%10000)-5000)/5000.0f)*maxValue); 
  else
    return (((rand()%10000)/10000.0f)*maxValue);
}

bool PathPointLight::Init()
{
  Vector3 position;
  position.x = GetRandomNumber(1200.0f, true);
  position.y = GetRandomNumber(900.0f, false)+50.0f;
  position.z = GetRandomNumber(550.0f, true);

  TiledDeferred *tiledDeferredPP = (TiledDeferred*)Demo::renderer->GetPostProcessor("TiledDeferred");
  if(!tiledDeferredPP)
    return false;

  pointLight = tiledDeferredPP->CreatePointLight(position, 256.0f, colors[rand() % NUM_COLORS]*2.5f);
  if(!pointLight)
    return false;
  
  direction.x = GetRandomNumber(1.0f, true);
  direction.y = GetRandomNumber(1.0f, true);
  direction.z = GetRandomNumber(1.0f, true);
  direction.Normalize();

  return true;
}

void PathPointLight::Update()
{
  if((!pointLight->IsActive()) || (paused))
    return;

  Vector3 position = pointLight->GetPosition();
  if(position.x < boundingBox.mins.x)
  {
    position.x = boundingBox.mins.x;
    direction.x = -direction.x;
  }
  if(position.x > boundingBox.maxes.x)
  {
    position.x = boundingBox.maxes.x;
    direction.x = -direction.x;
  }
  if(position.y < boundingBox.mins.y)
  {
    position.y = boundingBox.mins.y;
    direction.y = -direction.y;
  }
  if(position.y > boundingBox.maxes.y)
  {
    position.y = boundingBox.maxes.y;
    direction.y = -direction.y;
  }
  if(position.z < boundingBox.mins.z)
  {
    position.z = boundingBox.mins.z;
    direction.z = -direction.z;
  }
  if(position.z > boundingBox.maxes.z)
  {
    position.z = boundingBox.maxes.z;
    direction.z = -direction.z;
  }

  // prevent large values at beginning of application
  float frameInterval = (float)Demo::timeManager->GetFrameInterval();
  if(frameInterval > 1000.0f)
    return;

  position += direction*frameInterval*PATH_POINTLIGHT_MOVE_SPEED;
  pointLight->SetPosition(position);
}
