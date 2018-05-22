#include <stdafx.h>
#include <DEMO.h>
#include <PATH_POINT_LIGHT.h>

float PATH_POINT_LIGHT::controlPoints[4] = { 0.0f,0.0f,0.0f,0.0f };

bool PATH_POINT_LIGHT::Init(const VECTOR3D &position,float radius,const COLOR &color,float multiplier,const VECTOR3D &direction)
{
	pointLight = DEMO::renderer->CreatePointLight(position,radius,color,multiplier);
  if(!pointLight)
		return false;
	this->direction = direction;
	return true;
}

void PATH_POINT_LIGHT::Update()
{
	if((!pointLight->IsActive())||(paused))
		return;

	VECTOR3D position = pointLight->GetPosition();
	if(position.x > controlPoints[1])
	{
		position.x = controlPoints[1];
		direction.Set(0.0f,0.0f,-1.0f);
	}
	if(position.x < controlPoints[0])
	{
		position.x = controlPoints[0];
		direction.Set(0.0f,0.0f,1.0f);
	}
	if(position.z > controlPoints[3])
	{
		position.z = controlPoints[3];
		direction.Set(1.0f,0.0f,0.0f);
	}
	if(position.z < controlPoints[2])
	{
		position.z = controlPoints[2];
		direction.Set(-1.0f,0.0f,0.0f);
	}

	// prevent large values at beginning of application
	float frameInterval = (float)DEMO::timeManager->GetFrameInterval();
  if(frameInterval>1000.0f)
    return;

	position += direction*frameInterval*PATH_POINTLIGHT_MOVE_SPEED;
	pointLight->SetPosition(position);
}
