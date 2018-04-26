#include "DXUT.h"
#include "SceneManager.h"
#include "Scene.h"

SceneManager::SceneManager(Theatre* theatre)
:Cueable(theatre)
{
	this->scene = NULL;
}

SceneManager::~SceneManager(void)
{
}

void SceneManager::render(const RenderContext& context)
{
	scene->render(context);
}

void SceneManager::animate(double dt, double t)
{
	scene->animate(dt, t);
}

void SceneManager::control(const ControlContext& context)
{
	scene->control(context);
}

void SceneManager::processMessage( const MessageContext& context)
{
	scene->processMessage(context);
}
