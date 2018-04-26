#include "DXUT.h"
#include "RenderCueable.h"
#include "Cueable.h"
#include "RenderContext.h"

RenderCueable::RenderCueable(Cueable* cue, Cueable* cameraCue, const Role role)
:role(role)
{
	this->cue = cue;
	this->cameraCue = cameraCue;
}

void RenderCueable::execute(const TaskContext& context)
{
	Camera* camera = NULL;
	if(cameraCue)
		camera = cameraCue->getCamera();
	cue->render(RenderContext(context.theatre, context.localResourceOwner, camera, NULL, role, 1));
}