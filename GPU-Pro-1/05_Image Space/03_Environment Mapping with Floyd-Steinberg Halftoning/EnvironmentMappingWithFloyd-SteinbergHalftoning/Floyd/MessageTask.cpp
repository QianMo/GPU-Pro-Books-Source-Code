#include "DXUT.h"
#include "MessageTask.h"
#include "MessageContext.h"
#include "Cueable.h"
#include "TaskContext.h"

MessageTask::MessageTask(Cueable* cue, Cueable* cameraCue)
{
	this->cue = cue;
	this->cameraCue = cameraCue;
}


void MessageTask::execute(const TaskContext& context)
{
	const MessageContext* messageContext = context.asMessageContext();
	Camera* camera = messageContext->camera;
	if(cameraCue)
		camera = cameraCue->getCamera();
	cue->processMessage(MessageContext(
		context.theatre, context.localResourceOwner, 
		messageContext->controlStatus, 
		camera,
		messageContext->hWnd,
		messageContext->uMsg,
		messageContext->wParam,
		messageContext->lParam,
		messageContext->trapped));
}