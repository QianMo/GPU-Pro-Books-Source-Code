#include "DXUT.h"
#include "Act.h"
#include "Play.h"
#include "Cueable.h"
#include "RenderContext.h"
#include "MessageContext.h"
#include "ControlContext.h"
#include "XMLparser.h"
#include "Theatre.h"

Act::Act(Play* play, XMLNode& actNode)
{
	this->play = play;

//	initializationTaskBlock.loadTasks(play, actNode.getChildNode(L"initialization"), play);
//	taskBlock.loadTasks(play, actNode.getChildNode(L"rendering"), play);
	loadTaskBlocks(play, actNode);
	loadAnimationTasks(actNode.getChildNode(L"animation"));
	loadControlTasks(actNode.getChildNode(L"control"));
	loadMessageTasks(actNode.getChildNode(L"messaging"));

	executeEventTasks(EventType::createAct, TaskContext(play->getTheatre(), this));
//	initializationTaskBlock.execute(play->getTheatre(), this);
}

Act::~Act(void)
{

}

void Act::loadAnimationTasks(XMLNode& scriptNode)
{
	int iAnimationTask = 0;
	XMLNode animationTaskNode;
	while( !(animationTaskNode = scriptNode.getChildNode(L"animate", iAnimationTask)).isEmpty() )
	{
		const wchar_t* cue = animationTaskNode|L"cue";

		if(cue)
		{
			Cueable* cueable = play->getCueable(cue);
			if(cueable)
				animationTaskList.push_back( AnimationTask(cueable));
		}
		iAnimationTask++;
	}
}

void Act::loadControlTasks(XMLNode& scriptNode)
{
	int iControlTask = 0;
	XMLNode ControlTaskNode;
	while( !(ControlTaskNode = scriptNode.getChildNode(L"control", iControlTask)).isEmpty() )
	{
		const wchar_t* cue = ControlTaskNode|L"cue";
		const wchar_t* interactorCue = ControlTaskNode|L"interactorCue";

		if(cue)
		{
			Cueable* cueable = play->getCueable(cue);
			Cueable* interactorCueable = NULL;
			if(interactorCue)
				interactorCueable = play->getCueable(interactorCue);
			if(cueable)
				controlTaskList.push_back( ControlTask(cueable, interactorCueable));
		}
		iControlTask++;
	}
}

void Act::loadMessageTasks(XMLNode& scriptNode)
{
	int iMessageTask = 0;
	XMLNode messageTaskNode;
	while( !(messageTaskNode = scriptNode.getChildNode(L"message", iMessageTask)).isEmpty() )
	{
		const wchar_t* cue = messageTaskNode|L"cue";
		const wchar_t* cameraCue = messageTaskNode|L"cameraCue";
		if(cue)
		{
			Cueable* cueable = play->getCueable(cue);
			Cueable* cameraCueable = NULL;
			if(cameraCue)
				cameraCueable = play->getCueable(cameraCue);
			if(cueable)
				messageTaskList.push_back( MessageTask(cueable, cameraCueable));
		}
		iMessageTask++;
	}
}

void Act::processMessage( const MessageContext& context)
{
	MessageTaskList::iterator iMessageTask = messageTaskList.begin();
	while(iMessageTask != messageTaskList.end())
	{
		iMessageTask->execute(context);
//		if(context.trapped) break;
		iMessageTask++;
	}
}

void Act::animate(double dt, double t)
{
	AnimationTaskList::iterator iAnimationTask = animationTaskList.begin();
	while(iAnimationTask != animationTaskList.end())
	{
		iAnimationTask->cue->animate(dt, t);
		iAnimationTask++;
	}
}

void Act::control(const ControlStatus& status, double dt)
{
	ControlTaskList::iterator iControlTask = controlTaskList.begin();
	while(iControlTask != controlTaskList.end())
	{
		Node* interactors = NULL;
		if(iControlTask->interactorCue)
			interactors = iControlTask->interactorCue->getInteractors();
		iControlTask->cue->control(ControlContext(status, dt, interactors));
		iControlTask++;
	}
}

void Act::render()
{
	executeEventTasks(EventType::renderFrame, TaskContext(play->getTheatre(), this));
//	taskBlock.execute(play->getTheatre(), this);
}

ResourceSet* Act::getResourceSet()
{
	return play->getResourceSet();
}
