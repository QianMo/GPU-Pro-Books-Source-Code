#include "DXUT.h"
#include "ResourceOwner.h"
#include "TaskBlock.h"
#include "Theatre.h"
#include "xmlParser.h"
#include "TaskContext.h"

ResourceOwner::~ResourceOwner()
{
	taskBlockDirectory.deleteAll();
}

void ResourceOwner::executeEventTasks(const EventType& eventType, TaskContext& context)
{
	ResourceOwner* parent = getParentResourceOwner(context.theatre);
	if(parent)
	{
		TaskContext parentContext = context;
		parentContext.localResourceOwner = parent;
		parent->executeEventTasks(eventType, parentContext);
	}
	TaskBlockDirectory::iterator i = taskBlockDirectory.find(eventType);
	if(i != taskBlockDirectory.end())
	{
		i->second->execute(context);
	}
}

ResourceOwner* ResourceOwner::getParentResourceOwner(Theatre* theatre)
{
	return theatre->getPlay();
}

void ResourceOwner::loadTaskBlock(Play* play, XMLNode& scriptNode)
{
	const wchar_t* eventName = scriptNode|L"onEvent";
	const EventType& eventType = EventType::fromString(eventName);
	TaskBlockDirectory::iterator i = taskBlockDirectory.find(eventType);
	if(i != taskBlockDirectory.end())
		delete i->second;
	taskBlockDirectory[eventType] = new TaskBlock();
	taskBlockDirectory[eventType]->loadTasks(play, scriptNode, this);
}

void ResourceOwner::loadTaskBlocks(Play* play, XMLNode& ownerNode)
{
	int iScript = 0;
	XMLNode scriptNode;
	while( !(scriptNode = ownerNode.getChildNode(L"Script", iScript)).isEmpty() )
	{
		loadTaskBlock(play, scriptNode);
		iScript++;
	}
}