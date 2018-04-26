#pragma once

class ResourceSet;
class TaskBlock;
class Theatre;
class Play;
class XMLNode;
class TaskContext;
#include "EventType.h"
#include "Directory.h"

class ResourceOwner
{
	typedef CompositMap<EventType, TaskBlock*> TaskBlockDirectory;
	TaskBlockDirectory taskBlockDirectory;
public:
	~ResourceOwner();
	/// Returns the resource set of the resource owner. (Creating one if necessary.)
	virtual ResourceSet* getResourceSet()=0;

	void executeEventTasks(const EventType& eventType, TaskContext& context);

	virtual ResourceOwner* getParentResourceOwner(Theatre* theatre);

	void loadTaskBlock(Play* play, XMLNode& scriptNode);
	void loadTaskBlocks(Play* play, XMLNode& ownerNode);
};
