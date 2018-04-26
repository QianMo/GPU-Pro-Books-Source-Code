#pragma once

class XMLNode;
class Play;
class ResourceOwner;
#include "Task.h"
#include "Directory.h"
typedef CompositList<Task*> TaskList;

class TaskBlock
{
	TaskList taskList;
public:
	TaskBlock(void);
	~TaskBlock(void);
	void loadTasks(Play* play, XMLNode& scriptNode, ResourceOwner* localResourceOwner);
	void execute(const TaskContext& context);
};
