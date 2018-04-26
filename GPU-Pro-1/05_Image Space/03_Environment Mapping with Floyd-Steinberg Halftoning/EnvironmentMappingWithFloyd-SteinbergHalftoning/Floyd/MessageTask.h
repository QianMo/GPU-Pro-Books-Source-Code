#pragma once
#include "Task.h"

class Cueable;
class MessageContext;

class MessageTask : public Task
{
	friend class Act;
	Cueable* cue;
	Cueable* cameraCue;
public:
	MessageTask(Cueable* cue, Cueable* cameraCue);
	void execute(const TaskContext& context);
};

