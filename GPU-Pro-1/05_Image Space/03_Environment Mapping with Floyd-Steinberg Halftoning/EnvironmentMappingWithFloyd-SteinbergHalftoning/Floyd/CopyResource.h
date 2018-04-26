#pragma once
#include "task.h"

class ScriptResourceVariable;

class CopyResource :
	public Task
{
	ScriptResourceVariable* source;
	ScriptResourceVariable* destination;
public:
	CopyResource(ScriptResourceVariable* source, ScriptResourceVariable* destination);

	void execute(const TaskContext& context);
};
