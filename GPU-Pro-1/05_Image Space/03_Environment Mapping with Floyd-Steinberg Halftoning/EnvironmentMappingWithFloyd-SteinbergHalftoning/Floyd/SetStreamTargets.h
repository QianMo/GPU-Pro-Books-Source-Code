#pragma once
#include "Task.h"

class ScriptResourceVariable;

class SetStreamTargets :
	public Task
{
	std::vector<ScriptResourceVariable*> buffers;
	std::vector<unsigned int> offsets;
public:
	SetStreamTargets();
	void addBuffer(ScriptResourceVariable* buffer, unsigned int offset);
	void execute(const TaskContext& context);
};
