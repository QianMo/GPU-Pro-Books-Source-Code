#pragma once
#include "Task.h"

class ScriptResourceVariable;

class SetVertexBuffers :
	public Task
{
	std::vector<ScriptResourceVariable*> buffers;
	std::vector<unsigned int> offsets;
	std::vector<unsigned int> strides;
public:
	SetVertexBuffers();
	void addBuffer(ScriptResourceVariable* buffer, unsigned int stride, unsigned int offset);
	void execute(const TaskContext& context);
};
