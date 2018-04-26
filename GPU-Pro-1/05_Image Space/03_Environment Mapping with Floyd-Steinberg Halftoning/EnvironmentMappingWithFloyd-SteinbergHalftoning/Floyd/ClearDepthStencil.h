#pragma once
#include "Task.h"

class ScriptDepthStencilViewVariable;

class ClearDepthStencil :
	public Task
{
	ScriptDepthStencilViewVariable* dsv;
	float depth;
	unsigned char stencil;
	unsigned int flags;
public:
	ClearDepthStencil(ScriptDepthStencilViewVariable* dsv, unsigned int flags, float depth, unsigned char stencil);
	void execute(const TaskContext& context);
};
