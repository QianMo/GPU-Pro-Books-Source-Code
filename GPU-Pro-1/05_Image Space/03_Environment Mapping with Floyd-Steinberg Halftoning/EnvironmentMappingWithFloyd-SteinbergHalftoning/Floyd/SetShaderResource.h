#pragma once
#include "Task.h"

class ScriptShaderResourceViewVariable;

class SetShaderResource :
	public Task
{
	ID3D10EffectShaderResourceVariable* effectVariable;
	ScriptShaderResourceViewVariable* srv;
public:
	SetShaderResource(ID3D10EffectShaderResourceVariable* effectVariable, ScriptShaderResourceViewVariable* srv);
	void execute(const TaskContext& context);
};
