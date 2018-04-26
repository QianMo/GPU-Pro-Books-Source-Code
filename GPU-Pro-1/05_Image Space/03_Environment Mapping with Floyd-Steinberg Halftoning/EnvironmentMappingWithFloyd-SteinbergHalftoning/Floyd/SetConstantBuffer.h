#pragma once
#include "task.h"


class ScriptResourceVariable;

class SetConstantBuffer :
	public Task
{
	ID3D10EffectConstantBuffer* effectConstantBuffer;
	ScriptResourceVariable* resource;
public:
	SetConstantBuffer(ID3D10EffectConstantBuffer* effectConstantBuffer, ScriptResourceVariable* resource);
	void execute(const TaskContext& context);
};