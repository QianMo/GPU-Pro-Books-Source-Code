#pragma once
#include "Task.h"

class ScriptRenderTargetViewVariable;

class ClearTarget :
	public Task
{
	ScriptRenderTargetViewVariable* rtv;
	D3DXVECTOR4 color;
public:
	ClearTarget(ScriptRenderTargetViewVariable* rtv, const D3DXVECTOR4& color);
	void execute(const TaskContext& context);
};
