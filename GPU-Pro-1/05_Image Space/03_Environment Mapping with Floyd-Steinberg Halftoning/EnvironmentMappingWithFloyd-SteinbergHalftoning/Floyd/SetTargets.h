#pragma once
#include "Task.h"

class ScriptRenderTargetViewVariable;
class ScriptDepthStencilViewVariable;

class SetTargets :
	public Task
{
	std::vector<ScriptRenderTargetViewVariable*> rtvs;	//< might contain NULL
	ScriptDepthStencilViewVariable* dsv;				//< might be NULL
public:
	SetTargets(ScriptDepthStencilViewVariable* dsv);
	void addRenderTargetView(ScriptRenderTargetViewVariable* rtv);
	void execute(const TaskContext& context);
};
