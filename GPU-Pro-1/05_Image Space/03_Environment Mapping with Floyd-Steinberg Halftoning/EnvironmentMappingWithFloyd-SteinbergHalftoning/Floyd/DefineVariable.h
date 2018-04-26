#pragma once
#include "Task.h"
#include "ScriptVariableClass.h"

class DefineVariable :
	public Task
{
	const ScriptVariableClass& type;
	const std::wstring name;
public:
	DefineVariable(const ScriptVariableClass& type,	const wchar_t* name);
	void execute(const TaskContext& context);
};
