#pragma once
#include "Task.h"
#include "Role.h"

class Cueable;

class RenderCueable :
	public Task
{
	Cueable* cue;
	Cueable* cameraCue;
	const Role role;
public:
	RenderCueable(Cueable* cue, Cueable* cameraCue, const Role role);
	void execute(const TaskContext& context);
};
