#pragma once

class TaskContext;

class Task
{
public:
	virtual void execute(const TaskContext& context)=0;
	virtual ~Task();
};
