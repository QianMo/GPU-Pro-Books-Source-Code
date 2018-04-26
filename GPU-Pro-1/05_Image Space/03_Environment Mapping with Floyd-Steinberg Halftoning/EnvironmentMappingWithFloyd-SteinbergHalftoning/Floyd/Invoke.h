#pragma once
#include "Task.h"

class Invoke :
	public Task
{
public:
	virtual void execute(const TaskContext& context)=0;
	virtual ~Invoke(void);
};
