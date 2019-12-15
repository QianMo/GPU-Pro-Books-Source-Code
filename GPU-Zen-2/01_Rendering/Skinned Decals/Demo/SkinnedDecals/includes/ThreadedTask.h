#ifndef THREADED_TASK_H
#define THREADED_TASK_H

enum taskStates
{
  TASK_FINISHED=0,
	TASK_PENDING,
	TASK_IN_PROGRESS
};

// ThreadedTask
//
class ThreadedTask
{
public:
	friend class ThreadManager;

	ThreadedTask():
	  taskState(TASK_FINISHED)
	{
	}

	virtual void Run()=0;

	taskStates GetState() const
	{
    return taskState;
	}

private:
  volatile taskStates taskState;

};

#endif