/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_TASK
#define BE_CORE_TASK

#include "beCore.h"

namespace beCore
{

/// Task interface.
class LEAN_INTERFACE Task
{
	LEAN_INTERFACE_BEHAVIOR(Task)

public:
	/// Runs the task.
	BE_CORE_API virtual void Run() = 0;
	/// Calls Run() to run the task.
	LEAN_INLINE void operator ()() { Run(); }
};

} // namespace

#endif