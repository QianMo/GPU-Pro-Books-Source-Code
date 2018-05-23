/*****************************************************/
/* lean Logging                 (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_LOGGING_LOG_TARGET
#define LEAN_LOGGING_LOG_TARGET

#include "../lean.h"
#include "../strings/types.h"
#include "../tags/noncopyable.h"

namespace lean
{
namespace logging
{

class log;

/// Log target interface.
class LEAN_INTERFACE log_target
{
	LEAN_INTERFACE_BEHAVIOR(log_target)

public:
	/// Prints the given message.
	virtual void print(const char_ntri &message) = 0;
};

} // namespace

using logging::log_target;

} // namespace

#endif