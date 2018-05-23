/*****************************************************/
/* lean Logging                 (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_LOGGING
#define LEAN_LOGGING

namespace lean
{
	/// Provides utility functions and classes, automatically logging raised exceptions,
	/// redirecting logged output to various targets, enhancing logged informtion by
	/// source information (caller file and line), etc.
	/// @see LoggingMacros
	/// @see ExceptionMacros
	namespace logging { }
}

#include "streamconv.h"

#include "log_target.h"
#include "log_file.h"
#include "log_debugger.h"
#include "log_stream.h"

#include "log.h"

#include "errors.h"

#endif