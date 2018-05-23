#ifdef LEAN_BUILD_LIB
#include "../../depconfig.h"
#endif

#include "../log_debugger.h"
#include <windows.h>

// Gets the log target instance.
LEAN_ALWAYS_LINK lean::logging::log_debugger& lean::logging::log_debugger::get()
{
	static log_debugger debugger;
	return debugger;
}

// Gets the log target instance.
LEAN_ALWAYS_LINK lean::logging::log_debugger* lean::logging::log_debugger::get_if_attached()
{
	return (IsDebuggerPresent())
		? &get()
		: nullptr;
}

// Prints the given message to the debug output window. This method is thread-safe.
LEAN_ALWAYS_LINK void lean::logging::print_debugger(const char_ntri &message)
{
	// Thread-safe: http://www.unixwiz.net/techtips/outputdebugstring.html
	::OutputDebugStringA(message.c_str());
}
