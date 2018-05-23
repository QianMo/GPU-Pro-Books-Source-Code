/*****************************************************/
/* lean Logging                 (c) Tobias Zirr 2011 */
/*****************************************************/

// Opaque value requires this to go here
#ifdef LEAN_INCLUDE_LINKED
	#ifndef LEAN_LOGGING_LOG_FILE_CPP
		#define LEAN_LOGGING_LOG_FILE_CPP
		#include "source/log_file.cpp"
	#endif
#endif

#ifndef LEAN_LOGGING_LOG_FILE
#define LEAN_LOGGING_LOG_FILE

#include "../lean.h"
#include "../strings/types.h"
#include "../io/win_types.h"
#include "log_target.h"

namespace lean
{
namespace logging
{

/// Log file class that prints any given input to a given file.
class log_file : public noncopyable, public log_target
{
private:
	windows_file_handle m_handle;

public:
	/// Opens the given file for logging.
	LEAN_MAYBE_EXPORT explicit log_file(const utf8_ntri &name);
	/// Closes the log file.
	LEAN_MAYBE_EXPORT ~log_file();

	/// Gets whether the file was opened successfully.
	LEAN_MAYBE_EXPORT bool valid() const;
	
	/// Prints the given message to the log file. This method is thread-safe.
	LEAN_MAYBE_EXPORT void print(const char_ntri &message);
};

} // namespace

using logging::log_file;

} // namespace

#endif