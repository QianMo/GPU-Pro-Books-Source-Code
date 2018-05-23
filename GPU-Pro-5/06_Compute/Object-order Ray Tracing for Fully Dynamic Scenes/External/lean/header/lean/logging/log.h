/*****************************************************/
/* lean Logging                 (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_LOGGING_LOG
#define LEAN_LOGGING_LOG

#include "../lean.h"
#include "../tags/noncopyable.h"
#include "../strings/types.h"

#include <ostream>

namespace lean
{
namespace logging
{

class log_target;
class log_stream_out;

/// Log class.
class LEAN_INTERFACE log
{
	LEAN_INTERFACE_BEHAVIOR(log)

friend class log_stream_out;

public:
	typedef std::basic_ostream<char> output_stream;
	typedef std::basic_string<char> string_type;

	/// Acquires a stream to write to. This method is thread-safe.
	virtual output_stream& acquireStream() = 0;
	/// Prints the contents of the given stream and releases the stream for further re-use. This method is thread-safe.
	virtual void flushAndReleaseStream(output_stream &stream) = 0;

public:
	/// Adds the given target to this log. This method is thread-safe.
	virtual void add_target(log_target *target) = 0;
	/// Removes the given target from this log. This method is thread-safe.
	virtual void remove_target(log_target *target) = 0;

	/// Prints the given message. This method is thread-safe.
	virtual void print(const char_ntri &message) = 0;
};

/// Log stream class.
class log_stream_out : public noncopyable
{
private:
	log &m_log;
	log::output_stream &m_stream;

public:
	/// Output stream type.
	typedef log::output_stream output_stream;
	/// Basic output stream type.
	typedef std::basic_ostream<output_stream::char_type, output_stream::traits_type> basic_output_stream;
	/// Basic output ios type.
	typedef std::basic_ios<output_stream::char_type, output_stream::traits_type> basic_output_ios;

	/// Constructs a new TEMPORARY log stream from the log.
	explicit log_stream_out(log &log)
		: m_log(log),
		m_stream(log.acquireStream()) { }
	/// Prints the contents of this log stream and releases it for further re-use.
	~log_stream_out()
	{
		m_log.flushAndReleaseStream(m_stream);
	}

	/// Outputs the given value to this log stream.
	template <class Value>
	log_stream_out& operator <<(const Value &value)
	{
		m_stream << value;
		return *this;
	}
	// Passes the given manipulator to this log stream.
	log_stream_out& operator <<(std::ios_base& (*manip)(::std::ios_base&))
	{
		m_stream << manip;
		return *this;
	}
	// Passes the given manipulator to this log stream.
	log_stream_out& operator<<(basic_output_stream& (*manip)(basic_output_stream&))
	{
		m_stream << manip;
		return *this;
	}
	// Passes the given manipulator to this log stream.
	log_stream_out& operator<<(basic_output_ios& (*manip)(basic_output_ios&))
	{
		m_stream << manip;
		return *this;
	}

	/// Gets the underlying output stream.
	output_stream& std()
	{
		return m_stream;
	}
};

/// Gets the error log.
LEAN_MAYBE_EXPORT log& error_log();
/// Gets the debug log.
LEAN_MAYBE_EXPORT log& debug_log();
/// Gets the info log.
LEAN_MAYBE_EXPORT log& info_log();

struct error_stream : public log_stream_out
{
	error_stream()
		: log_stream_out(error_log()) { }
};

struct debug_stream : public log_stream_out
{
	debug_stream()
		: log_stream_out(debug_log()) { }
};

struct info_stream : public log_stream_out
{
	info_stream()
		: log_stream_out(info_log()) { }
};

} // namespace

using logging::log;
using logging::log_stream_out;

using logging::error_log;
using logging::debug_log;
using logging::info_log;

using logging::error_stream;
using logging::debug_stream;
using logging::info_stream;

} // namespace

/// @addtogroup LoggingMacros Logging macros
/// @see lean::logging
/// @{

/// Logs the given message, prepending the caller's file and line.
#define LEAN_LOG(msg) ::lean::log_stream_out(::lean::info_log()) << LEAN_SOURCE_STRING << ": " << msg << ::std::endl
/// Adds a line break.
#define LEAN_LOG_BREAK() ::lean::log_stream_out(::lean::info_log()) << ::std::endl

/// Logs the given error message, prepending the caller's file and line.
#define LEAN_LOG_ERROR(msg) ::lean::log_stream_out(::lean::error_log()) << LEAN_SOURCE_STRING << ": " << msg << ::std::endl
/// Adds a line break.
#define LEAN_LOG_ERROR_BREAK() ::lean::log_stream_out(::lean::error_log()) << ::std::endl

/// @}

#endif