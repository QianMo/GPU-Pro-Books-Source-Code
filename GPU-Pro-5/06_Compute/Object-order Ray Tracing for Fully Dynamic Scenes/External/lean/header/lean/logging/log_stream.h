/*****************************************************/
/* lean Logging                 (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_LOGGING_LOG_STREAM
#define LEAN_LOGGING_LOG_STREAM

#include "../lean.h"
#include "../strings/types.h"
#include "../concurrent/spin_lock.h"
#include "log_target.h"
#include <ostream>

namespace lean
{
namespace logging
{

/// Log stream class that prints any given input to a given stream.
/// As streams are not guaranteed to be thread-safe, it is recommended
/// to only ever hold one wrapper for any given stream at a time.
template <class Char = char, class Traits = std::char_traits<Char> >
class basic_log_stream : public log_target
{
public:
	/// Type of the stream wrapped.
	typedef ::std::basic_ostream<Char, Traits> stream_type;

private:
	stream_type &m_stream;

	spin_lock<> m_printLock;

public:
	/// Wraps the given stream for logging.
	explicit basic_log_stream(stream_type *stream)
		: m_stream(*stream) { };
	
	/// Prints the given message to the log stream. This method is thread-safe.
	void print(const char_ntri &message)
	{
		scoped_sl_lock lock(m_printLock);
		// Use streaming operator to allow for implicit character widening
		m_stream << message.c_str();
	}
};

/// Std stream logging wrapper.
/// @see lean::logging::basic_log_stream
typedef basic_log_stream<> log_stream;
/// Wide-char std stream logging wrapper.
/// @see lean::logging::basic_log_stream
typedef basic_log_stream<wchar_t> wlog_stream;

} // namespace

using logging::basic_log_stream;

using logging::log_stream;
using logging::wlog_stream;

} // namespace

#endif