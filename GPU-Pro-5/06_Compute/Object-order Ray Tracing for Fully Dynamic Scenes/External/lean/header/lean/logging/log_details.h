/*****************************************************/
/* lean Logging                 (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_LOGGING_LOGDETAILS
#define LEAN_LOGGING_LOGDETAILS

#include "log.h"
#include "../tags/noncopyable.h"
#include "../strings/types.h"
#include "../concurrent/spin_lock.h"
#include "../concurrent/shareable_spin_lock.h"
#include <vector>
#include <iosfwd>
#include <ostream>

namespace lean
{
namespace logging
{

/// Log class.
class log_details : public log
{
friend class log_stream_out;

private:
	typedef std::basic_stringstream<char> string_stream;
	
	struct string_storage;
	typedef std::vector<string_storage*> stream_vector;
	stream_vector m_streams;

	typedef std::vector<output_stream*> free_stream_vector;
	free_stream_vector m_freeStreams;
	spin_lock<> m_streamLock;

	typedef std::vector<log_target*> target_vector;
	target_vector m_targets;
	shareable_spin_lock<> m_targetLock;

	/// Acquires a stream to write to. This method is thread-safe.
	LEAN_MAYBE_EXPORT output_stream& acquireStream();
	/// Prints the contents of the given stream and releases the stream for further re-use. This method is thread-safe.
	LEAN_MAYBE_EXPORT void flushAndReleaseStream(output_stream &stream);

public:
	/// Constructor.
	LEAN_MAYBE_EXPORT log_details(log_target *initialTarget = nullptr);
	/// Destructor.
	LEAN_MAYBE_EXPORT ~log_details();

	/// Adds the given target to this log. This method is thread-safe.
	LEAN_MAYBE_EXPORT void add_target(log_target *target);
	/// Removes the given target from this log. This method is thread-safe.
	LEAN_MAYBE_EXPORT void remove_target(log_target *target);

	/// Prints the given message. This method is thread-safe.
	LEAN_MAYBE_EXPORT void print(const char_ntri &message);
};

} // namespace

using logging::log_details;

} // namespace

#ifdef LEAN_INCLUDE_LINKED
#include "source/log.cpp"
#endif

#endif