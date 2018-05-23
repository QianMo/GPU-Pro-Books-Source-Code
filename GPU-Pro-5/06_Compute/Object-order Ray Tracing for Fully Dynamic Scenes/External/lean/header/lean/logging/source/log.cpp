#ifdef LEAN_BUILD_LIB
#include "../../depconfig.h"
#include "../log_details.h"
#endif

#include <sstream>
#include <algorithm>
#include "../log_debugger.h"

namespace lean
{
namespace logging
{

struct log_details::string_storage
{
	string_stream stream;
	string_type string;
};

// Destructor.
LEAN_ALWAYS_LINK log_details::log_details(log_target *initialTarget)
{
	if (initialTarget)
		m_targets.push_back(initialTarget);
}

// Constructor.
LEAN_ALWAYS_LINK log_details::~log_details()
{
	for (stream_vector::iterator itStream = m_streams.begin();
		itStream != m_streams.end(); ++itStream)
		delete *itStream;
}

// Adds the given target to this log. This method is thread-safe.
LEAN_ALWAYS_LINK void log_details::add_target(log_target *target)
{
	if (target != nullptr)
	{
		scoped_ssl_lock lock(m_targetLock);
		m_targets.push_back(target);
	}
}

// Removes the given target from this log. This method is thread-safe.
LEAN_ALWAYS_LINK void log_details::remove_target(log_target *target)
{
	if (target != nullptr)
	{
		scoped_ssl_lock lock(m_targetLock);
		m_targets.erase(
			std::remove(m_targets.begin(), m_targets.end(), target),
			m_targets.end() );
	}
}

// Prints the given message. This method is thread-safe.
LEAN_ALWAYS_LINK void log_details::print(const char_ntri &message)
{
	scoped_ssl_lock_shared lock(m_targetLock);

	for (target_vector::const_iterator itTarget = m_targets.begin();
		itTarget != m_targets.end(); ++itTarget)
		(*itTarget)->print(message);
}

// Acquires a stream to write to. This method is thread-safe.
LEAN_ALWAYS_LINK log_details::output_stream& log_details::acquireStream()
{
	scoped_sl_lock lock(m_streamLock);

	if (m_freeStreams.empty())
	{
		std::auto_ptr<string_storage> newStorage(new string_storage());

		// Make sure m_freeStreams.capacity() >= m_streams.size()
		// + Never add invalid element to m_streams
		m_freeStreams.reserve(m_streams.size() + 1);
		m_streams.push_back(newStorage.get());

		return newStorage.release()->stream;
	}
	else
	{
		output_stream *stream = m_freeStreams.back();
		m_freeStreams.pop_back();
		return *stream;
	}
}

// Prints the contents of the given stream and releases the stream for further re-use. This method is thread-safe.
LEAN_ALWAYS_LINK void log_details::flushAndReleaseStream(output_stream &stream)
{
	string_storage& storage = reinterpret_cast<string_storage&>(static_cast<string_stream&>(stream));

	// Retrieve message
	storage.stream.clear();
	storage.string.resize((size_t) storage.stream.tellp());
	storage.stream.seekg(0);
	storage.stream.read(&storage.string[0], storage.string.size());

	// Print stream message
	print(storage.string);

	// Reset stream
	storage.stream.str(std::string());
	storage.stream.clear();

	// Release stream (if reset successfully)
	{
		scoped_sl_lock lock(m_streamLock);
		m_freeStreams.push_back(&stream);
	}
}

// Gets the error log.
LEAN_ALWAYS_LINK log& error_log()
{
	static log_details errorLog(log_debugger::get_if_attached());
	return errorLog;
}

// Gets the debug log.
LEAN_ALWAYS_LINK log& debug_log()
{
	static log_details debugLog(log_debugger::get_if_attached());
	return debugLog;
}

// Gets the info log.
LEAN_ALWAYS_LINK log& info_log()
{
	static log_details infoLog(nullptr);
	return infoLog;
}

} // namespace
} // namespace