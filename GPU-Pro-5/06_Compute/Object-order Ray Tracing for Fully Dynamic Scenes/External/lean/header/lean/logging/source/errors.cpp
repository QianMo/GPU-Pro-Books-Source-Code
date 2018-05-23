#ifdef LEAN_BUILD_LIB
#include "../../depconfig.h"
#endif

#include "../errors.h"
#include <stdexcept>
#include "../log.h"
#include "../../io/numeric.h"
#include "../../strings/utility.h"

namespace lean
{
namespace logging
{
namespace impl
{

	/// Makes the given source string valid in any case.
	inline const char* make_source_valid(const char *source)
	{
		return (source) ? source : "Unknown source";
	}

} // namespace
} // namespace
} // namespace

// Throws a runtime_error exception.
LEAN_ALWAYS_LINK LEAN_NORETURN void lean::logging::throw_error(const char *source)
{
	// Store valid source, re-used in throw clause!
	source = impl::make_source_valid(source);
	log_stream_out(error_log()) << source << ": An error occurred." << std::endl;
	throw std::runtime_error(source);
}

// Throws a runtime_error exception.
LEAN_ALWAYS_LINK LEAN_NORETURN void lean::logging::throw_error(const char *source, const char *reason)
{
	if (!reason)
		throw_error(source);

	log_stream_out(error_log()) << impl::make_source_valid(source) << ": An error occurred: " << reason << std::endl;
	throw std::runtime_error(reason);
}

// Throws a runtime_error exception.
LEAN_ALWAYS_LINK LEAN_NORETURN void lean::logging::throw_error(const char *source, const char *reason, const char *context)
{
	if (!context)
		throw_error(source, reason);
	if (!reason)
		throw_error(source, context);

	log_stream_out(error_log()) << impl::make_source_valid(source) << ": An error occurred: " << reason << " (" << context << ")" << std::endl;
	throw std::runtime_error(reason);
}

// Throws a runtime_error exception.
LEAN_ALWAYS_LINK LEAN_NORETURN void lean::logging::throw_error_ex(const char *source, const char *reason, const char *origin)
{
	if (!origin)
		throw_error(source, reason);
	if (!reason)
		throw_error(source, origin);

	log_stream_out(error_log()) << impl::make_source_valid(source) << ": An error occurred: " << reason << " << " << origin << std::endl;
	throw std::runtime_error(reason);
}

// Throws a runtime_error exception.
LEAN_ALWAYS_LINK LEAN_NORETURN void lean::logging::throw_error_ex(const char *source, const char *reason, const char *origin, const char *context)
{
	if (!context)
		throw_error_ex(source, reason, origin);
	if (!origin)
		throw_error(source, reason, context);
	if (!reason)
		throw_error(source, origin, context);

	log_stream_out(error_log()) << impl::make_source_valid(source) << ": An error occurred: " << reason
		<< " << " << origin << " (" << context << ")" << std::endl;
	throw std::runtime_error(reason);
}

// Throws an invalid_argument exception.
LEAN_ALWAYS_LINK LEAN_NORETURN void lean::logging::throw_invalid(const char *source)
{
	// Store valid source, re-used in throw clause!
	source = impl::make_source_valid(source);
	log_stream_out(error_log()) << source << ": Invalid argument." << std::endl;
	throw std::invalid_argument(source);
}

// Throws an invalid_argument exception.
LEAN_ALWAYS_LINK LEAN_NORETURN void lean::logging::throw_invalid(const char *source, const char *reason)
{
	if (!reason)
		throw_invalid(source);

	log_stream_out(error_log()) << impl::make_source_valid(source) << ": Invalid argument: " << reason << std::endl;
	throw std::invalid_argument(reason);
}

// Throws a bad_alloc exception.
LEAN_ALWAYS_LINK LEAN_NORETURN void lean::logging::throw_bad_alloc(const char *source)
{
	char msg[2048];
	size_t msgLen = 0;

	// Log without allocating further memory
	msgLen += strmcpy(msg + msgLen, impl::make_source_valid(source), sizeof(msg) - msgLen);
	msgLen += strmcpy(msg + msgLen, ": Out of memory.\n", sizeof(msg) - msgLen);
	
	msg[msgLen - 1] = '\n';
	error_log().print(msg);
	
	throw std::bad_alloc();
}

// Throws a bad_alloc exception.
LEAN_ALWAYS_LINK LEAN_NORETURN void lean::logging::throw_bad_alloc(const char *source, size_t size)
{
	char msg[2048];
	size_t msgLen = 0;

	// Log without allocating further memory
	msgLen += strmcpy(msg + msgLen, impl::make_source_valid(source), sizeof(msg) - msgLen);
	msgLen += strmcpy(msg + msgLen, ": Out of memory while allocating ", sizeof(msg) - msgLen);

	if (sizeof(msg) - msgLen > max_int_string_length<size_t>::value)
	{
		msgLen = int_to_char(msg + msgLen, size) - msg;
		msgLen += strmcpy(msg + msgLen, " bytes.\n", sizeof(msg) - msgLen);
	}
	
	msg[msgLen - 1] = '\n';
	error_log().print(msg);

	throw std::bad_alloc();
}

// Logs an error.
LEAN_ALWAYS_LINK void lean::logging::log_error(const char *source)
{
	log_stream_out(error_log()) << impl::make_source_valid(source) << ": An error occurred." << std::endl;
}

// Logs an error.
LEAN_ALWAYS_LINK void lean::logging::log_error(const char *source, const char *reason)
{
	if (!reason)
		return log_error(source);

	log_stream_out(error_log()) << impl::make_source_valid(source) << ": An error occurred: " << reason << std::endl;
}

// Logs an error.
LEAN_ALWAYS_LINK void lean::logging::log_error(const char *source, const char *reason, const char *context)
{
	if (!context)
		return log_error(source, reason);
	if (!reason)
		return log_error(source, context);

	log_stream_out(error_log()) << impl::make_source_valid(source) << ": An error occurred: " << reason << " (" << context << ")" << std::endl;
}

// Logs an error.
LEAN_ALWAYS_LINK void lean::logging::log_error_ex(const char *source, const char *reason, const char *origin)
{
	if (!origin)
		return log_error(source, reason);
	if (!reason)
		return log_error(source, origin);

	log_stream_out(error_log()) << impl::make_source_valid(source) << ": An error occurred: " << reason << " << " << origin << std::endl;
}

// Logs an error.
LEAN_ALWAYS_LINK void lean::logging::log_error_ex(const char *source, const char *reason, const char *origin, const char *context)
{
	if (!context)
		return log_error_ex(source, reason, origin);
	if (!origin)
		return log_error(source, reason, context);
	if (!reason)
		return log_error(source, origin, context);

	log_stream_out(error_log()) << impl::make_source_valid(source) << ": An error occurred: " << reason
		<< " << " << origin << " (" << context << ")" << std::endl;
}
