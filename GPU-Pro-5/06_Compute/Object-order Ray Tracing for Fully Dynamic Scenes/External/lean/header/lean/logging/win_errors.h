/*****************************************************/
/* lean Logging                 (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_LOGGING_WIN_ERRORS
#define LEAN_LOGGING_WIN_ERRORS

#include "../lean.h"
#include "../strings/types.h"
#include "../strings/conversions.h"
#include "../logging/errors.h"
#include <sstream>
#include "streamconv.h"

namespace lean
{
namespace logging
{

/// Gets an error message describing the last WinAPI error that occurred. Returns the number of characters used.
LEAN_MAYBE_EXPORT size_t get_last_win_error_msg(utf16_t *buffer, size_t maxCount);

/// Gets an error message describing the last WinAPI error that occurred.
template <class String>
String get_last_win_error_msg();

#ifndef DOXYGEN_SKIP_THIS

/// Gets an error message describing the last WinAPI error that occurred.
template <>
inline utf16_string get_last_win_error_msg()
{
	utf16_string msg;
	msg.resize(1024);
	msg.erase(get_last_win_error_msg(&msg[0], msg.size()));
	return msg;
}
/// Gets an error message describing the last WinAPI error that occurred.
template <>
inline utf8_string get_last_win_error_msg()
{
	return utf_to_utf8(get_last_win_error_msg<utf16_string>());
}

#endif

/// Logs the last WinAPI error.
template <class String1>
LEAN_NOTINLINE void log_last_win_error(const String1 &source)
{
	log_error(source, get_last_win_error_msg<utf8_string>().c_str());
}
/// Logs the last WinAPI error.
template <class String1, class String2>
LEAN_NOTINLINE void log_last_win_error(const String1 &source, const String2 &reason)
{
	log_error_ex(source, get_last_win_error_msg<utf8_string>().c_str(), reason);
}
/// Logs the last WinAPI error.
template <class String1, class String2, class String3>
LEAN_NOTINLINE void log_last_win_error(const String1 &source, const String2 &reason, const String3 &context)
{
	log_error_ex(source, get_last_win_error_msg<utf8_string>().c_str(), reason, context);
}

/// Throws a runtime_error exception containing the last WinAPI error.
template <class String1>
LEAN_NOTINLINE void throw_last_win_error(const String1 &source)
{
	throw_error(source, get_last_win_error_msg<utf8_string>().c_str());
}
/// Throws a runtime_error exception containing the last WinAPI error.
template <class String1, class String2>
LEAN_NOTINLINE void throw_last_win_error(const String1 &source, const String2 &reason)
{
	throw_error_ex(source, get_last_win_error_msg<utf8_string>().c_str(), reason);
}
/// Throws a runtime_error exception containing the last WinAPI error.
template <class String1, class String2, class String3>
LEAN_NOTINLINE void throw_last_win_error(const String1 &source, const String2 &reason, const String3 &context)
{
	throw_error_ex(source, get_last_win_error_msg<utf8_string>().c_str(), reason, context);
}

} // namespace

using logging::log_last_win_error;
using logging::throw_last_win_error;
using logging::get_last_win_error_msg;

} // namespace

/// @addtogroup LoggingMacros
/// @see lean::logging
/// @{

/// Logs an error message, prepending the caller's file and line.
#define LEAN_LOG_WIN_ERROR() ::lean::logging::log_last_win_error(__FILE__ " (" LEAN_QUOTE_VALUE(__LINE__) ")")
/// Logs the given error message, prepending the caller's file and line.
#define LEAN_LOG_WIN_ERROR_MSG(msg) ::lean::logging::log_last_win_error(__FILE__ " (" LEAN_QUOTE_VALUE(__LINE__) ")", msg)
/// Logs the given error message and context, prepending the caller's file and line.
#define LEAN_LOG_WIN_ERROR_CTX(msg, ctx) ::lean::logging::log_last_win_error(__FILE__ " (" LEAN_QUOTE_VALUE(__LINE__) ")", msg, ctx)
/// Logs the given error message and context, prepending the caller's file and line.
#define LEAN_LOG_WIN_ERROR_ANY(msg) LEAN_THROW_WIN_ERROR_MSG(static_cast<::std::ostringstream&>(::std::ostringstream() << msg).str().c_str())

/// @}

/// @addtogroup ExceptionMacros
/// @see lean::logging
/// @{

/// Throws a runtime_error exception.
#define LEAN_THROW_WIN_ERROR() ::lean::logging::throw_last_win_error(__FILE__ " (" LEAN_QUOTE_VALUE(__LINE__) ")")
/// Throws a runtime_error exception.
#define LEAN_THROW_WIN_ERROR_MSG(msg) ::lean::logging::throw_last_win_error(__FILE__ " (" LEAN_QUOTE_VALUE(__LINE__) ")", msg)
/// Throws a runtime_error exception.
#define LEAN_THROW_WIN_ERROR_CTX(msg, ctx) ::lean::logging::throw_last_win_error(__FILE__ " (" LEAN_QUOTE_VALUE(__LINE__) ")", msg, ctx)
/// Throws a runtime_error exception.
#define LEAN_THROW_WIN_ERROR_ANY(msg) LEAN_THROW_WIN_ERROR_MSG(static_cast<::std::ostringstream&>(::std::ostringstream() << msg).str().c_str())

/// @}

#ifdef LEAN_INCLUDE_LINKED
#include "source/win_errors.cpp"
#endif

#endif