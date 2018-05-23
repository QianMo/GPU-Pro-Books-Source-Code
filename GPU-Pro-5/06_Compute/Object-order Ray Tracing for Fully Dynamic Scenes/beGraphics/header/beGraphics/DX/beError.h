/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_ERROR_DX
#define BE_GRAPHICS_ERROR_DX

#include "beGraphics.h"
#include <lean/strings/conversions.h>
#include <lean/logging/errors.h>

namespace beGraphics
{

namespace DX
{

/// Gets an error message describing the given DirectX error.
BE_GRAPHICS_API const utf8_t* GetDXError(HRESULT error);

/// Gets an error message describing the given DirectX error.
template <class String>
String GetDXError(HRESULT error);

#ifndef DOXYGEN_SKIP_THIS

/// Gets an error message describing the given DirectX error.
template <>
inline utf16_string GetDXError(HRESULT error)
{
	return lean::strings::utf_to_utf16( GetDXError(error) );
}
/// Gets an error message describing the given DirectX error.
template <>
inline utf8_string GetDXError(HRESULT error)
{
	return utf8_string( GetDXError(error) );
}

#endif

/// Logs the given DirectX error.
template <class String1>
LEAN_NOTINLINE void LogDXError(HRESULT error, const String1 &source)
{
	lean::log_error(source, GetDXError(error));
}
/// Logs the given DirectX error.
template <class String1, class String2>
LEAN_NOTINLINE void LogDXError(HRESULT error, const String1 &source, const String2 &reason)
{
	lean::log_error_ex(source, GetDXError(error), reason);
}
/// Logs the given DirectX error.
template <class String1, class String2, class String3>
LEAN_NOTINLINE void LogDXError(HRESULT error, const String1 &source, const String2 &reason, const String3 &context)
{
	lean::log_error_ex(source, GetDXError(error), reason, context);
}

/// Logs the given DirectX error on failure.
template <class String1>
LEAN_INLINE bool GuardLogDXError(HRESULT error, const String1 &source)
{
	return FAILED(error)
		? (LogDXError(error, source), false)
		: true;
}
/// Logs the given DirectX error on failure.
template <class String1, class String2>
LEAN_INLINE bool GuardLogDXError(HRESULT error, const String1 &source, const String2 &reason)
{
	return FAILED(error)
		? (LogDXError(error, source, reason), false)
		: true;
}
/// Logs the given DirectX error on failure.
template <class String1, class String2, class String3>
LEAN_INLINE bool GuardLogDXError(HRESULT error, const String1 &source, const String2 &reason, const String3 &context)
{
	return FAILED(error)
		? (LogDXError(error, source, reason, context), false)
		: true;
}


/// Throws a runtime_error exception containing the given DirectX error.
template <class String1>
LEAN_NOTINLINE void ThrowDXError(HRESULT error, const String1 &source)
{
	lean::throw_error(source, GetDXError(error));
}
/// Throws a runtime_error exception containing the given DirectX error.
template <class String1, class String2>
LEAN_NOTINLINE void ThrowDXError(HRESULT error, const String1 &source, const String2 &reason)
{
	lean::throw_error_ex(source, GetDXError(error), reason);
}
/// Throws a runtime_error exception containing the given DirectX error.
template <class String1, class String2, class String3>
LEAN_NOTINLINE void ThrowDXError(HRESULT error, const String1 &source, const String2 &reason, const String3 &context)
{
	lean::throw_error_ex(source, GetDXError(error), reason, context);
}

/// Throws a runtime_error exception containing the given DirectX error on failure.
template <class String1>
LEAN_INLINE void GuardThrowDXError(HRESULT error, const String1 &source)
{
	if (FAILED(error))
		ThrowDXError(error, source);
}
/// Throws a runtime_error exception containing the given DirectX error on failure.
template <class String1, class String2>
LEAN_INLINE void GuardThrowDXError(HRESULT error, const String1 &source, const String2 &reason)
{
	if (FAILED(error))
		ThrowDXError(error, source, reason);
}
/// Throws a runtime_error exception containing the given DirectX error on failure.
template <class String1, class String2, class String3>
LEAN_INLINE void GuardThrowDXError(HRESULT error, const String1 &source, const String2 &reason, const String3 &context)
{
	if (FAILED(error))
		ThrowDXError(error, source, reason, context);
}

} // namespace

} // namespace

/// @addtogroup LoggingMacros
/// @{

/// Logs an error message, prepending the caller's file and line.
#define BE_LOG_DX_ERROR(err) ::beGraphics::DX::GuardLogDXError(err, LEAN_SOURCE_STRING)
/// Logs the given error message, prepending the caller's file and line.
#define BE_LOG_DX_ERROR_MSG(err, msg) ::beGraphics::DX::GuardLogDXError(err, LEAN_SOURCE_STRING, msg)
/// Logs the given error message and context, prepending the caller's file and line.
#define BE_LOG_DX_ERROR_CTX(err, msg, ctx) ::beGraphics::DX::GuardLogDXError(err, LEAN_SOURCE_STRING, msg, ctx)
/// Logs the given error message and context, prepending the caller's file and line.
#define BE_LOG_DX_ERROR_ANY(err, msg) BE_LOG_DX_ERROR_MSG(err, static_cast<::std::ostringstream&>(::std::ostringstream() << msg).str().c_str())

/// @}

/// @addtogroup ExceptionMacros
/// @{

/// Throws a runtime_error exception on failure.
#define BE_THROW_DX_ERROR(err) ::beGraphics::DX::GuardThrowDXError(err, LEAN_SOURCE_STRING)
/// Throws a runtime_error exception on failure.
#define BE_THROW_DX_ERROR_MSG(err, msg) ::beGraphics::DX::GuardThrowDXError(err, LEAN_SOURCE_STRING, msg)
/// Throws a runtime_error exception on failure.
#define BE_THROW_DX_ERROR_CTX(err, msg, ctx) ::beGraphics::DX::GuardThrowDXError(err, LEAN_SOURCE_STRING, msg, ctx)
/// Throws a runtime_error exception on failure.
#define BE_THROW_DX_ERROR_ANY(err, msg) BE_THROW_DX_ERROR_MSG(err, static_cast<::std::ostringstream&>(::std::ostringstream() << msg).str().c_str())

/// @}

#endif