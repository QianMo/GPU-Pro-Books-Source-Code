#pragma once

#include "Tracing.h"
#include <cuda_runtime.h>
#include <lean/logging/errors.h>

/// Gets an error message describing the given CUDA error.
LEAN_NOINLINE const lean::utf8_t* GetCUDAError(cudaError error)
{
	return cudaGetErrorString(error);
}

/// Gets an error message describing the given CUDA error.
template <class String>
String GetCUDAError(cudaError error);

#ifndef DOXYGEN_SKIP_THIS

/// Gets an error message describing the given CUDA error.
template <>
inline lean::utf16_string GetCUDAError(cudaError error)
{
	return lean::strings::utf_to_utf16( GetCUDAError(error) );
}
/// Gets an error message describing the given CUDA error.
template <>
inline lean::utf8_string GetCUDAError(cudaError error)
{
	return lean::utf8_string( GetCUDAError(error) );
}

#endif

/// Logs the given CUDA error.
template <class String1>
LEAN_NOTINLINE void LogCUDAError(cudaError error, const String1 &source)
{
	lean::log_error(source, GetCUDAError(error));
}
/// Logs the given CUDA error.
template <class String1, class String2>
LEAN_NOTINLINE void LogCUDAError(cudaError error, const String1 &source, const String2 &reason)
{
	lean::log_error_ex(source, GetCUDAError(error), reason);
}
/// Logs the given CUDA error.
template <class String1, class String2, class String3>
LEAN_NOTINLINE void LogCUDAError(cudaError error, const String1 &source, const String2 &reason, const String3 &context)
{
	lean::log_error_ex(source, GetCUDAError(error), reason, context);
}

/// Logs the given CUDA error on failure.
template <class String1>
LEAN_INLINE bool GuardLogCUDAError(cudaError error, const String1 &source)
{
	return (error != cudaSuccess)
		? (LogCUDAError(error, source), false)
		: true;
}
/// Logs the given CUDA error on failure.
template <class String1, class String2>
LEAN_INLINE bool GuardLogCUDAError(cudaError error, const String1 &source, const String2 &reason)
{
	return (error != cudaSuccess)
		? (LogCUDAError(error, source, reason), false)
		: true;
}
/// Logs the given CUDA error on failure.
template <class String1, class String2, class String3>
LEAN_INLINE bool GuardLogCUDAError(cudaError error, const String1 &source, const String2 &reason, const String3 &context)
{
	return (error != cudaSuccess)
		? (LogCUDAError(error, source, reason, context), false)
		: true;
}


/// Throws a runtime_error exception containing the given CUDA error.
template <class String1>
LEAN_NOTINLINE void ThrowCUDAError(cudaError error, const String1 &source)
{
	lean::throw_error(source, GetCUDAError(error));
}
/// Throws a runtime_error exception containing the given CUDA error.
template <class String1, class String2>
LEAN_NOTINLINE void ThrowCUDAError(cudaError error, const String1 &source, const String2 &reason)
{
	lean::throw_error_ex(source, GetCUDAError(error), reason);
}
/// Throws a runtime_error exception containing the given CUDA error.
template <class String1, class String2, class String3>
LEAN_NOTINLINE void ThrowCUDAError(cudaError error, const String1 &source, const String2 &reason, const String3 &context)
{
	lean::throw_error_ex(source, GetCUDAError(error), reason, context);
}

/// Throws a runtime_error exception containing the given CUDA error on failure.
template <class String1>
LEAN_INLINE void GuardThrowCUDAError(cudaError error, const String1 &source)
{
	if (error != cudaSuccess)
		ThrowCUDAError(error, source);
}
/// Throws a runtime_error exception containing the given CUDA error on failure.
template <class String1, class String2>
LEAN_INLINE void GuardThrowCUDAError(cudaError error, const String1 &source, const String2 &reason)
{
	if (error != cudaSuccess)
		ThrowCUDAError(error, source, reason);
}
/// Throws a runtime_error exception containing the given CUDA error on failure.
template <class String1, class String2, class String3>
LEAN_INLINE void GuardThrowCUDAError(cudaError error, const String1 &source, const String2 &reason, const String3 &context)
{
	if (error != cudaSuccess)
		ThrowCUDAError(error, source, reason, context);
}

/// @addtogroup LoggingMacros
/// @{

/// Logs an error message, prepending the caller's file and line.
#define BE_LOG_CUDA_ERROR(err) ::GuardLogCUDAError(err, LEAN_SOURCE_STRING)
/// Logs the given error message, prepending the caller's file and line.
#define BE_LOG_CUDA_ERROR_MSG(err, msg) ::GuardLogCUDAError(err, LEAN_SOURCE_STRING, msg)
/// Logs the given error message and context, prepending the caller's file and line.
#define BE_LOG_CUDA_ERROR_CTX(err, msg, ctx) ::GuardLogCUDAError(err, LEAN_SOURCE_STRING, msg, ctx)
/// Logs the given error message and context, prepending the caller's file and line.
#define BE_LOG_CUDA_ERROR_ANY(err, msg) BE_LOG_CUDA_ERROR_MSG(err, static_cast<::std::ostringstream&>(::std::ostringstream() << msg).str().c_str())

/// @}

/// @addtogroup ExceptionMacros
/// @{

/// Throws a runtime_error exception on failure.
#define BE_THROW_CUDA_ERROR(err) ::GuardThrowCUDAError(err, LEAN_SOURCE_STRING)
/// Throws a runtime_error exception on failure.
#define BE_THROW_CUDA_ERROR_MSG(err, msg) ::GuardThrowCUDAError(err, LEAN_SOURCE_STRING, msg)
/// Throws a runtime_error exception on failure.
#define BE_THROW_CUDA_ERROR_CTX(err, msg, ctx) ::GuardThrowCUDAError(err, LEAN_SOURCE_STRING, msg, ctx)
/// Throws a runtime_error exception on failure.
#define BE_THROW_CUDA_ERROR_ANY(err, msg) BE_THROW_CUDA_ERROR_MSG(err, static_cast<::std::ostringstream&>(::std::ostringstream() << msg).str().c_str())

/// @}
