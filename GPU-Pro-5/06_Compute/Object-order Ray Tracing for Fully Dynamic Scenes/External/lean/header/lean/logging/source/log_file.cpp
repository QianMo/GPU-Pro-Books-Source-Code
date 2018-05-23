#ifdef LEAN_BUILD_LIB
#include "../../depconfig.h"
#endif

// Use short file names in logging
#ifndef LEAN_DEFAULT_FILE_MACRO
	#line __LINE__ "log_file.cpp"
#endif

#include <windows.h>
#include "../log_file.h"
#include "../../strings/conversions.h"
#include "../../logging/win_errors.h"

// Opens the given file for logging.
LEAN_ALWAYS_LINK lean::logging::log_file::log_file(const utf8_ntri &name)
	: m_handle(
		::CreateFileW(utf_to_utf16(name).c_str(),
			GENERIC_WRITE, FILE_SHARE_READ,
			nullptr, CREATE_ALWAYS,
			0, NULL) )
{
	if (m_handle == INVALID_HANDLE_VALUE)
		LEAN_LOG_WIN_ERROR_CTX("CreateFile()", name.c_str());
}

// Closes the log file.
LEAN_ALWAYS_LINK lean::logging::log_file::~log_file()
{
	if (m_handle != INVALID_HANDLE_VALUE)
		::CloseHandle(m_handle);
}

// Gets whether the file was opened successfully.
LEAN_ALWAYS_LINK bool lean::logging::log_file::valid() const
{
	return (m_handle != INVALID_HANDLE_VALUE);
}

// Prints the given message to the log file. This method is thread-safe.
LEAN_ALWAYS_LINK void lean::logging::log_file::print(const char_ntri &message)
{
	DWORD written;

	// Thread-safe: http://msdn.microsoft.com/en-us/library/ms810467
	if (m_handle != INVALID_HANDLE_VALUE)
	{
		::WriteFile(m_handle, message.c_str(), static_cast<DWORD>(message.size()), &written, nullptr);
		::FlushFileBuffers(m_handle);
	}
}
