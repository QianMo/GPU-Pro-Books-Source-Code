#ifdef LEAN_BUILD_LIB
#include "../../depconfig.h"
#endif

// Use short file names in logging
#ifndef LEAN_DEFAULT_FILE_MACRO
	#line __LINE__ "filesystem.cpp"
#endif

#include <windows.h>
#include "../filesystem.h"
#include "../../logging/win_errors.h"
#include "../../smart/handle_guard.h"

#include "common.cpp"

// Checks whether the given file exists.
LEAN_MAYBE_LINK bool lean::io::file_exists(const utf16_nti& file)
{
	return (::GetFileAttributesW(file.c_str()) != INVALID_FILE_ATTRIBUTES);
}

// Gets the last modification time in microseconds since 1/1/1970. Returns 0 on error.
LEAN_MAYBE_LINK lean::uint8 lean::io::file_revision(const utf16_nti& file)
{
	uint8 revision = 0;

	handle_guard<HANDLE> hFile( ::CreateFileW(file.c_str(),
		FILE_READ_ATTRIBUTES,
		FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
		nullptr,
		OPEN_EXISTING,
		FILE_FLAG_BACKUP_SEMANTICS,
		NULL) );

	if (hFile == INVALID_HANDLE_VALUE)
		LEAN_LOG_WIN_ERROR_CTX("CreateFile()", file);
	else
	{
		LEAN_ASSERT(sizeof(uint8) == sizeof(::FILETIME));

		if (!::GetFileTime(hFile, nullptr, nullptr, reinterpret_cast<FILETIME*>(&revision)))
			LEAN_LOG_WIN_ERROR_CTX("GetFileTime()", file);

		revision = impl::get_revision_from_filetime(revision);
	}

	return revision;
}

// Gets the size of the given file, in bytes. Returns 0 on error.
LEAN_MAYBE_LINK lean::uint8 lean::io::file_size(const utf16_nti& file)
{
	uint8 size = 0;

	LEAN_ASSERT(sizeof(uint8) == sizeof(::LARGE_INTEGER));

	handle_guard<HANDLE> hFile( ::CreateFileW(file.c_str(),
		FILE_READ_ATTRIBUTES,
		FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
		nullptr,
		OPEN_EXISTING,
		FILE_FLAG_BACKUP_SEMANTICS,
		NULL) );

	if (hFile == INVALID_HANDLE_VALUE)
		LEAN_LOG_WIN_ERROR_CTX("CreateFile()", file);
	else if (!::GetFileSizeEx(hFile, reinterpret_cast<LARGE_INTEGER*>(&size)))
		LEAN_LOG_WIN_ERROR_CTX("GetFileSizeEx()", file);

	return size;
}

// Gets the current directory. Will return the buffer size required to store the
// current directory, if the given buffer is too small, the number of actual
// characters written, otherwise (excluding the terminating null appended).
LEAN_MAYBE_LINK size_t lean::io::current_directory(utf16_t *buffer, size_t bufferSize)
{
	return ::GetCurrentDirectoryW(
			(bufferSize > static_cast<DWORD>(-1)) ? static_cast<DWORD>(-1) : static_cast<DWORD>(bufferSize),
			buffer
		);
}
