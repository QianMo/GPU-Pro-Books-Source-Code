#ifdef LEAN_BUILD_LIB
#include "../../depconfig.h"
#endif

// Use short file names in logging
#ifndef LEAN_DEFAULT_FILE_MACRO
	#line __LINE__ "file.cpp"
#endif

#include <windows.h>
#include "../file.h"
#include "../../strings/conversions.h"
#include "../../logging/win_errors.h"

#include "common.cpp"

namespace lean
{
namespace io
{
namespace impl
{
	/// Converts the given access flags into valid windows access flags.
	inline DWORD get_windows_access_flags(uint4 access)
	{
		DWORD winAccess = 0;
		
		if (access & file::read)
			winAccess |= GENERIC_READ;
		if (access & file::write)
			winAccess |= GENERIC_WRITE;

		// Always require some kind of access
		if (!winAccess)
			winAccess = GENERIC_READ;

		return winAccess;
	}

	/// Converts the given sharing flags into valid windows sharing flags.
	inline DWORD get_windows_sharing_flags(uint4 share, uint4 access)
	{
		DWORD winShare = 0;
		
		if ((share & file::read) || (share & file::share_default))
			winShare |= FILE_SHARE_READ;
		// Share for writing, if default & access read-only
		if (share & file::write /*|| (share & file::share_default) && !(access & file::write)*/)
			winShare |= FILE_SHARE_WRITE;

		return winShare;
	}

	/// Converts the given open mode into the corresponding windows open mode.
	inline DWORD get_windows_open_mode(file::open_mode mode, uint4 access)
	{
		if (access & file::write)
			switch (mode)
			{
			case file::append:
				return OPEN_EXISTING;
			case file::create:
				return CREATE_NEW;
			case file::overwrite:
				return CREATE_ALWAYS;
			case file::open:
			default:
				return OPEN_ALWAYS;
			}
		else
			return OPEN_EXISTING;
	}

	/// Converts the given optimization hints into the corresponding windows flags.
	inline DWORD get_windows_optimization_flags(uint4 hints)
	{
		if (hints & file::sequential)
			return FILE_FLAG_SEQUENTIAL_SCAN;
		else if (hints & file::random)
			return FILE_FLAG_RANDOM_ACCESS;
		else
			return 0;
	}

} // namespace
} // namespace
} // namespace

// Opens the given file according to the given flags. Throws a runtime_exception on error.
LEAN_MAYBE_INLINE lean::io::file::file(const utf8_ntri &name,
	uint4 access, open_mode mode, uint4 hints, uint4 share)
	: m_name(name.to<utf8_string>()),
	m_handle(
		::CreateFileW(utf_to_utf16(name).c_str(),
			impl::get_windows_access_flags(access),
			impl::get_windows_sharing_flags(share, access),
			nullptr,
			impl::get_windows_open_mode(mode, access),
			impl::get_windows_optimization_flags(hints),
			NULL) )
{
	if (m_handle == INVALID_HANDLE_VALUE)
		LEAN_THROW_WIN_ERROR_CTX("CreateFile()", m_name.c_str());
}

// Closes this file.
LEAN_MAYBE_INLINE lean::io::file::~file()
{
	::CloseHandle(m_handle);
}

// Sets the current file cursor position. Throws a runtime_exception on error.
LEAN_MAYBE_INLINE void lean::io::file::pos(uint8 newPos)
{
	LEAN_ASSERT(sizeof(uint8) == sizeof(::LARGE_INTEGER));

	if (!::SetFilePointerEx(handle(), reinterpret_cast<LARGE_INTEGER&>(newPos), nullptr, FILE_BEGIN))
		LEAN_THROW_WIN_ERROR_CTX("SetFilePointerEx()", name().c_str());
}

// Gets the current file cursor position. Returns 0 on error.
LEAN_MAYBE_INLINE lean::uint8 lean::io::file::pos() const
{
	uint8 pos = 0;

	LEAN_ASSERT(sizeof(uint8) == sizeof(::LARGE_INTEGER));

	::SetFilePointerEx(handle(), reinterpret_cast<const LARGE_INTEGER&>(static_cast<const uint8&>(0)),
		reinterpret_cast<LARGE_INTEGER*>(&pos), FILE_CURRENT);
	return pos;
}

// Resizes the file, either extending or truncating it. Throws a runtime_exception on error.
LEAN_MAYBE_INLINE void lean::io::file::resize(uint8 newSize)
{
	pos(newSize);

	if (!::SetEndOfFile(handle()))
		LEAN_THROW_WIN_ERROR_CTX("SetEndOfFile()", name().c_str());

	pos(0);
}

// Gets the last modification time in microseconds since 1/1/1970. Returns 0 if file is currently open for writing.
LEAN_MAYBE_INLINE lean::uint8 lean::io::file::revision() const
{
	uint8 revision = 0;

	LEAN_ASSERT(sizeof(uint8) == sizeof(::FILETIME));

	if (!::GetFileTime(m_handle, nullptr, nullptr, reinterpret_cast<FILETIME*>(&revision)))
		LEAN_LOG_WIN_ERROR_CTX("GetFileTime()", name().c_str());

	return impl::get_revision_from_filetime(revision);
}

// Gets the size of this file, in bytes.
LEAN_MAYBE_INLINE lean::uint8 lean::io::file::size() const
{
	uint8 size = 0;

	LEAN_ASSERT(sizeof(uint8) == sizeof(::LARGE_INTEGER));

	if (!::GetFileSizeEx(m_handle, reinterpret_cast<LARGE_INTEGER*>(&size)))
		LEAN_LOG_WIN_ERROR_CTX("GetFileSizeEx()", name().c_str());

	return size;
}

// Marks this file modified.
LEAN_MAYBE_INLINE void lean::io::file::touch()
{
	::FILETIME currentFileTime;
	::GetSystemTimeAsFileTime(&currentFileTime);
	
	if (!::SetFileTime(m_handle, nullptr, nullptr, &currentFileTime))
		LEAN_LOG_WIN_ERROR_CTX("SetFileTime()", name().c_str());
}