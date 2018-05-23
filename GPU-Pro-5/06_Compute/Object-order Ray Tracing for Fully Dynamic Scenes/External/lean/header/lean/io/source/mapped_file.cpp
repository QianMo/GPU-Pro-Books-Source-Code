#ifdef LEAN_BUILD_LIB
#include "../../depconfig.h"
#endif

// Use short file names in logging
#ifndef LEAN_DEFAULT_FILE_MACRO
	#line __LINE__ "mapped_file.cpp"
#endif

#include <windows.h>
#include "../mapped_file.h"
#include "../../logging/win_errors.h"

namespace lean
{
namespace io
{
namespace impl
{
	/// Gets the memory alignment required in offset map calls.
	inline DWORD get_map_alignment()
	{
		struct helper
		{
			static DWORD system_allocation_granularity()
			{
				SYSTEM_INFO sysInfo;
				GetSystemInfo(&sysInfo);

				// Guarantee a power of two
				LEAN_ASSERT( (sysInfo.dwAllocationGranularity & (sysInfo.dwAllocationGranularity - 1)) == 0 );
				
				return sysInfo.dwAllocationGranularity;
			}
		};
		
		static const DWORD alignment = helper::system_allocation_granularity();
		return alignment;
	}

	/// Aligns the given map offset.
	inline uint8 align_map_offset(uint8 offset)
	{
		return offset & ~static_cast<uint8>(get_map_alignment() - 1);
	}

	/// Aligns the given pointer to mapped memory.
	template <class Type>
	inline Type* align_mapped_memory(Type *memory)
	{
		return reinterpret_cast<Type*>(
			reinterpret_cast<uintptr_t>(memory) &  ~static_cast<uintptr_t>(get_map_alignment() - 1) );
	}

	/// Aligns the given map size.
	inline size_t align_map_size(size_t size)
	{
		const size_t alignment_minus_1 = get_map_alignment() - 1;
		return (size + alignment_minus_1) & ~alignment_minus_1;
	}

	/// Clamps the given map size.
	inline size_t clamp_map_size(uint8 size)
	{
		return (size < static_cast<size_t>(-1))
			? static_cast<size_t>(size)
			: static_cast<size_t>(-1);
	}

} // namespace
} // namespace
} // namespace

// Opens the given file according to the given flags. Throws a runtime_exception on error.
LEAN_MAYBE_INLINE lean::io::mapped_file_base::mapped_file_base(const utf8_ntri &name,
		bool readonly, uint8 size,
		open_mode mode, uint4 hints, uint4 share)
	: file(name,
		(readonly) ? file::read : (file::read | file::write),
		mode, hints, share),
	m_mappingHandle( createMapping(*this, readonly, size) )
{
}

// Closes this file.
LEAN_MAYBE_INLINE lean::io::mapped_file_base::~mapped_file_base()
{
	if (m_mappingHandle != NULL)
		::CloseHandle(m_mappingHandle);
}

// Creates a file mapping. Throws a runtime_exception on error.
// A size of 0 equals the current file size. Sets the file size to the given size if not read-only.
LEAN_MAYBE_INLINE lean::io::windows_file_handle lean::io::mapped_file_base::createMapping(file &file, bool readonly, uint8 size)
{
	// Resize first to avoid creating the mapping twice
	if (!readonly && size != 0 && size != file.size())
		file.resize(size);

	// Handles size of 0 equal to current file size
	HANDLE mapping = ::CreateFileMappingW(
		file.handle(),
		nullptr,
		(readonly) ? PAGE_READONLY : PAGE_READWRITE,
		static_cast<DWORD>(size >> size_info<DWORD>::bits),
		static_cast<DWORD>(size),
		nullptr);

	if (mapping == NULL)
		LEAN_THROW_WIN_ERROR_CTX("CreateFileMapping()", file.name().c_str());

	return mapping;
}

// Resizes the file, either extending or truncating it. Throws a runtime_exception on error.
// Automatically destroys the mapping, re-mapping is only possible again after this method has returned successfully.
LEAN_MAYBE_INLINE void lean::io::mapped_file_base::resize(uint8 newSize)
{
	if (m_mappingHandle != NULL)
	{
		::CloseHandle(m_mappingHandle);
		m_mappingHandle = NULL;
	}

	// Try to re-create mapping with new size
	m_mappingHandle = createMapping(*this, false, newSize);
}

// Maps the given view of this file. A size of 0 maps the entire file starting at the given offset.
// Throws a runtime_exception on error.
LEAN_MAYBE_INLINE void* lean::io::mapped_file_base::map(bool readonly, uint8 offset, size_t size)
{
	// Offset is required to be aligned using system allocation granularity
	uint8 alignedOffset = impl::align_map_offset(offset);
	size_t alignmentDelta = static_cast<size_t>(offset - alignedOffset);
	// WARNING: DON'T align size, easily out-of-bounds otherwise
	size_t alignedSize = /*impl::align_map_size(*/ size + alignmentDelta /*)*/;

	// Handles size of 0 equal to end of file
	void *memory = (m_mappingHandle != NULL)
		? ::MapViewOfFile(m_mappingHandle,
			(readonly) ? FILE_MAP_READ : (FILE_MAP_READ | FILE_MAP_WRITE),
			static_cast<DWORD>(alignedOffset >> size_info<DWORD>::bits),
			static_cast<DWORD>(alignedOffset),
			alignedSize)
		: nullptr;

	if (!memory)
		LEAN_THROW_WIN_ERROR_CTX("MapViewOfFile()", name().c_str());

	// Return requested unaligned address
	return reinterpret_cast<char*>(memory) + alignmentDelta;
}

// Unmaps the given view of this file.
LEAN_MAYBE_INLINE void lean::io::mapped_file_base::unmap(void *memory)
{
	if (memory)
		::UnmapViewOfFile(impl::align_mapped_memory(memory));
}

// Opens the given file according to the given flags. Throws a runtime_exception on error.
LEAN_MAYBE_INLINE lean::io::rmapped_file::rmapped_file(const utf8_ntri &name,
		bool mapWhole, open_mode mode, uint4 hints, uint4 share)
	: mapped_file_base(name, true, 0, mode, hints, share),
	m_memory( (mapWhole) ? mapped_file_base::map(true, 0, 0) : nullptr )
{
}

// Closes this file.
LEAN_MAYBE_INLINE lean::io::rmapped_file::~rmapped_file()
{
	unmap();
}

// (Re-)maps the file using the given parameters. A size of 0 maps the entire file starting at the given offset.
// Throws a runtime_exception on error.
LEAN_MAYBE_INLINE const void* lean::io::rmapped_file::map(uint8 offset, size_t size)
{
	if (m_memory)
		unmap();

	m_memory = mapped_file_base::map(true, offset, size);
	return m_memory;
}

// Unmaps the file.
LEAN_MAYBE_INLINE void lean::io::rmapped_file::unmap()
{
	mapped_file_base::unmap(const_cast<void*>(m_memory));
	m_memory = nullptr;
}

// Opens the given file according to the given flags. Throws a runtime_exception on error.
// A size of 0 equals the current file size.
LEAN_MAYBE_INLINE lean::io::mapped_file::mapped_file(const utf8_ntri &name,
		uint8 size, bool mapWhole, open_mode mode, uint4 hints, uint4 share)
	: mapped_file_base(name, false, size, mode, hints, share),
	m_memory( (mapWhole)
		? mapped_file_base::map(false, 0, impl::clamp_map_size(size))
		: nullptr )
{
}

// Closes this file.
LEAN_MAYBE_INLINE lean::io::mapped_file::~mapped_file()
{
	unmap();
}

// (Re-)maps the file using the given parameters. A size of 0 maps the entire file starting at the given offset.
// Throws a runtime_exception on error.
LEAN_MAYBE_INLINE void* lean::io::mapped_file::map(uint8 offset, size_t size)
{
	if (m_memory)
		unmap();

	m_memory = mapped_file_base::map(false, offset, size);
	return m_memory;
}

// Unmaps the file.
LEAN_MAYBE_INLINE void lean::io::mapped_file::unmap()
{
	mapped_file_base::unmap(m_memory);
	m_memory = nullptr;
}

// Flushes the file contents in the mapped range to disk, returing true on success. Ignored, if currently unmapped.
LEAN_MAYBE_INLINE bool lean::io::mapped_file::flush()
{
	return (m_memory)
		? (::FlushViewOfFile(m_memory, 0) != FALSE)
		: false;
}

// Resizes the file, either extending or truncating it. Throws a runtime_exception on error.
// Automatically unmaps the file, re-mapping is only possible again after this method has returned successfully.
LEAN_MAYBE_INLINE void lean::io::mapped_file::resize(uint8 newSize)
{
	if (m_memory)
		unmap();

	mapped_file_base::resize(newSize);
}
