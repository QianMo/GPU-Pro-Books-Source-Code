/*****************************************************/
/* lean I/O                     (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_LOGGING_IO_MAPPED_FILE
#define LEAN_LOGGING_IO_MAPPED_FILE

#include "../lean.h"
#include "../strings/types.h"
#include "file.h"

namespace lean
{
namespace io
{

/// Base class for mapped files.
class mapped_file_base : public file
{
private:
	windows_file_handle m_mappingHandle;

	/// Creates a file mapping. Throws a runtime_exception on error.
	/// A size of 0 equals the current file size. Sets the file size to the given size if not read-only.
	static LEAN_MAYBE_EXPORT windows_file_handle createMapping(file &file, bool readonly, uint8 size);

protected:
	/// Opens the given file according to the given flags. Throws a runtime_exception on error.
	LEAN_MAYBE_EXPORT explicit mapped_file_base(const utf8_ntri &name,
		bool readonly, uint8 size, open_mode mode,
		uint4 hints, uint4 share);
	/// Closes this file.
	LEAN_MAYBE_EXPORT ~mapped_file_base();

	/// Maps the given view of this file. A size of 0 maps the entire file starting at the given offset.
	/// Throws a runtime_exception on error.
	LEAN_MAYBE_EXPORT void* map(bool readonly, uint8 offset, size_t size);
	/// Unmaps the given view of this file.
	LEAN_MAYBE_EXPORT void unmap(void *memory);

	/// Resizes the file, either extending or truncating it. Throws a runtime_exception on error.
	/// Destroys the mapping, re-mapping is only possible again after this method has returned successfully.
	LEAN_MAYBE_EXPORT void resize(uint8 newSize);
};

/// File class that allows for memory-mapped read access to a given file.
class rmapped_file : public mapped_file_base
{
private:
	const void *m_memory;

public:
	/// Opens the given file according to the given flags. Throws a runtime_exception on error.
	LEAN_MAYBE_EXPORT explicit rmapped_file(const utf8_ntri &name,
		bool mapWhole = true, open_mode mode = file::open,
		uint4 hints = file::none, uint4 share = file::share_default);
	/// Closes this file.
	LEAN_MAYBE_EXPORT ~rmapped_file();

	/// (Re-)maps the file using the given parameters. A size of 0 maps the entire file starting at the given offset.
	/// Throws a runtime_exception on error.
	LEAN_MAYBE_EXPORT const void* map(uint8 offset = 0, size_t size = 0);
	/// Unmaps the file.
	LEAN_MAYBE_EXPORT void unmap();

	/// Gets a pointer to the file in memory, nullptr if currently unmapped.
	LEAN_INLINE const void* data() const { return m_memory; };

	/// Gets whether the file is currently mapped.
	LEAN_INLINE bool mapped() const { return (m_memory != nullptr); }
};

/// File class that allows for memory-mapped read/write access to a given file.
class mapped_file : public mapped_file_base
{
private:
	void *m_memory;

public:
	/// Opens the given file according to the given flags. Throws a runtime_exception on error.
	/// A size of 0 equals the current file size.
	LEAN_MAYBE_EXPORT explicit mapped_file(const utf8_ntri &name,
		uint8 size = 0, bool mapWhole = true, open_mode mode = file::open,
		uint4 hints = file::none, uint4 share = file::share_default);
	/// Closes this file.
	LEAN_MAYBE_EXPORT ~mapped_file();

	/// (Re-)maps the file using the given parameters. Throws a runtime_exception on error.
	/// A size of 0 maps the entire file starting at the given offset.
	LEAN_MAYBE_EXPORT void* map(uint8 offset = 0, size_t size = 0);
	/// Unmaps the file.
	LEAN_MAYBE_EXPORT void unmap();

	/// Flushes the file contents in the mapped range to disk, returing true on success. Ignored, if currently unmapped.
	LEAN_MAYBE_EXPORT bool flush();

	/// Resizes the file, either extending or truncating it. Throws a runtime_exception on error.
	/// Unmaps the file, re-mapping is only possible again after this method has returned successfully.
	LEAN_MAYBE_EXPORT void resize(uint8 newSize);

	/// Gets a pointer to the file in memory, nullptr if currently unmapped.
	LEAN_INLINE void* data() { return m_memory; }
	/// Gets a pointer to the file in memory, nullptr if currently unmapped.
	LEAN_INLINE const void* data() const { return m_memory; }

	/// Gets whether the file is currently mapped.
	LEAN_INLINE bool mapped() const { return (m_memory != nullptr); }
};

} // namespace

using io::rmapped_file;
using io::mapped_file;

} // namespace

#ifdef LEAN_INCLUDE_INLINED
#include "source/mapped_file.cpp"
#endif

#endif