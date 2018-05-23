/*****************************************************/
/* lean I/O                     (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_LOGGING_IO_RAW_FILE
#define LEAN_LOGGING_IO_RAW_FILE

#include "../lean.h"
#include "../strings/types.h"
#include "file.h"

namespace lean
{
namespace io
{

/// File class that allows for raw read/write operations on a given file.
class raw_file : public file
{
public:
	/// Opens the given file according to the given flags. Throws a runtime_exception on error.
	LEAN_MAYBE_EXPORT explicit raw_file(const utf8_ntri &name,
		uint4 access = file::read | file::write, open_mode mode = file::open,
		uint4 hints = file::none, uint4 share = file::share_default);
	/// Closes this file.
	LEAN_MAYBE_EXPORT ~raw_file();

	/// Reads the given number of bytes from the file, returning the number of bytes read. This method is thread-safe.
	LEAN_MAYBE_EXPORT size_t read(char *begin, size_t count) const;
	/// Writes the given number of bytes to the file, returning the number of bytes written. This method is thread-safe.
	LEAN_MAYBE_EXPORT size_t write(const char *begin, size_t count);

	/// Prints the given range of characters to the file. This method is thread-safe.
	LEAN_MAYBE_EXPORT size_t print(const char_ntri &message);
};

} // namespace

using io::raw_file;

} // namespace

#ifdef LEAN_INCLUDE_INLINED
#include "source/raw_file.cpp"
#endif

#endif