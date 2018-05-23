/*****************************************************/
/* lean I/O                     (c) Tobias Zirr 2011 */
/*****************************************************/

// Opaque value requires this to go here
#ifdef LEAN_INCLUDE_INLINED
	#ifndef LEAN_LOGGING_IO_FILE_CPP
		#define LEAN_LOGGING_IO_FILE_CPP
		#include "source/file.cpp"
	#endif
#endif

#ifndef LEAN_LOGGING_IO_FILE
#define LEAN_LOGGING_IO_FILE

#include "../lean.h"
#include <string>
#include "../strings/types.h"
#include "win_types.h"
#include "../tags/noncopyable.h"

namespace lean
{
namespace io
{

/// File class that provides managed access to a given file.
class file : public noncopyable
{
private:
	utf8_string m_name;
	windows_file_handle m_handle;

public:
	/// Access modes.
	enum access_modes
	{
		read = 0x1,					///< Read mode.
		write = 0x2,				///< Write mode.
		readwrite = read | write	///< Read-write mode.
	};
	/// Additional sharing modes (enhancing access modes which may also be used as sharing modes).
	enum share_modes
	{
		dont_share = 0x0,		///< Exclusive access mode.
		share_default = 0x4,		///< Default sharing mode (read-write on read-only access, read otherwise).
	};
	/// Open modes.
	enum open_mode
	{
		create,			///< Create file, if not existent (write access only).
		append,			///< Open file, if existent (enforced on read-only access).
		open,			///< Open file, if existent, otherwise create new file (write access only).
		overwrite		///< Overwrite file, if existent, otherwise create new file (write access only).
	};
	/// Optimization hints.
	enum hints
	{
		none = 0x0,			///< Make no assumptions on future file access.
		sequential = 0x1,	///< The file is about to be read strictly front-to-back.
		random = 0x2		///< The file is about to be accessed randomly.
	};

	/// Opens the given file according to the given flags. Throws a runtime_exception on error.
	LEAN_MAYBE_EXPORT explicit file(const utf8_ntri &name,
		uint4 access = read | write, open_mode mode = open,
		uint4 hints = none, uint4 share = share_default);
	/// Closes this file.
	LEAN_MAYBE_EXPORT virtual ~file();
	
	/// Sets the current file cursor position. Throws a runtime_exception on error.
	LEAN_MAYBE_EXPORT void pos(uint8 newPos);
	/// Gets the current file cursor position. Returns 0 on error.
	LEAN_MAYBE_EXPORT uint8 pos() const;

	/// Resizes the file, either extending or truncating it. Throws a runtime_exception on error.
	LEAN_MAYBE_EXPORT void resize(uint8 newSize);

	/// Gets the last modification time in microseconds since 1/1/1970. Returns 0 if file is currently open for writing.
	LEAN_MAYBE_EXPORT uint8 revision() const;
	/// Gets the size of this file, in bytes.
	LEAN_MAYBE_EXPORT uint8 size() const;

	/// Marks this file modified.
	LEAN_MAYBE_EXPORT void touch();

	/// Gets the file handle of this file.
	LEAN_INLINE windows_file_handle handle() const { return m_handle; };
	/// Gets the name of this file.
	LEAN_INLINE const utf8_string& name() const { return m_name; };
};

} // namespace

using io::file;

} // namespace

#endif