/*****************************************************/
/* lean I/O                     (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_LOGGING_IO_RAW_FILE_INSERTER
#define LEAN_LOGGING_IO_RAW_FILE_INSERTER

#include "../lean.h"
#include "../tags/noncopyable.h"
#include "raw_file.h"

namespace lean
{
namespace io
{

/// File inserter class that follows the STL output iterator concept to allow for convenient buffered file output.
template <size_t BufferSize = 4096>
class raw_file_inserter : public noncopyable
{
private:
	raw_file *m_file;
	char m_buffer[BufferSize];
	char *m_end;

	LEAN_STATIC_ASSERT_MSG_ALT(BufferSize > 0,
		"Buffer size is required to be greater than 0",
		Buffer_size_is_required_to_be_greater_than_0);

	/// Flushes all buffered output to file.
	void flush()
	{
		size_t count = m_end - m_buffer;

		if (count != 0)
		{
			m_file->write(m_buffer, count);
			m_end = m_buffer;
		}
	}

public:
	/// Character reference type.
	typedef char& reference;
	/// Character reference type.
	typedef const char& const_reference;

	/// Iterator type.
	class iterator
	{
	private:
		raw_file_inserter *m_inserter;

	public:
		/// Constructs a file-inserter-based output iterator.
		LEAN_INLINE explicit iterator(raw_file_inserter &inserter)
			: m_inserter(&inserter) { }

		/// Assigns the given value to the current character.
		LEAN_INLINE iterator& operator =(const_reference value)
		{
			m_inserter->insert(value);
			return *this;
		}

		/// Inserts a new character into the underlying file.
		LEAN_INLINE iterator& operator ++()
		{
			// Follow the STL pattern and fake increment
			return *this;
		}
		/// Inserts a new character into the underlying file.
		LEAN_INLINE iterator& operator ++(int)
		{
			// Follow the STL pattern and fake increment
			return *this;
		}
		/// Gets a reference to the current character.
		LEAN_INLINE iterator operator *()
		{
			// Follow the STL pattern and fake dereferencing
			return *this;
		}
	};

	/// Constructs a file inserter from the given raw file.
	LEAN_INLINE explicit raw_file_inserter(raw_file &file)
		: m_file(&file),
		m_end(m_buffer) { }
	/// Copy constructor.
/*	LEAN_INLINE raw_file_inserter(const raw_file_inserter &right)
		: m_file(right.m_file),
		m_end(m_buffer) { }
*/	/// Flushes all buffered output.
	LEAN_INLINE ~raw_file_inserter()
	{
		flush();
	}

	/// Assigns the given file inserter to this file inserter.
/*	LEAN_INLINE raw_file_inserter& operator =(const raw_file_inserter &right)
	{
		flush();
		m_file = right.m_file;
		return *this;
	}
*/
	/// Inserts the given value.
	LEAN_INLINE void insert(const_reference value)
	{
		*(m_end++) = value;

		if (m_end == m_buffer + BufferSize)
			flush();
	}

	/// Gets an output iterator.
	LEAN_INLINE iterator iter()
	{
		return iterator(*this);
	}
};

} // namespace

using io::raw_file_inserter;

} // namespace

#endif