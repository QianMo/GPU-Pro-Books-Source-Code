/*****************************************************/
/* lean Strings                 (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_STRINGS_CHARSTREAM
#define LEAN_STRINGS_CHARSTREAM

#include "../lean.h"
#include <iostream>

namespace lean
{
namespace strings
{

/// Character stream buffer class that is required to allow for allocation-free usage of STL streaming facilities.
template < class Elem, class Traits = std::char_traits<Elem> >
class basic_charbuf : public std::basic_streambuf<Elem, Traits>
{
protected:
	virtual void __CLR_OR_THIS_CALL _Lock() { }
	virtual void __CLR_OR_THIS_CALL _Unlock() { }

public:
	/// Constructs a character stream buffer from the given character range.
	basic_charbuf(char_type *begin, char_type *end)
	{
		setp(begin, end);
		setg(begin, begin, end);
	}

	/// Resets the character stream buffer.
	void reset()
	{
		setp(pbase(), epptr());
		setg(eback(), eback(), egptr());
	}

	/// Gets the beginning of the underlying buffer.
	char_type* begin() const { return pbase(); }
	/// Gets the end of the underlying buffer.
	char_type* end() const { return epptr(); }
	/// Gets the current write position in the underlying buffer.
	char_type* write_end() const { return pptr(); }
	/// Gets the current read position in the underlying buffer.
	char_type* read_end() const { return gptr(); }

};

namespace impl
{
	/// Helper class that allows for automatic stream buffer construction before stream base class construction.
	template < class Elem, class Traits = std::char_traits<Elem> >
	class charbuf_holder
	{
	protected:
		/// Stream buffer type.
		typedef basic_charbuf<Elem, Traits> stream_buffer;
		/// Stream buffer.
		stream_buffer m_buffer;

		/// Constructs a character stream buffer from the given character range.
		charbuf_holder(typename stream_buffer::char_type *begin, typename stream_buffer::char_type *end)
			: m_buffer(begin, end) { }
	};
}

/// Character stream class that allows for allocation-free usage of STL streaming facilities.
template < class Elem, class Traits = std::char_traits<Elem> >
class basic_charstream : private impl::charbuf_holder<Elem, Traits>, public std::basic_iostream<Elem, Traits>
{
private:
	typedef impl::charbuf_holder<char_type, traits_type> holder_base_type;
	typedef std::basic_iostream<char_type, traits_type> stream_base_type;

public:
	/// Stream buffer type.
	typedef typename holder_base_type::stream_buffer stream_buffer;

	/// Constructs a character stream from the given character range.
	basic_charstream(char_type *begin, char_type *end)
			: holder_base_type(begin, end),
			stream_base_type(&m_buffer) { }
	/// Constructs an unlimited character stream from the given character buffer pointer.
	basic_charstream(char_type *begin)
			: holder_base_type(begin, begin + std::numeric_limits<int>::max()), // required to be int
			stream_base_type(&m_buffer) { }

	/// Resets the character stream.
	basic_charstream& reset()
	{
		m_buffer.reset();
		return *this;
	}

	/// Returns the address of the stored stream buffer object.
	stream_buffer* rdbuf() const { return static_cast<stream_buffer*>(stream_base_type::rdbuf()); }

	/// Gets the beginning of the underlying buffer.
	char_type* begin() const { return m_buffer.begin(); }
	/// Gets the end of the underlying buffer.
	char_type* end() const { return m_buffer.end(); }
	/// Gets the current write position in the underlying buffer.
	char_type* write_end() const { return m_buffer.write_end(); }
	/// Gets the current read position in the underlying buffer.
	char_type* read_end() const { return m_buffer.read_end(); }

};

/// Character stream class.
typedef basic_charstream<char> charstream;
/// Wide-character stream class.
typedef basic_charstream<wchar_t> wcharstream;

} // namespace

using strings::basic_charbuf;

using strings::basic_charstream;
using strings::charstream;
using strings::wcharstream;

} // namespace

#endif