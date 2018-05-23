//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (C) Tobias Zirr.  All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>

namespace D3DEffectsLite
{

struct CharRange
{
	const char *begin;
	const char *end;

	CharRange()
		: begin(), end() { }
	CharRange(const char *begin, const char *end)
		: begin(begin), end(end)
	{
		assert(begin == end || begin && end && begin <= end);
	}
	CharRange(const char *begin, size_t count)
		: begin(begin), end(begin + count) { }
	template <size_t Count>
	CharRange(const char (&arr)[Count])
		: begin(arr), end(arr + Count - !arr[Count - 1]) { }
	CharRange(const char *str)
		: begin(str), end(str + strlen(str)) { }
	template <class R>
	explicit CharRange(const R &r, typename R::iterator* = nullptr)
		: begin(&r[0]), end(&r[r.size()]) { }
};

inline bool empty(CharRange range)
{
	return range.end == range.begin;
}
inline size_t size(CharRange range)
{
	return range.end - range.begin;
}
inline CharRange deflate(CharRange range)
{
	assert(size(range) >= 2);
	return CharRange(range.begin + 1, range.end - 1);
}

inline bool operator ==(CharRange r1, CharRange r2)
{
	return size(r1) == size(r2) && memcmp(r1.begin, r2.begin, size(r1)) == 0;
}
inline bool operator !=(CharRange r1, CharRange r2)
{
	return !(r1 == r2);
}

struct CharRangeAnnotation
{
	const char *name;
	int line;

	CharRangeAnnotation(const char *name, int line)
		: name(name), line(line) { }
};

struct AnnotedCharRange : CharRange
{
	CharRangeAnnotation annot;

	AnnotedCharRange()
		: annot(nullptr, -1) { }

	template <class Range>
	AnnotedCharRange(const Range &range, const char *name, int line)
		: CharRange(range), annot(name, line) { }

	template <class Range>
	AnnotedCharRange(const Range &range, const CharRangeAnnotation &annot)
		: CharRange(range), annot(annot) { }
};

} // namespace
