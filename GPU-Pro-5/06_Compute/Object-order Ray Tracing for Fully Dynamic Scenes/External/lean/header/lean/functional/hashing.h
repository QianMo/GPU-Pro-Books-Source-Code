/*****************************************************/
/* lean Functional              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_FUNCTIONAL_HASHING
#define LEAN_FUNCTIONAL_HASHING

#ifdef DOXYGEN_READ_THIS

#include <functional>

namespace lean
{
namespace functional
{

/// Computes hash values from elements of the given type.
template<class Element>
struct hash : public std::unary_function<Element, size_t>
{
	/// Computes a hash value for the given element.
	size_t operator()(const Element &element) const;
};

} // namespace
} // namespace

#endif

#ifndef DOXYGEN_SKIP_THIS

// Use stdext hashing in pre-2010 VS
#if (_MSC_VER < 1600)

#include <hash_map>

namespace lean
{
namespace functional
{

template<class Element>
struct hash : public std::unary_function<Element, size_t>
{
	typedef stdext::hash_compare<Element> stdext_hasher;

	LEAN_INLINE size_t operator()(const Element &element) const
	{
		return stdext_hasher()(element);
	}
};

} // namespace
} // namespace

// MONITOR: Add other pre-TR1 compilers here

// Try to use default C++0x hashing
#else 

#include <functional>

namespace lean
{
namespace functional
{

template<class Element>
struct hash : public std::unary_function<Element, size_t>
{
	typedef std::hash<Element> std_hash;

	LEAN_INLINE size_t operator()(const Element &element) const
	{
		return std_hash()(element);
	}
};

} // namespace
} // namespace

#endif

#endif

namespace lean
{

using functional::hash;

}

#endif