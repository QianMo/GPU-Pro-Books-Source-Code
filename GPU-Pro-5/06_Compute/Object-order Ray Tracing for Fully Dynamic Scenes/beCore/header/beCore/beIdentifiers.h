/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_IDENTIFIERS
#define BE_CORE_IDENTIFIERS

#include "beCore.h"
#include <lean/tags/noncopyable.h>
#include <vector>

namespace beCore
{

/// Identifier manager.
class Identifiers : public lean::noncopyable
{
private:
	typedef std::vector<utf8_string> identifier_vector;
	identifier_vector m_identifiers;

public:
	/// Invalid ID.
	static const uint4 InvalidID = static_cast<uint4>(-1);
	/// Invalid short ID.
	static const uint2 InvalidShortID = static_cast<uint2>(-1);

	/// Constructor.
	BE_CORE_API Identifiers();
	/// Destructor.
	BE_CORE_API ~Identifiers();

	/// Adds the given identifier to this identifier manager.
	BE_CORE_API uint4 GetID(const utf8_ntri &name);
	/// Adds the given identifier to this identifier manager.
	BE_CORE_API uint4 GetID(const utf8_ntri &name) const;

	/// Adds the given identifier to this identifier manager.
	LEAN_INLINE utf8_string GetName(uint4 id) const { return (id < m_identifiers.size()) ? m_identifiers[id] : ""; }
};

}

#endif