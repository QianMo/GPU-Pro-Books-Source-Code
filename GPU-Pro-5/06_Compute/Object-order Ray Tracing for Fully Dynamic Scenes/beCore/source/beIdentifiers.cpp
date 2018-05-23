/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beCoreInternal/stdafx.h"
#include "beCore/beIdentifiers.h"
#include <algorithm>

namespace beCore
{

// Constructor.
Identifiers::Identifiers()
{
}

// Destructor.
Identifiers::~Identifiers()
{
}

// Adds the given identifier to this identifier manager.
uint4 Identifiers::GetID(const utf8_ntri &name)
{
	uint4 id = static_cast<const Identifiers*>(this)->GetID(name);

	if (id == InvalidID)
	{
		id = static_cast<uint4>( m_identifiers.size() );
		m_identifiers.push_back( name.to<utf8_string>() );
	}

	return id;
}

// Adds the given identifier to this identifier manager.
uint4 Identifiers::GetID(const utf8_ntri &name) const
{
	identifier_vector::const_iterator it = std::find(m_identifiers.begin(), m_identifiers.end(), name);

	return (it != m_identifiers.end())
		? static_cast<uint4>( it - m_identifiers.begin() )
		: InvalidID;
}

} // namespace
