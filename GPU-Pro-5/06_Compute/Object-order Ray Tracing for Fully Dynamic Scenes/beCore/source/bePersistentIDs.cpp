/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beCoreInternal/stdafx.h"
#include "beCore/bePersistentIDs.h"
#include <lean/functional/algorithm.h>
#include <lean/logging/errors.h>

namespace beCore
{

namespace
{

struct ReferenceOrder
{
	LEAN_INLINE bool operator ()(const PersistentIDs::Reference &left, const PersistentIDs::Reference &right) const
	{
		return left.id < right.id;
	}
	LEAN_INLINE bool operator ()(uint8 leftID, const PersistentIDs::Reference &right) const
	{
		return leftID < right.id;
	}
	LEAN_INLINE bool operator ()(const PersistentIDs::Reference &left, uint8 rightID) const
	{
		return left.id < rightID;
	}
};

}

// Constructor.
PersistentIDs::PersistentIDs(uint8 startID)
	: m_nextID(startID)
{
}

// Destructor.
PersistentIDs::~PersistentIDs()
{
}

// Reserves an ID.
uint8 PersistentIDs::ReserveID()
{
	return m_nextID++;
}

// Gets the next ID.
uint8 PersistentIDs::GetNextID() const
{
	return m_nextID;
}

// Skips all IDs up to the given the next ID.
void PersistentIDs::SkipIDs(uint8 nextID)
{
	if (nextID > m_nextID && nextID != InvalidID)
		m_nextID = nextID;
}

// Adds a new reference.
uint8 PersistentIDs::AddReference(void *ptr, const std::type_info &type)
{
	m_references.push_back( Reference(m_nextID, ptr, type) );
	return m_nextID++;
}

// Updates a reference.
bool PersistentIDs::SetReference(uint8 id, void *ptr, const std::type_info &type, bool bNoOverwrite)
{
	if (id >= m_nextID)
	{
		if (id != InvalidID)
		{
			// Jump ahead
			m_nextID = id;

			// Append
			lean::check( AddReference(ptr, type) == id );
			return true;
		}
		else
		{
			LEAN_LOG_ERROR_MSG("Cannot set reference for InvalidID");
			LEAN_ASSERT_DEBUG( id != InvalidID );
			return false;
		}
	}
	else
	{
		ref_vector::iterator itReference = lean::find_sorted(
				m_references.begin(), m_references.end(),
				id, ReferenceOrder()
			);

		// Update
		if (itReference != m_references.end())
		{
			if (!bNoOverwrite || !itReference->pointer)
			{
				itReference->pointer = ptr;
				itReference->type = &type;
			}
			else
				return (itReference->pointer == ptr);
		}
		// Insert
		else
			lean::push_sorted( m_references, Reference(id, ptr, type), ReferenceOrder() );

		return true;
	}
}

// Gets a reference.
void* PersistentIDs::GetReference(uint8 id, const std::type_info &type) const
{
	ref_vector::const_iterator itReference = lean::find_sorted(
			m_references.begin(), m_references.end(),
			id, ReferenceOrder()
		);

	return (itReference != m_references.end() && *itReference->type == type)
		? itReference->pointer
		: nullptr;
}

// Unsets a reference.
void PersistentIDs::UnsetReference(uint8 id, const void *ptr, bool bErase)
{
	ref_vector::iterator itReference = lean::find_sorted(
			m_references.begin(), m_references.end(),
			id, ReferenceOrder()
		);

	if (itReference != m_references.end() && (!ptr || itReference->pointer == ptr))
	{
		if (bErase)
			m_references.erase(itReference);
		else
		{
			itReference->pointer = nullptr;
			itReference->type = &typeid(void);
		}
	}
}

} // namespace
