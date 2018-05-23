/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beCoreInternal/stdafx.h"
#include "beCore/bePropertyProvider.h"

#include "beCore/beComponentObservation.h"

#include "beCore/bePropertyVisitor.h"
#include <lean/functional/predicates.h>

namespace beCore
{

struct ComponentObserverCollection::ListenerEntry
{
	ComponentObserver *listener;
	listeners_t next;

	ListenerEntry(ComponentObserver *listener)
		: listener(listener) { }
};

// Constructor.
ComponentObserverCollection::ComponentObserverCollection()
{
}

// Copies the given collection.
ComponentObserverCollection::ComponentObserverCollection(const ComponentObserverCollection &right)
{
	listeners_t *listPtr = &m_listeners;

	for (const ListenerEntry *it = right.m_listeners; it; it = it->next)
	{
		*listPtr = new ListenerEntry(it->listener);
		listPtr = &(**listPtr).next;
	}
}

#ifndef LEAN0X_NO_RVALUE_REFERENCES
// Takes over the given collection.
ComponentObserverCollection::ComponentObserverCollection(ComponentObserverCollection &&right)
	: m_listeners( std::move(right.m_listeners) )
{
}
#endif

// Destructor.
ComponentObserverCollection::~ComponentObserverCollection()
{
}

// Adds a property listener.
void ComponentObserverCollection::AddObserver(ComponentObserver *listener)
{
	listeners_t *listEndPtr = &m_listeners;

	for (; *listEndPtr; listEndPtr = &(**listEndPtr).next)
		if ((**listEndPtr).listener == listener)
			return;

	*listEndPtr = new ListenerEntry( LEAN_ASSERT_NOT_NULL(listener) );
}

// Removes a property listener.
void ComponentObserverCollection::RemoveObserver(ComponentObserver *pListener)
{
	listeners_t *listPtr = &m_listeners;

	for (ListenerEntry *it = m_listeners; it; listPtr = &it->next, it = *listPtr)
		if (it->listener == pListener)
		{
			*listPtr = it->next.detach();
			return;
		}
}

// Calls all listeners.
void ComponentObserverCollection::EmitPropertyChanged(const PropertyProvider &provider) const
{
	for (const ListenerEntry *it = m_listeners; it; it = it->next)
		it->listener->PropertyChanged(provider);
}

// Calls all listeners.
void ComponentObserverCollection::EmitChildChanged(const ReflectedComponent &provider) const
{
	for (const ListenerEntry *it = m_listeners; it; it = it->next)
		it->listener->ChildChanged(provider);
}

// Calls all listeners.
void ComponentObserverCollection::EmitStructureChanged(const Component &provider) const
{
	for (const ListenerEntry *it = m_listeners; it; it = it->next)
		it->listener->StructureChanged(provider);
}

// Calls all listeners.
void ComponentObserverCollection::EmitComponentReplaced(const Component &previous) const
{
	for (const ListenerEntry *it = m_listeners; it; it = it->next)
		it->listener->ComponentReplaced(previous);
}

namespace
{

struct PropertyTransfer : public PropertyVisitor
{
	PropertyProvider *dest;
	uint4 destID;

	PropertyTransfer(PropertyProvider &dest, uint4 id)
		: dest(&dest), destID(id) { }

	void Visit(const PropertyProvider &provider, uint4 propertyID, const PropertyDesc &desc, const void *values)
	{
		dest->SetProperty(destID, desc.TypeDesc->Info.type, values, desc.Count);
	}
};

} // namespace

// Transfers all from the given source property provider to the given destination property provider.
void TransferProperties(PropertyProvider &dest, const PropertyProvider &source)
{
	const uint4 count = dest.GetPropertyCount();
	const uint4 srcCount = source.GetPropertyCount();

	uint4 nextID = 0;

	for (uint4 srcID = 0; srcID < srcCount; ++srcID)
	{
		utf8_ntr srcName = source.GetPropertyName(srcID);

		uint4 lowerID = nextID;
		uint4 upperID = nextID;

		for (uint4 i = 0; i < count; ++i)
		{
			// Perform bi-directional search: even == forward; odd == backward
			uint4 id = (lean::is_odd(i) | (upperID == count)) & (lowerID != 0)
				? --lowerID
				: upperID++;

			if (dest.GetPropertyName(id) == srcName)
			{
				PropertyTransfer transfer(dest, id);
				source.ReadProperty(srcID, transfer);

				// Start next search with next property
				nextID = id + 1;
				break;
			}
		}
	}
}

} // namespace
