/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#include "beEntitySystemInternal/stdafx.h"
#include "beEntitySystem/beSynchronizedHost.h"
#include <lean/functional/algorithm.h>
#include <lean/logging/errors.h>

namespace beEntitySystem
{

// Constructor.
SynchronizedHost::SynchronizedHost()
{
}

// Destructor.
SynchronizedHost::~SynchronizedHost()
{
}

// Synchronizes synchronized objects with the simulation.
void SynchronizedHost::Flush()
{
	for (synchronized_vector::const_iterator it = m_synchFlush.begin(); it != m_synchFlush.end(); ++it)
		(*it)->Flush();
}

// Synchronizes the simulation with synchronized objects.
void SynchronizedHost::Fetch()
{
	for (synchronized_vector::const_iterator it = m_synchFetch.begin(); it != m_synchFetch.end(); ++it)
		(*it)->Fetch();
}

// Adds a synchronized controller.
void SynchronizedHost::AddSynchronized(Synchronized *synchronized, uint4 flags)
{
	if (!synchronized)
	{
		LEAN_LOG_ERROR_MSG("synchronized may not be nullptr");
		return;
	}

	if (flags & SynchronizedFlags::Flush)
		lean::push_unique(m_synchFlush, synchronized);
	if (flags & SynchronizedFlags::Fetch)
		lean::push_unique(m_synchFetch, synchronized);
}

// Removes a synchronized controller.
void SynchronizedHost::RemoveSynchronized(Synchronized *synchronized, uint4 flags)
{
	if (flags & SynchronizedFlags::Flush)
		lean::remove(m_synchFlush, synchronized);
	if (flags & SynchronizedFlags::Fetch)
		lean::remove(m_synchFetch, synchronized);
}

} // namespace
