/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beCoreInternal/stdafx.h"
#include "beCore/beComponentMonitor.h"
#include <string>
#include <lean/functional/algorithm.h>
#include <lean/concurrent/critical_section.h>

namespace beCore
{

struct ComponentMonitorChannel::M
{
	lean::critical_section cs;
};

// Constructor.
ComponentMonitorChannel::ComponentMonitorChannel()
	: m(new M)
{
}

// Destructor.
ComponentMonitorChannel::~ComponentMonitorChannel()
{
}

// Marks components of the given type as changed.
void ComponentMonitorChannel::AddChanged(const ComponentType *type)
{
	lean::scoped_cs_lock lock(m->cs);

	lean::push_sorted_unique(m_changedTypes, type);
}

// Processes the next batch of changes.
void ComponentMonitorChannel::Process()
{
	lean::scoped_cs_lock lock(m->cs);

	m_processedTypes.clear();
	swap(m_processedTypes, m_changedTypes);
}

// Checks if there are more changes to process.
bool ComponentMonitorChannel::ChangesPending() const
{
	return !m_changedTypes.empty();
}

// Checks if components of the given type have been changed.
bool ComponentMonitorChannel::HasChanged(const ComponentType *type) const
{
	return lean::find_sorted(m_processedTypes.begin(), m_processedTypes.end(), type) != m_processedTypes.end();
}

} // namespace
