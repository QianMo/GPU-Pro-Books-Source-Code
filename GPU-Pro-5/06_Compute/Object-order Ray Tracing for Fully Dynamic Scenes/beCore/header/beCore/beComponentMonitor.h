/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_COMPONENT_MONITOR
#define BE_CORE_COMPONENT_MONITOR

#include "beCore.h"
#include "beShared.h"
#include <lean/tags/noncopyable.h>
#include <vector>
#include <lean/pimpl/pimpl_ptr.h>

namespace beCore
{

struct ComponentType;

/// Tracks component changes by type.
class ComponentMonitorChannel : public lean::noncopyable_chain<Shared>
{
	struct M;

private:
	typedef std::vector<const ComponentType*> types_t;
	types_t m_changedTypes;
	types_t m_processedTypes;

	lean::pimpl_ptr<M> m;

public:
	/// Constructor.
	BE_CORE_API ComponentMonitorChannel();
	/// Destructor.
	BE_CORE_API ~ComponentMonitorChannel();

	/// Marks components of the given type as changed.
	BE_CORE_API void AddChanged(const ComponentType *type);
	/// Processes the next batch of changes.
	BE_CORE_API void Process();
	/// Checks if there are more changes to process.
	BE_CORE_API bool ChangesPending() const;
	/// Checks if any components have changed.
	BE_CORE_API bool HasChanges() const { return !m_processedTypes.empty(); }
	/// Checks if components of the given type have been changed.
	BE_CORE_API bool HasChanged(const ComponentType *type) const;
};

/// Tracks component changes by type.
struct ComponentMonitor : public Resource
{
	ComponentMonitorChannel Data;			///< Some data has changed.
	ComponentMonitorChannel Structure;		///< The structure / layout has changed.
	ComponentMonitorChannel Replacement;	///< Some resources have been replaced.
	ComponentMonitorChannel Management;		///< The ownership / metadata of some resources has changed.

	/// Processes the next batch of changes.
	void Process()
	{
		Data.Process();
		Structure.Process();
		Replacement.Process();
		Management.Process();
	}

	/// Checks for more changes.
	bool ChangesPending() const
	{
		return Data.ChangesPending()
			|| Structure.ChangesPending() 
			|| Replacement.ChangesPending()
			|| Management.ChangesPending();
	}
};

} // namespace

#endif