/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beCoreInternal/stdafx.h"
#include "beCore/beSerializationJobs.h"

#include <boost/ptr_container/ptr_list.hpp>

namespace beCore
{

/// Implementation.
struct SaveJobs::M
{
	// NOTE: ptr_list template does not support const pointers?
	typedef boost::ptr_sequence_adapter< const SaveJob, std::list<const void*> > job_list;
	job_list jobs;
};

// Constructor.
SaveJobs::SaveJobs()
	: m( new M() )
{
}

// Destructor.
SaveJobs::~SaveJobs()
{
}

// Takes ownership of the given serialization job.
void SaveJobs::AddSerializationJob(const SaveJob *pJob)
{
	m->jobs.push_back( LEAN_ASSERT_NOT_NULL(pJob) );
}

// Saves anything, e.g. to the given XML root node.
void SaveJobs::Save(rapidxml::xml_node<lean::utf8_t> &root, beCore::ParameterSet &parameters)
{
	// Execute all jobs
	for (M::job_list::const_iterator it = m->jobs.begin(); it != m->jobs.end(); ++it)
		// NOTE: New jobs might be added any time
		it->Save(root, parameters, *this);
}

/// Implementation.
struct LoadJobs::M
{
	// NOTE: ptr_list template does not support const pointers?
	typedef boost::ptr_sequence_adapter< const LoadJob, std::list<const void*> > job_list;
	job_list jobs;
};

// Constructor.
LoadJobs::LoadJobs()
	: m( new M() )
{
}

// Destructor.
LoadJobs::~LoadJobs()
{
}

// Takes ownership of the given serialization job.
void LoadJobs::AddSerializationJob(const LoadJob *pJob)
{
	m->jobs.push_back( LEAN_ASSERT_NOT_NULL(pJob) );
}

// Loads anything, e.g. from the given XML root node.
void LoadJobs::Load(const rapidxml::xml_node<lean::utf8_t> &root, beCore::ParameterSet &parameters)
{
	// Execute all jobs
	for (M::job_list::const_iterator it = m->jobs.begin(); it != m->jobs.end(); ++it)
		// NOTE: New jobs might be added any time
		it->Load(root, parameters, *this);
}

} // namespace
