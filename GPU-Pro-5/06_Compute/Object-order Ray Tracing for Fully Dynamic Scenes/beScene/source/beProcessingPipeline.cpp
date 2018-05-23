/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beProcessingPipeline.h"
#include <lean/functional/algorithm.h>

namespace beScene
{

// Constructor.
ProcessingPipeline::ProcessingPipeline(const utf8_ntri &name)
	: m_name(name.to<utf8_string>())
{
}

// Destructor.
ProcessingPipeline::~ProcessingPipeline()
{
}

// Adds another processor.
void ProcessingPipeline::Add(const Processor *pProcessor)
{
	LEAN_ASSERT(pProcessor != nullptr);

	m_processors.push_back(pProcessor);
}

// Removes a processor.
void ProcessingPipeline::Remove(const Processor *pProcessor)
{
	lean::remove(m_processors, pProcessor);
}

// Adds another processor.
void ProcessingPipeline::Add(const PipelineProcessor *pProcessor)
{
	LEAN_ASSERT(pProcessor != nullptr);

	m_pipelineProcessors.push_back(pProcessor);
	m_processors.push_back(pProcessor);
}

// Removes a processor.
void ProcessingPipeline::Remove(const PipelineProcessor *pProcessor)
{
	lean::remove(m_pipelineProcessors, pProcessor);
	lean::remove(m_processors, pProcessor);
}

// Applies this processing pipeline.
void ProcessingPipeline::Render(const Perspective *pPerspective, const RenderContext &context) const
{
	for (processor_vector::const_iterator it = m_processors.begin(); it != m_processors.end(); ++it)
		(*it)->Render(pPerspective, context);
}

// Applies this processing pipeline (classified).
void ProcessingPipeline::Render(uint4 stageID, uint4 queueID, const Perspective *pPerspective, const RenderContext &context) const
{
	for (pipeline_processor_vector::const_iterator it = m_pipelineProcessors.begin(); it != m_pipelineProcessors.end(); ++it)
		(*it)->Render(stageID, queueID, pPerspective, context);
}

// Sets the name.
void ProcessingPipeline::SetName(const utf8_ntri &name)
{
	m_name.assign(name.begin(), name.end());
}

} // namespace
