/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_PROCESSINGPIPELINE
#define BE_SCENE_PROCESSINGPIPELINE

#include "beScene.h"
#include <beCore/beShared.h>
#include "bePipelineProcessor.h"
#include <vector>
#include <lean/smart/resource_ptr.h>

namespace beScene
{

// Prototypes
class RenderContext;

/// Processing Pipeline.
class ProcessingPipeline : public PipelineProcessor
{
private:
	utf8_string m_name;

	typedef std::vector< lean::resource_ptr<const Processor> > processor_vector;
	processor_vector m_processors;

	typedef std::vector< lean::resource_ptr<const PipelineProcessor> > pipeline_processor_vector;
	pipeline_processor_vector m_pipelineProcessors;

public:
	/// Constructor.
	BE_SCENE_API ProcessingPipeline(const utf8_ntri &name);
	/// Destructor.
	BE_SCENE_API ~ProcessingPipeline();

	/// Adds another processor.
	BE_SCENE_API void Add(const Processor *pProcessor);
	/// Removes a processor.
	BE_SCENE_API void Remove(const Processor *pProcessor);

	/// Adds another processor.
	BE_SCENE_API void Add(const PipelineProcessor *pProcessor);
	/// Removes a processor.
	BE_SCENE_API void Remove(const PipelineProcessor *pProcessor);

	/// Applies this processing pipeline (unclassified).
	BE_SCENE_API void Render(const Perspective *pPerspective, const RenderContext &context) const;
	/// Applies this processing pipeline (classified).
	BE_SCENE_API void Render(uint4 stageID, uint4 queueID, const Perspective *pPerspective, const RenderContext &context) const;

	/// Sets the name.
	BE_SCENE_API void SetName(const utf8_ntri &name);
	/// Gets the name.
	LEAN_INLINE const utf8_string& GetName() const { return m_name; }
};

} // nmaespace

#endif