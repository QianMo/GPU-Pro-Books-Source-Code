/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_PIPELINE_PROCESSOR
#define BE_SCENE_PIPELINE_PROCESSOR

#include "beScene.h"
#include "beProcessor.h"

namespace beScene
{

// Prototypes.
class Perspective;
class RenderContext;

/// Processor base.
class PipelineProcessor : public Processor
{
protected:
	PipelineProcessor& operator =(const PipelineProcessor&) { return *this; }

public:
	using Processor::Render;
	/// Applies this processor.
	virtual void Render(uint4 stageID, uint4 queueID, const Perspective *pPerspective, const RenderContext &context) const = 0;
};

} // namespace

#endif