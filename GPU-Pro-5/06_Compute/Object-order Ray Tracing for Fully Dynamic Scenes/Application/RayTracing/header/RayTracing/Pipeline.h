#pragma once

#include "Tracing.h"

#include <beScene/beResourceManager.h>
#include <beScene/beEffectDrivenRenderer.h>

#include <beScene/beRenderingController.h>

#include <beScene/bePipelinePerspective.h>

#include <lean/pimpl/pimpl_ptr.h>
#include <lean/smart/resource_ptr.h>

namespace app
{

namespace tracing
{

/// Tracing pipeline class.
class Pipeline
{
public:
	struct M;

private:
	lean::pimpl_ptr<M> m;

public:
	/// Constructor.
	Pipeline(besc::EffectDrivenRenderer *renderer, besc::ResourceManager *resourceManager);
	/// Destructor.
	~Pipeline();

	typedef std::vector<float> benchmark_vector;

	/// Processes / commits changes.
	void Commit();

	/// Renders the scene from the given perspective.
	void Render(besc::RenderingController *scene, besc::PipelinePerspective &perspective, benchmark_vector *pBenchmark = nullptr);

	/// Sets the bounds of the scene.
	void SetBounds(const bem::fvec3 &min, const bem::fvec3 &max);
	/// Sets the number of triangles.
	void SetBatchSize(uint4 triCount);

	/// Gets the number of warm-up passes.
	uint4 GetWarmupPassCount() const;

	/// Gets a modified tracing renderer.
	besc::EffectDrivenRenderer *GetTracingRenderer();
};

} // namespace

} // namespace