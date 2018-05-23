/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_QUAD_PROCESSOR
#define BE_SCENE_QUAD_PROCESSOR

#include "beScene.h"
#include "bePipelineProcessor.h"
#include <lean/smart/resource_ptr.h>
#include <lean/pimpl/forward_val.h>
#include <beGraphics/beDevice.h>
#include "beMaterialDriven.h"

namespace beScene
{

// Prototypes
class Mesh;
class AbstractProcessingEffectDriver;
template <class EffectBinder>
class EffectDriverCache;
class Perspective;
class RenderContext;

/// Quad processor.
class QuadProcessor : public PipelineProcessor, public MaterialDriven
{
private:
	lean::resource_ptr< EffectDriverCache<AbstractProcessingEffectDriver> > m_pEffectBinderCache;

protected:
	/// Device.
	lean::resource_ptr<const beGraphics::Device> m_pDevice;

	/// Quad mesh.
	lean::resource_ptr<Mesh> m_pQuad;

private:
	/// Pass.
	struct Layer;
	typedef std::vector<Layer> layer_vector;
	layer_vector m_layers;

	/// Applies this processor.
	void Render(uint4 layerIdx, uint4 stageID, uint4 queueID, const Perspective *pPerspective, const RenderContext &context, bool &bProcessorReady) const;

public:
	/// Constructor.
	BE_SCENE_API QuadProcessor(const beGraphics::Device *pDevice, EffectDriverCache<AbstractProcessingEffectDriver> *pEffectBinderCache);
	/// Destructor.
	BE_SCENE_API ~QuadProcessor();

	/// Applies this processor (unclassified passes only).
	BE_SCENE_API void Render(const Perspective *pPerspective, const RenderContext &context) const;
	/// Applies this processor (classified passes only).
	BE_SCENE_API void Render(uint4 stageID, uint4 queueID, const Perspective *pPerspective, const RenderContext &context) const;

	/// Applies this processor (unclassified passes only).
	BE_SCENE_API void Render(uint4 layerIdx, const Perspective *pPerspective, const RenderContext &context) const;
	/// Applies this processor (classified passes only).
	BE_SCENE_API void Render(uint4 layerIdx, uint4 stageID, uint4 queueID, const Perspective *pPerspective, const RenderContext &context) const;

	/// Sets the processing effect.
	BE_SCENE_API virtual void SetMaterial(beGraphics::Material *pSetup);
};

} // namespace

#endif