#pragma once

#include "Tracing.h"

#include <beScene/beResourceManager.h>
#include <beScene/beEffectDrivenRenderer.h>

#include <beEntitySystem/beSimulation.h>
#include <beScene/beRenderingController.h>

#include <beEntitySystem/beWorld.h>

#include <lean/smart/resource_ptr.h>
#include <lean/smart/scoped_ptr.h>

#include <beMath/beMatrixDef.h>

namespace app
{

/// Scene class.
class Scene
{
private:
	lean::resource_ptr<besc::EffectDrivenRenderer> m_pRenderer;
	lean::resource_ptr<besc::ResourceManager> m_pResourceManager;

	lean::resource_ptr<bees::World> m_pWorld;
	besc::RenderingController *m_pScene;

	lean::resource_ptr<bees::Simulation> m_pSimulation;

	bem::fvec3 m_min;
	bem::fvec3 m_max;

	uint4 m_triangleCount, m_maxTriangleCount, m_batchCount;

public:
	/// Constructor.
	Scene(const utf8_ntri &file, besc::EffectDrivenRenderer *pRenderer, besc::ResourceManager *pResourceManager);
	/// Destructor.
	~Scene();

	/// Steps the scene.
	void Step(float timeStep);
	/// Renders the scene.
	void Render();

	/// Gets the bounds.
	const bem::fvec3& GetMin() const { return m_min; };
	/// Gets the bounds.
	const bem::fvec3& GetMax() const { return m_max; };
	/// Gets the number of triangles.
	uint4 GetTriangleCount() const { return m_triangleCount; };
	/// Gets the max number of triangles in one batch.
	uint4 GetMaxTriangleCount() const { return m_maxTriangleCount; };
	/// Gets the number of batches.
	uint4 GetBatchCount() const { return m_batchCount; };

	// Gets the world.
	bees::World* GetWorld() const { return m_pWorld; }

	/// Gets the serialization parameters.
	bec::ParameterSet GetSerializationParameters() const;

	/// Gets the scene.
	besc::RenderingController* GetScene() const { return m_pScene; }
	/// Gets the simulation.
	bees::Simulation* GetSimulation() const { return m_pSimulation; }

	/// Gets the renderer.
	besc::EffectDrivenRenderer* GetRenderer() const { return m_pRenderer; }
	/// Gets the resource manager.
	besc::ResourceManager* GetResourceManager() const { return m_pResourceManager; }
};

} // namespace