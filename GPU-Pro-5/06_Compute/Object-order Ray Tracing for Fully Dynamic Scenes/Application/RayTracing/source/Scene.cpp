/******************************************************/
/* Object-order Ray Tracing Demo (c) Tobias Zirr 2013 */
/******************************************************/

#include "stdafx.h"

#include "Scene.h"

#include <beScene/beRenderContext.h>

#include <beEntitySystem/beSerializationParameters.h>
#include <beScene/beSerializationParameters.h>

#include <beEntitySystem/beEntities.h>
#include <beEntitySystem/beWorld.h>
#include <beEntitySystem/beWorldControllers.h>
#include <beScene/beMeshControllers.h>
#include <beScene/beRenderableMesh.h>
#include <beScene/DX11/beMesh.h>
#include <beScene/beMeshCompound.h>

#include <beGraphics/Any/beDevice.h>
#include <beGraphics/Any/beTexture.h>

#include <beMath/beAAB.h>

#include <lean/logging/log.h>

namespace app
{

// Constructor.
Scene::Scene(const utf8_ntri &file, besc::EffectDrivenRenderer *pRenderer, besc::ResourceManager *pResourceManager)
	: m_pRenderer( LEAN_ASSERT_NOT_NULL(pRenderer) ),
	m_pResourceManager( LEAN_ASSERT_NOT_NULL(pResourceManager) )
{
	lean::scoped_ptr<bees::WorldControllers> controllers = new_scoped bees::WorldControllers();
	m_pScene = new besc::RenderingController(
			m_pRenderer->Pipeline(),
			besc::CreateRenderContext(*m_pRenderer->ImmediateContext()).get()
		);
	controllers->AddControllerConsume(m_pScene);
	
	m_pWorld = new_resource bees::World("World", file, GetSerializationParameters(), controllers.move_ptr());
	
	m_pSimulation = new_resource bees::Simulation("Scene");
	Attach(m_pWorld, m_pSimulation);

	m_triangleCount = 0;
	m_maxTriangleCount = 0;
	m_batchCount = 0;

	m_min = 2.0e32f, m_max = -2.0e32f;

	for (bees::Entities::Range entities = m_pWorld->Entities()->GetEntities(); entities.Begin < entities.End; ++entities.Begin)
	{
		bees::Entity *entity = *entities.Begin;

		if (entity->GetName() == "Sky")
			continue;

		besc::MeshController *meshController = entity->GetController<besc::MeshController>();
		auto transform = entity->GetTransformation();
		auto matrix =  mat_transform(
				transform.Position,
				transform.Orientation[2] * transform.Scaling[2],
				transform.Orientation[1] * transform.Scaling[1],
				transform.Orientation[0] * transform.Scaling[0]
			);

		if (meshController)
		{
			for (besc::RenderableMesh::MeshRange meshes = meshController->GetMesh()->GetMeshes(); meshes.Begin < meshes.End; ++meshes.Begin)
			{
				uint4 batchTriCount = ToImpl(*meshes.Begin)->GetIndexCount() / 3;
				m_triangleCount += batchTriCount;
				m_maxTriangleCount = max(batchTriCount, m_maxTriangleCount);
				++m_batchCount;
			}

//			meshController->UpdateFromSubsets();
			besc::RenderableMesh const* meshes = meshController->GetMesh();
			bem::faab3 b = besc::ComputeBounds(meshes->GetMeshes().Begin, Size4(meshes->GetMeshes()));
			b = mulh(b, matrix);
			m_min = min_cw(b.min, m_min);
			m_max = max_cw(b.max, m_max);

//			bem::fsphere3 s = meshController->GetLocalBounds();
//			bem::fvec3 scaling = entity->GetScaling();
//			s.center *= scaling;
//			s.radius *= lean::max( lean::max(abs(scaling[0]), abs(scaling[1])), abs(scaling[2]) );
//			s.center += entity->GetPosition();

//			m_min = min_cw(s.center - s.radius, m_min);
//			m_max = max_cw(s.center + s.radius, m_max);
		}
	}

	lean::debug_stream() << "Total number of triangles: " << m_triangleCount << std::endl;
	lean::debug_stream() << "Max batch number of triangles: " << m_maxTriangleCount << std::endl;
}

// Destructor.
Scene::~Scene()
{
}

// Gets the serialization parameters.
bec::ParameterSet Scene::GetSerializationParameters() const
{
	bec::ParameterSet parameters( &bees::GetSerializationParameters() );
	
	bees::SetEntitySystemParameters(
			parameters,
			bees::EntitySystemParameters(m_pWorld)
		);

	besc::SetSceneParameters(
			parameters,
			besc::SceneParameters(m_pResourceManager, m_pRenderer, m_pScene)
		);

	return parameters;
}

// Steps the simulation.
void Scene::Step(float timeStep)
{
	m_pSimulation->Fetch();

	m_pSimulation->Step(timeStep);

	m_pSimulation->Flush();
}

// Renders the scene.
void Scene::Render()
{
	m_pRenderer->InvalidateCaches();
	m_pSimulation->Render();
}

} // namespace