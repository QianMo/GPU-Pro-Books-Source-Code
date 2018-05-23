/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2013 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beMeshControllers.h"

#include <beEntitySystem/beEntities.h>

#include <beCore/beReflectionProperties.h>
#include <beCore/bePersistentIDs.h>

#include "beScene/beRenderingPipeline.h"
#include "beScene/bePipelinePerspective.h"
#include "beScene/bePerspectiveStatePool.h"
#include "beScene/beQueueStatePool.h"

#include "beScene/DX11/beMesh.h"
#include "beScene/beRenderableMesh.h"
#include "beScene/beRenderableMaterial.h"
#include "beScene/beRenderableEffectData.h"
#include "beScene/beAbstractRenderableEffectDriver.h"

#include "beScene/beRenderContext.h"
#include <beGraphics/Any/beDevice.h>
#include <beGraphics/Any/beStateManager.h>
#include <beGraphics/Any/beDeviceContext.h>

#include <beMath/beVector.h>
#include <beMath/beMatrix.h>
#include <beMath/beSphere.h>
#include <beMath/bePlane.h>
#include <beMath/beUtility.h>

#include <lean/functional/algorithm.h>

#include <lean/memory/chunk_pool.h>
#include <lean/containers/simple_vector.h>
#include <lean/containers/multi_vector.h>

#include <lean/io/numeric.h>

#include <beGraphics/DX/beError.h>

namespace beScene
{

BE_CORE_PUBLISH_COMPONENT(MeshController)
BE_CORE_PUBLISH_COMPONENT(MeshControllers)

class MeshControllers::M : public MeshControllers
{
public:
	// Controller data
	struct Record
	{
		MeshController *Reflected;
		lean::resource_ptr<RenderableMesh> Mesh;
		uint8 PersistentID;

		Record(MeshController *reflected)
			: Reflected(reflected),
			PersistentID(-1) { }
	};

	struct Configuration
	{
		bem::fsphere3 LocalBounds;
	};

	struct State
	{
		Configuration Config;
		bool Visible : 1;
		bool Attached : 1;

		State()
			 : Visible(true),
			Attached(false) { }
	};
	
	enum record_tag { record };
	enum state_tag { state };
	enum bounds_tag { bounds };
	enum renderable_data_tag { renderableData };
	
	typedef lean::chunk_pool<MeshController, 128> handle_pool;
	handle_pool handles;

	typedef lean::multi_vector_t< lean::simple_vector_binder<lean::vector_policies::semipod> >::make<
			Record, record_tag,
			State, state_tag,
			bem::fsphere3, bounds_tag,
			RenderableEffectData, renderable_data_tag
		>::type controllers_t;
	

	// Rendering data
	struct Data
	{
		struct Queue : public QueueStateBase
		{
			struct Pass
			{
				lean::com_ptr<ID3D11InputLayout> inputLayout;
				const DX11::Mesh *mesh;
				const AbstractRenderableEffectDriver *effectDriver;
				const QueuedPass *pass;
				const beGraphics::MaterialTechnique *material;
			};

			typedef lean::simple_vector<uint4, lean::vector_policies::inipod> offsets_t;
			typedef lean::simple_vector<Pass, lean::vector_policies::semipod> passes_t;

			offsets_t meshLODsToPasses;
			passes_t passes;

			void Reset(PipelineQueueID id)
			{
				base_type::Reset(id);
				this->meshLODsToPasses.clear();
				this->passes.clear();
			}
		};
		
		struct MeshLOD
		{
			float distance;

			MeshLOD(float distance)
				: distance(distance) { }
		};
		
		typedef QueueStatePool<Queue, lean::vector_policies::semipod> queues_t;
		typedef lean::simple_vector<RenderableMesh*, lean::vector_policies::inipod> meshes_t;
		typedef lean::simple_vector<MeshLOD, lean::vector_policies::inipod> mesh_lods_t;
		typedef lean::simple_vector<bec::Range<uint4>, lean::vector_policies::inipod> lod_ranges_t;

		uint4 structureRevision;

		meshes_t uniqueMeshes;
		mesh_lods_t meshLODs;

		uint4 activeControllerCount;
		controllers_t controllers;
		lod_ranges_t controllersToMeshLODs;
		
		queues_t queues;

		Data()
			: structureRevision(0),
			activeControllerCount(0) { }
	};
	Data dataSets[2];
	Data *data, *dataAux;
	uint4 controllerRevision;
	
	struct PerspectiveState;
	mutable PerspectiveStatePool<PerspectiveState> perspectiveState;

	lean::resource_ptr<beCore::ComponentMonitor> pComponentMonitor;

	M()
		: data(dataSets),
		dataAux(&dataSets[1]),
		controllerRevision(0) { }

	/// Verifies the given handle.
	friend LEAN_INLINE bool VerifyHandle(const M &m, const MeshControllerHandle handle) { return handle.Index < m.data->controllers.size(); }

	/// Fixes all controller handles to match the layout of the given controller vector.
	static void FixControllerHandles(controllers_t &controllers, uint4 internalIdx = 0)
	{
		// Fix subsequent handles
		for (; internalIdx < controllers.size(); ++internalIdx)
			controllers(record)[internalIdx].Reflected->Handle().SetIndex(internalIdx);
	}

	/// Gets the number of child components.
	uint4 GetComponentCount() const { return static_cast<uint4>(data->controllers.size()); }
	/// Gets the name of the n-th child component.
	beCore::Exchange::utf8_string GetComponentName(uint4 idx) const { return MeshController::GetComponentType()->Name; }
	/// Gets the n-th reflected child component, nullptr if not reflected.
	lean::com_ptr<const ReflectedComponent, lean::critical_ref> GetReflectedComponent(uint4 idx) const { return data->controllers[idx].Reflected; }
};

/// Creates a collection of mesh controllers.
lean::scoped_ptr<MeshControllers, lean::critical_ref> CreateMeshControllers(beCore::PersistentIDs *persistentIDs)
{
	return lean::make_scoped<MeshControllers::M>();
}

namespace
{

void CommitExternalChanges(MeshControllers::M &m, beCore::ComponentMonitor &monitor)
{
	LEAN_FREE_PIMPL(MeshControllers);
	M::Data &data = *m.data;

	bool bHasChanges = monitor.Structure.HasChanged(RenderableMesh::GetComponentType()) ||
		monitor.Data.HasChanged(RenderableMesh::GetComponentType());

	if (monitor.Replacement.HasChanged(RenderableMesh::GetComponentType()))
	{
		uint4 controllerCount = (uint4) data.controllers.size();

		for (uint4 internalIdx = 0; internalIdx < controllerCount; ++internalIdx)
		{
			RenderableMesh *oldMesh = data.controllers(M::record)[internalIdx].Mesh;
			RenderableMesh *mesh = bec::GetSuccessor(oldMesh);

			if (mesh != oldMesh)
			{
				data.controllers(M::record)[internalIdx].Reflected->SetMesh(mesh);
				bHasChanges |= true;
			}
		}
	}

	if (bHasChanges)
		++m.controllerRevision;
}

struct MeshMaterialSorter
{
	const MeshControllers::M::controllers_t &v;

	MeshMaterialSorter(const MeshControllers::M::controllers_t &v)
		: v(v) { }

	LEAN_INLINE bool operator ()(uint4 l, uint4 r) const
	{
		LEAN_FREE_PIMPL(MeshControllers);

		const RenderableMesh *leftMesh = v[l].Mesh;
		bool leftAttached = leftMesh && v(M::state)[l].Attached;
		const RenderableMesh *rightMesh = v[r].Mesh;
		bool rightAttached = rightMesh && v(M::state)[r].Attached;

		// Move null meshes outwards
		if (!leftAttached)
			return false;
		else if (!rightAttached)
			return true;
		// Sort by mesh
		else if (leftMesh < rightMesh)
			return true;
		else if (leftMesh == rightMesh)
		{
			const RenderableMesh::MaterialRange leftMaterials = leftMesh->GetMaterials();
			const RenderableMesh::MaterialRange rightMaterials = rightMesh->GetMaterials();

			const RenderableMaterial *leftMaterial = (leftMaterials.Begin != leftMaterials.End) ? leftMaterials.Begin[0] : nullptr;
			const RenderableMaterial *rightMaterial = (rightMaterials.Begin != rightMaterials.End) ? rightMaterials.Begin[0] : nullptr;

			// Sort by material
			if (leftMaterial < rightMaterial)
				return true;
			else if (leftMaterial == rightMaterial)
			{
				const AbstractRenderableEffectDriver *leftDriver = (leftMaterial) ? leftMaterial->GetTechniques().Begin[0].TypedDriver() : nullptr;
				const AbstractRenderableEffectDriver *rightDriver = (rightMaterial) ? rightMaterial->GetTechniques().Begin[0].TypedDriver() : nullptr;

				// Sort by effect
				return leftDriver < rightDriver;
			}
		}
		return false;
	}
};

/// Sorts controllers by mesh, material and shader (moving null meshes outwards).
void SortControllers(MeshControllers::M::controllers_t &destControllers, const MeshControllers::M::controllers_t &srcControllers)
{
	LEAN_FREE_PIMPL(MeshControllers);
	
	uint4 controllerCount = (uint4) srcControllers.size();
	
	lean::scoped_ptr<uint4[]> sortIndices( new uint4[controllerCount] );
	std::generate_n(&sortIndices[0], controllerCount, lean::increment_gen<uint4>(0));
	std::sort(&sortIndices[0], &sortIndices[controllerCount], MeshMaterialSorter(srcControllers));
	
	destControllers.clear();
	lean::append_swizzled(srcControllers, &sortIndices[0], &sortIndices[controllerCount], destControllers);
}

uint4 LinkControllersToUniqueMeshesAndCountLODs(MeshControllers::M::Data &data)
{
	LEAN_FREE_PIMPL(MeshControllers);

	const uint4 controllerCount = (uint4) data.controllers.size();
	data.controllersToMeshLODs.resize(controllerCount);
	data.uniqueMeshes.clear();
	data.activeControllerCount = 0;

	const RenderableMesh *prevMesh = nullptr;
	bec::Range<uint4> prevMeshLODRange;

	for (uint4 internalIdx = 0; internalIdx < controllerCount; ++internalIdx)
	{
		const M::Record &controller = data.controllers[internalIdx];

		// Ignore null & detached meshes at the back
		if (!controller.Mesh || !data.controllers(M::state)[internalIdx].Attached)
			break;

		++data.activeControllerCount;

		// Add new unique mesh
		if (prevMesh != controller.Mesh)
		{
			data.uniqueMeshes.push_back(controller.Mesh);
			
			prevMesh = controller.Mesh;
			prevMeshLODRange.Begin = prevMeshLODRange.End;
			prevMeshLODRange.End += Size4(prevMesh->GetLODs());
		}

		// Let controllers reference unique mesh LOD ranges
		data.controllersToMeshLODs[internalIdx] = prevMeshLODRange;
	}

	// NOTE: Mesh LODs initialized during construction of queues, see below
	return prevMeshLODRange.End;
}

MeshControllers::M::Data::Queue::Pass ConstructPass(const DX11::Mesh *mesh, const beGraphics::MaterialTechnique *material,
	const AbstractRenderableEffectDriver *effectDriver, const QueuedPass *meshPass)
{
	LEAN_FREE_PIMPL(MeshControllers);

	M::Data::Queue::Pass pass;
	
	uint4 passSignatureSize = 0;
	const char *passSignature = meshPass->GetInputSignature(passSignatureSize);
	BE_THROW_DX_ERROR_MSG(
		beGraphics::Any::GetDevice(*mesh->GetVertexBuffer())->CreateInputLayout(
			mesh->GetVertexElementDescs(),
			mesh->GetVertexElementDescCount(),
			passSignature, passSignatureSize,
			pass.inputLayout.rebind()),
		"ID3D11Device::CreateInputLayout()");

	pass.mesh = mesh;
	pass.material = material;
	pass.pass = meshPass;
	pass.effectDriver = effectDriver;

	return pass;
}

void AddTechniquePasses(MeshControllers::M::Data &data, uint4 meshLODIdx, const DX11::Mesh *mesh, RenderableMaterial::Technique technique)
{
	LEAN_FREE_PIMPL(MeshControllers);

	AbstractRenderableEffectDriver::PassRange passes = technique.TypedDriver()->GetPasses();

	for (uint4 passIdx = 0, passCount = Size4(passes); passIdx < passCount; ++passIdx)
	{
		const QueuedPass *pass = &passes[passIdx];
		PipelineQueueID queueID(pass->GetStageID(), pass->GetQueueID());
		M::Data::Queue &queue = data.queues.GetQueue(queueID);

		// Link mesh LOD to beginning of pass range & insert pass
		queue.meshLODsToPasses.resize( meshLODIdx + 1, (uint4) queue.passes.size());
		queue.passes.push_back( ConstructPass(mesh, technique.Technique, technique.TypedDriver(), pass) );
	}
}

void AddSubsetPasses(MeshControllers::M::Data &data, uint4 meshLODIdx,
					 bec::Range<uint4> subsets, RenderableMesh::MeshRange meshes, RenderableMesh::MaterialRange materials)
{
	LEAN_FREE_PIMPL(MeshControllers);

	for (uint4 subsetIdx = subsets.Begin; subsetIdx < subsets.End; ++subsetIdx)
	{
		const DX11::Mesh *mesh = ToImpl(meshes[subsetIdx]);
		const RenderableMaterial *material = materials[subsetIdx];

		// NOTE: Ignore incomplete subsets
		if (!mesh || !material)
			continue;

		RenderableMaterial::TechniqueRange meshTechniques = material->GetTechniques();

		for (uint4 techniqueIdx = 0, techniqueCount = Size4(meshTechniques); techniqueIdx < techniqueCount; ++techniqueIdx)
			AddTechniquePasses(data, meshLODIdx, mesh, meshTechniques[techniqueIdx]);
	}
}

void CollectLODsAndBuildQueues(MeshControllers::M::Data &data, uint4 totalLODCount)
{
	LEAN_FREE_PIMPL(MeshControllers);

	data.meshLODs.clear();
	data.meshLODs.reserve(totalLODCount);

	data.queues.Clear();

	// Build queues from meshes
	for (uint4 uniqueMeshIdx = 0, uniqueMeshCount = (uint4) data.uniqueMeshes.size(); uniqueMeshIdx < uniqueMeshCount; ++uniqueMeshIdx)
	{
		const RenderableMesh *meshCompound = data.uniqueMeshes[uniqueMeshIdx];
		RenderableMesh::LODRange meshLODs = meshCompound->GetLODs();
		RenderableMesh::MeshRange meshes = meshCompound->GetMeshes();
		RenderableMesh::MaterialRange materials = meshCompound->GetMaterials();

		for (uint4 lod = 0, lodCount = Size4(meshLODs); lod < lodCount; ++lod)
		{
			const RenderableMesh::LOD &meshLOD = meshLODs[lod];

			uint4 meshLODIdx = (uint4) data.meshLODs.size();
			data.meshLODs.push_back( M::Data::MeshLOD(meshLOD.Distance) );

			AddSubsetPasses(data, meshLODIdx, meshLOD.Subsets, meshes, materials);
		}
	}

	// Discard unused queues
	data.queues.Shrink();

	// IMPORTANT: Finish implicit mesh LOD to pass offset ranges
	for (M::Data::Queue *it = data.queues.begin(), *itEnd = data.queues.end(); it < itEnd; ++it)
		it->meshLODsToPasses.resize(totalLODCount + 1, (uint4) it->passes.size());
}

} // namespace

// Commits changes.
void MeshControllers::Commit()
{
	LEAN_STATIC_PIMPL();

	if (m.pComponentMonitor)
		CommitExternalChanges(m, *m.pComponentMonitor);

	const M::Data &prevData = *m.data;
	M::Data &data = *m.dataAux;

	if (prevData.structureRevision != m.controllerRevision)
	{
		// Rebuild internal data structures in swap buffer
		SortControllers(data.controllers, prevData.controllers);
		uint4 totalLODCount = LinkControllersToUniqueMeshesAndCountLODs(data);
		CollectLODsAndBuildQueues(data, totalLODCount);
	
		data.structureRevision = m.controllerRevision;

		// Swap current data with updated data
		std::swap(m.data, m.dataAux);
		M::FixControllerHandles(m.data->controllers);
	}
}

struct MeshControllers::M::PerspectiveState : public PerspectiveStateBase<const MeshControllers::M, PerspectiveState>
{
	struct VisibleLOD
	{
		uint4 controllerIdx;
		uint4 lodIdx;

		VisibleLOD(uint4 controllerIdx, uint4 lodIdx)
			: controllerIdx(controllerIdx),
			lodIdx(lodIdx) { }
	};
	
	struct Queue : public QueueStateBase
	{
		struct VisiblePass
		{
			uint4 controllerIdx;
			uint4 passIdx;

			VisiblePass(uint4 controllerIdx, uint4 passIdx)
				: controllerIdx(controllerIdx),
				passIdx(passIdx) { }
		};

		typedef lean::simple_vector<VisiblePass, lean::vector_policies::inipod> visible_passes_t;
		visible_passes_t visiblePasses;

		void Reset(PipelineQueueID id)
		{
			base_type::Reset(id);
			this->visiblePasses.clear();
		}
	};

	struct ExternalPass
	{
		uint4 controllerIdx;
		const M::Data::Queue::Pass *pass;

		ExternalPass(uint4 controllerIdx, const M::Data::Queue::Pass *pass)
			: controllerIdx(controllerIdx),
			pass(pass) { }
	};

	typedef lean::simple_vector<VisibleLOD, lean::vector_policies::inipod> visible_lods_t;
	typedef lean::simple_vector<float, lean::vector_policies::inipod> distances_t;
	typedef QueueStatePool<Queue, lean::vector_policies::semipod> queues_t;
	typedef lean::simple_vector<ExternalPass, lean::vector_policies::inipod> external_passes_t;

	visible_lods_t visibleLODs;
	distances_t distances;
	queues_t queues;
	external_passes_t externalPasses;

	PerspectiveState(const MeshControllers::M *parent)
		: base_type(parent) { }
	
	void Reset(Perspective *perspective)
	{
		base_type::Reset(perspective);
		this->visibleLODs.clear();
		this->distances.clear();
		this->queues.Reset();
		this->externalPasses.clear();
	}

	void Synchronize(const MeshControllers::M::Data &data)
	{
		this->queues.CopyFrom(data.queues);
	}
};

// Perform visiblity culling.
void MeshControllers::Cull(PipelinePerspective &perspective) const
{
	LEAN_STATIC_PIMPL_CONST();
	M::PerspectiveState &state = m.perspectiveState.GetState(perspective, &m);
	const M::Data &data = *m.data;

	// Initialize frame & cull state
	state.Synchronize(data);
	state.visibleLODs.clear();
	state.distances.clear();
	
	const beMath::fplane3 *planes = perspective.GetDesc().Frustum;
	beMath::fvec3 center = perspective.GetDesc().CamPos;

	// Find visible controllers
	for (uint4 controllerIdx = 0; controllerIdx < data.activeControllerCount; ++controllerIdx)
	{
		const bem::fsphere3 &bounds = data.controllers(M::bounds)[controllerIdx];
		
		bool visible = true;

		// Cull bounding sphere against frustum
		for (int i = 0; i < 6; ++i)
			visible &= ( sdist(planes[i], bounds.center) <= bounds.radius );

		// Build list of visible controllers
		if (visible)
		{
			float distSquared = distSq(bounds.center, center);

			bec::Range<uint4> meshLODs = data.controllersToMeshLODs[controllerIdx];

			// Select level of detail
			for (uint4 meshLODIdx = meshLODs.Begin; meshLODIdx < meshLODs.End; ++meshLODIdx)
			{
				const M::Data::MeshLOD &meshLOD = data.meshLODs[meshLODIdx];
				
				if (distSquared >= meshLOD.distance * meshLOD.distance)
				{
					state.visibleLODs.push_back( M::PerspectiveState::VisibleLOD(controllerIdx, meshLODIdx) );
					state.distances.push_back( distSquared );
					break;
				}
			}
		}
	}

	// IMPORTANT: Clear external passes ONCE before preparation of individual queues
	state.externalPasses.clear();
}

// Prepares the given render queue for the given perspective.
bool MeshControllers::Prepare(PipelinePerspective &perspective, PipelineQueueID queueID,
							  const PipelineStageDesc &stageDesc, const RenderQueueDesc &queueDesc) const
{
	LEAN_STATIC_PIMPL_CONST();
	M::PerspectiveState &state = m.perspectiveState.GetExistingState(perspective, &m);
	const M::Data &data = *m.data;
	
	const M::Data::Queue *pDataQueue = data.queues.GetExistingQueue(queueID);
	if (!pDataQueue) return false;
	const M::Data::Queue &dataQueue = *pDataQueue;
	
	if (!queueDesc.DepthSort)
	{
		// Not utilizing frame coherence so far
		M::PerspectiveState::Queue &stateQueue = state.queues.GetParallelQueue(data.queues, pDataQueue);
		stateQueue.visiblePasses.clear();

		for (M::PerspectiveState::visible_lods_t::iterator it = state.visibleLODs.begin(), itEnd = state.visibleLODs.end(); it < itEnd; ++it)
		{
			const M::PerspectiveState::VisibleLOD &visibleMeshLOD = *it;

			uint4 passStartIdx = dataQueue.meshLODsToPasses[visibleMeshLOD.lodIdx];
			uint4 passEndIdx = dataQueue.meshLODsToPasses[visibleMeshLOD.lodIdx + 1];

			for (uint4 passIdx = passStartIdx; passIdx < passEndIdx; ++passIdx)
				stateQueue.visiblePasses.push_back( M::PerspectiveState::Queue::VisiblePass(visibleMeshLOD.controllerIdx, passIdx) );
		}

		// Check for queue passes
		return !stateQueue.visiblePasses.empty();
	}
	else
	{
		PipelinePerspective::QueueHandle jobQueue = perspective.QueueRenderJobs(queueID);
		size_t prevExtPassCount = state.externalPasses.size();

		uint4 backToFront = queueDesc.Backwards ? -1 : 0;

		for (uint4 i = 0, count = (uint4) state.visibleLODs.size(); i < count; ++i)
		{
			const M::PerspectiveState::VisibleLOD &visibleMeshLOD = state.visibleLODs[i];

			uint4 passStartIdx = dataQueue.meshLODsToPasses[visibleMeshLOD.lodIdx];
			uint4 passEndIdx = dataQueue.meshLODsToPasses[visibleMeshLOD.lodIdx + 1];

			float distance = state.distances[i];

			for (uint4 passIdx = passStartIdx; passIdx < passEndIdx; ++passIdx)
			{
				uint4 externalPassIdx = (uint4) state.externalPasses.size();
				state.externalPasses.push_back( M::PerspectiveState::ExternalPass(visibleMeshLOD.controllerIdx, &dataQueue.passes[passIdx]) );
				perspective.AddRenderJob( jobQueue, OrderedRenderJob(this, externalPassIdx, backToFront ^ *reinterpret_cast<const uint4*>(&distance)) );
			}
		}

		// Check for NEW passes
		return state.externalPasses.size() > prevExtPassCount;
	} 
}

namespace
{

struct PassMaterialSorter
{
	const MeshControllers::M::Data::Queue::passes_t &passes;

	PassMaterialSorter(const MeshControllers::M::Data::Queue::passes_t &passes)
		: passes(passes) { }

	LEAN_INLINE bool operator ()(const MeshControllers::M::PerspectiveState::Queue::VisiblePass &l,
								 const MeshControllers::M::PerspectiveState::Queue::VisiblePass &r) const
	{
		LEAN_FREE_PIMPL(MeshControllers);

		const M::Data::Queue::Pass &leftPass = passes[l.passIdx];
		const M::Data::Queue::Pass &rightPass = passes[r.passIdx];

		// TODO: Input layout?
		// TODO: Hash?!

		if (leftPass.pass < rightPass.pass)
			return true;
		else if (leftPass.pass == rightPass.pass)
		{
			if (leftPass.material < rightPass.material)
				return true;
			else if (leftPass.material == rightPass.material)
				return leftPass.mesh < rightPass.mesh;
		}

		return false;
	}
};

} // namespace

// Performs optional optimization such as sorting.
void MeshControllers::Optimize(const PipelinePerspective &perspective, PipelineQueueID queueID) const
{
	LEAN_STATIC_PIMPL_CONST();
	M::PerspectiveState &state =  m.perspectiveState.GetExistingState(perspective, &m);
	const M::Data &data = *m.data;
	
	const M::Data::Queue *pDataQueue = data.queues.GetExistingQueue(queueID);
	if (!pDataQueue) return;
	const M::Data::Queue &dataQueue = *pDataQueue;
	M::PerspectiveState::Queue &stateQueue = state.queues.GetParallelQueue(data.queues, pDataQueue);

	std::sort(stateQueue.visiblePasses.begin(), stateQueue.visiblePasses.end(), PassMaterialSorter(dataQueue.passes));
}

namespace
{

void RenderPass(const RenderableEffectData &data, const PipelinePerspective &perspective, uint4 controllerIdx,
				const MeshControllers::M::Data::Queue::Pass &pass, const RenderContext &renderContext)
{
	const beGraphics::Any::DeviceContext &deviceContext = ToImpl(renderContext.Context());
	beGraphics::Any::StateManager &stateManager = ToImpl(renderContext.StateManager());

	deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	deviceContext->IASetInputLayout(pass.inputLayout);

	const DX11::Mesh &mesh = *pass.mesh;
	UINT vertexStride = mesh.GetVertexSize();
	UINT vertexOffset = 0;
	deviceContext->IASetVertexBuffers(0, 1, &mesh.GetVertexBuffer().GetBuffer(), &vertexStride, &vertexOffset);
	deviceContext->IASetIndexBuffer(mesh.GetIndexBuffer(), mesh.GetIndexFormat(), 0);

	stateManager.Revert();
	pass.material->Apply(deviceContext);

	struct DrawJob : lean::vcallable_base<AbstractRenderableEffectDriver::DrawJobSignature, DrawJob>
	{
		uint4 indexCount;

		void operator ()(uint4 passIdx, beGraphics::StateManager &stateManager, const beGraphics::DeviceContext &context)
		{
			ToImpl(stateManager).Reset();
			ToImpl(context)->DrawIndexed(indexCount, 0, 0);
		}
	};

	DrawJob drawJob;
	drawJob.indexCount = mesh.GetIndexCount();

	// TODO: Do something sensible
	RenderableEffectDataEx rde;
	rde.Data = data;
	rde.ElementCount = drawJob.indexCount;

	pass.effectDriver->Render(pass.pass, &rde.Data, perspective, drawJob, stateManager, deviceContext);
}

} // namespace

// Renders the given render queue for the given perspective.
void MeshControllers::Render(const PipelinePerspective &perspective, PipelineQueueID queueID, const RenderContext &context) const
{
	LEAN_STATIC_PIMPL_CONST();
	M::PerspectiveState &state =  m.perspectiveState.GetExistingState(perspective, &m);
	const M::Data &data = *m.data;
	
	const M::Data::Queue *pDataQueue = data.queues.GetExistingQueue(queueID);
	if (!pDataQueue) return;
	const M::Data::Queue &dataQueue = *pDataQueue;
	M::PerspectiveState::Queue &stateQueue = state.queues.GetParallelQueue(data.queues, pDataQueue);

	const RenderableEffectData *renderableData = &data.controllers(M::renderableData)[0];
	const M::Data::Queue::Pass *passes = &dataQueue.passes[0];

	for (M::PerspectiveState::Queue::visible_passes_t::iterator it = stateQueue.visiblePasses.begin(), itEnd = stateQueue.visiblePasses.end();
		it != itEnd; ++it)
	{
		const M::PerspectiveState::Queue::VisiblePass &visiblePass = *it;
		RenderPass(renderableData[visiblePass.controllerIdx], perspective, visiblePass.controllerIdx, passes[visiblePass.passIdx], context);
	}
}

// Renders the given single object for the given perspective.
void MeshControllers::Render(uint4 objectID, const PipelinePerspective &perspective, PipelineQueueID queueID, const RenderContext &context) const
{
	LEAN_STATIC_PIMPL_CONST();
	M::PerspectiveState &state =  m.perspectiveState.GetExistingState(perspective, &m);
	const M::Data &data = *m.data;

	LEAN_ASSERT(objectID < state.externalPasses.size());
	const M::PerspectiveState::ExternalPass &job = state.externalPasses[objectID];

	RenderPass(data.controllers(M::renderableData)[job.controllerIdx], perspective, job.controllerIdx, *job.pass, context);
}

namespace
{

/// Computes a sphere containing the given box.
beMath::fsphere3 ComputeSphere(const beMath::faab3 &box)
{
	return beMath::fsphere3(
			(box.min + box.max) * 0.5f,
			length(box.max - box.min) * 0.5f
		);
}

// Updates the transformed bounding sphere of the given controller.
void UpdateBounds(MeshControllers::M &m, uint4 internalIdx, lean::tristate extVisible)
{
	LEAN_FREE_PIMPL(MeshControllers);

	M::Data &data = *m.data;

	const M::State &state = data.controllers(M::state)[internalIdx];
	const RenderableEffectData &trafo = data.controllers(M::renderableData)[internalIdx];
	bem::fsphere3 &bounds = data.controllers(M::bounds)[internalIdx];

	bool bEntityVisible = (extVisible != lean::dontcare)
		? (extVisible != lean::carefalse)
		: (bounds.radius >= 0.0f);
	
	float maxScaling = sqrt(bem::max(
			lengthSq(trafo.Transform[0]),
			lengthSq(trafo.Transform[1]),
			lengthSq(trafo.Transform[2])
		));

	bounds.center = state.Config.LocalBounds.center * maxScaling + bem::fvec3(trafo.Transform[3]);
	// MONITOR: Hide by ensuring that mesh is always culled
	bounds.radius = (state.Visible && bEntityVisible)
		? state.Config.LocalBounds.radius * maxScaling
		: -FLT_MAX * 0.5f;
}

} // namespace

// Sets the mesh.
void MeshControllers::SetMesh(MeshControllerHandle controller, RenderableMesh *pMesh)
{
	BE_STATIC_PIMPL_HANDLE(controller);
	
	M::Record &record = m.data->controllers(M::record)[controller.Index];

	if (record.Mesh != pMesh)
	{
		record.Mesh = pMesh;
		++m.controllerRevision;

		if (record.Mesh)
		{
			const RenderableMesh::MeshRange meshes = record.Mesh->GetMeshes();
			SetLocalBounds(controller, ComputeSphere(ComputeBounds(meshes.Begin, Size(meshes))));
		}
	}
}

// Gets the mesh.
RenderableMesh* MeshControllers::GetMesh(const MeshControllerHandle controller)
{
	BE_STATIC_PIMPL_HANDLE_CONST(controller);
	return m.data->controllers(M::record)[controller.Index].Mesh;
}

// Sets the visibility.
void MeshControllers::SetVisible(MeshControllerHandle controller, bool bVisible)
{
	BE_STATIC_PIMPL_HANDLE(controller);
	m.data->controllers(M::state)[controller.Index].Visible = bVisible;

	UpdateBounds(m, controller.Index, lean::dontcare);
}

// Gets the visibility.
bool MeshControllers::IsVisible(const MeshControllerHandle controller)
{
	BE_STATIC_PIMPL_HANDLE_CONST(controller);
	return m.data->controllers(M::state)[controller.Index].Visible;
}

// Sets the local bounding sphere.
void MeshControllers::SetLocalBounds(MeshControllerHandle controller, const beMath::fsphere3 &bounds)
{
	BE_STATIC_PIMPL_HANDLE(controller);
	m.data->controllers(M::state)[controller.Index].Config.LocalBounds = bounds;

	UpdateBounds(m, controller.Index, lean::dontcare);
}

// Gets the local bounding sphere.
const beMath::fsphere3& MeshControllers::GetLocalBounds(const MeshControllerHandle controller)
{
	BE_STATIC_PIMPL_HANDLE_CONST(controller);
	return m.data->controllers(M::state)[controller.Index].Config.LocalBounds;
}

// Adds a controller
MeshController* MeshControllers::AddController()
{
	LEAN_STATIC_PIMPL();
	M::Data &data = *m.data;

	uint4 internalIdx = static_cast<uint4>(data.controllers.size());
	MeshController *handle = new(m.handles.allocate()) MeshController( MeshControllerHandle(&m, internalIdx) );
	try { data.controllers.push_back( M::Record(handle) ); } LEAN_ASSERT_NOEXCEPT

	++m.controllerRevision;

	return handle;
}

// Clones the given controller.
MeshController* MeshControllers::CloneController(const MeshControllerHandle controller)
{
	BE_STATIC_PIMPL_HANDLE(const_cast<MeshControllerHandle&>(controller));
	M::Data &data = *m.data;

	lean::scoped_ptr<MeshController> clone( m.AddController() );

	// ORDER: First mesh then config (mesh resets local bounds)
	SetMesh(clone->Handle(), data.controllers[controller.Index].Mesh);
	data.controllers(M::state)[clone->Handle().Index].Config = data.controllers(M::state)[controller.Index].Config;
	
	return clone.detach();
}

// Removes a controller.
void MeshControllers::RemoveController(MeshController *pController)
{
	if (!pController || !pController->Handle().Group)
		return;

	BE_STATIC_PIMPL_HANDLE(pController->Handle());
	M::Data &data = *m.data;

	uint4 internalIdx  = pController->Handle().Index;
	LEAN_ASSERT(internalIdx < data.controllers.size());

	try
	{
		data.controllers.erase(internalIdx);
		m.handles.free(pController);
	}
	LEAN_ASSERT_NOEXCEPT

	// Fix subsequent handles
	M::FixControllerHandles(data.controllers, internalIdx);

	++m.controllerRevision;
}

// Attaches the controller to the given entity.
void MeshControllers::Attach(MeshControllerHandle controller, beEntitySystem::Entity *entity)
{
	BE_STATIC_PIMPL_HANDLE(controller);
	M::State &state = m.data->controllers(M::state)[controller.Index];

	if (!state.Attached)
	{
		state.Attached = true;
		++m.controllerRevision;
	}
}

// Detaches the controller from the given entity.
void MeshControllers::Detach(MeshControllerHandle controller, beEntitySystem::Entity *entity)
{
	BE_STATIC_PIMPL_HANDLE(controller);
	M::State &state = m.data->controllers(M::state)[controller.Index];

	if (state.Attached)
	{
		state.Attached = false;
		++m.controllerRevision;
	}
}

// Sets the component monitor.
void MeshControllers::SetComponentMonitor(beCore::ComponentMonitor *componentMonitor)
{
	LEAN_STATIC_PIMPL();
	m.pComponentMonitor = componentMonitor;
}

// Gets the component monitor.
beCore::ComponentMonitor* MeshControllers::GetComponentMonitor() const
{
	LEAN_STATIC_PIMPL_CONST();
	return m.pComponentMonitor;
}

// Synchronizes this controller with the given entity controlled.
void MeshController::Flush(const beEntitySystem::EntityHandle entity)
{
	BE_FREE_STATIC_PIMPL_HANDLE(MeshControllers, m_handle);
	M::Data &data = *m.data;

	using beEntitySystem::Entities;

	const Entities::Transformation& entityTrafo = Entities::GetTransformation(entity);
	const M::State &state = data.controllers(M::state)[m_handle.Index];
	RenderableEffectData &renderableData = data.controllers(M::renderableData)[m_handle.Index];
	
	renderableData.ID = Entities::GetCurrentID(entity);

	renderableData.Transform = mat_transform(
			entityTrafo.Position,
			entityTrafo.Orientation[2] * entityTrafo.Scaling[2],
			entityTrafo.Orientation[1] * entityTrafo.Scaling[1],
			entityTrafo.Orientation[0] * entityTrafo.Scaling[0]
		);
	renderableData.TransformInv = mat_transform_inverse(
			entityTrafo.Position,
			entityTrafo.Orientation[2] / entityTrafo.Scaling[2],
			entityTrafo.Orientation[1] / entityTrafo.Scaling[1],
			entityTrafo.Orientation[0] / entityTrafo.Scaling[0]
		);

	UpdateBounds(m, m_handle.Index, Entities::IsVisible(entity));
}

// Gets the number of child components.
uint4 MeshController::GetComponentCount() const
{
	return 1;
}

// Gets the name of the n-th child component.
beCore::Exchange::utf8_string MeshController::GetComponentName(uint4 idx) const
{
	return "Mesh";
}

// Gets the n-th reflected child component, nullptr if not reflected.
lean::com_ptr<const beCore::ReflectedComponent, lean::critical_ref> MeshController::GetReflectedComponent(uint4 idx) const
{
	return GetMesh();
}


// Gets the type of the n-th child component.
const beCore::ComponentType* MeshController::GetComponentType(uint4 idx) const
{
	return RenderableMesh::GetComponentType();
}

// Gets the n-th component.
lean::cloneable_obj<lean::any, true> MeshController::GetComponent(uint4 idx) const
{
	return bec::any_resource_t<RenderableMesh>::t( GetMesh() );
}

// Returns true, if the n-th component can be replaced.
bool MeshController::IsComponentReplaceable(uint4 idx) const
{
	return true;
}

// Sets the n-th component.
void MeshController::SetComponent(uint4 idx, const lean::any &pComponent)
{
	SetMesh( any_cast<RenderableMesh*>(pComponent) );
}

} // namespace

#include "beScene/beResourceManager.h"
#include "beScene/beEffectDrivenRenderer.h"
#include "beScene/beRenderableMaterialCache.h"

namespace beScene
{

namespace
{

/// Default effect.
inline utf8_string& GetDefaultEffectFile()
{
	static utf8_string defaultEffectFile = "Materials/Simple.fx";
	return defaultEffectFile;
}

} // namespace

// Sets the default mesh effect file.
void SetMeshDefaultEffect(const utf8_ntri &file)
{
	GetDefaultEffectFile().assign(file.begin(), file.end());
}

// Gets the default mesh effect file.
beCore::Exchange::utf8_string GetMeshDefaultEffect()
{
	const utf8_string &file = GetDefaultEffectFile();
	return beCore::Exchange::utf8_string(file.begin(), file.end());
}

// Gets the default material for meshes.
RenderableMaterial* GetMeshDefaultMaterial(ResourceManager &resources, EffectDrivenRenderer &renderer)
{
	const char *materialName = "beMeshController.Material";

	beg::Material *material = resources.MaterialCache()->GetByName(materialName);
	if (!material)
		material = resources.MaterialCache()->NewByFile(GetDefaultEffectFile(), materialName);

	return renderer.RenderableMaterials()->GetMaterial(material);
}

} // namespace