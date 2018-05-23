/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beLightControllers.h"

#include <beEntitySystem/beEntities.h>

#include <beCore/beReflectionProperties.h>
#include <beCore/bePersistentIDs.h>

#include "beScene/beRenderingPipeline.h"
#include "beScene/bePipelinePerspective.h"
#include "beScene/bePerspectiveStatePool.h"
#include "beScene/beQueueStatePool.h"

#include "beScene/bePerspectivePool.h"
#include "beScene/bePipePool.h"

#include "beScene/beLightMaterial.h"
#include "beScene/beLightEffectData.h"
#include "beScene/beAbstractLightEffectDriver.h"

#include "beScene/beRenderContext.h"
#include <beGraphics/Any/beAPI.h>
#include <beGraphics/Any/beDevice.h>
#include <beGraphics/Any/beBuffer.h>
#include <beGraphics/Any/beTexture.h>
#include <beGraphics/Any/beStateManager.h>
#include <beGraphics/Any/beDeviceContext.h>

#include <beMath/beUtility.h>
#include <beMath/beVector.h>
#include <beMath/beMatrix.h>
#include <beMath/beSphere.h>
#include <beMath/bePlane.h>
#include <beMath/beAAB.h>

#include <lean/containers/multi_vector.h>
#include <lean/containers/simple_vector.h>
#include <lean/memory/chunk_pool.h>

#include <lean/functional/algorithm.h>
#include <lean/io/numeric.h>

#include <beGraphics/DX/beError.h>

namespace beScene
{

template <class LightController>
struct LightInternals
{
	static const beCore::ReflectionProperty Properties[];
	static const beCore::ComponentType ControllerType;
	static const beCore::ComponentTypePlugin ControllerTypePlugin;

	static const uint4 LightTypeID;
	static utf8_string DefaultShadowStageName;
	static utf8_string DefaultEffectFile;

	static bem::fsphere3 GetDefaultBounds();

	struct Constants;
	struct ShadowConstants;
	struct AuxConfiguration { };

	struct Configuration : AuxConfiguration
	{
		bem::fsphere3 LocalBounds;
		uint4 ShadowResolution;
		PipelineStageMask ShadowStageMask;
		bool Shadow : 1;

		Configuration(const bem::fsphere3 &bounds, uint4 defaultShadowRes, PipelineStageMask defaultShadowStages)
			: LocalBounds(bounds),
			ShadowResolution(defaultShadowRes),
			ShadowStageMask(defaultShadowStages),
			Shadow(false) { }
	};

	struct State
	{
		Configuration Config;
		bool Visible : 1;
		bool Attached : 1;

		State(const bem::fsphere3 &bounds, uint4 defaultShadowRes, PipelineStageMask defaultShadowStages)
			: Config(bounds, defaultShadowRes, defaultShadowStages),
			Visible(true),
			Attached(false) { }
	};

	struct ShadowStateBase
	{
		uint4 ControllerIdx;

		ShadowStateBase(uint4 controllerIdx)
			: ControllerIdx(controllerIdx) { }
	};
	struct ShadowState : ShadowStateBase { };

	typedef lean::simple_vector<ShadowState, lean::vector_policies::semipod> shadow_state_t;
};

template <class LightController>
const beCore::ComponentTypePlugin LightInternals<LightController>::ControllerTypePlugin( &LightInternals<LightController>::ControllerType );

template <class LightController>
utf8_string LightInternals<LightController>::DefaultShadowStageName = "ShadowPipelineStage";

} // namespace

#include "beDirectionalLight.hpp"
#include "bePointLight.hpp"
#include "beSpotLight.hpp"

namespace beScene
{

struct LightControllersBase::M
{
	enum record_tag { record };
	enum state_tag { state };
	enum bounds_tag { bounds };
	enum constants_tag { constants };
	enum observers_tag { observers };
	
	struct Data
	{
		struct Queue : public QueueStateBase
		{
			struct Pass
			{
				lean::resource_ptr<const AbstractLightEffectDriver> effectDriver;
				const QueuedPass *pass;
				const beGraphics::MaterialTechnique *material;
			};

			typedef lean::simple_vector<uint4, lean::vector_policies::inipod> pass_offsets_t;
			typedef lean::simple_vector<Pass, lean::vector_policies::semipod> passes_t;

			pass_offsets_t materialsToPasses;
			passes_t passes;

			void Reset(PipelineQueueID id)
			{
				base_type::Reset(id);
				this->materialsToPasses.clear();
				this->passes.clear();
			}
		};
	
		typedef QueueStatePool<Queue, lean::vector_policies::semipod> queues_t;
	};

	struct PerspectiveState;
};

template <class LightController>
class LightControllers<LightController>::M : public LightControllersBase::M, public LightControllers<LightController>
{
public:
	lean::com_ptr<beg::API::Device> device;
	lean::resource_ptr<PerspectivePool> perspectivePool;
	
	uint4 defaultShadowResolution;
	PipelineStageMask defaultShadowStageMask;
	
	typedef LightController LightController;
	typedef typename LightControllers<LightController>::LightControllerHandle LightControllerHandle;

	struct Record
	{
		LightController *Reflected;
		lean::resource_ptr<LightMaterial> Material;

		uint8 PersistentID;

		Record(LightController *reflected)
			: Reflected(reflected),
			PersistentID(-1) { }
	};

	typedef typename LightInternals<LightController>::Configuration Configuration;
	typedef typename LightInternals<LightController>::State State;
	typedef typename LightInternals<LightController>::Constants Constants;
	
	typedef lean::chunk_pool<LightControllerHandle, 128> handle_pool;
	handle_pool handles;

	typedef typename lean::multi_vector_t< lean::simple_vector_binder<lean::vector_policies::semipod> >::make<
			Record, record_tag,
			State, state_tag,
			bem::fsphere3, bounds_tag,
			Constants, constants_tag,
			bec::ComponentObserverCollection, observers_tag
		>::type controllers_t;

	struct Data : public LightControllersBase::M::Data
	{
		typedef lean::simple_vector<LightMaterial*, lean::vector_policies::inipod> materials_t;
		typedef lean::simple_vector<uint4, lean::vector_policies::inipod> offsets_t;

		uint4 structureRevision;

		materials_t uniqueMaterials;
		
		controllers_t controllers;
		uint4 activeControllerCount;
		uint4 shadowControllerCount;
		offsets_t controllersToMaterials;

		queues_t queues;
		
		lean::com_ptr<beg::API::Buffer> lightConstantBuffer;
		lean::com_ptr<beg::API::ShaderResourceView> lightConstantSRV;

		Data()
			: structureRevision(-1),
			activeControllerCount(0),
			shadowControllerCount(0) { }
	};
	Data dataSets[2];
	Data *data, *dataAux;
	uint4 controllerRevision;

	struct PerspectiveState;
	mutable PerspectiveStatePool<PerspectiveState> perspectiveState;

	lean::resource_ptr<beCore::ComponentMonitor> pComponentMonitor;

	M(beCore::PersistentIDs *persistentIDs, PerspectivePool *perspectivePool, const RenderingPipeline &pipeline, const beGraphics::Device &device)
		: device(ToImpl(device)),
		perspectivePool( LEAN_ASSERT_NOT_NULL(perspectivePool) ),
		defaultShadowResolution(512),
		defaultShadowStageMask( ComputeStageMask( GetDefaultShadowStage<LightController>(pipeline) ) ),
		data(dataSets),
		dataAux(&dataSets[1]),
		controllerRevision(0) { }

	/// Gets the number of child components.
	uint4 GetComponentCount() const { return static_cast<uint4>(data->controllers.size()); }
	/// Gets the name of the n-th child component.
	beCore::Exchange::utf8_string GetComponentName(uint4 idx) const { return LightController::GetComponentType()->Name; }
	/// Gets the n-th reflected child component, nullptr if not reflected.
	lean::com_ptr<const beCore::ReflectedComponent, lean::critical_ref> GetReflectedComponent(uint4 idx) const { return data->controllers[idx].Reflected; }

	/// Fixes all controller handles to match the layout of the given controller vector.
	static void FixControllerHandles(controllers_t &controllers, uint4 internalIdx = 0)
	{
		// Fix subsequent handles
		for (; internalIdx < controllers.size(); ++internalIdx)
			controllers(record)[internalIdx].Reflected->Handle().SetIndex(internalIdx);
	}
	/// Verifies the given handle.
	friend LEAN_INLINE bool VerifyHandle(const M &m, const LightControllerHandle handle) { return handle.Index < m.data->controllers.size(); }
};

/// Creates a collection of mesh controllers.
template <class LightController>
BE_SCENE_API lean::scoped_ptr<LightControllers<LightController>, lean::critical_ref> CreateLightControllers(beCore::PersistentIDs *persistentIDs,
	PerspectivePool *perspectivePool, const RenderingPipeline &pipeline, const beGraphics::Device &device)
{
	return new_scoped LightControllers<LightController>::M(persistentIDs, perspectivePool, pipeline, device);
}

namespace
{

template <class LightControllers>
void CommitExternalChanges(typename LightControllers::M &m, beCore::ComponentMonitor &monitor)
{
	LEAN_FREE_PIMPL(typename LightControllers);
	typename M::Data &data = *m.data;

	if (monitor.Replacement.HasChanged(LightMaterial::GetComponentType()))
	{
		uint4 controllerCount = (uint4) data.controllers.size();

		for (uint4 internalIdx = 0; internalIdx < controllerCount; ++internalIdx)
		{
			typename M::Record &record = data.controllers[internalIdx];
			LightMaterial *newMaterial = record.Material;
			
			while (LightMaterial *successor = newMaterial->GetSuccessor())
				newMaterial = successor;

			if (newMaterial != record.Material)
				LightControllers::SetMaterial(record.Reflected->Handle(), newMaterial);
		}
	}
}

template <class Controllers>
struct MaterialSorter
{
	const Controllers &v;

	MaterialSorter(const Controllers &v)
		: v(v) { }

	LEAN_INLINE bool operator ()(uint4 l, uint4 r) const
	{
		const LightMaterial *left = v[l].Material;
		bool leftAttached = left && v(LightControllersBase::M::state)[l].Attached;
		bool leftShadow = v(LightControllersBase::M::state)[l].Config.Shadow;
		const LightMaterial *right = v[r].Material;
		bool rightAttached = right && v(LightControllersBase::M::state)[r].Attached;
		bool rightShadow = v(LightControllersBase::M::state)[r].Config.Shadow;

		// Move null meshes outwards
		if (!leftAttached)
			return false;
		else if (!rightAttached)
			return true;
		// Group by state
		else if (leftShadow && !rightShadow)
			return true;
		else if (leftShadow == rightShadow)
		{
			// Sort by material
			if (left < right)
				return true;
			else if (left == right)
			{
				const AbstractLightEffectDriver *leftDriver = left->GetTechniques().Begin[0].TypedDriver();
				const AbstractLightEffectDriver *rightDriver = right->GetTechniques().Begin[0].TypedDriver();

				// Sort by effect
				return leftDriver < rightDriver;
			}
		}
		return false;
	}
};

/// Sort controllers by material and shader (moving null materials outwards).
template <class LightControllers>
void SortControllers(typename LightControllers::M::controllers_t &destControllers, const typename LightControllers::M::controllers_t &srcControllers)
{
	LEAN_FREE_PIMPL(typename LightControllers);
	
	uint4 controllerCount = (uint4) srcControllers.size();
	
	lean::scoped_ptr<uint4[]> sortIndices( new uint4[controllerCount] );
	std::generate_n(&sortIndices[0], controllerCount, lean::increment_gen<uint4>(0));
	std::sort(&sortIndices[0], &sortIndices[controllerCount], MaterialSorter<typename M::controllers_t>(srcControllers));
	
	destControllers.clear();
	lean::append_swizzled(srcControllers, &sortIndices[0], &sortIndices[controllerCount], destControllers);
}

template <class LightControllers>
void LinkControllersToUniqueMaterials(typename LightControllers::M::Data &data)
{
	LEAN_FREE_PIMPL(typename LightControllers);

	const uint4 controllerCount = (uint4) data.controllers.size();
	data.controllersToMaterials.resize(controllerCount);
	data.uniqueMaterials.clear();
	data.activeControllerCount = 0;
	data.shadowControllerCount = 0;

	const besc::LightMaterial *prevMaterial = nullptr;
	uint4 materialIdx = 0;

	for (uint4 internalIdx = 0; internalIdx < controllerCount; ++internalIdx)
	{
		typename M::Record &controller = data.controllers[internalIdx];
		typename M::State &controllerState = data.controllers(M::state)[internalIdx];

		// Ignore null & detached meshes at the back
		if (!controller.Material || !controllerState.Attached)
			break;

		data.shadowControllerCount += controllerState.Config.Shadow;
		++data.activeControllerCount;

		// Add new unique material
		if (prevMaterial != controller.Material)
		{
			materialIdx = (uint4) data.uniqueMaterials.size();
			data.uniqueMaterials.push_back(controller.Material);
			prevMaterial = controller.Material;
		}

		// Let controllers reference unique materials
		data.controllersToMaterials[internalIdx] = materialIdx;
	}
}

LightControllersBase::M::Data::Queue::Pass ConstructPass(const beGraphics::MaterialTechnique *material,
	const AbstractLightEffectDriver *effectDriver, const QueuedPass *driverPass)
{
	LightControllersBase::M::Data::Queue::Pass pass;

	pass.material = material;
	pass.pass = driverPass;
	pass.effectDriver = effectDriver;

	return pass;
}

template <class LightControllers>
void AddTechniquePasses(typename LightControllers::M::Data &data, uint4 materialIdx, besc::LightMaterial::Technique technique)
{
	LEAN_FREE_PIMPL(typename LightControllers);

	besc::AbstractLightEffectDriver::PassRange passes = technique.TypedDriver()->GetPasses();

	for (uint4 passIdx = 0, passCount = Size4(passes); passIdx < passCount; ++passIdx)
	{
		const besc::QueuedPass *pass = &passes[passIdx];
		besc::PipelineQueueID queueID(pass->GetStageID(), pass->GetQueueID());
		typename M::Data::Queue &queue = data.queues.GetQueue(queueID);

		// Link material to beginning of pass range & insert pass
		queue.materialsToPasses.resize( materialIdx + 1, (uint4) queue.passes.size() );
		queue.passes.push_back( ConstructPass(technique.Technique, technique.TypedDriver(), pass) );
	}
}

template <class LightControllers>
void BuildQueues(typename LightControllers::M::Data &data)
{
	LEAN_FREE_PIMPL(typename LightControllers);

	data.queues.Clear();

	const uint4 materialCount = (uint4) data.uniqueMaterials.size();

	// Build queues from meshes
	for (uint4 materialIdx = 0; materialIdx < materialCount; ++materialIdx)
	{
		const besc::LightMaterial *material = data.uniqueMaterials[materialIdx];
		besc::LightMaterial::TechniqueRange techniques = material->GetTechniques();

		for (uint4 techniqueIdx = 0, techniqueCount = Size4(techniques); techniqueIdx < techniqueCount; ++techniqueIdx)
			AddTechniquePasses<LightControllers>(data, materialIdx, techniques[techniqueIdx]);
	}
	
	// Discard unused queues
	data.queues.Shrink();

	// IMPORTANT: Finish implicit materials to pass offset ranges
	for (M::Data::Queue *it = data.queues.begin(), *itEnd = data.queues.end(); it < itEnd; ++it)
		it->materialsToPasses.resize(materialCount + 1, (uint4) it->passes.size());
}

} // namespace

// Commits changes.
template <class LightController>
void LightControllers<LightController>::Commit()
{
	LEAN_STATIC_PIMPL();

	if (m.pComponentMonitor)
		CommitExternalChanges<LightControllers>(m, *m.pComponentMonitor);

	typename M::Data &prevData = *m.data;
	typename M::Data &data = *m.dataAux;
	
	if (prevData.structureRevision != m.controllerRevision)
	{
		// Rebuild internal data structures in swap buffer
		SortControllers<LightControllers>(data.controllers, prevData.controllers);
		LinkControllersToUniqueMaterials<LightControllers>(data);
		BuildQueues<LightControllers>(data);

		if (data.activeControllerCount)
		{
			// Reallocate GPU buffers
			const size_t FloatRowSize = sizeof(float) * 4;

			LEAN_STATIC_ASSERT_MSG(
					sizeof(typename M::Constants) % FloatRowSize == 0,
					"Size of light constant buffer required to be multiple of 4 floats"
				);

			data.lightConstantBuffer = beg::Any::CreateStructuredBuffer(m.device, D3D11_BIND_SHADER_RESOURCE,
				sizeof(typename M::Constants), data.activeControllerCount, 0);
			data.lightConstantSRV = beg::Any::CreateSRV(data.lightConstantBuffer, DXGI_FORMAT_R32G32B32A32_FLOAT, 0,
				0, sizeof(typename M::Constants) / FloatRowSize * data.activeControllerCount);
		}

		data.structureRevision = m.controllerRevision;
		std::swap(m.data, m.dataAux);
		M::FixControllerHandles(m.data->controllers);

		prevData.lightConstantBuffer = nullptr;
		prevData.lightConstantSRV = nullptr;
	}
}

struct LightControllersBase::M::PerspectiveState
{
	struct VisibleMaterial
	{
		uint4 controllerIdx;
		uint4 materialIdx;

		VisibleMaterial(uint4 controllerIdx, uint4 materialIdx)
			: controllerIdx(controllerIdx),
			materialIdx(materialIdx) { }
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
			visiblePasses.clear();
		}
	};

	struct ExternalPass
	{
		uint4 controllerIdx;
		const Data::Queue::Pass *pass;

		ExternalPass(uint4 controllerIdx, const Data::Queue::Pass *pass)
			: controllerIdx(controllerIdx),
			pass(pass) { }
	};

	typedef lean::simple_vector<VisibleMaterial, lean::vector_policies::inipod> visible_materials_t;
	typedef QueueStatePool<Queue, lean::vector_policies::semipod> queues_t;
	typedef lean::simple_vector<ExternalPass, lean::vector_policies::inipod> external_passes_t;
};

template <class LightController>
struct LightControllers<LightController>::M::PerspectiveState : public LightControllersBase::M::PerspectiveState,
	public PerspectiveStateBase<const typename LightControllers<LightController>::M, typename LightControllers<LightController>::M::PerspectiveState>
{
	typedef typename LightInternals<LightController>::ShadowStateBase ShadowStateBase;
	typedef typename LightInternals<LightController>::ShadowState ShadowState;
	typedef typename LightInternals<LightController>::ShadowConstants ShadowConstants;

	typedef lean::simple_vector<float, lean::vector_policies::inipod> distances_t;
	typedef lean::simple_vector<ShadowConstants, lean::vector_policies::inipod> shadow_constants_t;
	typedef lean::simple_vector<ShadowState, lean::vector_policies::semipod> shadow_state_t;
	typedef lean::simple_vector<uint1, lean::vector_policies::inipod> active_t;
	typedef lean::simple_vector<beg::TextureViewHandle, lean::vector_policies::inipod> shadow_textures_t;

	uint4 structureRevision;

	visible_materials_t visibleMaterials;
	distances_t distances;
	queues_t queues;
	external_passes_t externalPasses;

	shadow_constants_t shadowConstants;
	shadow_textures_t shadowTextures;

	active_t shadowsActive;
	shadow_state_t shadowState;

	lean::com_ptr<beg::API::Buffer> shadowConstantBuffer;
	lean::com_ptr<beg::API::ShaderResourceView> shadowConstantSRV;

	PerspectiveState(const typename LightControllers::M *parent)
		: base_type(parent),
		structureRevision(-1) { }
	
	void Reset(Perspective *perspective)
	{
		base_type::Reset(perspective);

		this->visibleMaterials.clear();
		this->distances.clear();
		this->queues.Reset();
		this->externalPasses.clear();
		
		std::fill(this->shadowsActive.begin(), this->shadowsActive.end(), 0);
		this->shadowState.clear();
	}

	void Synchronize(const typename LightControllers::M &m)
	{
		LEAN_FREE_PIMPL(typename LightControllers);
		const typename M::Data &data = *m.data;

		if (this->structureRevision != data.structureRevision)
		{
			this->queues.CopyFrom(data.queues);

			this->shadowConstants.resize(data.shadowControllerCount);
			this->shadowTextures.resize(data.shadowControllerCount);
			this->shadowsActive.resize(data.shadowControllerCount);

			// Reallocate GPU buffers
			const size_t FloatRowSize = sizeof(float) * 4;
		
			LEAN_STATIC_ASSERT_MSG(
					sizeof(ShadowConstants) % FloatRowSize == 0,
					"Size of light constant buffer required to be multiple of 4 floats"
				);

			if (data.shadowControllerCount)
			{
				this->shadowConstantBuffer = beg::Any::CreateStructuredBuffer(m.device, D3D11_BIND_SHADER_RESOURCE,
					sizeof(typename M::PerspectiveState::ShadowConstants), data.shadowControllerCount, 0);
				this->shadowConstantSRV = beg::Any::CreateSRV(this->shadowConstantBuffer, DXGI_FORMAT_R32G32B32A32_FLOAT, 0,
					0, sizeof(typename M::PerspectiveState::ShadowConstants) / FloatRowSize * data.shadowControllerCount);
			}
			else
			{
				this->shadowConstantBuffer = nullptr;
				this->shadowConstantSRV = nullptr;
			}

			this->Reset(this->PerspectiveBinding);
			this->structureRevision = data.structureRevision;
		}
	}

};

// Perform visiblity culling.
template <class LightController>
void LightControllers<LightController>::Cull(PipelinePerspective &perspective) const
{
	LEAN_STATIC_PIMPL_CONST();
	typename M::PerspectiveState &state = m.perspectiveState.GetState(perspective, &m);
	const typename M::Data &data = *m.data;

	// Initialize frame & cull state, not utilizing frame coherency so far
	state.Synchronize(m);
	state.visibleMaterials.clear();
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

			state.visibleMaterials.push_back( typename M::PerspectiveState::VisibleMaterial(controllerIdx, data.controllersToMaterials[controllerIdx]) );
			state.distances.push_back( distSquared );
		}
	}

	// IMPORTANT: Clear external passes ONCE before preparation of individual queues
	state.externalPasses.clear();
}

// Prepares the given render queue for the given perspective.
template <class LightController>
bool LightControllers<LightController>::Prepare(PipelinePerspective &perspective, PipelineQueueID queueID,
												const PipelineStageDesc &stageDesc, const RenderQueueDesc &queueDesc) const
{
	LEAN_STATIC_PIMPL_CONST();
	typename M::PerspectiveState &state = m.perspectiveState.GetExistingState(perspective, &m);
	const typename M::Data &data = *m.data;
	
	const typename M::Data::Queue *pDataQueue = data.queues.GetExistingQueue(queueID);
	if (!pDataQueue) return false;
	const typename M::Data::Queue &dataQueue = *pDataQueue;
	
	// Prepare shadows for _rendered_ light passes
	for (typename M::PerspectiveState::visible_materials_t::iterator it = state.visibleMaterials.begin(), itEnd = state.visibleMaterials.end();
		it != itEnd && it->controllerIdx < data.shadowControllerCount; ++it)
	{
		const typename M::PerspectiveState::VisibleMaterial &visibleMaterial = *it;

		// Make sure *something* is rendered
		if (dataQueue.materialsToPasses[visibleMaterial.materialIdx] < dataQueue.materialsToPasses[visibleMaterial.materialIdx + 1])
		{
			uint1 &shadowActive = state.shadowsActive[visibleMaterial.controllerIdx];

			// Add missing shadow
			if (!shadowActive)
				AddShadow(
						state.shadowState, typename M::PerspectiveState::ShadowStateBase(visibleMaterial.controllerIdx),
						data.controllers(M::state)[visibleMaterial.controllerIdx].Config,
						perspective, *m.perspectivePool
					);
			
			// Mark shadows of visible lights as active
			shadowActive = true;
		}
	}

	if (!queueDesc.DepthSort)
	{
		// Not utilizing frame coherence so far
		typename M::PerspectiveState::Queue &stateQueue = state.queues.GetParallelQueue(data.queues, pDataQueue);
		stateQueue.visiblePasses.clear();

		for (M::PerspectiveState::visible_materials_t::iterator it = state.visibleMaterials.begin(), itEnd = state.visibleMaterials.end(); it < itEnd; ++it)
		{
			const typename M::PerspectiveState::VisibleMaterial &visibleMaterial = *it;

			uint4 passStartIdx = dataQueue.materialsToPasses[visibleMaterial.materialIdx];
			uint4 passEndIdx = dataQueue.materialsToPasses[visibleMaterial.materialIdx + 1];

			for (uint4 passIdx = passStartIdx; passIdx < passEndIdx; ++passIdx)
				stateQueue.visiblePasses.push_back( typename M::PerspectiveState::Queue::VisiblePass(visibleMaterial.controllerIdx, passIdx) );
		}

		return !stateQueue.visiblePasses.empty();
	}
	else
	{
		PipelinePerspective::QueueHandle jobQueue = perspective.QueueRenderJobs(queueID);
		size_t prevExtPassCount = state.externalPasses.size();

		uint4 backToFront = queueDesc.Backwards ? -1 : 0;

		for (uint4 i = 0, count = (uint4) state.visibleMaterials.size(); i < count; ++i)
		{
			const typename M::PerspectiveState::VisibleMaterial &visibleMaterial = state.visibleMaterials[i];

			uint4 passStartIdx = dataQueue.materialsToPasses[visibleMaterial.materialIdx];
			uint4 passEndIdx = dataQueue.materialsToPasses[visibleMaterial.materialIdx + 1];

			float distance = state.distances[i];

			for (uint4 passIdx = passStartIdx; passIdx < passEndIdx; ++passIdx)
			{
				uint4 externalPassIdx = (uint4) state.externalPasses.size();
				state.externalPasses.push_back( typename M::PerspectiveState::ExternalPass(visibleMaterial.controllerIdx, &dataQueue.passes[passIdx]) );
				perspective.AddRenderJob( jobQueue, OrderedRenderJob(this, externalPassIdx, backToFront ^ *reinterpret_cast<const uint4*>(&distance)) );
			}
		}

		return state.externalPasses.size() > prevExtPassCount;
	}
}

// Prepares the collected render queues for the given perspective.
template <class LightController>
void LightControllers<LightController>::Collect(PipelinePerspective &perspective) const
{
	LEAN_STATIC_PIMPL_CONST();
	typename M::PerspectiveState &state = m.perspectiveState.GetExistingState(perspective, &m);
	const typename M::Data &data = *m.data;

	uint4 inactiveShadowCount = 0;

	// Prepare shadows for _rendered_ light passes
	for (typename M::PerspectiveState::shadow_state_t::iterator it = state.shadowState.begin(), itEnd = state.shadowState.end(); it != itEnd; ++it)
	{
		typename M::PerspectiveState::ShadowState &shadowState = *it;

		if (state.shadowsActive[shadowState.ControllerIdx])
			// Update active shadows
			PrepareShadow(data.controllers(M::state)[shadowState.ControllerIdx].Config,
				data.controllers(M::constants)[shadowState.ControllerIdx],
				state.shadowConstants[shadowState.ControllerIdx],
				shadowState,
				perspective);
		else
			++inactiveShadowCount;
	}

	if (inactiveShadowCount)
		for (typename M::PerspectiveState::shadow_state_t::iterator it = state.shadowState.end(), itEnd = state.shadowState.end(); it-- != itEnd; )
			if (!state.shadowsActive[it->ControllerIdx])
				state.shadowState.erase(it);
}

namespace
{

struct PassMaterialSorter
{
	const LightControllersBase::M::Data::Queue::passes_t &passes;

	PassMaterialSorter(const LightControllersBase::M::Data::Queue::passes_t &passes)
		: passes(passes) { }

	LEAN_INLINE bool operator ()(const LightControllersBase::M::PerspectiveState::Queue::VisiblePass &l,
								 const LightControllersBase::M::PerspectiveState::Queue::VisiblePass &r) const
	{
		const LightControllersBase::M::Data::Queue::Pass &leftPass = passes[l.passIdx];
		const LightControllersBase::M::Data::Queue::Pass &rightPass = passes[r.passIdx];

		// TODO: Input layout?

		if (leftPass.pass < rightPass.pass)
			return true;
		else if (leftPass.pass == rightPass.pass)
			return leftPass.material < rightPass.material;

		return false;
	}
};

} // namespace

// Performs optional optimization such as sorting.
template <class LightController>
void LightControllers<LightController>::Optimize(const PipelinePerspective &perspective, PipelineQueueID queueID) const
{
	LEAN_STATIC_PIMPL_CONST();
	typename M::PerspectiveState &state = m.perspectiveState.GetExistingState(perspective, &m);
	const typename M::Data &data = *m.data;
	
	const typename M::Data::Queue *pDataQueue = data.queues.GetExistingQueue(queueID);
	if (!pDataQueue) return;
	const typename M::Data::Queue &dataQueue = *pDataQueue;
	typename M::PerspectiveState::Queue &stateQueue = state.queues.GetParallelQueue(data.queues, pDataQueue);

	std::sort(stateQueue.visiblePasses.begin(), stateQueue.visiblePasses.end(), PassMaterialSorter(dataQueue.passes));
}

namespace
{

void SetupRendering(const RenderContext &renderContext)
{
	beg::api::DeviceContext *deviceContext = ToImpl(renderContext.Context());

	deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
	deviceContext->IASetInputLayout(nullptr);

	beg::api::Buffer *noBuffer = nullptr;
	UINT vertexStride = 0;
	UINT vertexOffset = 0;
	deviceContext->IASetVertexBuffers(0, 1, &noBuffer, &vertexStride, &vertexOffset);
	deviceContext->IASetIndexBuffer(noBuffer, DXGI_FORMAT_R16_UINT, 0);
}

void RenderPass(const LightEffectData &data, const PipelinePerspective &perspective, uint4 controllerIdx,
				const LightControllersBase::M::Data::Queue::Pass &pass, const RenderContext &renderContext)
{
	const beGraphics::Any::DeviceContext &deviceContext = ToImpl(renderContext.Context());
	beGraphics::Any::StateManager &stateManager = ToImpl(renderContext.StateManager());

	stateManager.Revert();
	pass.material->Apply(deviceContext);

	struct DrawJob : lean::vcallable_base<AbstractLightEffectDriver::DrawJobSignature, DrawJob>
	{
		uint4 controllerIdx;

		void operator ()(uint4 passIdx, beGraphics::StateManager &stateManager, const beGraphics::DeviceContext &context)
		{
			ToImpl(stateManager).Reset();
			ToImpl(context)->DrawInstanced(4, 1, 0, controllerIdx);
		}
	};

	DrawJob drawJob;
	drawJob.controllerIdx = controllerIdx;

	pass.effectDriver->Render(pass.pass, &data, perspective, drawJob, stateManager, deviceContext);
}

} // namespace

// Prepares rendering from the collected render queues for the given perspective.
template <class LightController>
void LightControllers<LightController>::PreRender(const PipelinePerspective &perspective, const RenderContext &renderContext) const
{
	LEAN_STATIC_PIMPL_CONST();
	typename M::PerspectiveState &state = m.perspectiveState.GetExistingState(perspective, &m);
	const typename M::Data &data = *m.data;
	beg::api::DeviceContext *deviceContext = ToImpl(renderContext.Context());

	// TODO: Update changed only
	if (data.lightConstantBuffer)
		deviceContext->UpdateSubresource(data.lightConstantBuffer, 0, nullptr, &data.controllers(M::constants)[0], 0, 0);
	if (state.shadowConstantBuffer)
		deviceContext->UpdateSubresource(state.shadowConstantBuffer, 0, nullptr, &state.shadowConstants[0], 0, 0);

	// Gather shadows 
	for (typename M::PerspectiveState::shadow_state_t::iterator it = state.shadowState.begin(), itEnd = state.shadowState.end(); it != itEnd; ++it)
		state.shadowTextures[it->ControllerIdx] = GatherShadow(*it);
}

// Renders the given render queue for the given perspective.
template <class LightController>
void LightControllers<LightController>::Render(const PipelinePerspective &perspective, PipelineQueueID queueID, const RenderContext &context) const
{
	LEAN_STATIC_PIMPL_CONST();
	typename M::PerspectiveState &state = m.perspectiveState.GetExistingState(perspective, &m);
	const typename M::Data &data = *m.data;
	
	const typename M::Data::Queue *pDataQueue = data.queues.GetExistingQueue(queueID);
	if (!pDataQueue) return;
	const typename M::Data::Queue &dataQueue = *pDataQueue;
	typename M::PerspectiveState::Queue &stateQueue = state.queues.GetParallelQueue(data.queues, pDataQueue);

	LightEffectData lightData;
	lightData.TypeID = LightInternals<LightController>::LightTypeID;
	lightData.Lights = beg::Any::TextureViewHandle(data.lightConstantSRV);
	lightData.Shadows = beg::Any::TextureViewHandle(state.shadowConstantSRV);
	
	SetupRendering(context);

	{
		typename M::PerspectiveState::Queue::visible_passes_t::iterator it = stateQueue.visiblePasses.begin(), itEnd = stateQueue.visiblePasses.end();

		lightData.ShadowMapCount = 1;

		// Draw lights with shadow
		for (; it != itEnd && it->controllerIdx < data.shadowControllerCount; ++it)
		{
			lightData.ShadowMaps = &state.shadowTextures[it->controllerIdx];
			RenderPass(lightData, perspective, it->controllerIdx, dataQueue.passes[it->passIdx], context);
		}

		lightData.ShadowMaps = nullptr;
		lightData.ShadowMapCount = 0;

		// Draw lights without shadow
		for (; it != itEnd; ++it)
		{
			// TODO: Allow multiple ...
			RenderPass(lightData, perspective, it->controllerIdx, dataQueue.passes[it->passIdx], context);
		}
	}
}

// Renders the given single object for the given perspective.
template <class LightController>
void LightControllers<LightController>::Render(uint4 objectID, const PipelinePerspective &perspective, PipelineQueueID queueID, const RenderContext &context) const
{
	LEAN_STATIC_PIMPL_CONST();
	typename M::PerspectiveState &state = m.perspectiveState.GetExistingState(perspective, &m);
	const typename M::Data &data = *m.data;

	LEAN_ASSERT(objectID < state.externalPasses.size());
	const typename M::PerspectiveState::ExternalPass &job = state.externalPasses[objectID];

	bool bHasShadow = job.controllerIdx < data.shadowControllerCount;

	LightEffectData lightData;
	lightData.TypeID = LightInternals<LightController>::LightTypeID;
	lightData.Lights = beg::Any::TextureViewHandle(data.lightConstantSRV);
	lightData.Shadows = beg::Any::TextureViewHandle(state.shadowConstantSRV);
	lightData.ShadowMaps = (bHasShadow) ? &state.shadowTextures[job.controllerIdx] : nullptr;
	lightData.ShadowMapCount = bHasShadow;
	
	SetupRendering(context);
	RenderPass(lightData, perspective, job.controllerIdx, *job.pass, context);
}

namespace
{

template <class M>
void PropertyChanged(M &m, uint4 internalIdx)
{
	typename M::Data &data = *m.data;
	const bec::ComponentObserverCollection &observers = data.controllers(M::observers)[internalIdx];

	if (observers.HasObservers())
		observers.EmitPropertyChanged(*data.controllers(M::record)[internalIdx].Reflected);
}

template <class M>
void ComponentChanged(M &m, uint4 internalIdx)
{
	typename M::Data &data = *m.data;
	const bec::ComponentObserverCollection &observers = data.controllers(M::observers)[internalIdx];

	if (observers.HasObservers())
		observers.EmitChildChanged(*data.controllers(M::record)[internalIdx].Reflected);
}

/// Computes a sphere containing the given box.
beMath::fsphere3 ComputeSphere(const beMath::faab3 &box)
{
	return beMath::fsphere3(
			(box.min + box.max) * 0.5f,
			length(box.max - box.min) * 0.5f
		);
}

// Updates the transformed bounding sphere of the given controller.
template <class M>
void UpdateBounds(M &m, uint4 internalIdx, lean::tristate extVisible)
{
	typename M::Data &data = *m.data;

	const typename M::State &state = data.controllers(M::state)[internalIdx];
	const typename M::Constants &constants = data.controllers(M::constants)[internalIdx];
	bem::fsphere3 &bounds = data.controllers(M::bounds)[internalIdx];

	bool bEntityVisible = (extVisible != lean::dontcare)
		? (extVisible != lean::carefalse)
		: (bounds.radius >= 0.0f);
	
	float maxScaling = sqrt(bem::max(
			lengthSq(constants.Transformation[0]),
			lengthSq(constants.Transformation[1]),
			lengthSq(constants.Transformation[2])
		));

	bounds.center = state.Config.LocalBounds.center * maxScaling + bem::fvec3(constants.Transformation[3]);
	// MONITOR: Hide by ensuring that mesh is always culled
	bounds.radius = (state.Visible && bEntityVisible)
		? state.Config.LocalBounds.radius * maxScaling
		: -FLT_MAX * 0.5f;
}

} // namespace

// Sets the material.
template <class LightController>
void LightControllers<LightController>::SetMaterial(LightControllerHandle controller, LightMaterial *pMaterial)
{
	BE_STATIC_PIMPL_HANDLE(controller);
	
	typename M::Record &record = m.data->controllers(M::record)[controller.Index];

	if (record.Material != pMaterial)
	{
		record.Material = pMaterial;
		++m.controllerRevision;

		ComponentChanged(m, controller.Index);
	}
}

// Gets the material.
template <class LightController>
LightMaterial* LightControllers<LightController>::GetMaterial(const LightControllerHandle controller)
{
	BE_STATIC_PIMPL_HANDLE_CONST(controller);
	return m.data->controllers(M::record)[controller.Index].Material;
}

// Sets the visibility.
template <class LightController>
void LightControllers<LightController>::SetVisible(LightControllerHandle controller, bool bVisible)
{
	BE_STATIC_PIMPL_HANDLE(controller);
	m.data->controllers(M::state)[controller.Index].Visible = bVisible;

	UpdateBounds(m, controller.Index, lean::dontcare);
	PropertyChanged(m, controller.Index);
}

// Gets the visibility.
template <class LightController>
bool LightControllers<LightController>::IsVisible(const LightControllerHandle controller)
{
	BE_STATIC_PIMPL_HANDLE_CONST(controller);
	return m.data->controllers(M::state)[controller.Index].Visible;
}

/// Sets the color.
template <class LightController>
void LightControllersColorBase<LightController>::SetColor(LightControllerHandle controller, const beMath::fvec4 &color)
{
	BE_FREE_STATIC_PIMPL_HANDLE(typename LightControllers, controller);
	m.data->controllers(M::constants)[controller.Index].Color = color;
	PropertyChanged(m, controller.Index);
}

// Gets the color.
template <class LightController>
const beMath::fvec4& LightControllersColorBase<LightController>::GetColor(const LightControllerHandle controller)
{
	BE_FREE_STATIC_PIMPL_HANDLE_CONST(typename LightControllers, controller);
	return m.data->controllers(M::constants)[controller.Index].Color;
}

// Sets the (indirect) color.
template <class LightController>
void LightControllersColorBase<LightController>::SetIndirectColor(LightControllerHandle controller, const beMath::fvec4 &color)
{
	BE_FREE_STATIC_PIMPL_HANDLE(typename LightControllers, controller);
	m.data->controllers(M::constants)[controller.Index].IndirectColor = color;
	PropertyChanged(m, controller.Index);
}

// Gets the (indirect) color.
template <class LightController>
const beMath::fvec4& LightControllersColorBase<LightController>::GetIndirectColor(const LightControllerHandle controller)
{
	BE_FREE_STATIC_PIMPL_HANDLE_CONST(typename LightControllers, controller);
	return m.data->controllers(M::constants)[controller.Index].IndirectColor;
}

// Sets the attenuation.
template <class LightController>
void LightControllersPointBase<LightController>::SetAttenuation(LightControllerHandle controller, float attenuation)
{
	BE_FREE_STATIC_PIMPL_HANDLE(typename LightControllers, controller);
	m.data->controllers(M::constants)[controller.Index].Attenuation = attenuation;
	PropertyChanged(m, controller.Index);
}

// Gets the attenuation.
template <class LightController>
float LightControllersPointBase<LightController>::GetAttenuation(const LightControllerHandle controller)
{
	BE_FREE_STATIC_PIMPL_HANDLE_CONST(typename LightControllers, controller);
	return m.data->controllers(M::constants)[controller.Index].Attenuation;
}

// Sets the attenuation offset.
template <class LightController>
void LightControllersPointBase<LightController>::SetAttenuationOffset(LightControllerHandle controller, float offset)
{
	BE_FREE_STATIC_PIMPL_HANDLE(typename LightControllers, controller);
	m.data->controllers(M::constants)[controller.Index].AttenuationOffset = offset;
	PropertyChanged(m, controller.Index);
}
// Gets the attenuation offset.
template <class LightController>
float LightControllersPointBase<LightController>::GetAttenuationOffset(const LightControllerHandle controller)
{
	BE_FREE_STATIC_PIMPL_HANDLE_CONST(typename LightControllers, controller);
	return m.data->controllers(M::constants)[controller.Index].AttenuationOffset;
}

// Sets the range.
template <class LightController>
void LightControllersPointBase<LightController>::SetRange(LightControllerHandle controller, float range)
{
	BE_FREE_STATIC_PIMPL_HANDLE(typename LightControllers, controller);
	m.data->controllers(M::constants)[controller.Index].Range = range;
	PropertyChanged(m, controller.Index);
}
// Gets the range.
template <class LightController>
float LightControllersPointBase<LightController>::GetRange(LightControllerHandle controller)
{
	BE_FREE_STATIC_PIMPL_HANDLE_CONST(typename LightControllers, controller);
	return m.data->controllers(M::constants)[controller.Index].Range;
}

// Sets the angles.
template <class LightController>
void LightControllersSpotBase<LightController>::SetInnerAngle(LightControllerHandle controller, float angle)
{
	BE_FREE_STATIC_PIMPL_HANDLE(typename LightControllers, controller);
	m.data->controllers(M::state)[controller.Index].Config.InnerAngle = angle;
	
	typename M::Constants &constants = m.data->controllers(M::constants)[controller.Index];
	constants.SinInnerAngle = sin(angle);
	constants.CosInnerAngle = cos(angle);

	PropertyChanged(m, controller.Index);
}
// Gets the angles.
template <class LightController>
float LightControllersSpotBase<LightController>::GetInnerAngle(LightControllerHandle controller)
{
	BE_FREE_STATIC_PIMPL_HANDLE_CONST(typename LightControllers, controller);
	return m.data->controllers(M::state)[controller.Index].Config.InnerAngle;
}

// Sets the angles.
template <class LightController>
void LightControllersSpotBase<LightController>::SetOuterAngle(LightControllerHandle controller, float angle)
{
	BE_FREE_STATIC_PIMPL_HANDLE(typename LightControllers, controller);
	m.data->controllers(M::state)[controller.Index].Config.OuterAngle = angle;

	typename M::Constants &constants = m.data->controllers(M::constants)[controller.Index];
	constants.SinOuterAngle = sin(angle);
	constants.CosOuterAngle = cos(angle);

	PropertyChanged(m, controller.Index);
}
// Gets the angles.
template <class LightController>
float LightControllersSpotBase<LightController>::GetOuterAngle(LightControllerHandle controller)
{
	BE_FREE_STATIC_PIMPL_HANDLE_CONST(typename LightControllers, controller);
	return m.data->controllers(M::state)[controller.Index].Config.OuterAngle;
}

// Sets the local bounding sphere.
template <class LightController>
void LightControllers<LightController>::SetLocalBounds(LightControllerHandle controller, const beMath::fsphere3 &bounds)
{
	BE_STATIC_PIMPL_HANDLE(controller);
	m.data->controllers(M::state)[controller.Index].Config.LocalBounds = bounds;

	UpdateBounds(m, controller.Index, lean::dontcare);
}

// Gets the local bounding sphere.
template <class LightController>
const beMath::fsphere3& LightControllers<LightController>::GetLocalBounds(const LightControllerHandle controller)
{
	BE_STATIC_PIMPL_HANDLE_CONST(controller);
	return m.data->controllers(M::state)[controller.Index].Config.LocalBounds;
}

// Sets the local bounding sphere.
template <class LightController>
void LightControllers<LightController>::EnableShadow(LightControllerHandle controller, bool bEnable)
{
	BE_STATIC_PIMPL_HANDLE(controller);
	typename M::Configuration &state = m.data->controllers(M::state)[controller.Index].Config;

	if (state.Shadow != bEnable)
	{
		state.Shadow = bEnable;
		++m.controllerRevision;

		PropertyChanged(m, controller.Index);
	}
}

// Gets the local bounding sphere.
template <class LightController>
bool LightControllers<LightController>::IsShadowEnabled(const LightControllerHandle controller)
{
	BE_STATIC_PIMPL_HANDLE_CONST(controller);
	return m.data->controllers(M::state)[controller.Index].Config.Shadow;
}

// Sets the local bounding sphere.
template <class LightController>
void LightControllers<LightController>::SetShadowResolution(LightControllerHandle controller, uint4 resolution)
{
	BE_STATIC_PIMPL_HANDLE(controller);
	m.data->controllers(M::state)[controller.Index].Config.ShadowResolution = resolution;
	PropertyChanged(m, controller.Index);
}

// Gets the local bounding sphere.
template <class LightController>
uint4 LightControllers<LightController>::GetShadowResolution(const LightControllerHandle controller)
{
	BE_STATIC_PIMPL_HANDLE_CONST(controller);
	return m.data->controllers(M::state)[controller.Index].Config.ShadowResolution;
}

// Sets the local bounding sphere.
template <class LightController>
void LightControllers<LightController>::SetShadowStages(LightControllerHandle controller, PipelineStageMask stages)
{
	BE_STATIC_PIMPL_HANDLE(controller);
	m.data->controllers(M::state)[controller.Index].Config.ShadowStageMask = stages;
}

// Gets the local bounding sphere.
template <class LightController>
PipelineStageMask LightControllers<LightController>::GetShadowStages(const LightControllerHandle controller)
{
	BE_STATIC_PIMPL_HANDLE_CONST(controller);
	return m.data->controllers(M::state)[controller.Index].Config.ShadowStageMask;
}

// Adds a controller
template <class LightController>
LightController* LightControllers<LightController>::AddController()
{
	LEAN_STATIC_PIMPL();
	typename M::Data &data = *m.data;

	uint4 internalIdx = static_cast<uint4>(data.controllers.size());
	LightController *handle = new(m.handles.allocate()) LightController( LightControllerHandle(&m, internalIdx) );

	try
	{
		data.controllers.push_back(
				typename M::Record(handle),
				typename M::State(LightInternals<LightController>::GetDefaultBounds(), m.defaultShadowResolution, m.defaultShadowStageMask)
			);
	}
	catch (...)
	{
		m.handles.free(handle);
		throw;
	}

	++m.controllerRevision;

	return handle;
}

// Clones the given controller.
template <class LightController>
LightController* LightControllers<LightController>::CloneController(const LightControllerHandle controller)
{
	BE_STATIC_PIMPL_HANDLE(const_cast<LightControllerHandle&>(controller));
	typename M::Data &data = *m.data;

	lean::scoped_ptr<LightController> clone( m.AddController() );
	LightControllerHandle cloneHandle = clone->Handle();
	
	// IMPORTANT: Clone only "data" part of state
	data.controllers(M::state)[cloneHandle.Index].Config = data.controllers(M::state)[controller.Index].Config;
	data.controllers(M::constants)[cloneHandle.Index] = data.controllers(M::constants)[controller.Index];
	SetMaterial(clone->Handle(), data.controllers[controller.Index].Material);
	
	return clone.detach();
}

// Removes a controller.
template <class LightController>
void LightControllers<LightController>::RemoveController(LightController *pController)
{
	if (!pController || !pController->Handle().Group)
		return;

	BE_STATIC_PIMPL_HANDLE(pController->Handle());
	typename M::Data &data = *m.data;

	uint4 internalIdx  = pController->Handle().Index;

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
template <class LightController>
void LightControllers<LightController>::Attach(LightControllerHandle controller, beEntitySystem::Entity *entity)
{
	BE_STATIC_PIMPL_HANDLE(controller);
	typename M::State &state = m.data->controllers(M::state)[controller.Index];

	if (!state.Attached)
	{
		state.Attached = true;
		++m.controllerRevision;
	}
}

// Detaches the controller from the given entity.
template <class LightController>
void LightControllers<LightController>::Detach(LightControllerHandle controller, beEntitySystem::Entity *entity)
{
	BE_STATIC_PIMPL_HANDLE(controller);
	typename M::State &state = m.data->controllers(M::state)[controller.Index];

	if (state.Attached)
	{
		state.Attached = false;
		++m.controllerRevision;
	}
}
// Sets the component monitor.
template <class LightController>
void LightControllers<LightController>::SetComponentMonitor(beCore::ComponentMonitor *componentMonitor)
{
	LEAN_STATIC_PIMPL();
	m.pComponentMonitor = componentMonitor;
}

// Gets the component monitor.
template <class LightController>
beCore::ComponentMonitor* LightControllers<LightController>::GetComponentMonitor() const
{
	LEAN_STATIC_PIMPL_CONST();
	return m.pComponentMonitor;
}

// Adds a property listener.
template <class LightController>
void LightControllerBase<LightController>::AddObserver(beCore::ComponentObserver *listener)
{
	BE_FREE_STATIC_PIMPL_HANDLE(typename LightControllers, m_handle);
	m.data->controllers(M::observers)[m_handle.Index].AddObserver(listener);
}

// Removes a property listener.
template <class LightController>
void LightControllerBase<LightController>::RemoveObserver(beCore::ComponentObserver *pListener)
{
	BE_FREE_STATIC_PIMPL_HANDLE(typename LightControllers, m_handle);
	m.data->controllers(M::observers)[m_handle.Index].RemoveObserver(pListener);
}

// Synchronizes this controller with the given entity controlled.
template <class LightController>
void LightControllerBase<LightController>::Flush(const beEntitySystem::EntityHandle entity)
{
	BE_FREE_STATIC_PIMPL_HANDLE(typename LightControllers, m_handle);
	typename M::Data &data = *m.data;

	using beEntitySystem::Entities;

	const Entities::Transformation& entityTrafo = Entities::GetTransformation(entity);
	const typename M::State &state = data.controllers(M::state)[m_handle.Index];
	typename M::Constants &constants = data.controllers(M::constants)[m_handle.Index];
	
	constants.Transformation = mat_transform(
			entityTrafo.Position,
			entityTrafo.Orientation[2] * entityTrafo.Scaling[2],
			entityTrafo.Orientation[1] * entityTrafo.Scaling[1],
			entityTrafo.Orientation[0] * entityTrafo.Scaling[0]
		);

	UpdateBounds(m, m_handle.Index, Entities::IsVisible(entity));
}

// Gets the number of child components.
template <class LightController>
uint4 LightControllerBase<LightController>::GetComponentCount() const
{
	return 1;
}

// Gets the name of the n-th child component.
template <class LightController>
beCore::Exchange::utf8_string LightControllerBase<LightController>::GetComponentName(uint4 idx) const
{
	return "Material";
}

// Gets the n-th reflected child component, nullptr if not reflected.
template <class LightController>
lean::com_ptr<const beCore::ReflectedComponent, lean::critical_ref> LightControllerBase<LightController>::GetReflectedComponent(uint4 idx) const
{
	return Reflect(GetMaterial());
}


// Gets the type of the n-th child component.
template <class LightController>
const beCore::ComponentType* LightControllerBase<LightController>::GetComponentType(uint4 idx) const
{
	return LightMaterial::GetComponentType();
}

// Gets the n-th component.
template <class LightController>
lean::cloneable_obj<lean::any, true> LightControllerBase<LightController>::GetComponent(uint4 idx) const
{
	return bec::any_resource_t<LightMaterial>::t( GetMaterial() );
}

// Returns true, if the n-th component can be replaced.
template <class LightController>
bool LightControllerBase<LightController>::IsComponentReplaceable(uint4 idx) const
{
	return true;
}

// Sets the n-th component.
template <class LightController>
void LightControllerBase<LightController>::SetComponent(uint4 idx, const lean::any &pComponent)
{
	SetMaterial( any_cast<LightMaterial*>(pComponent) );
}

// Gets the controller type.
template <class LightController>
const beCore::ComponentType* LightControllers<LightController>::GetComponentType()
{
	return &LightInternals< LightControllers<LightController> >::ControllerType;
}

// Gets the controller type.
template <class LightController>
const beCore::ComponentType* LightControllers<LightController>::GetType() const
{
	// IMPORTANT: Instantiate LightInternals::Plugin!
	return LightInternals< LightControllers<LightController> >::ControllerTypePlugin.Type;
}

/// Gets the controller properties.
template <class LightController>
beCore::ReflectionPropertyProvider::Properties LightControllerBase<LightController>::GetOwnProperties()
{
	return ToPropertyRange(LightInternals<LightController>::Properties);
}

// Gets the controller properties.
template <class LightController>
beCore::ReflectionPropertyProvider::Properties LightControllerBase<LightController>::GetReflectionProperties() const
{
	return ToPropertyRange(LightInternals<LightController>::Properties);
}

// Gets the controller type.
template <class LightController>
const beCore::ComponentType* LightControllerBase<LightController>::GetComponentType()
{
	return &LightInternals<LightController>::ControllerType;
}

// Gets the controller type.
template <class LightController>
const beCore::ComponentType* LightControllerBase<LightController>::GetType() const
{
	// IMPORTANT: Instantiate LightInternals::Plugin!
	return LightInternals<LightController>::ControllerTypePlugin.Type;
}

} // namespace

#include "beScene/beResourceManager.h"
#include "beScene/beEffectDrivenRenderer.h"
#include "beScene/beLightMaterialCache.h"
#include <beGraphics/beMaterialCache.h>

namespace beScene
{

// Sets the default light effect file.
template <class LightController>
BE_SCENE_API void SetLightDefaultEffect(const utf8_ntri &file)
{
	LightInternals<LightController>::DefaultEffectFile = file.to<utf8_string>();
}

// Gets the default mesh effect file.
template <class LightController>
BE_SCENE_API utf8_ntr GetLightDefaultEffect()
{
	return utf8_ntr(LightInternals<LightController>::DefaultEffectFile);
}

// Gets the default material for meshes.
template <class LightController>
BE_SCENE_API LightMaterial* GetLightDefaultMaterial(ResourceManager &resources, EffectDrivenRenderer &renderer)
{
	utf8_string materialName = utf8_string(LightInternals<LightController>::ControllerType.Name) + ".DefaultMaterial";
	beg::Material *material = resources.MaterialCache()->GetByName(materialName);

	if (!material)
		material = resources.MaterialCache()->NewByFile(LightInternals<LightController>::DefaultEffectFile, materialName);

	return renderer.LightMaterials()->GetMaterial(material);
}

// Sets the default shadow stage for directional lights.
template <class LightController>
BE_SCENE_API void SetDefaultShadowStage(const utf8_ntri &name)
{
	LightInternals<LightController>::DefaultShadowStageName = name.to<utf8_string>();
}

// Gets the default shadow stage for directional lights.
template <class LightController>
BE_SCENE_API utf8_ntr GetDefaultShadowStage()
{
	return utf8_ntr(LightInternals<LightController>::DefaultShadowStageName);
}

// Gets the default shadow stage for directional lights.
template <class LightController>
BE_SCENE_API PipelineStageMask GetDefaultShadowStage(const RenderingPipeline &pipeline)
{
	return pipeline.GetStageID(LightInternals<LightController>::DefaultShadowStageName);
}

} // namespace