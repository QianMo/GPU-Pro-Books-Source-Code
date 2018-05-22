#ifndef __FLUID_EMITTER_MANAGER__H__
#define __FLUID_EMITTER_MANAGER__H__

#include "../Util/Singleton.h"

#include "Fluid.h"

#include <vector>

#include "NxPhysics.h"


// -----------------------------------------------------------------------------
/// FluidMetaDataManager
// -----------------------------------------------------------------------------
/// \ingroup 
/// 
/// 
// -----------------------------------------------------------------------------
class FluidMetaDataManager : public Singleton<FluidMetaDataManager>
{
	friend class Singleton<FluidMetaDataManager>;

public:

	enum DynamicObjectType
	{
		DOT_SPHERE=0,
		DOT_BOX,
		DOT_COUNT
	};

	struct DynamicObject
	{
		DynamicObjectType type;
		Vector3 position;
		float size;

		DynamicObject(DynamicObjectType _type, const Vector3& _position, float _size) : type(_type), position(_position), size(_size) {}
	};

	FluidMetaDataManager(void);
	~FluidMetaDataManager(void);

	/// Init stuff
	void Init();

	/// Destroy stuff
	void Exit();

	/// Returns the emitter description for the given scene index
	NxFluidEmitterDesc* GetEmitterDesc(unsigned int index) const;

	/// Flags if drain is used
	bool HasCustomDrain(unsigned int index) const;

	/// Returns the drain actor description for the given scene index
	NxActorDesc GetDrainActorDesc(unsigned int index) const;

	/// Files the given ParticleData for the given scene index
	NxParticleData GetInitParticleData(unsigned int index) const;

	const std::vector<DynamicObject*>& GetDynamicObjects(unsigned int index) const;

public:

	/// Create emitter descriptions for different scenes
	void CreateDescriptions(void);

	void CreateParticleSphere(NxParticleData& pd, unsigned maxParticles, bool append, const NxVec3& pos, const NxVec3 vel, float lifeTime, float distance, unsigned sideNum) const;
	void CreateParticleBox(NxParticleData& pd, unsigned int maxParticles, bool append, const NxVec3& pos, const NxVec3 vel, const NxVec3 extend, float distance, float lifeTime) const;

	struct MetaData
	{
		NxFluidEmitterDesc* fluidEmitterDesc;

		bool hasCustomDrain;
		NxActorDesc fluidDrainActorDesc;

		unsigned int initParticlesNum;
		Fluid::FluidParticle* initParticles;
		NxParticleData initParticleData;

		std::vector<DynamicObject*> dynamicObjects;

		MetaData(void) : fluidEmitterDesc(NULL), hasCustomDrain(false), fluidDrainActorDesc(), initParticlesNum(0), initParticles(NULL) {}

	};

	bool isInit;

	std::vector<MetaData*> metaData;

};

#endif

