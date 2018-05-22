#ifndef __FLUID__H__
#define __FLUID__H__

#include "NxPhysics.h"

#include "../Util/Vector3.h"
#include "../Util/AntTweakBar.h"

class FluidDescription;


// -----------------------------------------------------------------------------
/// 
// -----------------------------------------------------------------------------
/// \ingroup 
/// 
/// 
// -----------------------------------------------------------------------------
class Fluid {

public:

	struct FluidParticle
	{
		Vector3			position;
		float			density;
		Vector3			velocity;
		float			lifetime;
		float			foam;
		unsigned int	id;
		//Vector3		collisionNormal;
	};
	
	/// Create fluid
	Fluid(NxScene* _scene, bool hardware, unsigned int sceneIndex, FluidDescription* fluidDescription);
	/// Destructor
	~Fluid();

	/// Update fluid
	void Update(float deltaTime);

	NxFluid* GetNxFluid(void) { return fluidSystem; }

	unsigned int GetFluidBufferNum(void) const { return fluidBufferNum; }
	FluidParticle* GetFluidBuffer(void) const { return fluidBuffer; }

	unsigned int GetFluidCreatedParticleIdsNum(void) const { return fluidCreatedParticleIdsNum; }
	const unsigned int* GetFluidCreatedParticleIds(void) const { return fluidCreatedParticleIds; }

	unsigned int GetFluidDeletedParticleIdsNum(void) const { return fluidDeletedParticleIdsNum; }
	const unsigned int* GetFluidDeletedParticleIds(void) const { return fluidDeletedParticleIds; }

	unsigned int GetFluidNumPackets(void) const { return fluidNumPackets; }
	const NxFluidPacket* GetFluidPackets(void) const { return fluidPacketData.bufferFluidPackets; }

	unsigned int GetMaxParticles(void) const { return fluidSystem->getMaxParticles(); }
	unsigned int GetMaxPackets(void) const;

private:

	void Init(bool hardware, unsigned int sceneIndex, FluidDescription* fluidDescription);
	void InitFluid(bool hardware, unsigned int sceneIndex, FluidDescription* fluidDescription);
	void InitEmitters(unsigned int sceneIndex);
	void InitDrains(unsigned int sceneIndex);

	void ExitFluid(void);

	/// Physics scene
	NxScene* scene;
	bool hardwareSimulation;

	NxFluid*			fluidSystem;
	NxFluidEmitter*		emitter;

	unsigned int		fluidBufferNum;
	FluidParticle*		fluidBuffer;

	unsigned int		fluidCreatedParticleIdsNum;
	unsigned int*		fluidCreatedParticleIds;

	unsigned int		fluidDeletedParticleIdsNum;
	unsigned int*		fluidDeletedParticleIds;

	NxFluidPacketData	fluidPacketData;
	unsigned int		fluidNumPackets;

	float emitterTimer;
};

#endif
