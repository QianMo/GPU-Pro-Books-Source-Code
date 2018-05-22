#include "Fluid.h"
#include "FluidMetaDataManager.h"

#include "../Main/DemoManager.h"

#include "../Input/InputManager.h"

#include "../Util/Math.h"
#include "../Util/ConfigLoader.h"

#include "../XMLParser/tinyxml.h"
#include "../XMLParser/tinyutil.h"


// -----------------------------------------------------------------------------
// -------------------------------- Fluid::Fluid -------------------------------
// -----------------------------------------------------------------------------
Fluid::Fluid(NxScene* _scene, bool hardware, unsigned int sceneIndex, FluidDescription* fluidDescription) :
	scene(_scene),
	hardwareSimulation(false),
	fluidSystem(NULL),
	emitter(NULL),
	fluidBufferNum(0),
	fluidBuffer(NULL),
	fluidCreatedParticleIdsNum(0),
	fluidCreatedParticleIds(NULL),
	fluidDeletedParticleIdsNum(0),
	fluidDeletedParticleIds(NULL),
	fluidNumPackets(0),
	emitterTimer(0.0f)
{
	Init(hardware, sceneIndex, fluidDescription);
}

// -----------------------------------------------------------------------------
// ------------------------------- Fluid::~Fluid -------------------------------
// -----------------------------------------------------------------------------
Fluid::~Fluid()
{
	ExitFluid();
}

// -----------------------------------------------------------------------------
// ------------------------------- Fluid::Update -------------------------------
// -----------------------------------------------------------------------------
void Fluid::Update(float deltaTime)
{
	if (InputManager::Instance()->IsKeyPressedAndReset(KEY_ENTER))
	{
		NX_BOOL enabled = emitter->getFlag(NX_FEF_ENABLED);
		emitter->setFlag(NX_FEF_ENABLED, !enabled);

		if (enabled)
			DemoManager::Instance()->SetDebugMsg("Fluid Emitter disabled");
		else
			DemoManager::Instance()->SetDebugMsg("Fluid Emitter enabled");
	}

	emitterTimer += deltaTime;
}


// -----------------------------------------------------------------------------
// ---------------------------- Fluid::GetMaxPackets ---------------------------
// -----------------------------------------------------------------------------
unsigned int Fluid::GetMaxPackets(void) const
{
	return (unsigned int)scene->getPhysicsSDK().getParameter(NX_CONSTANT_FLUID_MAX_PACKETS);
}


// -----------------------------------------------------------------------------
// -------------------------------- Fluid::Init --------------------------------
// -----------------------------------------------------------------------------
void Fluid::Init(bool hardware, unsigned int sceneIndex, FluidDescription* fluidDescription)
{
	hardwareSimulation = hardware;

	InitFluid(hardwareSimulation, sceneIndex, fluidDescription);
	InitEmitters(sceneIndex);
	InitDrains(sceneIndex);
}


// -----------------------------------------------------------------------------
// --------------------------- Fluid::InitFluid --------------------------------
// -----------------------------------------------------------------------------
void Fluid::InitFluid(bool hardware, unsigned int sceneIndex, FluidDescription* fluidDescription)
{
	//Pre cook hotspots
	//NxBounds3 precookAABB;
	//precookAABB.setCenterExtents(NxVec3(0.0f, 20.0f, 0.0f), NxVec3(100.0f, 100.0f, 100.0f));
	//scene->cookFluidMeshHotspot(precookAABB, 8, fluidDescription->restParticlesPerMeter,
	//	fluidDescription->kernelRadiusMultiplier, fluidDescription->motionLimitMultiplier, fluidDescription->collisionDistanceMultiplier);

	//////////////////////////////////////////////////////////////////////////

	// setup fluid descriptor
	NxFluidDesc fluidDesc;
	fluidDesc.setToDefault();

	fluidDesc.maxParticles						= fluidDescription->maxParticles;
	fluidDesc.numReserveParticles				= fluidDescription->numReserveParticles;
	fluidDesc.restParticlesPerMeter				= fluidDescription->restParticlesPerMeter;
	fluidDesc.restDensity						= fluidDescription->restDensity;
	fluidDesc.kernelRadiusMultiplier			= fluidDescription->kernelRadiusMultiplier;
	fluidDesc.motionLimitMultiplier				= fluidDescription->motionLimitMultiplier;
	fluidDesc.collisionDistanceMultiplier		= fluidDescription->collisionDistanceMultiplier;
	//fluidDesc.packetSizeMultiplier			= fluidDescription->packetSizeMultiplier;
	fluidDesc.packetSizeMultiplier				= 8;
	fluidDesc.stiffness							= fluidDescription->stiffness;
	fluidDesc.viscosity							= fluidDescription->viscosity;
	fluidDesc.surfaceTension					= fluidDescription->surfaceTension;
	fluidDesc.damping							= fluidDescription->damping;
	fluidDesc.fadeInTime						= fluidDescription->fadeInTime;
	fluidDesc.restitutionForStaticShapes		= fluidDescription->restitutionForStaticShapes;
	fluidDesc.dynamicFrictionForStaticShapes	= fluidDescription->dynamicFrictionForStaticShapes;
	fluidDesc.restitutionForDynamicShapes		= fluidDescription->restitutionForDynamicShapes;
	fluidDesc.dynamicFrictionForDynamicShapes	= fluidDescription->dynamicFrictionForDynamicShapes;
	fluidDesc.collisionResponseCoefficient		= fluidDescription->collisionResponseCoefficient;
	fluidDesc.simulationMethod					= fluidDescription->simulationMethod;
	
	// enable collision with static and dynamic actors
	fluidDesc.collisionMethod					= NX_F_STATIC|NX_F_DYNAMIC;

	// 2-way interaction between fluid and scene
	fluidDesc.flags |= NX_FF_COLLISION_TWOWAY;

	fluidDesc.initialParticleData = FluidMetaDataManager::Instance()->GetInitParticleData(sceneIndex);

	if (!hardware)
		fluidDesc.flags &= ~NX_FF_HARDWARE;

	fluidPacketData.bufferFluidPackets = new NxFluidPacket[(NxU32)scene->getPhysicsSDK().getParameter(NX_CONSTANT_FLUID_MAX_PACKETS)];
	fluidPacketData.numFluidPacketsPtr = &fluidNumPackets;
	fluidDesc.fluidPacketData = fluidPacketData;

	//////////////////////////////////////////////////////////////////////////

	// create user fluid.
	fluidBuffer = new FluidParticle[fluidDescription->maxParticles];

	// setup particle write data.
	NxParticleData particleData;
	particleData.numParticlesPtr = &fluidBufferNum;
	particleData.bufferPos = &fluidBuffer[0].position.x;
	particleData.bufferPosByteStride = sizeof(FluidParticle);
	particleData.bufferVel = &fluidBuffer[0].velocity.x;
	particleData.bufferVelByteStride = sizeof(FluidParticle);
	particleData.bufferDensity = &fluidBuffer[0].density;
	particleData.bufferDensityByteStride = sizeof(FluidParticle);
	particleData.bufferLife = &fluidBuffer[0].lifetime;
	particleData.bufferLifeByteStride = sizeof(FluidParticle);
	particleData.bufferId = &fluidBuffer[0].id;
	particleData.bufferIdByteStride = sizeof(FluidParticle);

	fluidDesc.particlesWriteData = particleData;

	//////////////////////////////////////////////////////////////////////////

	fluidCreatedParticleIds = new unsigned int[fluidDescription->maxParticles];
	fluidDeletedParticleIds = new unsigned int[fluidDescription->maxParticles];

	//Setup id write data.
	NxParticleIdData idData;

	//Creation
	idData.numIdsPtr = &fluidCreatedParticleIdsNum;
	idData.bufferId = fluidCreatedParticleIds;
	idData.bufferIdByteStride = sizeof(NxU32);
	fluidDesc.particleCreationIdWriteData = idData;

	//Deletion
	idData.numIdsPtr = &fluidDeletedParticleIdsNum;
	idData.bufferId = fluidDeletedParticleIds;
	idData.bufferIdByteStride = sizeof(NxU32);
	fluidDesc.particleDeletionIdWriteData = idData;

	//////////////////////////////////////////////////////////////////////////

	fluidSystem = scene->createFluid(fluidDesc);
	assert(fluidSystem);

	if (fluidSystem == NULL)
	{
		printf("Could not create fluid!\n");
		exit(0);
	}
}

// -----------------------------------------------------------------------------
// ------------------------- Fluid::InitEmitters -------------------------------
// -----------------------------------------------------------------------------
void Fluid::InitEmitters(unsigned int sceneIndex)
{
	NxFluidEmitterDesc* desc = FluidMetaDataManager::Instance()->GetEmitterDesc(sceneIndex);

	emitter = fluidSystem->createEmitter(*desc);
	emitterTimer = 0.0f;

	if (emitter == NULL)
	{
		printf("Could not create fluid emitter!\n");
		exit(0);
	}

	emitter->setFlag(NX_FEF_ENABLED, true);
}

// -----------------------------------------------------------------------------
// -------------------------- Fluid::InitDrains --------------------------------
// -----------------------------------------------------------------------------
void Fluid::InitDrains(unsigned int sceneIndex)
{
	// Drain that deletes the fluid particles
	if (FluidMetaDataManager::Instance()->HasCustomDrain(sceneIndex))
	{
		NxActorDesc drainActorDesc = FluidMetaDataManager::Instance()->GetDrainActorDesc(sceneIndex);
		scene->createActor(drainActorDesc);
	}

	// Create ground plane
	NxPlaneShapeDesc planeDesc;
	planeDesc.normal = NxVec3(0.0f, 1.0f, 0.0f);
	planeDesc.d = -160.0f;
	planeDesc.shapeFlags |= NX_SF_FLUID_DRAIN;

	NxActorDesc actorDesc;
	actorDesc.shapes.pushBack(&planeDesc);
	scene->createActor(actorDesc);
}

// -----------------------------------------------------------------------------
// --------------------------- Fluid::ExitFluid --------------------------------
// -----------------------------------------------------------------------------
void Fluid::ExitFluid(void)
{
	fluidSystem->releaseEmitter(*emitter);
	emitter = NULL;

	delete[] fluidCreatedParticleIds;
	fluidCreatedParticleIds = NULL;

	delete[] fluidDeletedParticleIds;
	fluidDeletedParticleIds = NULL;

	delete[] fluidPacketData.bufferFluidPackets;
	delete[] fluidBuffer;
	fluidBuffer = NULL;
	fluidPacketData.bufferFluidPackets = NULL;

	scene->releaseFluid(*fluidSystem);
	fluidSystem = NULL;
}
