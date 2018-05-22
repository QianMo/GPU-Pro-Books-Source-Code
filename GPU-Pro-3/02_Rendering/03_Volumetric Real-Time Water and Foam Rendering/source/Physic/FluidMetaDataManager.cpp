
#include "../Physic/FluidMetaDataManager.h"

#include "../Util/Matrix4.h"

#include "../Util/Math.h"

#include "NxPhysics.h"


// -----------------------------------------------------------------------------
// ----------------- FluidMetaDataManager::FluidMetaDataManager ----------------
// -----------------------------------------------------------------------------
FluidMetaDataManager::FluidMetaDataManager(void) :
	isInit(false)
{
	
}


// -----------------------------------------------------------------------------
// ---------------- FluidMetaDataManager::~FluidMetaDataManager ----------------
// -----------------------------------------------------------------------------
FluidMetaDataManager::~FluidMetaDataManager(void)
{

}


// -----------------------------------------------------------------------------
// ------------------------- FluidMetaDataManager::Init ------------------------
// -----------------------------------------------------------------------------
void FluidMetaDataManager::Init(void)
{
	assert(!isInit);

	CreateDescriptions();

	isInit = true;
}


// -----------------------------------------------------------------------------
// ------------------------- FluidMetaDataManager::Exit ------------------------
// -----------------------------------------------------------------------------
void FluidMetaDataManager::Exit(void)
{
	assert(isInit);

	unsigned int i;
	for (i=0; i<metaData.size(); i++)
	{
		delete(metaData[i]->fluidEmitterDesc);

		if (metaData[i]->hasCustomDrain)
		{
			unsigned int j;
			for (j=0; j<metaData[i]->fluidDrainActorDesc.shapes.size(); j++)
				delete(metaData[i]->fluidDrainActorDesc.shapes[j]);
		}

		delete(metaData[i]->initParticles);

		unsigned int j;
		for (j=0; j<metaData[i]->dynamicObjects.size(); j++)
			delete(metaData[i]->dynamicObjects[j]);
		metaData[i]->dynamicObjects.clear();

		delete(metaData[i]);
	}
	metaData.clear();

	isInit = false;
}


// -----------------------------------------------------------------------------
// -------------------- FluidMetaDataManager::GetEmitterDesc -------------------
// -----------------------------------------------------------------------------
NxFluidEmitterDesc* FluidMetaDataManager::GetEmitterDesc(unsigned int index) const
{
	assert(isInit);
	assert(index < metaData.size());

	return metaData[index]->fluidEmitterDesc;
}


// -----------------------------------------------------------------------------
// -------------------- FluidMetaDataManager::HasCustomDrain -------------------
// -----------------------------------------------------------------------------
bool FluidMetaDataManager::HasCustomDrain(unsigned int index) const
{
	assert(isInit);
	assert(index < metaData.size());

	return metaData[index]->hasCustomDrain;
}


// -----------------------------------------------------------------------------
// ------------------ FluidMetaDataManager::GetDrainActorDesc ------------------
// -----------------------------------------------------------------------------
NxActorDesc FluidMetaDataManager::GetDrainActorDesc(unsigned int index) const
{
	assert(isInit);
	assert(index < metaData.size());

	return metaData[index]->fluidDrainActorDesc;
}


// -----------------------------------------------------------------------------
// ----------------- FluidMetaDataManager::GetInitParticleData -----------------
// -----------------------------------------------------------------------------
NxParticleData FluidMetaDataManager::GetInitParticleData(unsigned int index) const
{
	assert(isInit);
	assert(index < metaData.size());

	return metaData[index]->initParticleData;
}


// -----------------------------------------------------------------------------
// ---- std::vector<DynamicObject*>& FluidMetaDataManager::GetDynamicObjects ---
// -----------------------------------------------------------------------------
const std::vector<FluidMetaDataManager::DynamicObject*>& FluidMetaDataManager::GetDynamicObjects(unsigned int index) const
{
	assert(isInit);
	assert(index < metaData.size());

	return metaData[index]->dynamicObjects;
}


// -----------------------------------------------------------------------------
// -------------- FluidMetaDataManager::CreateEmitterDescriptions --------------
// -----------------------------------------------------------------------------
void FluidMetaDataManager::CreateDescriptions(void)
{
	// scene0
	{
		MetaData* data = new MetaData();

		// setup emitter
		data->fluidEmitterDesc = new NxFluidEmitterDesc();
		{
			data->fluidEmitterDesc->relPose.M.rotX(NxHalfPiF32);
			data->fluidEmitterDesc->relPose.t = NxVec3(0.0f, 63.5f, -63.5f);

			data->fluidEmitterDesc->maxParticles = 0;
			data->fluidEmitterDesc->dimensionX = 7.5f;
			data->fluidEmitterDesc->dimensionY = 7.5f;
			data->fluidEmitterDesc->randomAngle = 0.0f;
			data->fluidEmitterDesc->randomPos = NxVec3(0.0f,0.0f,0.0f);
			data->fluidEmitterDesc->shape = NX_FE_RECTANGULAR;
			data->fluidEmitterDesc->type = NX_FE_CONSTANT_PRESSURE; //NX_FE_CONSTANT_FLOW_RATE;
			data->fluidEmitterDesc->fluidVelocityMagnitude = 50.0f;
			data->fluidEmitterDesc->particleLifetime = 30.0f;
		}

		// setup init particle data
		{
			data->initParticles = new Fluid::FluidParticle[1<<16];

			data->initParticleData.numParticlesPtr		= &data->initParticlesNum;
			data->initParticleData.bufferPos			= &data->initParticles[0].position.x;
			data->initParticleData.bufferPosByteStride	= sizeof(Fluid::FluidParticle);
			data->initParticleData.bufferVel			= &data->initParticles[0].velocity.x;
			data->initParticleData.bufferVelByteStride	= sizeof(Fluid::FluidParticle);
			data->initParticleData.bufferLife			= &data->initParticles[0].lifetime;
			data->initParticleData.bufferLifeByteStride	= sizeof(Fluid::FluidParticle);
		}

		// setup dynamic objects
		{
			data->dynamicObjects.push_back(new DynamicObject(DOT_SPHERE, Vector3(0.0f, 50.0f, 0.0f), 5.0f));
			data->dynamicObjects.push_back(new DynamicObject(DOT_SPHERE, Vector3(-10.0f, 50.0f, 0.0f), 5.0f));
			data->dynamicObjects.push_back(new DynamicObject(DOT_SPHERE, Vector3(10.0f, 50.0f, 0.0f), 5.0f));
		}

		metaData.push_back(data);
	}

	// scene1
	{
		MetaData* data = new MetaData();

		// setup emitter
		data->fluidEmitterDesc = new NxFluidEmitterDesc();
		{
			data->fluidEmitterDesc->relPose.M.rotX(NxHalfPiF32);
			data->fluidEmitterDesc->relPose.t = NxVec3(0.0f, 63.5f, -70.0f);

			data->fluidEmitterDesc->maxParticles = 0;
			data->fluidEmitterDesc->dimensionX = 6.0f;
			data->fluidEmitterDesc->dimensionY = 6.0f;
			data->fluidEmitterDesc->randomAngle = 0.0f;
			data->fluidEmitterDesc->randomPos = NxVec3(0.0f,0.0f,0.0f);
			data->fluidEmitterDesc->shape = NX_FE_RECTANGULAR;
			data->fluidEmitterDesc->type = NX_FE_CONSTANT_FLOW_RATE; //NX_FE_CONSTANT_FLOW_RATE;
			data->fluidEmitterDesc->rate = 2000;
			data->fluidEmitterDesc->fluidVelocityMagnitude = 50.0f;
			data->fluidEmitterDesc->particleLifetime = 240.0f;
		}

		// setup init particle data
		{
			data->initParticles = new Fluid::FluidParticle[1<<16];

			data->initParticleData.numParticlesPtr		= &data->initParticlesNum;
			data->initParticleData.bufferPos			= &data->initParticles[0].position.x;
			data->initParticleData.bufferPosByteStride	= sizeof(Fluid::FluidParticle);
			data->initParticleData.bufferVel			= &data->initParticles[0].velocity.x;
			data->initParticleData.bufferVelByteStride	= sizeof(Fluid::FluidParticle);
			data->initParticleData.bufferLife			= &data->initParticles[0].lifetime;
			data->initParticleData.bufferLifeByteStride	= sizeof(Fluid::FluidParticle);
		}

		metaData.push_back(data);
	}

	// scene2
	{
		MetaData* data = new MetaData();

		// setup emitter
		data->fluidEmitterDesc = new NxFluidEmitterDesc();
		{
			data->fluidEmitterDesc->relPose.M.rotX(NxHalfPiF32);
			data->fluidEmitterDesc->relPose.t = NxVec3(0.0f, 63.5f, -63.5f);

			data->fluidEmitterDesc->maxParticles = 0;
			data->fluidEmitterDesc->dimensionX = 7.5f;
			data->fluidEmitterDesc->dimensionY = 7.5f;
			data->fluidEmitterDesc->randomAngle = 0.0f;
			data->fluidEmitterDesc->randomPos = NxVec3(0.0f,0.0f,0.0f);
			data->fluidEmitterDesc->shape = NX_FE_RECTANGULAR;
			data->fluidEmitterDesc->type = NX_FE_CONSTANT_FLOW_RATE;
			data->fluidEmitterDesc->rate = 5000;
			data->fluidEmitterDesc->fluidVelocityMagnitude = 50.0f;
			data->fluidEmitterDesc->particleLifetime = 10.0f;
		}

		// setup drain
		{
			data->hasCustomDrain = true;

			NxBoxShapeDesc* drainDesc = new NxBoxShapeDesc();
			drainDesc->setToDefault();
			drainDesc->shapeFlags |= NX_SF_FLUID_DRAIN;
			drainDesc->dimensions.set(20.0f, 10.0f, 10.0f);

			data->fluidDrainActorDesc.shapes.pushBack(drainDesc);
			data->fluidDrainActorDesc.globalPose.t = NxVec3(60.0f, -23.0f, -50.0f);

			Matrix4 rot = Matrix4::Matrix4Rotation(5.0f*Math::DEG_TO_RAD, 0.0f, 0.0f);
			NxMat33 rotation;
			rotation.setColumn(0, NxVec3(rot.entry[0], rot.entry[1], rot.entry[2]));
			rotation.setColumn(1, NxVec3(rot.entry[4], rot.entry[5], rot.entry[6]));
			rotation.setColumn(2, NxVec3(rot.entry[8], rot.entry[9], rot.entry[10]));

			data->fluidDrainActorDesc.globalPose.M = rotation;
		}

		// setup init particle data
		{
			data->initParticles = new Fluid::FluidParticle[1<<16];

			data->initParticleData.numParticlesPtr		= &data->initParticlesNum;
			data->initParticleData.bufferPos			= &data->initParticles[0].position.x;
			data->initParticleData.bufferPosByteStride	= sizeof(Fluid::FluidParticle);
			data->initParticleData.bufferVel			= &data->initParticles[0].velocity.x;
			data->initParticleData.bufferVelByteStride	= sizeof(Fluid::FluidParticle);
			data->initParticleData.bufferLife			= &data->initParticles[0].lifetime;
			data->initParticleData.bufferLifeByteStride	= sizeof(Fluid::FluidParticle);

			//Create particle filled sphere in buffer.
			NxVec3 fluidPos(-39.8f, 50.0f, 47.0f);
			float distance = 1.25f;
			CreateParticleBox(data->initParticleData, 1<<16, false, fluidPos, NxVec3(0.0f, 0.0f, 0.0f), NxVec3(32.0f, 96.0f, 29.0f), distance, 20.0f);
		}

		metaData.push_back(data);
	}
}


void FluidMetaDataManager::CreateParticleSphere(NxParticleData& pd, unsigned maxParticles, bool append, const NxVec3& pos,
												const NxVec3 vel, float lifeTime, float distance, unsigned sideNum) const
{
	float rad = sideNum*distance*0.5f;

	char* bufferPos = reinterpret_cast<char*>(pd.bufferPos);
	char* bufferVel = reinterpret_cast<char*>(pd.bufferVel);
	char* bufferLife = reinterpret_cast<char*>(pd.bufferLife);

	if(bufferPos == NULL && bufferVel == NULL && bufferLife == NULL)
		return;

	if(!append)
		(*pd.numParticlesPtr) = 0;
	else
	{
		bufferPos += pd.bufferPosByteStride * (*pd.numParticlesPtr);
		bufferVel += pd.bufferVelByteStride * (*pd.numParticlesPtr);
		bufferLife += pd.bufferLifeByteStride * (*pd.numParticlesPtr);
	}

	for(unsigned i=0; i<sideNum; i++)
		for(unsigned j=0; j<sideNum; j++)
			for(unsigned k=0; k<sideNum; k++)
			{
				if(*pd.numParticlesPtr >= maxParticles)
					break;

				NxVec3 p = NxVec3(i*distance,j*distance,k*distance);
				if(p.distance(NxVec3(rad,rad,rad)) < rad)
				{
					p += pos;

					if(bufferPos)
					{
						NxVec3& position = *reinterpret_cast<NxVec3*>(bufferPos);
						position = p;
						bufferPos += pd.bufferPosByteStride;
					}

					if(bufferVel)
					{
						NxVec3& velocity = *reinterpret_cast<NxVec3*>(bufferVel);
						velocity = vel;
						bufferVel += pd.bufferVelByteStride;
					}

					if(bufferLife)
					{
						NxReal& life = *reinterpret_cast<NxReal*>(bufferLife);
						life = lifeTime;
						bufferLife += pd.bufferLifeByteStride;
					}

					(*pd.numParticlesPtr)++;
				}
			}
}


// -----------------------------------------------------------------------------
// ------------------ FluidMetaDataManager::CreateParticleBox ------------------
// -----------------------------------------------------------------------------
void FluidMetaDataManager::CreateParticleBox(NxParticleData& pd, unsigned int maxParticles, bool append, const NxVec3& pos,
											 const NxVec3 vel, const NxVec3 extend, float distance, float lifeTime) const
{
	char* bufferPos = reinterpret_cast<char*>(pd.bufferPos);
	char* bufferVel = reinterpret_cast<char*>(pd.bufferVel);
	char* bufferLife = reinterpret_cast<char*>(pd.bufferLife);

	if(bufferPos == NULL && bufferVel == NULL && bufferLife == NULL)
		return;

	if(!append)
		(*pd.numParticlesPtr) = 0;
	else
	{
		bufferPos += pd.bufferPosByteStride * (*pd.numParticlesPtr);
		bufferVel += pd.bufferVelByteStride * (*pd.numParticlesPtr);
		bufferLife += pd.bufferLifeByteStride * (*pd.numParticlesPtr);
	}

	unsigned int numX = (int)(extend.x/distance);
	unsigned int numY = (int)(extend.y/distance);
	unsigned int numZ = (int)(extend.z/distance);

	NxVec3 corner = pos-extend*0.5f;

	for(unsigned i=0; i<numX; i++)
	{
		for(unsigned j=0; j<numY; j++)
		{
			for(unsigned k=0; k<numZ; k++)
			{
				if(*pd.numParticlesPtr >= maxParticles)
					break;

				NxVec3 p = NxVec3(i*distance,j*distance,k*distance);
				//if(p.distance(NxVec3(rad,rad,rad)) < rad)
				{
					p += corner;

					if(bufferPos)
					{
						NxVec3& position = *reinterpret_cast<NxVec3*>(bufferPos);
						position = p;
						bufferPos += pd.bufferPosByteStride;
					}

					if(bufferVel)
					{
						NxVec3& velocity = *reinterpret_cast<NxVec3*>(bufferVel);
						velocity = vel;
						bufferVel += pd.bufferVelByteStride;
					}

					if(bufferLife)
					{
						NxReal& life = *reinterpret_cast<NxReal*>(bufferLife);
						life = lifeTime*Math::RandomFloat(0.0f, 1.0f);
						bufferLife += pd.bufferLifeByteStride;
					}

					(*pd.numParticlesPtr)++;
				}
			}
		}
	}
}
