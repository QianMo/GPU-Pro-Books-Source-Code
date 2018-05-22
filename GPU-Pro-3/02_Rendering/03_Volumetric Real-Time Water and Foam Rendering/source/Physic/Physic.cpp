#include <stdio.h>

#include <vector>
#include <limits>

#include "NxPhysics.h"
#include "NxCooking.h"

#include "../Physic/Physic.h"
#include "../Physic/Stream.h"
#include "../Physic/Fluid.h"

#include "../Util/Matrix4.h"
#include "../Util/Vector3.h"
#include "../Util/Vector4.h"
#include "../Util/Math.h"
#include "../Util/ConfigLoader.h"

#include "../Render/RenderObject.h"
#include "../Render/ShaderManager.h"

#include "../Main/DemoManager.h"

#include "../Level/Camera.h"
#include "../Level/Light.h"

#include "../Input/InputManager.h"

#include <IL/il.h>
#include <IL/ilu.h>

//#define THREAD_SDK

class ErrorOutputStream : public NxUserOutputStream
{
	void reportError (NxErrorCode code, const char *message, const char* file, int line)
	{
		//this should be routed to the application
		//specific error handling. If this gets hit
		//then you are in most cases using the SDK
		//wrong and you need to debug your code!
		//however, code may  just be a warning or
		//information.

		if (code < NXE_DB_INFO)
		{
			//MessageBox(NULL, message, "SDK Error", MB_OK),
			print(message);
		}
	}

	NxAssertResponse reportAssertViolation (const char *message, const char *file,int line)
	{
		//this should not get hit by
		// a properly debugged SDK!
		assert(0);
		return NX_AR_CONTINUE;
	}

	void print (const char *message)
	{
		printf("SDK says: %s\n", message);
	}

} errorOutputStream;

const char* getNxSDKCreateError(const NxSDKCreateError& errorCode) 
{
	switch(errorCode) 
	{
	case NXCE_NO_ERROR: return "NXCE_NO_ERROR";
	case NXCE_PHYSX_NOT_FOUND: return "NXCE_PHYSX_NOT_FOUND";
	case NXCE_WRONG_VERSION: return "NXCE_WRONG_VERSION";
	case NXCE_DESCRIPTOR_INVALID: return "NXCE_DESCRIPTOR_INVALID";
	case NXCE_CONNECTION_ERROR: return "NXCE_CONNECTION_ERROR";
	case NXCE_RESET_ERROR: return "NXCE_RESET_ERROR";
	case NXCE_IN_USE_ERROR: return "NXCE_IN_USE_ERROR";
	default: return "Unknown error";
	}
};

NxActor* CreateDynamicBox(const NxVec3& pos, const NxVec3& boxDim, const NxReal density, NxScene* scene)
{
	// Add a single-shape actor to the scene
	NxActorDesc actorDesc;
	NxBodyDesc bodyDesc;

	// The actor has one shape, a box
	NxBoxShapeDesc boxDesc;
	boxDesc.dimensions.set(boxDim.x,boxDim.y,boxDim.z);
	boxDesc.localPose.t = NxVec3(0,boxDim.y,0);
	actorDesc.shapes.pushBack(&boxDesc);

	if (density)
	{
		actorDesc.body = &bodyDesc;
		actorDesc.density = density;
	}
	else
	{
		actorDesc.body = NULL;
	}
	actorDesc.globalPose.t = pos;
	return scene->createActor(actorDesc);	
}

// -----------------------------------------------------------------------------
// --------------------------- Physic::Physic ----------------------------------
// -----------------------------------------------------------------------------
Physic::Physic(void) :
	hardwareSimulation(false),
	sceneRunning(false),
	physicsSDK(NULL),
	scene(NULL),
	fluid(NULL),
	pauseSimulation(false)
{
}


// -----------------------------------------------------------------------------
// --------------------------- Physic::Physic ----------------------------------
// -----------------------------------------------------------------------------
Physic::~Physic(void)
{
	Exit();
}


// -----------------------------------------------------------------------------
// --------------------------- Physic::Init ------------------------------------
// -----------------------------------------------------------------------------
void Physic::Init(void)
{
	// Initialize PhysicsSDK
	NxPhysicsSDKDesc desc;
	NxSDKCreateError errorCode = NXCE_NO_ERROR;

	physicsSDK = NxCreatePhysicsSDK(NX_PHYSICS_SDK_VERSION, NULL, &errorOutputStream, desc, &errorCode);
	assert(physicsSDK != NULL);

	physicsSDK->getFoundationSDK().getRemoteDebugger()->connect ("localhost", 5425);

	physicsSDK->setParameter(NX_SKIN_WIDTH, 0.01f);
	physicsSDK->setParameter(NX_ADAPTIVE_FORCE, 1);

	// Create a scene
	NxSceneDesc sceneDesc;
	sceneDesc.gravity = NxVec3(0.0f, -98.1f, 0.0f);

#if defined(THREAD_SDK)
	sceneDesc.flags |= NX_SF_ENABLE_MULTITHREAD;
	sceneDesc.internalThreadCount = 2;
#else 
	// No threading
#endif

	if (physicsSDK->getHWVersion() != NX_HW_VERSION_NONE)
	{
		sceneDesc.simType = NX_SIMULATION_HW;
		hardwareSimulation = true;
	}
	else
	{
		sceneDesc.simType = NX_SIMULATION_SW;
		hardwareSimulation = false;
	}

	scene = physicsSDK->createScene(sceneDesc);
	if(scene == NULL) 
	{
		assert(false);
	}

	scene->setTiming(1.0f/30.0f, 8, NX_TIMESTEP_FIXED);

	// Set default material
	NxMaterial* defaultMaterial = scene->getMaterialFromIndex(0);
	defaultMaterial->setRestitution(0.1f);
	defaultMaterial->setStaticFriction(0.2f);
	defaultMaterial->setDynamicFriction(0.2f);

	// Create ground plane
	NxPlaneShapeDesc planeDesc;
	planeDesc.normal = NxVec3(0.0f, 1.0f, 0.0f);
	planeDesc.d = -200.0f;
	//planeDesc.d = 0.0f;
	NxActorDesc actorDesc;
	actorDesc.shapes.pushBack(&planeDesc);
	scene->createActor(actorDesc);

	time = 0.0f;
}

// -----------------------------------------------------------------------------
// ------------------------- Physic::InitFluid ---------------------------------
// -----------------------------------------------------------------------------
void Physic::InitFluid(unsigned int sceneIndex, FluidDescription* fluidDescription)
{
	// create fluid
	fluid = new Fluid(scene, hardwareSimulation, sceneIndex, fluidDescription);
}

// -----------------------------------------------------------------------------
// ------------------------- Physic::ExitFluid ---------------------------------
// -----------------------------------------------------------------------------
void Physic::ExitFluid(void)
{
	// delete fluid
	delete fluid;
	fluid = NULL;
}

// -----------------------------------------------------------------------------
// --------------------------- Physic::Update ----------------------------------
// -----------------------------------------------------------------------------
void Physic::Update(float deltaTime)
{
	if (scene)
	{
		time += deltaTime;

		fluid->Update(deltaTime);

		if (InputManager::Instance()->IsKeyPressedAndReset('p'))
		{
			pauseSimulation = !pauseSimulation;

			if (pauseSimulation)
				DemoManager::Instance()->SetDebugMsg("Simulation paused");
			else
				DemoManager::Instance()->SetDebugMsg("Simulation resumed");
		}
	}
}

// -----------------------------------------------------------------------------
// --------------------------- Physic::Render ----------------------------------
// -----------------------------------------------------------------------------
void Physic::Render(bool useMaterials)
{
	glDisable(GL_CULL_FACE);
	glDisable(GL_BLEND);

	if (useMaterials)
	{
		Vector3 eyePosition = DemoManager::Instance()->GetCamera()->GetCameraPosition();
		Vector3 lightPosition = DemoManager::Instance()->GetLight()->GetLightPosition();

		ShaderManager::Instance()->SetParameter3fv(ShaderManager::SP_EYE_POS_OBJ_SPACE, eyePosition.comp);
		ShaderManager::Instance()->SetParameter3fv(ShaderManager::SP_LIGHT_POS_OBJ_SPACE, lightPosition.comp);
	}

	glEnable(GL_CULL_FACE);
}

// -----------------------------------------------------------------------------
// --------------------------- Physic::Exit ------------------------------------
// -----------------------------------------------------------------------------
void Physic::Exit(void)
{
	WaitForPhysics();
	ExitFluid();

	if(physicsSDK != NULL)
	{
		if(scene != NULL) physicsSDK->releaseScene(*scene);

		physicsSDK->release();
	}

	physicsSDK = NULL;
	scene = NULL;
}


// -----------------------------------------------------------------------------
// ------------------------- Physic::Simulate ----------------------------------
// -----------------------------------------------------------------------------
void Physic::Simulate(float deltaTime)
{
	timer.Start();

	if (sceneRunning || pauseSimulation)
		return;

	{
		// Run collision and dynamics for delta time since the last frame
		scene->simulate(1.0f/30.0f);
		sceneRunning = true;

		DemoManager::Instance()->PerformCpuCaluations(deltaTime);

		WaitForPhysics();
	}
}

// -----------------------------------------------------------------------------
// ---------------------- Physic::WaitForPhysics -------------------------------
// -----------------------------------------------------------------------------
void Physic::WaitForPhysics(void)
{
	if (sceneRunning)
	{
		NxU32 error;

		scene->flushStream();
		scene->fetchResults(NX_RIGID_BODY_FINISHED, true, &error);

		//assert(error == 0);
		sceneRunning = false;
	}
}


// -----------------------------------------------------------------------------
// ------------------------- Physic::GetNumOfActors ----------------------------
// -----------------------------------------------------------------------------
int Physic::GetNumOfActors(void)
{
	return scene->getNbActors();
}

// -----------------------------------------------------------------------------
// ------------------------- Physic::GetActorsMatrix ---------------------------
// -----------------------------------------------------------------------------
void Physic::SetActorsPosition(const int& actorId, const Vector3& position)
{
	int nbActors = scene->getNbActors();

	NxActor** actors = scene->getActors();
	while (nbActors--)
	{
		NxActor* actor = *actors++;

		if (actor->userData == NULL) continue;

		if (int(size_t(actor->userData)) != actorId) continue;

		actor->setGlobalPosition(NxVec3(position.x, position.y, position.z));
		actor->setLinearVelocity(NxVec3(0.0f, 0.0f, 0.0f));
		actor->setAngularVelocity(NxVec3(0.0f, 0.0f, 0.0f));
	}
}

// -----------------------------------------------------------------------------
// ------------------------- Physic::GetActorsMatrix ---------------------------
// -----------------------------------------------------------------------------
Matrix4 Physic::GetActorsMatrix(const int& actorId)
{
	int nbActors = scene->getNbActors();

	NxActor** actors = scene->getActors();
	while (nbActors--)
	{
		NxActor* actor = *actors++;

		if (actor->userData == NULL) continue;

		if (int(size_t(actor->userData)) != actorId) continue;

		float glMat[16];
		actor->getGlobalPose().getColumnMajor44(glMat);

		return Matrix4(glMat[0],  glMat[1],  glMat[2],  glMat[3],
					   glMat[4],  glMat[5],  glMat[6],  glMat[7],
					   glMat[8],  glMat[9],  glMat[10], glMat[11],
					   glMat[12], glMat[13], glMat[14], glMat[15]);
	}

	return Matrix4::IDENTITY;
}

// -----------------------------------------------------------------------------
// ----------------------- Physic::GetActorsVelocity ---------------------------
// -----------------------------------------------------------------------------
Vector3 Physic::GetActorsVelocity( const int& actorId )
{
	int nbActors = scene->getNbActors();

	NxActor** actors = scene->getActors();
	while (nbActors--)
	{
		NxActor* actor = *actors++;

		if (actor->userData == NULL) continue;

		if (int(size_t(actor->userData)) != actorId) continue;

		NxVec3 ret = actor->getLinearVelocity();
		return Vector3(ret.x, ret.y, ret.z);
	}

	return Vector3(0.0f, 0.0f, 0.0f);
}


// -----------------------------------------------------------------------------
// ------------------------------ Physic::GetActor -----------------------------
// -----------------------------------------------------------------------------
NxActor* Physic::GetActor(const int& actorId)
{
	int nbActors = scene->getNbActors();

	NxActor** actors = scene->getActors();
	while (nbActors--)
	{
		NxActor* actor = *actors++;

		if (actor->userData == NULL) continue;

		if (int(size_t(actor->userData)) != actorId) continue;

		return actor;
	}

	return NULL;
}



// -----------------------------------------------------------------------------
// ---------------------------- Physic::ReleaseActor ---------------------------
// -----------------------------------------------------------------------------
void Physic::ReleaseActor(const int& actorId)
{
	int nbActors = scene->getNbActors();

	NxActor** actors = scene->getActors();
	while (nbActors--)
	{
		NxActor* actor = *actors++;

		if (actor->userData == NULL) continue;

		if (int(size_t(actor->userData)) != actorId) continue;

		scene->releaseActor(*actor);
	}
}

// -----------------------------------------------------------------------------
// ------------------------------ Physic::CreateLevel --------------------------
// -----------------------------------------------------------------------------
void Physic::CreateLevel(list<physicData*>& levelData, bool createJoint)
{
	physicData* pData = NULL;
	
	list<physicData*>::iterator iter;
	for (iter=levelData.begin(); iter!=levelData.end(); ++iter)
	{
		// get the first element
		pData = *iter;

		if (pData->type == "box")
		{
			NxActorDesc actorDesc;
			actorDesc.globalPose.t = NxVec3(pData->position.x, pData->position.y, pData->position.z);
			
			Matrix4 rot = Matrix4::Matrix4Rotation(pData->rotation.x, pData->rotation.y, pData->rotation.z);
			NxMat33 rotation;
			rotation.setColumn(0, NxVec3(rot.entry[0], rot.entry[1], rot.entry[2]));
			rotation.setColumn(1, NxVec3(rot.entry[4], rot.entry[5], rot.entry[6]));
			rotation.setColumn(2, NxVec3(rot.entry[8], rot.entry[9], rot.entry[10]));
			actorDesc.globalPose.M = rotation;

			NxBoxShapeDesc boxShapeDesc;
			boxShapeDesc.dimensions = NxVec3(pData->measures.x / 2, pData->measures.y / 2, pData->measures.z / 2);

			NxBodyDesc bodyDesc;
			bodyDesc.angularDamping	= 0.5f;
			bodyDesc.flags = NX_BF_KINEMATIC | NX_BF_ENERGY_SLEEP_TEST;
			//bodyDesc.flags = NX_BF_ENERGY_SLEEP_TEST;

			actorDesc.body = &bodyDesc;
			actorDesc.density = 10.0f;
			actorDesc.shapes.pushBack(&boxShapeDesc);

			NxActor* actor = scene->createActor(actorDesc);

			const int id = pData->matrixId;
			actor->userData = (void*)size_t(id);
		}
		else if (pData->type == "sphere")
		{
			NxActorDesc actorDesc;
			actorDesc.globalPose.t = NxVec3(pData->position.x, pData->position.y, pData->position.z);

			Matrix4 rot = Matrix4::Matrix4Rotation(pData->rotation.x, pData->rotation.y, pData->rotation.z);
			NxMat33 rotation;
			rotation.setColumn(0, NxVec3(rot.entry[0], rot.entry[1], rot.entry[2]));
			rotation.setColumn(1, NxVec3(rot.entry[4], rot.entry[5], rot.entry[6]));
			rotation.setColumn(2, NxVec3(rot.entry[8], rot.entry[9], rot.entry[10]));
			actorDesc.globalPose.M = rotation;

			NxSphereShapeDesc sphereShapeDesc;
			sphereShapeDesc.radius = pData->radius;
			sphereShapeDesc.shapeFlags |= NX_SF_FLUID_TWOWAY;

			NxBodyDesc bodyDesc;
			bodyDesc.angularDamping	= 0.5f;
			bodyDesc.flags = NX_BF_ENERGY_SLEEP_TEST | NX_BF_KINEMATIC;

			actorDesc.body = &bodyDesc;
			actorDesc.density = pData->density;
			actorDesc.shapes.pushBack(&sphereShapeDesc);

			NxActor* actor = scene->createActor(actorDesc);

			const int id = pData->matrixId;
			actor->userData = (void*)size_t(id);
		}
		else if (pData->type == "dynamic_sphere")
		{
			NxActorDesc actorDesc;
			actorDesc.globalPose.t = NxVec3(pData->position.x, pData->position.y, pData->position.z);

			Matrix4 rot = Matrix4::Matrix4Rotation(pData->rotation.x, pData->rotation.y, pData->rotation.z);
			NxMat33 rotation;
			rotation.setColumn(0, NxVec3(rot.entry[0], rot.entry[1], rot.entry[2]));
			rotation.setColumn(1, NxVec3(rot.entry[4], rot.entry[5], rot.entry[6]));
			rotation.setColumn(2, NxVec3(rot.entry[8], rot.entry[9], rot.entry[10]));
			actorDesc.globalPose.M = rotation;

			NxSphereShapeDesc sphereShapeDesc;
			sphereShapeDesc.radius = pData->radius;
			sphereShapeDesc.shapeFlags |= NX_SF_FLUID_TWOWAY;

			NxBodyDesc bodyDesc;
			bodyDesc.angularDamping	= 0.5f;
			bodyDesc.flags = NX_BF_ENERGY_SLEEP_TEST;

			actorDesc.body = &bodyDesc;
			actorDesc.density = pData->density;
			actorDesc.shapes.pushBack(&sphereShapeDesc);

			NxActor* actor = scene->createActor(actorDesc);

			const int id = pData->matrixId;
			actor->userData = (void*)size_t(id);
		}
		else if ((pData->type == "mesh") || (pData->type == "convex_mesh") || (pData->type == "dynamic_mesh") || (pData->type == "dynamic_convex_mesh"))
		{
			NxActorDesc actorDesc;
			actorDesc.globalPose.t = NxVec3(pData->position.x, pData->position.y, pData->position.z);

			Matrix4 rot = Matrix4::Matrix4Rotation(pData->rotation.x, pData->rotation.y, pData->rotation.z);
			NxMat33 rotation;
			rotation.setColumn(0, NxVec3(rot.entry[0], rot.entry[1], rot.entry[2]));
			rotation.setColumn(1, NxVec3(rot.entry[4], rot.entry[5], rot.entry[6]));
			rotation.setColumn(2, NxVec3(rot.entry[8], rot.entry[9], rot.entry[10]));
			actorDesc.globalPose.M = rotation;

			NxBodyDesc bodyDesc;
			bodyDesc.angularDamping	= 0.5f;
			bodyDesc.flags = NX_BF_ENERGY_SLEEP_TEST;

			if (pData->type == "dynamic_mesh" || pData->type == "dynamic_convex_mesh")
				actorDesc.body = &bodyDesc;
			actorDesc.density = pData->density;
			
			if (pData->type == "convex_mesh" || pData->type == "dynamic_convex_mesh")
			{
				unsigned int i;
				for (i=0; i<pData->objects.size(); i++)
				{
					NxConvexShapeDesc* convexShapeDesc = new NxConvexShapeDesc;
					CreateConvexShape(pData->objects[i], convexShapeDesc);
					actorDesc.shapes.pushBack(convexShapeDesc);
				}
			}
			else
			{
				unsigned int i;
				for (i=0; i<pData->objects.size(); i++)
				{
					NxTriangleMeshShapeDesc* triangleShapeDesc = new NxTriangleMeshShapeDesc;
					CreateTriangleShape(pData->objects[i], triangleShapeDesc);
					actorDesc.shapes.pushBack(triangleShapeDesc);
				}
			}

			if (pData->collMesh)
			{
				unsigned int i;
				for (i=0; i<pData->objects.size(); i++)
				{
					pData->objects[i]->Exit();
					delete pData->objects[i];
				}
				pData->objects.clear();
			}

			NxActor* actor = scene->createActor(actorDesc);

			unsigned int i;
			for (i=0; i<actorDesc.shapes.size(); i++)
			{
				delete actorDesc.shapes[i];
			}

			const int id = pData->matrixId;
			actor->userData = (void*)size_t(id);
		}
		else
		{
			assert(false);
		}
	}

	if (createJoint)
	{
		NxRevoluteJointDesc revDesc;

		revDesc.actor[0] = GetActor(2);
		revDesc.actor[1] = NULL;

		revDesc.setGlobalAxis(NxVec3(1.0f, 0.0f, 0.0f)); //The direction of the axis the bodies revolve around.
		revDesc.setGlobalAnchor(NxVec3(0.626f, 30.566f, 52.389f)); //Reference point that the axis passes through.

		revDesc.isValid();

		scene->createJoint(revDesc);
	}
}

// -----------------------------------------------------------------------------
// ------------------------------ Physic::ReleaseLevel -------------------------
// -----------------------------------------------------------------------------
void Physic::ReleaseLevel(void)
{
	if (scene)
	{
		vector<JointStructure>::iterator i;

		int nbActors = scene->getNbActors();

		NxActor** actors = scene->getActors();
		while (nbActors--)
		{
			NxActor* actor = *actors++;
			scene->releaseActor(*actor);
		}

		time = 0.0f;
	}
}

// -----------------------------------------------------------------------------
// -------------------------- Physic::CreateConvexShape ------------------------
// -----------------------------------------------------------------------------
void Physic::CreateConvexShape(RenderObject* mesh, NxConvexShapeDesc* desc)
{
	NxVec3* points = NULL;

	int numVertices = mesh->GetNumVertices();
	int numIndices = mesh->GetNumIndices();

	const RenderObject::Vertex* vertices = mesh->GetVertices();
	const unsigned int* indices = mesh->GetIndices();

	points = new NxVec3[numVertices];

	int i;
	for (i=0; i<numVertices; i++)
	{
		points[i] = NxVec3(vertices[i].vertex.x, vertices[i].vertex.y, vertices[i].vertex.z);
	}

	NxConvexMeshDesc convexMeshDesc;
	convexMeshDesc.numVertices = numVertices;
	convexMeshDesc.numTriangles = numIndices / 3;
	convexMeshDesc.pointStrideBytes = sizeof(NxVec3);
	convexMeshDesc.triangleStrideBytes = 3*sizeof(NxU32);//Number of bytes from one triangle to the next.
	convexMeshDesc.points = points;
	convexMeshDesc.triangles = indices;
	convexMeshDesc.flags = NX_MF_HARDWARE_MESH;

	static NxCookingInterface *cooking = NxGetCookingLib(NX_PHYSICS_SDK_VERSION);
	cooking->NxInitCooking();

	MemoryWriteBuffer buffer;
	cooking->NxCookConvexMesh(convexMeshDesc, buffer);
	desc->meshData = physicsSDK->createConvexMesh(MemoryReadBuffer(buffer.data));
	desc->shapeFlags |= NX_SF_FLUID_TWOWAY;

	if (points)
	{
		delete[] points;
		points = NULL;
	}
}

// -----------------------------------------------------------------------------
// ------------------------ Physic::CreateTriangleShape ------------------------
// -----------------------------------------------------------------------------
void Physic::CreateTriangleShape(RenderObject* mesh, NxTriangleMeshShapeDesc* desc)
{
	NxVec3* points = NULL;

	int numVertices = mesh->GetNumVertices();
	int numIndices = mesh->GetNumIndices();

	const RenderObject::Vertex* vertices = mesh->GetVertices();
	const unsigned int* indices = mesh->GetIndices();

	points = new NxVec3[numVertices];

	int i;
	for (i=0; i<numVertices; i++)
	{
	points[i] = NxVec3(vertices[i].vertex.x, vertices[i].vertex.y, vertices[i].vertex.z);
	}

	NxTriangleMeshDesc meshDesc;
	meshDesc.numVertices			= numVertices;
	meshDesc.numTriangles			= numIndices / 3;
	meshDesc.pointStrideBytes		= sizeof(NxVec3);
	meshDesc.triangleStrideBytes	= 3*sizeof(NxU32);
	meshDesc.points					= points;
	meshDesc.triangles				= indices;							
	meshDesc.flags					= 0;
	meshDesc.flags					= NX_MF_HARDWARE_MESH;

	static NxCookingInterface *cooking = NxGetCookingLib(NX_PHYSICS_SDK_VERSION);
	cooking->NxInitCooking();

	MemoryWriteBuffer buffer;
	cooking->NxCookTriangleMesh(meshDesc, buffer);
	desc->meshData = physicsSDK->createTriangleMesh(MemoryReadBuffer(buffer.data));

	desc->shapeFlags |= NX_SF_FLUID_TWOWAY;
	desc->meshPagingMode = NX_MESH_PAGING_AUTO;

	if (points)
	{
		delete[] points;
		points = NULL;
	}
}

// -----------------------------------------------------------------------------
// ------------------------- Physic::CreateSphere ------------------------------
// -----------------------------------------------------------------------------
void Physic::CreateSphere(int id, const NxVec3& pos, int size, const NxVec3* initialVelocity)
{
	if(scene == NULL)
		return;

	if (GetNumOfActors() >= 1000)
		return;

	// Create body
	NxBodyDesc bodyDesc;
	bodyDesc.angularDamping	= 0.1f;
	if(initialVelocity)
		bodyDesc.linearVelocity = *initialVelocity;

	NxSphereShapeDesc sphereDesc;
	sphereDesc.radius = (NxReal) size;
	sphereDesc.shapeFlags |= NX_SF_FLUID_TWOWAY;

	NxActorDesc actorDesc;
	actorDesc.shapes.pushBack(&sphereDesc);
	actorDesc.body			= &bodyDesc;
	actorDesc.density		= 3.0f;
	actorDesc.globalPose.t  = pos;
	NxActor* actor = scene->createActor(actorDesc);

	actor->userData = (void*)size_t(id);
	actor->setLinearVelocity(NxVec3(0.0f, 0.0f, 0.0f));
}

// -----------------------------------------------------------------------------
// --------------------------- Physic::CreateBox -------------------------------
// -----------------------------------------------------------------------------
void Physic::CreateBox(int id, const NxVec3& pos, const NxVec3& dimension, const NxVec3* initialVelocity)
{
	if(scene == NULL)
		return;

	if (GetNumOfActors() >= 1000)
		return;

	// Create body
	NxBodyDesc bodyDesc;
	bodyDesc.angularDamping	= 0.1f;
	if(initialVelocity)
		bodyDesc.linearVelocity = *initialVelocity;

	NxBoxShapeDesc boxDesc;
	boxDesc.dimensions = dimension;
	boxDesc.localPose.t = pos;
	boxDesc.shapeFlags |= NX_SF_FLUID_TWOWAY;

	NxActorDesc actorDesc;
	actorDesc.shapes.pushBack(&boxDesc);
	actorDesc.body			= &bodyDesc;
	actorDesc.density		= 10.0f;
	//actorDesc.globalPose.t	= pos;
	NxActor* actor = scene->createActor(actorDesc);

	actor->userData = (void*)size_t(id);
	actor->setLinearVelocity(NxVec3(0.0f, 0.0f, 0.0f));
}

// -----------------------------------------------------------------------------
// ----------------------------- Physic::CreateBox -----------------------------
// -----------------------------------------------------------------------------
NxActor* Physic::CreateBox(const NxVec3& pos, const NxVec3& boxDim, const NxReal density)
{
	// Add a single-shape actor to the scene
	NxActorDesc actorDesc;
	NxBodyDesc bodyDesc;

	// The actor has one shape, a box
	NxBoxShapeDesc boxDesc;
	boxDesc.dimensions.set(boxDim.x,boxDim.y,boxDim.z);
	boxDesc.localPose.t = NxVec3(0,boxDim.y,0);
	actorDesc.shapes.pushBack(&boxDesc);

	if (density)
	{
		actorDesc.body = &bodyDesc;
		actorDesc.density = density;
	}
	else
	{
		actorDesc.body = NULL;
	}
	actorDesc.globalPose.t = pos;
	return scene->createActor(actorDesc);	
}