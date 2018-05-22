#ifndef __PHYSIC__H__
#define __PHYSIC__H__

#include <string>
#include <list>
#include <vector>

#include "NxPhysics.h"

#include "../Util/Singleton.h"
#include "../Util/Vector3.h"
#include "../Util/Vector4.h"
#include "../Util/Timer.h"

class RenderObject;

using namespace std;

class Matrix4;
class Fluid;

class FluidDescription;

class Physic : public Singleton<Physic>
{
	friend class Singleton<Physic>;

public:

	struct physicData 
	{
		int matrixId;
		string type;
		Vector3 position;
		Vector3 rotation;
		Vector3 measures;
		float radius;
		float density;
		bool collMesh;
		std::vector<RenderObject*> objects;
		list<physicData*> childs;
	};

	Physic(void);
	~Physic(void);

	/// Inits the physics engine
	void Init(void);
	
	/// Inits the fluid (must be called after the level was loaded)
	void InitFluid(unsigned int sceneIndex, FluidDescription* fluidDescription);

	/// Exits the fluid
	void ExitFluid(void);

	/// Updates the physics, not the simulation is updated,
	/// instead things like getting debug data for the next render call etc.
	void Update(float deltaTime);

	/// Renders physics stuff, eg. debug
	void Render(bool useMaterials);

	/// Exits the sdk
	void Exit(void);

	/// Makes a simulation step
	void Simulate(float deltaTime);

	/// Wait until everything is done
	void WaitForPhysics(void);

	/// Returns the number of actors in the physics world
	int GetNumOfActors(void);

	/// Sets a matrix for a actor
	void SetActorsPosition(const int& actorId, const Vector3& position);

	/// Returns the matrix of an actor
	Matrix4	GetActorsMatrix(const int& actorId);

	/// Returns the velocity of an actor
	Vector3 GetActorsVelocity(const int& actorId);

	/// Returns the actor with the given index
	NxActor* GetActor(const int& actorId);

	/// Releases the actor with the given id
	void ReleaseActor(const int& actorId);

	/// Creates the level with the given data
	void CreateLevel(list<physicData*>& levelData, bool createJoint);

	/// Releases the level
	void ReleaseLevel(void);

	/// Creates a sphere
	void CreateSphere(int id, const NxVec3& pos, int size=2, const NxVec3* initialVelocity=NULL);

	/// Creates a box
	void CreateBox(int id, const NxVec3& pos, const NxVec3& dimension, const NxVec3* initialVelocity=NULL);

	/// Returns the fluid
	Fluid* GetFluid(void) const { return fluid; }

private:

	struct JointStructure
	{
		NxJoint* joint;
		Vector3 position;
		float distanceFactor;
		float timeFactor;
		float speedFactor;
		int direction;
		float timer;
	};

	/// Creates a convex mesh shape
	void CreateConvexShape(RenderObject* mesh, NxConvexShapeDesc* desc);

	/// Creates a triangle mesh shape
	void CreateTriangleShape(RenderObject* mesh, NxTriangleMeshShapeDesc* desc);

	/// Creates a box
	NxActor* CreateBox(const NxVec3& pos, const NxVec3& boxDim, const NxReal density);

	/// Flags if hardware is used
	bool hardwareSimulation;

	/// Flags if the physics simulation is currently being calculated
	bool sceneRunning;

	/// Physics skd
	NxPhysicsSDK* physicsSDK;

	/// Physics scene
	NxScene* scene;

	/// Timestep
	NxReal timeStep;

	/// Time for updating joints
	float time;

	/// Fluid
	Fluid* fluid;

	bool pauseSimulation;

	Timer timer;
};

#endif