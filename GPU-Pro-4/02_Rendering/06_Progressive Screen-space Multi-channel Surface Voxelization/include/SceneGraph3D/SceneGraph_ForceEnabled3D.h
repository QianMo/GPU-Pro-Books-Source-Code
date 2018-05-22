#ifndef _SCENE_GRAPH_FORCEENABLED3D_
#define _SCENE_GRAPH_FORCEENABLED3D_

#include <libxml/tree.h>
#include <vector>
#include "Vector3D.h"
#include "SceneGraph_Node3D.h"

using namespace std;

class ForceEnabled3D
{
protected:
	float mass;
	float friction, init_friction;
	Vector3D speed;
	vector<class Force3D*> forces;
	vector<class Force3D*> init_forces;
	vector<char *> force_names;
	class World3D * global_world;

public:
	ForceEnabled3D();
	~ForceEnabled3D();
	void parse(xmlNodePtr pXMLNode);
	float getMass() {return mass;}
	virtual Vector3D getForcePoint() {return Vector3D(0,0,0);}
	int getNumForces() {return forces.size();}
	class Force3D* getForce(int i);
	Vector3D getTotalAcceleration();
	Vector3D getTotalForce();
	Vector3D getSpeed() {return speed;}
	void init(class World3D * w);
	void reset();
	void processMessage(char * msg);
};

#endif
