#ifndef _SCENE_GRAPH_FORCE3D_
#define _SCENE_GRAPH_FORCE3D_

#include "SceneGraph_Node3D.h"
#include "Vector3D.h"
#include <libxml/tree.h>
#include "SceneGraph_Timer3D.h"
#include "SceneGraph_ForceEnabled3D.h"

class Force3D: public Node3D
{

protected:
	Vector3D dir, transformed_dir, init_dir;
	float init_force;
	float force;
	float effective_radius;
	Vector3D center, transformed_center, init_center;
	Timer3D timer;
	float init_fade_time, fade_time;
	bool fading;
	static vector<char *> eventmap;

	virtual int getEventID (char * eventname) GET_EVENT_ID(eventname)
public:
	Force3D();
	~Force3D();
	virtual Vector3D getForce(Vector3D pos);
	virtual Vector3D getForce(ForceEnabled3D *node); // this version of the method can be used
	                                                // for more elaborate force fields such
													// as mass-dependent forces
	virtual Vector3D getAcceleration(ForceEnabled3D *node);
	virtual void app();
	virtual void init();
	virtual void reset();
	virtual void draw();
	virtual void parse(xmlNodePtr pXMLnode);
	virtual void processMessage(char * msg);
};


#endif