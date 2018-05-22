//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//  Scene Graph 3D                                                          //
//  Georgios Papaioannou, 2009                                              //
//                                                                          //
//  This is a free, extensible scene graph management library that works    //
//  along with the EaZD deferred renderer. Both libraries and their source  //
//  code are free. If you use this code as is or any part of it in any kind //
//  of project or product, please acknowledge the source and its author.    //
//                                                                          //
//  For manuals, help and instructions, please visit:                       //
//  http://graphics.cs.aueb.gr/graphics/                                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

#ifndef _SCENE_GRAPH_CAMERA3D_
#define _SCENE_GRAPH_CAMERA3D_

#include <libxml/tree.h>

#include "cglib.h"
#include "SceneGraph_Node3D.h"

class Camera3D: public Node3D
{
protected:
	Node3D * attached_to, * init_attached_to;
	float apperture, init_apperture,
		  cnear, init_near,
		  cfar, init_far,
		  focal_distance, init_focal_distance,
		  focal_range, init_focal_range;
	char * to_follow_name;
	Vector3D cop;
	Vector3D dir;

	static vector<char *> eventmap;
	virtual int getEventID (char * eventname) GET_EVENT_ID(eventname)

public:
	Camera3D();
	~Camera3D();
	virtual void parse(xmlNodePtr pXMLnode);
	virtual void init();
	virtual void reset();
	virtual void app();
	void setupViewProjection(int width, int height);
	void attachTo(Node3D * nd);
	Node3D * attachedTo() {return attached_to;}
	void attachTo(char * nname);
	void setNear(float n) {cnear=n;}
	void setFar(float f) {cfar=f;}
	float getNear() { return cnear;}
	float getFar() { return cfar;}
	Vector3D getCOP() {return Vector3D(cop);}
	Vector3D getDirection() {return dir;}
	virtual void processMessage(char * msg);

	static void registerEvent(char *evt) {eventmap.push_back(evt);}
};

#endif

