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

#ifndef _SCENE_GRAPH_USER3D_
#define _SCENE_GRAPH_USER3D_

#include <libxml/tree.h>

#include "cglib.h"
#include "SceneGraph_TransformationNode3D.h"

class User3D: public TransformNode3D
{
private:
	bool reseting;
protected:
	Vector3D position, init_position, next_position;
	Vector3D target, init_target;
	Vector3D direction, init_direction;
	Vector3D right, init_right;
	Vector3D up, init_up;
	float init_height, height;
	float radius, init_radius;
	float linear_speed, init_linear_speed;
	float angular_speed, init_angular_speed;
	bool upright, init_upright;
	bool free_roll, init_free_roll;
	int mode, init_mode;
	virtual void calcMatrix();
	Input3D * controller;
	char * controller_name;
	bool actuator[32];
	int num_actuators;
	static vector<char *> eventmap;
	double prev_time;

	virtual int getEventID (char * eventname) GET_EVENT_ID(eventname)
	void adjustPosition();
public:
	User3D();
	~User3D();
	virtual void parse(xmlNodePtr pXMLnode);
	virtual void init();
	virtual void reset();
	virtual void app();
	virtual void draw();
	void setPosition(float x, float y, float z);
	void setLookAt(float x, float y, float z);
	void setLinearSpeed(float lsp) {linear_speed = lsp;}
	void setAngularSpeed(float asp) {angular_speed = asp;}
	void move(float mult);
	void updown(float mult);
	void strafe(float mult);
	void turn(float mult);
	void tilt(float mult);
	void orbit(float mult);
	Vector3D getPosition() {return position;}
	Vector3D getLookAt() {return target;}
	Vector3D getRight() {return right;}
	Vector3D getUp() {return up;}
	virtual Vector3D getWorldPosition();
	int getNumActuators() {return num_actuators;}
	bool isActuatorEnabled(int ac) { return ((ac<32)&&(ac>=0))?actuator[ac]:false;}
	static void registerEvent(char *evt) {eventmap.push_back(evt);}
	virtual void processMessage(char * msg);
};

#endif

