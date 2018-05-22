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

#include <typeinfo>

#include "SceneGraph.h"

User3D::~User3D()
{
	if (controller_name)
		free (controller_name);
}

User3D::User3D()
{
	position = init_position = Vector3D(0,0,0);
	target = init_target = Vector3D(0,0,-1);
	linear_speed = init_linear_speed = 1.0f;
	angular_speed = init_angular_speed = 1.0f;
	direction = init_direction = Vector3D(0,0,-1);
	right = init_right = Vector3D(1,0,0);
	up = init_up = Vector3D(0,1,0);
	mode = init_mode = SCENE_GRAPH_USER_MODE_NAVIGATE;
	controller = NULL;
	controller_name = NULL;
	upright = init_upright = false;
	free_roll = init_free_roll = false;
	height = init_height = 1.7f;
	radius = init_radius = 0.3f;
	reseting = false;
}

void User3D::init()
{
	TransformNode3D::init();
	position = init_position;
	target = init_target;
	linear_speed = init_linear_speed;
	angular_speed = init_angular_speed;
	direction = target - position;
	direction.normalize();
	right = direction.cross(up);
	right.normalize();
	up = right.cross(direction);
	up.normalize();
	upright = init_upright;
	free_roll = init_free_roll;
	mode = init_mode;
	height = init_height;
	radius = init_radius;
	calcMatrix();
	prev_time = world->getTime();
	Node3D * nd = NULL;
	if (controller_name)
		nd = world->getNodeByName(controller_name);
	if (nd && typeid(*nd)==typeid(Input3D))
		controller = reinterpret_cast<Input3D*>(nd);
}

void User3D::reset()
{
	TransformNode3D::reset();
	position = init_position;
	target = init_target;
	linear_speed = init_linear_speed;
	angular_speed = init_angular_speed;
	direction = target - position;
	direction.normalize();
	right = direction.cross(up);

	if (!free_roll)
		right.y=0;

	right.normalize();
	up = right.cross(direction);
	up.normalize();
	mode = init_mode;
	upright = init_upright;
	free_roll = init_free_roll;
	height = init_height;
	radius = init_radius;
	calcMatrix();
	prev_time = world->getTime();
	Node3D * nd = NULL;
	if (controller_name)
		nd = world->getNodeByName(controller_name);
	if (nd && typeid(*nd)==typeid(Input3D))
		controller = reinterpret_cast<Input3D*>(nd);
	reseting = true;
}

void User3D::setPosition(float x, float y, float z)
{
	position = Vector3D(x,y,z);
	direction = target - position;
	direction.normalize();
	right = direction.cross(up);
	right.normalize();
	up = right.cross(direction);
	up.normalize();
	calcMatrix();
}

void User3D::setLookAt(float x, float y, float z)
{
	target = Vector3D(x,y,z);
	direction = target - position;
	direction.normalize();
	right = direction.cross(up);
	right.normalize();
	up = right.cross(direction);
	up.normalize();
	calcMatrix();
}

void User3D::adjustPosition()
{
	//Vector3D a = getTotalAcceleration();
	//float dt = (float)(world->getDeltaTime());
	//next_position+=a*(dt*dt);
	
	Vector3D offset = next_position-position;
	float dist_to_go = offset.length();
	if (dist_to_go<0.00001)
		return;

	position = next_position;
	
	target += offset;
	direction = target - position;
	direction.normalize();
	right = direction.cross(up);
	right.normalize();
	up = right.cross(direction);
	up.normalize();
	calcMatrix();
	this->getTransform();
}

void User3D::move(float mult)
{
	Vector3D offset;
	if (!upright)
		offset = direction*mult*linear_speed*(world->getDeltaTime());
	else
	{
		Vector3D planar_dir = direction;
		planar_dir.y=0.0f;
		planar_dir.normalize();
		offset = planar_dir*mult*linear_speed*(world->getDeltaTime());
	}
	next_position += offset;
//	target = target + offset;
	//calcMatrix();
//	printf("position = %f %f %f\n", position.x, position.y, position.z);
//	printf("target = %f %f %f\n", target.x, target.y, target.z);
	if (mult>0)
	{
		EVENT_OCCURED("moveforward");
	}
	else if (mult<0)
	{
		EVENT_OCCURED("movebackward");
	}
}

void User3D::strafe(float mult)
{
	Vector3D offset = right*mult*linear_speed*(world->getDeltaTime());
	next_position += offset;
//	target = target + offset;
	//calcMatrix();
	if (mult>0)
	{
		EVENT_OCCURED("moveright");
	}
	else if (mult<0)
	{
		EVENT_OCCURED("moveleft");
	}
}

void User3D::updown(float mult)
{
	Vector3D offset = up*mult*linear_speed*(world->getDeltaTime());
	next_position += offset;
	//target = target + offset;
	//calcMatrix();
	if (mult>0)
	{
		EVENT_OCCURED("moveup");
	}
	else if (mult<0)
	{
		EVENT_OCCURED("movedown");
	}
}

void User3D::turn(float mult)
{
	float len = (target-position).length();
	direction = direction + right*mult*angular_speed;
	direction.normalize();
	right = direction.cross(up);
	if (!free_roll)
		right.y=0;
	right.normalize();
	up = right.cross(direction);
	up.normalize();
	target = position + direction*len;
	calcMatrix();
	if (mult>0)
	{
		EVENT_OCCURED("turnright");
	}
	else if (mult<0)
	{
		EVENT_OCCURED("turnleft");
	}
}

void User3D::tilt(float mult)
{
	float len = (target-position).length();
	direction = direction + up*angular_speed*mult;
	direction.normalize();
	up = right.cross(direction);
	up.normalize();
	right = direction.cross(up);
	if (!free_roll)
		right.y=0;
	right.normalize();
	up = right.cross(direction);
	up.normalize();
	target = position + direction*len;
	calcMatrix();
	if (mult>0)
	{
		EVENT_OCCURED("turnup");
	}
	else if (mult<0)
	{
		EVENT_OCCURED("turndown");
	}
}

void User3D::orbit(float mult)
{
	float len = (target-position).length();
	direction -= right*mult*(world->getTime()-prev_time);
	direction.normalize();
	right = direction.cross(up);
	if (!free_roll)
		right.y=0;
	right.normalize();
	up = right.cross(direction);
	up.normalize();
	position = target - direction*len;
	calcMatrix();
}

void User3D::calcMatrix()
{
	DBL mdata[16];
	mdata[0] = right[0]; mdata[1] = up[0]; mdata[2] = -direction[0];
	mdata[4] = right[1]; mdata[5] = up[1]; mdata[6] = -direction[1];
	mdata[8] = right[2]; mdata[9] = up[2]; mdata[10] = -direction[2];
	mdata[12] = mdata[13] = mdata[14] = 0;
	mdata[3] = position[0]; mdata[7] = position[1]; mdata[11] = position[2];
	mdata[15] = 1;
	matrix.setData(mdata);
	matrix.transpose();
}

void User3D::parse(xmlNodePtr pXMLNode)
{
	char * val = NULL;
	
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"input");
	if (val)
	{
		controller_name = STR_DUP(val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"freeroll");
	if (val)
	{
		parseBoolean(init_free_roll, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"upright");
	if (val)
	{
		parseBoolean(init_upright, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"height");
	if (val)
	{
		parseFloat(init_height, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"radius");
	if (val)
	{
		parseFloat(init_radius, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"position");
	if (val)
	{
		parseVec3(init_position, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"lookat");
	if (val)
	{
		parseVec3(init_target, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"turn");
	if (val)
	{
		parseFloat(init_angular_speed, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"speed");
	if (val)
	{
		parseFloat(init_linear_speed, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"control");
	if (val)
	{
		if (STR_EQUAL(val,"navigate"))
			init_mode = SCENE_GRAPH_USER_MODE_NAVIGATE;	
		else if (STR_EQUAL(val,"orbit"))
			init_mode = SCENE_GRAPH_USER_MODE_ORBIT;	
		else if (STR_EQUAL(val,"passive"))
			init_mode = SCENE_GRAPH_USER_MODE_PASSIVE;	

		xmlFree(val);
	}

	TransformNode3D::parse(pXMLNode);
}

void User3D::app()
{
	if (reseting)
	{
		reseting = false;
		prev_time = world->getTime();
		return;
	}
	
	if (!isEnabled())
		return;
	
	TransformNode3D::app();
	// motion
	float movx, movz, movy, rotx, roty;
	if (!controller)
	{
		movx=movy=movz=roty=rotx=0.0f;
	}
	else
	{
		movx = controller->getNormalizedAxis(1);
		movy = controller->getNormalizedAxis(4);
		movz = controller->getNormalizedAxis(0);
		roty = controller->getNormalizedAxis(2);
		rotx = controller->getNormalizedAxis(3);
	}
	// EAZD_TRACE ("User3D::app(): mov(x,z) = (" << movx << ',' << movz << ") rot(y,x) = (" << roty << ',' << rotx << ") mov(x,z) = (" << controller->getRawAxis(1) << ',' << controller->getRawAxis(0) << ") rot(y,x) = (" << controller->getRawAxis(2) << ',' << controller->getRawAxis(3) << ')');
	next_position = position;

	switch(mode)
	{
	case SCENE_GRAPH_USER_MODE_NAVIGATE:
		if (movx!=0.0f)
			strafe(movx);
		if (movy!=0.0f)
			updown(movy);
		if (movz!=0.0f)
			move(movz);
		
		adjustPosition();
		
		if (roty!=0.0f)
			turn(roty);
		if (rotx!=0.0f)
			tilt(-rotx);
		break;

	case SCENE_GRAPH_USER_MODE_ORBIT:
		if (movx!=0.0f)
			strafe(movx);
		if (movy!=0.0f)
			updown(movy);
		if (movz!=0.0f)
			move(movz);
		
		adjustPosition();
		
		if (rotx!=0.0f)
			tilt(rotx);
		if (roty!=0.0f)
			orbit(roty);
		break;
	}

	Vector3D a = getTotalAcceleration();
	float dt = (float)(world->getDeltaTime());
	speed+=a*dt;
	speed-=(speed*friction);
	if (speed.length()<0.0001)
		speed = Vector3D(0,0,0);
	next_position=position+speed*dt;
	adjustPosition();
		
	// actuators (buttons)
	int i;
	num_actuators = controller->getNumButtons();
	for (i=0;i<num_actuators;i++)
		actuator[i] = controller->getButton(i);
	
	prev_time = world->getTime();

//	printf("position = %f %f %f\n", position.x, position.y, position.z);
//	printf("target = %f %f %f\n", target.x, target.y, target.z);
}

Vector3D User3D::getWorldPosition()
{
	Matrix4D m = getTransform();
	Vector3D vec = Vector3D(0,0,0)*m;
	
	return vec;
}

void User3D::processMessage(char * msg)
{
    char val[MAXSTRING];

	if (SUBSTR_EQUAL(msg,"control"))
	{
		sscanf (msg, "%*s%s", val);
		if (SUBSTR_EQUAL(val,"navigate"))
			mode = SCENE_GRAPH_USER_MODE_NAVIGATE;
		else if (SUBSTR_EQUAL(val,"orbit"))
			mode = SCENE_GRAPH_USER_MODE_ORBIT;
		else if (SUBSTR_EQUAL(val,"passive"))
			mode = SCENE_GRAPH_USER_MODE_PASSIVE;
	}
	else if (SUBSTR_EQUAL(msg,"freeroll"))
	{
		sscanf (msg, "%*s%s", val);
		parseBoolean(free_roll,val);
	}
    else if (SUBSTR_EQUAL(msg,"upright"))
	{
		sscanf (msg, "%*s%s", val);
		parseBoolean(upright,val);
	}
	else if (SUBSTR_EQUAL(msg,"speed"))
	{
		sscanf (msg, "%*s%s", val);
		parseFloat(linear_speed,val);
	}
	else if (SUBSTR_EQUAL(msg,"turn"))
	{
		sscanf (msg, "%*s%s", val);
		parseFloat(linear_speed,val);
	}
	else if (SUBSTR_EQUAL(msg,"position"))
	{
		parseVec3(position,skipParameterName(msg));
	}
	else if (SUBSTR_EQUAL(msg,"lookat"))
	{
		parseVec3(target,skipParameterName(msg));
	}
	else if (SUBSTR_EQUAL(msg,"input"))
	{
		sscanf (msg, "%*s%s", val);
		if (STR_EQUAL(val,"none"))
			controller = NULL;
		else
		{
			Node3D * nd = NULL;
			nd = world->getNodeByName(val);
			if (nd && typeid(*nd)==typeid(Input3D))
			    controller = reinterpret_cast<Input3D*>(nd);
		}
	}
	else
		Group3D::processMessage(msg);
}

void User3D::draw()
{
}
