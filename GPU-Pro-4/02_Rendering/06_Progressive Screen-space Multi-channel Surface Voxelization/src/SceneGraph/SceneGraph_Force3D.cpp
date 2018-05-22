
#include "SceneGraph.h"
#include <GL/glut.h>

Force3D::Force3D()
{
	dir = init_dir = transformed_dir = Vector3D(0.0f,-1.0f,0.0f);
	init_force = 10.0f;
	force = init_force;
	effective_radius = 0.0f;
	center = init_center = transformed_center = Vector3D(0.0f, 0.0f, 0.0f);
	init_fade_time = fade_time = 0.5f;
	timer.setDuration(init_fade_time);
	timer.setLooping(false);
	timer.setValueRange(0.0f,force);
	fading = false;
}

Force3D::~Force3D()
{
}

Vector3D Force3D::getForce(Vector3D pos)
{
	if (!active)
		return Vector3D(0,0,0);

	if (effective_radius>0.0f)
	{
		float dist = pos.distance(transformed_center);
		float contribution = dist>=effective_radius?0.0f:(effective_radius-dist)/effective_radius;
		return transformed_dir*contribution;
	}
	else
	{
		return transformed_dir*force;
	}
}

Vector3D Force3D::getForce(ForceEnabled3D *node)
{
	return getForce(node->getForcePoint());
}

Vector3D Force3D::getAcceleration(ForceEnabled3D *node)
{
	Vector3D f = getForce(node->getForcePoint());
	float m = node->getMass(); 
	return m>0?f*(1.0f/m):f;
}

void Force3D::app()
{
	if (!active)
		return;

	Node3D::app();
	timer.update();
	fading = timer.isRunning();
	if (fading)
	{
		force = (float)timer.getValue();
	}
	Matrix4D mat = getTransform();
	transformed_dir = dir;
	transformed_dir.xformVec(mat);
	transformed_center = center;
	transformed_center.xformPt(mat);
}

void Force3D::init()
{
	Node3D::init();
	force = init_force;
	fade_time = init_fade_time;
	timer.setDuration(init_fade_time);
	timer.setLooping(false);
	timer.setValueRange(0.0f,force);
	timer.stop();
	fading = false;
	center = init_center;
	dir = init_dir;
	center = init_center;
	dir = init_dir;
}

void Force3D::reset()
{
	Force3D::init();
	Node3D::reset();
}

void Force3D::parse(xmlNodePtr pXMLnode)
{
	char * val = NULL;

	val = (char *)xmlGetProp(pXMLnode, (xmlChar *)"value");
	if (val)
	{
		parseFloat(init_force,val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLnode, (xmlChar *)"direction");
	if (val)
	{
		parseVec3(init_dir,val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLnode, (xmlChar *)"fadetime");
	if (val)
	{
		parseFloat(init_fade_time,val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLnode, (xmlChar *)"range");
	if (val)
	{
		parseFloat(effective_radius,val);
		xmlFree(val);
	}
	Node3D::parse(pXMLnode);
}

void Force3D::processMessage(char * msg)
{
	char val[MAXSTRING];

	if (SUBSTR_EQUAL(msg,"fadeto"))
	{
		sscanf (msg, "%*s%s", val);
		float cur_force;
		if (fading)
			cur_force = (float)timer.getValue();
		else
			cur_force = force;
		parseFloat(force,val);
		timer.stop();
		timer.setValueRange(cur_force,force);
		timer.start();
	}
	else if (SUBSTR_EQUAL(msg,"direction"))
	{
		sscanf (msg, "%*s%s", val);
		parseVec3(dir,val);
	}
	else if (SUBSTR_EQUAL(msg,"center"))
	{
		sscanf (msg, "%*s%s", val);
		parseVec3(center,val);
	}
	else if (SUBSTR_EQUAL(msg,"fadetime"))
	{
		sscanf (msg, "%*s%s", val);
		parseFloat(fade_time,val);
	}
	else
		Node3D::processMessage(msg);
}

void Force3D::draw()
{
	if (getDebug ())
    {
		glColor4f(1.0,0.0,0.0,1.0);
		glPushMatrix();
		glTranslatef(center.x, center.y, center.z);
		glutSolidSphere(0.5f, 4, 4);
		if (effective_radius>0.0f)
			glutWireSphere(effective_radius,8,8);
		glBegin(GL_LINES);
		glVertex3f( center.x-dir.x*0.5, center.y-dir.y*0.5, center.z-dir.z*0.5);
		glVertex3f( center.x+dir.x*0.5, center.y+dir.y*0.5, center.z+dir.z*0.5);
		glColor4f(0.0,0.0,1.0,1.0);
		glVertex3f( center.x+dir.x*0.5, center.y+dir.y*0.5, center.z+dir.z*0.5);
		glVertex3f( center.x+dir.x*0.6, center.y+dir.y*0.6, center.z+dir.z*0.6);
		glEnd();
		glPopMatrix();
	}
}
