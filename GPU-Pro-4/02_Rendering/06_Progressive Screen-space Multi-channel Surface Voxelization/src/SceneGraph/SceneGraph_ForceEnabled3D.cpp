#include "SceneGraph.h"
#include <string.h>

void ForceEnabled3D::parse(xmlNodePtr pXMLnode)
{
	char * val = NULL;

	val = (char *)xmlGetProp(pXMLnode, (xmlChar *)"addforces");
	if (val)
	{
		char * tok = strtok(val," ,\t");
		while (tok)
		{
			force_names.push_back(strdup(tok));
			tok = strtok(NULL," ,\t");
		}
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLnode, (xmlChar *)"mass");
	if (val)
	{
		parseFloat(mass,val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLnode, (xmlChar *)"friction");
	if (val)
	{
		parseFloat(init_friction,val);
		xmlFree(val);
	}
}

ForceEnabled3D::ForceEnabled3D() 
{
	mass=1.0f;
	speed = Vector3D(0,0,0);
	friction = init_friction = 0.1f;
}

ForceEnabled3D::~ForceEnabled3D() 
{
	for (unsigned int i=0; i<force_names.size(); i++)
		SAFEFREE (force_names[i]);
	force_names.clear();
	forces.clear();
	init_forces.clear();
}

void ForceEnabled3D::init(World3D * w)
{
	global_world = w;

	if (!w)
		return;

	unsigned int i;
	for (i=0; i<force_names.size(); i++)
	{
		Force3D *f = dynamic_cast<Force3D*>(global_world->getNodeByName(force_names[i]));
		if (f)
			init_forces.push_back(f);
	}

	for (i=0; i<init_forces.size(); i++)
	{
		forces.push_back(init_forces[i]);
	}

	friction = init_friction;
}

void ForceEnabled3D::reset()
{
	forces.clear();
	for (unsigned int i=0; i<init_forces.size(); i++)
	{
		forces.push_back(init_forces[i]);
	}
	speed = Vector3D(0,0,0);
	friction = init_friction;
}

void ForceEnabled3D::processMessage(char * msg)
{
	if (!global_world)
		return;

	char val[MAXSTRING];

	if (SUBSTR_EQUAL(msg,"stop"))
	{
		sscanf (msg, "%*s%s", val);
		parseFloat(friction,val);
	}
	else if (SUBSTR_EQUAL(msg,"friction"))
	{
		speed = Vector3D(0,0,0);
	}
	else if (SUBSTR_EQUAL(msg,"addforces"))
	{
		sscanf (msg, "%*s%s", val);
		char * tok = strtok(val," ,\t");
		while (tok)
		{
			Force3D *f = dynamic_cast<Force3D*>(global_world->getNodeByName(tok));
			if (f)
				forces.push_back(f);
			tok = strtok(NULL," ,\t");
		}
	}
	else if (SUBSTR_EQUAL(msg,"removeforces"))
	{
		sscanf (msg, "%*s%s", val);
		char * tok = strtok(val," ,\t");
		while (tok)
		{
			Force3D *f = dynamic_cast<Force3D*>(global_world->getNodeByName(tok));
			if (f)
			{
				vector<Force3D*>::iterator iter = forces.begin();
				for (; iter != forces.end(); iter++)
			    {
					if ((*iter)==f)
					{
						forces.erase(iter);
						break;
					}
				}
			}
			tok = strtok(NULL," ,\t");
		}
	}
	
}

Vector3D ForceEnabled3D::getTotalAcceleration()
{
	Vector3D a=Vector3D(0.0f,0.0f,0.0f);
	for (unsigned int i=0; i<forces.size(); i++)
		a+=forces[i]->getAcceleration(this);
	return a;
}

Vector3D ForceEnabled3D::getTotalForce()
{
	Vector3D f=Vector3D(0.0f,0.0f,0.0f);
	for (unsigned int i=0; i<forces.size(); i++)
		f+=forces[i]->getForce(this);
	return f;
}
	