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

Camera3D::Camera3D()
{
	init_attached_to = attached_to = NULL;
	apperture = init_apperture = 90;
	cnear = init_near = 1.0f;
	cfar = init_far = 1000.0f;
	focal_distance = init_focal_distance = 200.0f;
	focal_range = init_focal_range = 200.0f;
	to_follow_name = NULL;
	cop = Vector3D(0,0,0);
}

Camera3D::~Camera3D()
{
	if (to_follow_name)
		free (to_follow_name);
}

void Camera3D::init()
{
	Node3D::init();
	apperture = init_apperture;
	cnear = init_near;
	cfar = init_far;
	focal_distance = init_focal_distance;
	focal_range = init_focal_range;
	world->getRenderer()->setDOF(focal_distance,focal_range);
	if (to_follow_name)
		attached_to = init_attached_to = world->getNodeByName(to_follow_name);
	if (attached_to == NULL)
	{	
		EAZD_TRACE ("Camera3D::init() : ERROR - Can not attach to user \"" << to_follow_name << "\".");
	}
}

void Camera3D::reset()
{
	Node3D::reset();
	apperture = init_apperture;
	cnear = init_near;
	cfar = init_far;
	focal_distance = init_focal_distance;
	focal_range = init_focal_range;
	attached_to = init_attached_to;
}

void Camera3D::app()
{
	Node3D::app();
}

void Camera3D::attachTo(Node3D *nd)
{
	if (nd)
		attached_to = nd;
}

void Camera3D::attachTo(char * nname)
{
	if (nname)
		attached_to = world->getNodeByName(nname);
}

void Camera3D::setupViewProjection(int width, int height)
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
    //gluPerspective(apperture,width/(float)height,cnear,cfar);
	float h=cnear*tan(PI*apperture/360.0f);
	float w=h*width/(float)height;
	glFrustum(-w,w,-h,h,cnear,cfar);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

    // setup viewing transformation
	
	if (attached_to) // if camera is attached to a node
	{
		User3D *user = dynamic_cast<User3D*>(attached_to);
		if (user) // if camera attached to user node
		{
			Matrix4D m = attached_to->getParent()->getTransform();
			m.transpose();
			Vector3D tgt = user->getLookAt();
			Vector3D pos = user->getPosition();
			Vector3D up  = user->getUp();
			tgt.xformPt(m);
			pos.xformPt(m);
			up.xformVec(m);
			gluLookAt(pos[0],pos[1],pos[2],tgt[0],tgt[1],tgt[2],up[0],up[1],up[2]);
			cop = pos;
			dir = tgt-pos;
		}
		else // if camera attached to a regular (non-user) node
		{
			Matrix4D m = attached_to->getTransform();
			Matrix4D mt = m;
			mt.transpose();
			cop = mt*Vector3D(0,0,0);
			m.invert();
			if (typeid(DBL)==typeid(float))
				glMultMatrixf((float*)m.a);
			else
				glMultMatrixd((double*)m.a);
			dir = Vector3D(0,0,-1);
			dir.xformVec(mt);
		}
	}
	else // if camera is free-standing
	{
		Matrix4D fm = parent->getTransform();
		Matrix4D fmt = fm;
		fmt.transpose();
		cop = fmt*Vector3D(0,0,0);
		fm.invert();

		if (typeid(DBL)==typeid(float))
			glMultMatrixf((float*)fm.a);
		else
			glMultMatrixd((double*)fm.a);
		dir = Vector3D(0,0,-1);
		dir.xformVec(fmt);
	}
	dir.normalize();
}

void Camera3D::parse(xmlNodePtr pXMLNode)
{
	char * val = NULL;
	
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"apperture");
	if (val)
	{
		parseFloat(init_apperture, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"focal_distance");
	if (val)
	{
		parseFloat(init_focal_distance, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"focal_range");
	if (val)
	{
		parseFloat(init_focal_range, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"near");
	if (val)
	{
		parseFloat(init_near, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"far");
	if (val)
	{
		parseFloat(init_far, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"primary");
	if (val)
	{
		bool primary;
		parseBoolean(primary, val);
		if (primary)
			world->setActiveCamera(this);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"follow");
	if (val)
	{
        to_follow_name = STR_DUP(val);
        xmlFree(val);
	}

	Node3D::parse(pXMLNode);
}

void Camera3D::processMessage(char * msg)
{
	char val[MAXSTRING];

	if (SUBSTR_EQUAL(msg,"follow"))
	{
		sscanf (msg, "%*s%s", val);
		Node3D * nd = world->getNodeByName(val);
		if (nd)
		{
			attached_to = nd;
			EVENT_OCCURED("follow");
		}
	}
	else if (SUBSTR_EQUAL(msg,"apperture"))
	{
		sscanf (msg, "%*s%s", val);
		parseFloat(apperture,val);
	}
	else if (SUBSTR_EQUAL(msg,"focal_distance"))
	{
		sscanf (msg, "%*s%s", val);
		parseFloat(focal_distance,val);
	}
	else if (SUBSTR_EQUAL(msg,"focal_range"))
	{
		sscanf (msg, "%*s%s", val);
		parseFloat(focal_range,val);
	}
	else if (SUBSTR_EQUAL(msg,"near"))
	{
		sscanf (msg, "%*s%s", val);
		parseFloat(cnear,val);
	}
	else if (SUBSTR_EQUAL(msg,"far"))
	{
		sscanf (msg, "%*s%s", val);
		parseFloat(cfar,val);
	}
	else
		Node3D::processMessage(msg);
}

