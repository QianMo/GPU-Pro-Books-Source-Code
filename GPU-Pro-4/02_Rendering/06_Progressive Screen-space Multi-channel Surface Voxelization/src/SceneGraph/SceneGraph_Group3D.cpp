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

#ifndef WIN32
	#include <cxxabi.h>     // for abi::__cxa_demangle()
#endif

#include "SceneGraph.h"

void Group3D::addChild(Node3D *nd)
{
	children.push_back(nd);
}

void Group3D::setRenderMode(int mode)
{
    render_mode = mode;
	vector<Node3D*>::iterator iter;
	for ( iter = children.begin(); iter!=children.end() ; iter++ )
		(*iter)->setRenderMode(mode);
}

void Group3D::removeChild(Node3D *nd)
{
	vector<Node3D*>::iterator iter;
	for ( iter = children.begin(); iter!=children.end() ; iter++ )
		if (nd==*iter)
		{
			children.erase(iter);
			break;
		}
}

void Group3D::removeChild(int index)
{
	children.erase(children.begin()+index);
}

void Group3D::parse(xmlNodePtr pXMLnode)
{
	xmlNodePtr cur;
	Node3D * sgnode = NULL;
	char * val = NULL;

	cur = pXMLnode->xmlChildrenNode;
	while (cur != NULL)
	{
	    sgnode = buildNode((char*)cur->name);
		if (sgnode)
		{
			sgnode->setParent(this);
			sgnode->setWorld(world);
			children.push_back(sgnode);
			sgnode->parse(cur);
		}
		cur = cur->next;
	}
	
	Node3D::parse(pXMLnode);
}

Group3D::Group3D()
{
	children.clear();
}

Group3D::~Group3D()
{
	for (unsigned int i=0; i<children.size();i++)
		delete children.at(i);
	children.clear();
}

void Group3D::preApp()
{
	Node3D::preApp();

	if (isEnabled())
	for (unsigned int i=0; i<children.size();i++)
		children.at(i)->preApp();
}

void Group3D::app()
{
	Node3D::app();

	if (isEnabled())
	{
		for (unsigned int i=0; i<children.size();i++)
			children.at(i)->app();
	}
}

void Group3D::postApp()
{
	Node3D::postApp();

	if (isEnabled())
	for (unsigned int i=0; i<children.size();i++)
		children.at(i)->postApp();
}

void Group3D::cull()
{
	if (!isVisible())
		return;
	for (unsigned int i=0; i<children.size();i++)
		children.at(i)->cull();
}

void Group3D::draw()
{
	if (!isVisible())
		return;
	for (unsigned int i=0; i<children.size();i++)
		children.at(i)->draw();
}

void Group3D::init()
{
	for (unsigned int i=0; i<children.size();i++)
		children.at(i)->init();
	Node3D::init();
}

void Group3D::reset()
{
	for (unsigned int i=0; i<children.size();i++)
		children.at(i)->reset();
	Node3D::reset();
}

void Group3D::processMessages()
{
	Node3D::processMessages();
	for (unsigned int i=0; i<children.size();i++)
		children.at(i)->processMessages();
}

void Group3D::dispatchMessages()
{
	Node3D::dispatchMessages();
	for (unsigned int i=0; i<children.size();i++)
		children.at(i)->dispatchMessages();
}

Node3D *Group3D::getNodeByType(char *classname)
{
	Node3D *nd = NULL;
	Group3D *group = NULL;
	for (unsigned int i=0; i<children.size();i++)
	{
		nd = children.at(i);
#ifdef WIN32
		char * cname = (char *)(typeid(*nd)).name();
		if (STR_EQUAL(classname,cname+6))
#else
		int status = 0;
		char *cname = abi::__cxa_demangle((char *)(typeid(*nd)).name(), 0, 0, &status);
		if (!status && STR_EQUAL(classname,cname))
#endif
		{
			return nd;
		}
		group = dynamic_cast<Group3D*>(nd);
		if (group)
		{
			nd = group->getNodeByType(classname);
			if (nd)
				return nd;
		}
	}
	return NULL;

}

vector<Node3D*> Group3D::getAllNodesByType(char *classname)
{
	vector<Node3D*> allNodes;
	vector<Node3D*> tmpAllNodes;
	vector <Node3D*>::iterator iter;
  	Node3D *nd = NULL;
	Group3D *group = NULL;
	for (unsigned int i=0; i<children.size();i++)
	{
		nd = children.at(i);
#ifdef WIN32
		char * cname = (char *)(typeid(*nd)).name();
		if (STR_EQUAL(classname,cname+6))
#else
		int status = 0;
		char *cname = abi::__cxa_demangle((char *)(typeid(*nd)).name(), 0, 0, &status);
		if (!status && STR_EQUAL(classname,cname))
#endif
		{
			allNodes.push_back(nd);
		}
		group = dynamic_cast<Group3D*>(nd);
		if (group)
		{
			tmpAllNodes = group->getAllNodesByType(classname);
			if (tmpAllNodes.size()!=0)
			{
				for (iter = tmpAllNodes.begin(); iter != tmpAllNodes.end(); iter++)
				{
					allNodes.push_back(*iter);
				}
			}
		}
	}
	return allNodes;
}
	
Node3D *Group3D::getNodeByName(char *node_name)
{
	Node3D *nd = NULL;
	Group3D *group = NULL;
	if (!node_name)
		return NULL;
	for (unsigned int i=0; i<children.size();i++)
	{
		nd = children.at(i);
		char * name= nd->getName();
		if (STR_EQUAL(node_name,name))
		{
			free (name);
			return nd;
		}
		group = dynamic_cast<Group3D*>(nd);
		if (group)
		{
			nd = group->getNodeByName(node_name);
			if (nd)
				return nd;
		}
	}
	return NULL;
}

BBox3D Group3D::getBBox()
{
	BBox3D box,tmp;
	Vector3D mn, mx, bmn, bmx;
	if (children.size()==0)
		return BBox3D();
	tmp = children.at(0)->getBBox();
	bmn =  tmp.getMin();
	bmx =  tmp.getMax();
	for (unsigned int i=1; i<children.size();i++)
	{
		tmp = children.at(i)->getBBox();
		mn =  tmp.getMin();
		mx =  tmp.getMax();
		if (mn.x<bmn.x)
			bmn.x = mn.x;
		if (mn.y<bmn.y)
			bmn.y = mn.y;
		if (mn.z<bmn.z)
			bmn.z = mn.z;
		if (mx.x>bmx.x)
			bmx.x = mx.x;
		if (mx.y>bmx.y)
			bmx.y = mx.y;
		if (mx.z>bmx.z)
			bmx.z = mx.z;
	}
	box.set(bmn,bmx);
	return box;
}
