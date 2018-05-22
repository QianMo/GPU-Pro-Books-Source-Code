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

#include "SceneGraph.h"

#ifdef WIN32
	#pragma warning (disable : 4996)
#endif

Node3D::Node3D()
{
	name = (char *) malloc(8);
	sprintf(name,"none");
	parent = NULL;
	init_visible = init_active = true;
	culled = false;
	event_messages_incoming.clear();
	event_messages_outgoing.clear();
	render_mode = SCENE_GRAPH_RENDER_MODE_NORMAL;
	no_ao = false;
	debug = false;
}

BBox3D Node3D::getBBox()
{
	return BBox3D();
}
	

Node3D::Node3D(Node3D * parent_node)
{
    Node3D();
	parent = parent_node;
}

void Node3D::setName(const char * str)
{
	if (!str)
		return;
	if (name)
		free (name);
	name = STR_DUP(str);
}

void Node3D::parse(xmlNodePtr pXMLnode)
{
	char * val = NULL;
	
	val = (char *)xmlGetProp(pXMLnode, (xmlChar *)"debug");
	if (val)
	{
		parseBoolean(debug,val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLnode, (xmlChar *)"name");
	if (val)
	{
		setName(val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLnode, (xmlChar *)"visible");
	if (val)
	{
		parseBoolean(init_visible,val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLnode, (xmlChar *)"no_ao");
	if (val)
	{
		parseBoolean(no_ao,val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLnode, (xmlChar *)"active");
	if (val)
	{
		parseBoolean(init_active,val);
		xmlFree(val);
	}

	parseEventMessages(pXMLnode);
}

Node3D::~Node3D()
{
	free (name);
#if 0
	for (unsigned int i=0; i<event_messages_registered.size(); i++)
		delete event_messages_registered.at(i);
#endif
	event_messages_registered.clear();
#if 0
	while (!event_messages_incoming.empty())
	{
		free(event_messages_incoming.front());
		event_messages_incoming.pop_front();
	}
#endif
	event_messages_incoming.clear();
#if 0
	while (!event_messages_outgoing.empty())
	{
		delete event_messages_outgoing.front();
		event_messages_outgoing.pop_front();
	}
#endif
	event_messages_outgoing.clear();
	
}

void Node3D::init()
{
	visible = init_visible;
	active = init_active;
	// resolve all recipient node addresses
	for (unsigned int i=0; i<event_messages_registered.size(); i++)
	{
		EventMessage3D *evt = event_messages_registered[i];
		if (STR_EQUAL(evt->recipient_name,"%world"))
			event_messages_registered.at(i)->recipient = world;
		else
			event_messages_registered.at(i)->recipient = world->getNodeByName(evt->recipient_name);
	}
	event_messages_incoming.clear();
	event_messages_outgoing.clear();
}

void Node3D::reset()
{
	visible = init_visible;
	active = init_active;

	while (!event_messages_incoming.empty())
	{
		free(event_messages_incoming.front());
		event_messages_incoming.pop_front();
	}
	event_messages_incoming.clear();
	while (!event_messages_outgoing.empty())
	{
		delete event_messages_outgoing.front();
		event_messages_outgoing.pop_front();
	}
	event_messages_outgoing.clear();
}

void Node3D::preApp()
{
}

void Node3D::app()
{
}

void Node3D::postApp()
{
}

void Node3D::cull()
{
}

void Node3D::draw()
{
}

char * Node3D::getName()
{
	return STR_DUP(name);
}

Matrix4D Node3D::getTransform()
{
	if (parent)
		return parent->getTransform();
	else
		return Matrix4D::identity();
}

Vector3D Node3D::getWorldPosition()
{
	Matrix4D m = getTransform();
	return Vector3D(0,0,0)*m;
}

void Node3D::processMessage(char * msg)
{
	char val[MAXSTRING];

	if (STR_EQUAL(msg,"enable") || STR_EQUAL(msg,"activate"))
		active = true;
	else if (STR_EQUAL(msg,"disable") || STR_EQUAL(msg,"deactivate"))
		active = false;
	else if (STR_EQUAL(msg,"hide"))
		visible = false;
	else if (STR_EQUAL(msg,"show"))
		visible = true;
	else if (SUBSTR_EQUAL(msg,"visible"))
	{
		sscanf (msg, "%*s%s", val);
		parseBoolean(visible,val);
	}
	else if (SUBSTR_EQUAL(msg,"active"))
	{
		sscanf (msg, "%*s%s", val);
		parseBoolean(active,val);
	}
}

void Node3D::postMessage(char * msg)
{
	char * incomming = STR_DUP(msg);
	event_messages_incoming.push_back(incomming);
}

void Node3D::parseEventMessages(xmlNodePtr pXMLnode)
{
	xmlNodePtr cur;
	char * recipient = NULL;
	char * message = NULL;
	char * eventname = NULL;
	float delay;
	char * val = NULL;
	
	cur = pXMLnode->xmlChildrenNode;
	while (cur != NULL)
	{
	    if (STR_EQUAL((char*)cur->name,"eventmessage"))
		{
			recipient = (char *)xmlGetProp(cur, (xmlChar *)"recipient");
			message = (char *)xmlGetProp(cur, (xmlChar *)"message");
			eventname = (char *)xmlGetProp(cur, (xmlChar *)"event");
			delay = 0;
			val = (char *)xmlGetProp(cur, (xmlChar *)"delay");
			if (val!=NULL)
			{
				parseFloat(delay,val);
				xmlFree(val);
			}
			if (recipient!=NULL && message!=NULL && eventname!=NULL)
			{
				int id = getEventID(eventname);
				if (id>=0)
				{
					EventMessage3D *evt = new EventMessage3D();
					evt->event_name = STR_DUP(eventname);
					evt->message = STR_DUP(message);
					evt->notification_delay = delay;
					evt->recipient_name = STR_DUP(recipient);
					evt->event_id = id;
					event_messages_registered.push_back(evt);
				}
			}
			xmlFree(recipient);
			xmlFree(message);
			xmlFree(eventname);
		}
		cur = cur->next;
	}
}

void Node3D::processMessages()
{
	char * message;
	char messagecopy[128];
	while (!event_messages_incoming.empty())
	{
		message = event_messages_incoming.front();
		strncpy(messagecopy,message,127);
		event_messages_incoming.pop_front();
		SAFEFREE(message);
		processMessage(messagecopy);
	}
}

void Node3D::dispatchMessages()
{
	double curtime = world->getTime();
	
	// Check event triggers and add all corresponding event messages
	// to the output queue with a timestamp
	int i,j;
	for (i=0;i<(int)event_states.size();i++)
	{
		if (event_states.at(i)==true)
		{
			for (j=0;j<(int)event_messages_registered.size();j++)
			{
				EventMessage3D *reg_evt = event_messages_registered.at(j);
				if (reg_evt->event_id==i)
				{
					EventMessage3D *evt = new EventMessage3D();
					evt->event_id = reg_evt->event_id;
					evt->event_name = STR_DUP(reg_evt->event_name);
					//evt->event_time = reg_evt->event_time;
					evt->message = STR_DUP(reg_evt->message);
					evt->notification_delay = reg_evt->notification_delay;
					evt->recipient = reg_evt->recipient;
					evt->recipient_name = reg_evt->recipient_name;
					evt->event_time = curtime;
					event_messages_outgoing.push_back(evt);
				}
			}
		}
		event_states.at(i)=false;
	}

	// Check scheduled outgoing message time stamps and post
	// all appropriate (expired) message notifications.
	for (i=0;i<(int)event_messages_outgoing.size();i++)
	{
		EventMessage3D *cur_evt = event_messages_outgoing.front();
		// if event notification expired or is immediate
		if (cur_evt->event_time+cur_evt->notification_delay<=curtime)
		{
			// notify node and consume event
			if (cur_evt->recipient)
				cur_evt->recipient->postMessage(cur_evt->message);
			//delete cur_evt;
		}
		else
			event_messages_outgoing.push_back(cur_evt);
		event_messages_outgoing.pop_front();
	}
}

