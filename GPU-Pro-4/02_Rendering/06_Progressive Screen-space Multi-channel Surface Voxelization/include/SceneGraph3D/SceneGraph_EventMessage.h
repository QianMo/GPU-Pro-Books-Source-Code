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

#ifndef _SCENE_GRAPH_EVENTMESSAGE3D_
#define _SCENE_GRAPH_EVENTMESSAGE3D_

#include <stdlib.h>
#include <string.h>

class EventMessage3D
{
public:
	char * event_name;
	char * recipient_name;
	char * message;
	class Node3D * recipient;
	int event_id;
	double event_time;
	double notification_delay;
	EventMessage3D()
	{
		event_name = recipient_name = message = NULL;
		recipient = NULL;
		event_id = -1;
		event_time = 0.0;
		notification_delay = 0.0;
	}

	~EventMessage3D()
	{
		if (event_name)
			free (event_name);
		if (recipient_name)
			free (recipient_name);
		if (message)
			free (message);
	}
};

void SceneGraphRegisterEvents();

#define INIT_MESSAGE_MAP(classname) vector<char *> classname::eventmap;
#define REGISTER_EVENT(classname, evt) classname::registerEvent((char *) evt);

#define EVENT_OCCURED(eventname) \
{ \
	/*printf("Event occured: %s\n", eventname);*/ \
    if (event_states.size()<eventmap.size()) \
		event_states.assign(eventmap.size(),false); \
	for (int evti=0; evti<(int)eventmap.size();evti++) \
	{ \
		char * evt = eventmap.at(evti); \
		if (strcmp(evt,eventname)==0) \
			event_states[evti] = true; \
	} \
};

#define GET_EVENT_ID(eventname) \
{ \
	for (int evti=0; evti<(int)eventmap.size();evti++) \
	{ \
		char * evt = eventmap.at(evti);	\
		if (strcmp(evt,eventname)==0) \
			return evti; \
	} \
	return -1; \
}
#endif

