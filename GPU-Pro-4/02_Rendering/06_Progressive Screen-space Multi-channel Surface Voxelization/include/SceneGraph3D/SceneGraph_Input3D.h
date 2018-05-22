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

#ifndef _SCENE_GRAPH_INPUT3D_
#define _SCENE_GRAPH_INPUT3D_

#include <libxml/tree.h>

#include "SceneGraph_Node3D.h"
#include "InputDevices3D.h"

class Input3D: public Node3D
{
private:
	InputDevice3D *device;
	static vector<char *> eventmap;
	bool *prev_button_state;

	virtual int getEventID (char * eventname) GET_EVENT_ID(eventname)
public:
	Input3D();
	~Input3D();
	virtual void parse(xmlNodePtr pXMLnode);
	virtual void init();
	virtual void reset();
	virtual void app();
	float getRawAxis(int ax) {return (device)?device->getValue(ax):0.0f;}
	float getNormalizedAxis(int ax) {return (device)?device->getNormalizedValue(ax):0.0f;}
	bool getButton(int b) {return (device)?device->getButton(b):false;}
	int getNumButtons() {return (device)?device->getButtons():0;}
	int getNumAxes() {return (device)?device->getAxes():0;}
	static void registerEvent(char *evt) {eventmap.push_back(evt);}
	virtual void processMessage(char * msg);
};

#endif

