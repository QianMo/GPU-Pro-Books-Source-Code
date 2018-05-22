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

#ifndef _SCENE_GRAPH_NODE3D_
#define _SCENE_GRAPH_NODE3D_

#include <libxml/tree.h>
#include <vector>
#include <list>
#include "SceneGraph_EventMessage.h"
#include "SceneGraph_Drawable3D.h"

#include "cglib.h"

using namespace std;

class Node3D : public Drawable3D
{
protected:
	Node3D *parent;
	class World3D *world;
	char *name;
	bool visible;
	bool active;
	bool culled;
	bool no_ao;
	bool debug;
	int render_mode;
	bool init_visible, init_active;
	static vector<char *> eventmap;
	list<char *>event_messages_incoming;
	vector<bool> event_states;
	vector<EventMessage3D*>event_messages_registered;
	list<EventMessage3D*>event_messages_outgoing;
		
	virtual int getEventID (char * eventname) GET_EVENT_ID(eventname)
	virtual void processMessage(char * msg);
	void parseEventMessages(xmlNodePtr pXMLnode);
public:
	Node3D();
	Node3D(Node3D * parent_node);
	virtual ~Node3D();
	virtual void parse(xmlNodePtr pXMLnode);
	virtual void init();
	virtual void reset();
	virtual void preApp();
	virtual void app();
	virtual void postApp();
	virtual void cull();
	virtual void draw();
	Node3D * getParent() {return parent;}
	class World3D * getWorld() {return world;}
	void setWorld(World3D *w) {world = w;}
	void setParent(Node3D *p) {parent = p;}
	char * getName();
	void setName(const char * str);
	virtual BBox3D getBBox();
	virtual Matrix4D getTransform();
	bool isEnabled() {return active;}
	bool isVisible() {return visible;}
	bool hasAO() {return !no_ao;}
	bool getDebug() {return debug;}
	virtual Vector3D getWorldPosition();
	static void registerEvent(char *evt) {eventmap.push_back(evt);}
	void postMessage(char * msg);
	virtual void processMessages();
	virtual void dispatchMessages();
	virtual void setRenderMode(int mode) {render_mode=mode;}
};

#endif

