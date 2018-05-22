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

#ifndef _SCENE_GRAPH_GROUP3D_
#define _SCENE_GRAPH_GROUP3D_

#include <vector>
#include <libxml/tree.h>

#include "SceneGraph_Node3D.h"

using namespace std;

class Group3D: public Node3D
{
protected:
	static vector<char *> eventmap;

	virtual int getEventID (char * eventname) GET_EVENT_ID(eventname)
	
public:
	Group3D();
	~Group3D();
	virtual void parse(xmlNodePtr pXMLnode);
	virtual void init();
	virtual void reset();
	virtual void preApp();
	virtual void app();
	virtual void postApp();
	virtual void cull();
	virtual void draw();
	virtual BBox3D getBBox();
	void addChild(Node3D *nd);
	void removeChild(Node3D *nd);
	void removeChild(int index);
	Node3D *getNodeByName(char *node_name);
	Node3D *getNodeByType(char * class_name);
	vector<Node3D*> getAllNodesByType(char *classname);
	static void registerEvent(char *evt) {eventmap.push_back(evt);}
	virtual void processMessages();
	virtual void dispatchMessages();
	virtual void setRenderMode(int mode);

	vector<Node3D*> children;
};

#endif

