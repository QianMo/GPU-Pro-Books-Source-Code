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

#ifndef _SCENE_GRAPH_GEOMETRY3D_
#define _SCENE_GRAPH_GEOMETRY3D_

#include <libxml/tree.h>

#include "cglib.h"
#include "SceneGraph_Node3D.h"

class Geometry3D: public Node3D
{
protected:
	Mesh3D    * mesh;
	bool        dirty;
	int         err_code;
	
	static vector<char *> eventmap;
	virtual int getEventID (char * eventname) GET_EVENT_ID(eventname)
	
public:
	Geometry3D();
	~Geometry3D();
	virtual void parse(xmlNodePtr pXMLnode);
	virtual void app();
	virtual void cull();
	virtual void draw();
	virtual void init();
	virtual BBox3D getBBox();
	virtual BSphere3D getBSphere();
	virtual Vector3D getWorldPosition();
	int getErrorCode() {return err_code;}
	void load(char * file);
	static void registerEvent(char *evt) {eventmap.push_back(evt);}
	virtual void processMessage(char * msg);
	Mesh3D * getMesh () { return mesh; }
	inline void setDirty(bool d) { dirty = d; }
	inline bool getDirty(void) { return dirty; }
	virtual float * flattenRawTriangles(long *number);
};

#endif
