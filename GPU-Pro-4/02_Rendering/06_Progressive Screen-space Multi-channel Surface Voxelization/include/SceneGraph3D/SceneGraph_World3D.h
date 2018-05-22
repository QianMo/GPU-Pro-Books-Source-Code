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

#ifndef _SCENE_GRAPH_WORLD3D_
#define _SCENE_GRAPH_WORLD3D_

#include <libxml/parser.h>
#include <vector>

#include "cglib.h"
#include "SceneGraph_Group3D.h"

using namespace std;

class World3D: public Group3D
{
protected:
	double app_time, init_time, dt;
	Vector3D ambient, init_ambient;
	int err_code;
	bool debug;
	xmlDocPtr doc;
	xmlNodePtr xml_root;
	vector<char*> paths;
	class Camera3D * active_camera;
	char worldfile[128];

	static vector<char *> eventmap;
	virtual int getEventID (char * eventname) GET_EVENT_ID(eventname)

#ifndef _USING_DEFERRED_RENDERING_
	GenericRenderer * dr;
#else
	DeferredRenderer * dr;
	int init_shadows;
	int	init_hdr_mode, hdr_mode;
	Vector3D init_hdr_white, hdr_white;
	float init_hdr_key, hdr_key;
	vector<char*> gi_light_names;
	int ao;
	float aoRadius;
	int aoNumRays;
#endif

public:
	World3D();
	~World3D();
	virtual void parse(xmlNodePtr pXMLnode);
	virtual void init();
	virtual void reset();
	virtual void app();
	virtual void processMessage(char * msg);
	virtual void cull();
	virtual void draw();
	void clear(long mask);
	void load(char *scenefile);
	char * getFullPath(char * file);
	void check();
	bool getDebug() { return debug; }
	double getTime() { return app_time; }
	double getDeltaTime() { return dt; }
	Camera3D * getActiveCamera() { return active_camera; }
	void setActiveCamera(Camera3D * cam) { if (cam) active_camera = cam; }
	vector<char *>getPaths() { return paths; }
	char * getWorldFile () { return worldfile; }
#ifndef _USING_DEFERRED_RENDERING_
	void setRenderer(GenericRenderer *renderer) { dr = renderer; }
	GenericRenderer * getRenderer() { return dr; }
#else
	void setRenderer(DeferredRenderer *renderer) { dr = renderer; }
	DeferredRenderer * getRenderer() { return dr; }
#endif
	
	static void registerEvent(char *evt) { eventmap.push_back(evt); }
};

#endif

