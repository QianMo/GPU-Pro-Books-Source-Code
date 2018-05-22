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

#ifndef _SCENE_GRAPH_SPINNER3D_
#define _SCENE_GRAPH_SPINNER3D_

#include <libxml/tree.h>

#include "cglib.h"
#include "SceneGraph_TransformationNode3D.h"
#include "SceneGraph_Timer3D.h"

class Spinner3D: public TransformNode3D
{
protected:
	Timer3D timer;
	int repeats, init_repeats;
	float period, init_period;
	static vector<char *> eventmap;

	virtual int getEventID (char * eventname) GET_EVENT_ID(eventname)
public:
	Spinner3D();
	~Spinner3D();
	virtual void app();
	virtual void init();
	virtual void reset();
	virtual void parse(xmlNodePtr pXMLnode);
	void setRotation(float theta, float ax, float ay, float az);
    void setTranslation(float ox, float oy, float oz);
	void setScale(float sx, float sy, float sz);
	GLfloat * getGLMatrix() {return matrix.a;}
	Matrix4D getMatrix() {return matrix;}
	static void registerEvent(char *evt) {eventmap.push_back(evt);}
	virtual void processMessage(char * msg);
};

#endif

