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

#ifndef _SCENE_GRAPH_TRANSFORMATION3D_
#define _SCENE_GRAPH_TRANSFORMATION3D_

#include <libxml/tree.h>

#include "cglib.h"
#include "SceneGraph_Group3D.h"
#include "SceneGraph_ForceEnabled3D.h"

class TransformNode3D: public Group3D, public ForceEnabled3D
{
protected:
	float angle, init_angle;
	Vector3D axis, init_axis;
	Vector3D scale, init_scale;
	Vector3D offset, init_offset;
	Matrix4D matrix;
	virtual void calcMatrix();

	static vector<char *> eventmap;
	virtual int getEventID (char * eventname) GET_EVENT_ID(eventname)

public:
	TransformNode3D();
	~TransformNode3D();
	virtual void parse(xmlNodePtr pXMLnode);
	virtual void init();
	virtual void reset();
	virtual void app();
	virtual void cull();
	virtual void draw();
	virtual Matrix4D getTransform();
	virtual BBox3D getBBox();
	void setRotation(float theta, float ax, float ay, float az);
	void setTranslation(float ox, float oy, float oz);
	void setScale(float sx, float sy, float sz);
	GLfloat * getGLMatrix() {return matrix.a;}
	Matrix4D getMatrix() {return matrix;}

	static void registerEvent(char *evt) {eventmap.push_back(evt);}
	virtual void processMessage(char * msg);
};

#endif

