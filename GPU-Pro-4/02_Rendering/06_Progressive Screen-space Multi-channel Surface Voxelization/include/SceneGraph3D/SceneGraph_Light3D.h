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

#ifndef _SCENE_GRAPH_LIGHT3D_
#define _SCENE_GRAPH_LIGHT3D_

#include <libxml/tree.h>

#include "cglib.h"
#include "SceneGraph_Node3D.h"

class Light3D: public Node3D
{
protected:
#ifdef	_USING_DEFERRED_RENDERING_
	class DeferredRenderer * renderer;
#else
	class GenericRenderer * renderer;
#endif
	Vector3D position, init_position;
	Vector3D target, init_target;
	Vector3D color, init_color;
	float range, init_range;
	float near_range, init_near_range;
	float apperture, init_apperture;
	int resolution;
	bool shadows, init_shadows;
	bool attenuation, init_attenuation;
	float size, init_size;
	int ID;
	int skip_frames;
	bool glowing;
	bool conical;
	float glow_radius, init_glow_radius;
	bool explicit_glow_radius;
	static vector<char *> eventmap;

	virtual int getEventID (char * eventname) GET_EVENT_ID(eventname)
public:
	Light3D();
	virtual void parse(xmlNodePtr pXMLnode);
	virtual void init();
	virtual void reset();
	virtual void app();
	virtual void cull();
	virtual void draw();
	void enable(bool en);
	void setSize(float);
	void enableShadows(bool en);
	void setShadowResolution(int res);
	void setConeApperture(float ap);
	void setRanges(float nr, float fr);
	void setAttenuation(bool attn);
	void setColor(float r, float g, float b);
	void setTarget(Vector3D tgt);
	void setPosition(Vector3D pos);
	bool isGlowing(){return glowing;}
	void setGlowRadius(float rad) {rad>0?glow_radius=rad:glow_radius=0.0f; }
	virtual Vector3D getWorldPosition();
	int getID() {return ID;}
	static void registerEvent(char *evt) {eventmap.push_back(evt);}
	virtual void processMessage(char * msg);
};

#endif

