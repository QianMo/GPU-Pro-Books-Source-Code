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

LinearMotion3D::LinearMotion3D()
{
	repeats_init = repeats = 0.0f;
	duration_init = duration = 0.0f;
	begin_init = begin = Vector3D(0, 0, 0);
	end_init = end = Vector3D(0, 0, 0);
	interp = 0;
	mode = SCENE_GRAPH_MOTION_ONEWAY;
}

LinearMotion3D::~LinearMotion3D() {}

void LinearMotion3D::app()
{
	timer.update();
	
	if (timer.isRunning())
	{
		interp = timer.getValue();
		if(mode == SCENE_GRAPH_MOTION_ONEWAY)
		{
			offset = begin*(1-interp) + end*interp;
		}
		if(mode == SCENE_GRAPH_MOTION_PINGPONG)
		{
			if (interp<=0.5f)
				offset = begin*(1.0f-2.0f*interp) + end*interp*2.0f;
			else
				offset = end*(2.0f-2.0f*interp) + begin*(interp*2.0f-1.0f);
		}

		TransformNode3D::calcMatrix();
	}
	TransformNode3D::app();
}

void LinearMotion3D::init()
{
	begin = begin_init;
	end = end_init;
	duration = duration_init;
	repeats = repeats_init;
	timer.setRepeats(repeats_init);
	timer.setDuration(duration_init);
	timer.setValueRange(0.0, 1.0);
	interp = (float) timer.getValue();
	TransformNode3D::offset = begin*(1.0f-interp) + end*interp;
	
	TransformNode3D::init();
}		

void LinearMotion3D::reset()
{
	begin = begin_init;
	end = end_init;
	duration = duration_init;
	repeats = repeats_init;
	timer.stop();
	timer.setRepeats(repeats_init);
	timer.setDuration(duration_init);
	interp = (float) timer.getValue();
	TransformNode3D::offset = begin*(1.0f-interp) + end*interp;
	TransformNode3D::reset();
}

void LinearMotion3D::parse(xmlNodePtr pXMLnode)
{
	char * val = NULL;
	
	val = (char *)xmlGetProp(pXMLnode, (xmlChar *)"begin");
	if (val) {
		parseVec3(begin_init, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLnode, (xmlChar *)"end");
	if (val) {
		parseVec3(end_init, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLnode, (xmlChar *)"repeats");
	if (val) {
		parseFloat(repeats_init, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLnode, (xmlChar *)"duration");
	if (val) {
		parseFloat(duration_init, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLnode, (xmlChar *)"mode");
	if (val) {
		if (STR_EQUAL(val,"oneway"))
			mode = SCENE_GRAPH_MOTION_ONEWAY;	
		else if (STR_EQUAL(val,"pingpong"))
			mode = SCENE_GRAPH_MOTION_PINGPONG;	

		xmlFree(val);
	}

	TransformNode3D::parse(pXMLnode);
}

void LinearMotion3D::processMessage(char * msg)
{
	char val[MAXSTRING];

	if (SUBSTR_EQUAL(msg,"begin")) {
		parseVec3(begin,skipParameterName(msg));
	}
	else if (SUBSTR_EQUAL(msg,"end")) {
		parseVec3(end,skipParameterName(msg));
	}
	else if (SUBSTR_EQUAL(msg,"start")) {
		timer.start();
	}
	else if (SUBSTR_EQUAL(msg,"stop")) {
		timer.stop();
	}
	else if (SUBSTR_EQUAL(msg,"pause")) {
		timer.pause();
	}
	else if (SUBSTR_EQUAL(msg,"toggle"))
	{
		if (timer.isRunning()) 
			timer.pause();
		else
			timer.start();
	}
	else if (SUBSTR_EQUAL(msg,"repeats")) {
		parseFloat(repeats,skipParameterName(msg));
	}
	else if (SUBSTR_EQUAL(msg,"duration")) {
		parseFloat(duration,skipParameterName(msg));
	}
	else if (SUBSTR_EQUAL(msg,"mode")) {
		if (SUBSTR_EQUAL(val,"oneway"))
			mode = SCENE_GRAPH_MOTION_ONEWAY;
		else if (SUBSTR_EQUAL(val,"pingpong"))
			mode = SCENE_GRAPH_MOTION_PINGPONG;
	}
	else
		TransformNode3D::processMessage(msg);
}
