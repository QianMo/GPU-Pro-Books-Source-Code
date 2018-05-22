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

Spinner3D::Spinner3D()
{
	init_repeats = repeats = 0;
	init_period = period = 1.0;
}

Spinner3D::~Spinner3D()
{
}

void Spinner3D::app()
{
	timer.update();
	if (timer.isRunning())
	{
		double s = timer.getValue();
		angle = (float) fmod(init_angle+s*360.0,360.0);
		calcMatrix();
	}
	TransformNode3D::app();
}

void Spinner3D::init()
{
	repeats = init_repeats;
	period = init_period;
	angle = init_angle;
	timer.setRepeats(repeats);
	timer.setDuration(period);
	TransformNode3D::init();
}

void Spinner3D::reset()
{
	repeats = init_repeats;
	period = init_period;
	timer.setRepeats(repeats);
	timer.setDuration(period);
	TransformNode3D::reset();
}

void Spinner3D::parse(xmlNodePtr pXMLnode)
{
	char * val = NULL;
	
	val = (char *)xmlGetProp(pXMLnode, (xmlChar *)"period");
	if (val)
	{
		parseFloat(init_period, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLnode, (xmlChar *)"repeats");
	if (val)
	{
		parseInteger(init_repeats, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLnode, (xmlChar *)"axis");
	if (val)
	{
		parseVec3(init_axis, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLnode, (xmlChar *)"start_angle");
	if (val)
	{
		parseFloat(init_angle, val);
		xmlFree(val);
	}

	TransformNode3D::parse(pXMLnode);
}

void Spinner3D::processMessage(char * msg)
{
	if (SUBSTR_EQUAL(msg,"start"))
	{
		timer.start();
	}
	else if (SUBSTR_EQUAL(msg,"stop"))
	{
		timer.stop();
	}
	else if (SUBSTR_EQUAL(msg,"pause"))
	{
		timer.pause();
	}
	else if (SUBSTR_EQUAL(msg,"toggle"))
	{
		if (timer.isRunning()) 
			timer.pause();
		else
			timer.start();
	}
	else if (SUBSTR_EQUAL(msg,"repeats"))
	{
		parseInteger(repeats,skipParameterName(msg));
	}
	else if (SUBSTR_EQUAL(msg,"period"))
	{
		parseFloat(period,skipParameterName(msg));
	}
	else
		TransformNode3D::processMessage(msg);
}

