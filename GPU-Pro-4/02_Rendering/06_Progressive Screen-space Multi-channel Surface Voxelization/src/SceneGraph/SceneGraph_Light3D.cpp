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

Light3D::Light3D()
{
	renderer = NULL;
	position = init_position = Vector3D(0,0,0);
	target = init_target = Vector3D(0,0,0);
	color = init_color = Vector3D(1,1,1);
	range = init_range = 1000.0f;
	near_range = init_near_range = 10.0f;
	apperture = init_apperture = 90.0f;
	resolution = 512;
	shadows = init_shadows = false;
	attenuation = init_attenuation = false;
	size = init_size = 0.0f;
	ID = -1;
	skip_frames = 0;
	explicit_glow_radius = false;
	glow_radius = init_glow_radius = range/2.0f;
	glowing=false;
	conical=false;
}

void Light3D::init()
{
	position = init_position;
	target = init_target;
	color = init_color;
	range = init_range;
	apperture = init_apperture;
	shadows = init_shadows;
	attenuation = init_attenuation;
	size = init_size;
	renderer = world->getRenderer();
	ID = world->getRenderer()->createLight();
	world->getRenderer()->attachLightData(ID,this);
	world->getRenderer()->enableLight(ID,active);
	setPosition(init_position);
	setTarget(init_target);
	setColor(init_color.x,init_color.y,init_color.z);
	setRanges(init_near_range,init_range);
	setConeApperture(init_apperture);
	enableShadows(init_shadows);
	setAttenuation(init_attenuation);
	setShadowResolution(resolution);
	setSize(init_size);
	world->getRenderer()->getLight(ID)->setConical(conical);
	if (explicit_glow_radius)
		setGlowRadius(init_glow_radius);
	else
		setGlowRadius(range);
	world->getRenderer()->setLightSkipFrames(ID,skip_frames);
	Node3D::init();
}

void Light3D::reset()
{
	setPosition(init_position);
	setTarget(init_target);
	setColor(init_color.x,init_color.y,init_color.z);
	setRanges(init_near_range,init_range);
	setConeApperture(init_apperture);
	enableShadows(init_shadows);
	setAttenuation(init_attenuation);
	setSize(init_size);
	setGlowRadius(init_glow_radius);
	
	Node3D::init();
}

void Light3D::draw()
{
	if (active && glowing)
	{
		//glBlendFunc(GL_ONE,GL_ONE);
		glBlendEquation(GL_ADD);
#ifdef	_USING_DEFERRED_RENDERING_
		if (render_mode==SCENE_GRAPH_RENDER_MODE_PARTICLES)
		{
			glMatrixMode(GL_TEXTURE);
			glActiveTexture(GL_TEXTURE0_ARB);
			glDisable(GL_TEXTURE_2D);
			glPushMatrix();
			glLoadIdentity();

			glEnable(GL_COLOR_MATERIAL);
			glBegin(GL_POINTS);
			glColor4f(color.x,color.y,color.z,0.2f);
			glVertex4f(position.x,position.y,position.z,glow_radius);
			glEnd();
			glDisable(GL_COLOR_MATERIAL);
			glPopMatrix();
			glMatrixMode(GL_MODELVIEW);
			glEnable(GL_TEXTURE_2D);
		}
#endif
	}
	
	Node3D::draw();
}

void Light3D::app()
{
	Node3D::app();
}

void Light3D::cull()
{
	float f, dist;
	if (attenuation)
	{
		f = world->getActiveCamera()->getFar();
		dist = world->getActiveCamera()->getCOP().distance(getWorldPosition());
		if (f+range<dist)
			culled = true;
		else
			culled = false;
	}
	else
		culled = false;
	renderer->enableLight(ID,culled?false:visible&active);
	Node3D::cull();
}

Vector3D Light3D::getWorldPosition()
{
	Matrix4D m = getTransform();
	return position*m;
}

void Light3D::enable(bool en)
{
	active=en;
	renderer->enableLight(ID,active);
}

void Light3D::setSize(float s)
{
	size = s;
	renderer->setLightSize(ID,size);
}

void Light3D::enableShadows(bool en)
{
	shadows = en;
	renderer->enableLightShadows(ID,shadows);
}

void Light3D::setShadowResolution(int res)
{
	resolution = res;
	renderer->setShadowResolution(ID,resolution);
}

void Light3D::setConeApperture(float ap)
{
	apperture = ap;
	renderer->setLightCone(ID,apperture);
}

void Light3D::setRanges(float nr, float fr)
{
	range = fr;
	near_range = nr;
	if (near_range>=range)
		near_range=range-1.0f;
	renderer->setLightRanges(ID,near_range,range);

}

void Light3D::setAttenuation(bool attn)
{
	attenuation = attn;
	renderer->setLightAttenuation(ID,attenuation);
}

void Light3D::setColor(float r, float g, float b)
{
	color = Vector3D(r,g,b);
	renderer->setLightColor(ID,r,g,b);
}

void Light3D::setTarget(Vector3D tgt)
{
	target = tgt;
	renderer->setLightTarget(ID,tgt.x, tgt.y, tgt.z);
}

void Light3D::setPosition(Vector3D pos)
{
	position = pos;
	renderer->setLightPosition(ID, pos.x, pos.y, pos.z);
}

void Light3D::parse(xmlNodePtr pXMLNode)
{
	char * val = NULL;
	
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"shadows");
	if (val)
	{
		parseBoolean(init_shadows, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"conical");
	if (val)
	{
		parseBoolean(conical, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"glow");
	if (val)
	{
		parseBoolean(glowing, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"color");
	if (val)
	{
		parseVec3(init_color, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"attenuation");
	if (val)
	{
		parseBoolean(init_attenuation, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"range");
	if (val)
	{
		parseFloat(init_range, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"glow_radius");
	if (val)
	{
		parseFloat(init_glow_radius, val);
		explicit_glow_radius = true;
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"near_range");
	if (val)
	{
		parseFloat(init_near_range, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"position");
	if (val)
	{
		parseVec3(init_position, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"target");
	if (val)
	{
		parseVec3(init_target, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"resolution");
	if (val)
	{
		parseInteger(resolution, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"skipframes");
	if (val)
	{
		parseInteger(skip_frames, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"apperture");
	if (val)
	{
		parseFloat(init_apperture, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"size");
	if (val)
	{
		parseFloat(init_size, val);
		xmlFree(val);
	}

	Node3D::parse(pXMLNode);
}

void Light3D::processMessage(char * msg)
{
	char val[MAXSTRING];
	
	Vector3D vec;
	if (SUBSTR_EQUAL(msg,"target"))
	{
		parseVec3(vec,skipParameterName(msg));
		setTarget(vec);
	}
	else if (SUBSTR_EQUAL(msg,"position"))
	{
		parseVec3(vec,skipParameterName(msg));
		setPosition(vec);
	}
	else if (SUBSTR_EQUAL(msg,"color"))
	{
		parseVec3(vec,skipParameterName(msg));
		setColor(vec.x,vec.y,vec.z);
	}
	else if (SUBSTR_EQUAL(msg,"range"))
	{
		sscanf (msg, "%*s%s", val);
		parseFloat(range,val);
		setRanges(near_range, range);
	}
	else if (SUBSTR_EQUAL(msg,"glow_radius"))
	{
		sscanf (msg, "%*s%s", val);
		parseFloat(glow_radius,val);
	}
	else if (SUBSTR_EQUAL(msg,"near_range"))
	{
		sscanf (msg, "%*s%s", val);
		parseFloat(near_range,val);
		setRanges(near_range, range);
	}
	else if (SUBSTR_EQUAL(msg,"shadows"))
	{
		sscanf (msg, "%*s%s", val);
		parseBoolean(shadows,val);
		enableShadows(shadows);
	}
	else if (SUBSTR_EQUAL(msg,"attenuation"))
	{
		sscanf (msg, "%*s%s", val);
		parseBoolean(attenuation,val);
		setAttenuation(attenuation);
	}
	else if (SUBSTR_EQUAL(msg,"apperture"))
	{
		sscanf (msg, "%*s%s", val);
		parseFloat(apperture,val);
		setConeApperture(apperture);
	}
	else if (SUBSTR_EQUAL(msg,"size"))
	{
		sscanf (msg, "%*s%s", val);
		parseFloat(size,val);
		setSize(size);
	}
	else if (SUBSTR_EQUAL(msg,"skipframes"))
	{
		sscanf (msg, "%*s%s", val);
		parseInteger(skip_frames,val);
		world->getRenderer()->setLightSkipFrames(ID,skip_frames);
	}
	else
		Node3D::processMessage(msg);
}

