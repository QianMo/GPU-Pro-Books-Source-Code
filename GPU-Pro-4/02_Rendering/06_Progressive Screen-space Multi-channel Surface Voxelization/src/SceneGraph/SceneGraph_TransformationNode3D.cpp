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

#include <typeinfo>

#include "SceneGraph.h"

TransformNode3D::TransformNode3D()
{
	init_axis = axis = Vector3D(1,0,0);
	init_scale = scale = Vector3D(1,1,1);
	init_angle = angle = 0;
	init_offset = offset = Vector3D(0,0,0);
	matrix=Matrix4D::identity();
}

TransformNode3D::~TransformNode3D()
{
}

void TransformNode3D::app()
{
	Group3D::app();
}

void TransformNode3D::cull()
{
	Group3D::cull();
}

void TransformNode3D::draw()
{
	if (!isVisible())
		return;

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	if (typeid(DBL)==typeid(float))
		glMultMatrixf((float*)matrix.a);
	else
		glMultMatrixd((double*)matrix.a);
	for (unsigned int i=0; i<children.size();i++)
		children.at(i)->draw();
	glPopMatrix();
}

void TransformNode3D::calcMatrix()
{
	Matrix4D S, T, R;
	S.makeScale(scale);
	R.makeRotate(axis,angle);
	T.makeTranslate(offset);
	matrix = T*R*S;
	matrix.transpose();
}

Matrix4D TransformNode3D::getTransform()
{
	if (parent)
		return matrix * parent->getTransform();
	else
		return matrix;
}

void TransformNode3D::setRotation(float theta, float ax, float ay, float az)
{
	angle = theta;
	axis = Vector3D(ax,ay,az);
	calcMatrix();
}

void TransformNode3D::setTranslation(float ox, float oy, float oz)
{
	offset = Vector3D(ox,oy,oz);
	calcMatrix();
}

void TransformNode3D::setScale(float sx, float sy, float sz)
{
	scale = Vector3D(sx,sy,sz);
	calcMatrix();
}

void TransformNode3D::init()
{
	scale = init_scale;
	angle = init_angle;
	offset = init_offset;
	axis = init_axis;
	calcMatrix();
	Group3D::init();
	ForceEnabled3D::init(world);
}

void TransformNode3D::reset()
{
	scale = init_scale;
	angle = init_angle;
	offset = init_offset;
	axis = init_axis;
	calcMatrix();
	Group3D::reset();
	ForceEnabled3D::reset();
}

void TransformNode3D::parse(xmlNodePtr pXMLnode)
{
	char * val = NULL;
	
	val = (char *)xmlGetProp(pXMLnode, (xmlChar *)"rotation");
	if (val)
	{
		parseVec4(init_angle, init_axis, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLnode, (xmlChar *)"translation");
	if (val)
	{
		parseVec3(init_offset, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLnode, (xmlChar *)"scale");
	if (val)
	{
		parseVec3(init_scale, val);
		xmlFree(val);
	}

	Group3D::parse(pXMLnode);
	ForceEnabled3D::parse(pXMLnode);
}

void TransformNode3D::processMessage(char * msg)
{
	if (SUBSTR_EQUAL(msg,"translation"))
	{
		parseVec3(offset,skipParameterName(msg));
	}
	else if (SUBSTR_EQUAL(msg,"scale"))
	{
		parseVec3(scale,skipParameterName(msg));
	}
	else if (SUBSTR_EQUAL(msg,"rotation"))
	{
		parseVec4(angle,axis,skipParameterName(msg));
	}
	else
		Group3D::processMessage(msg);
	ForceEnabled3D::processMessage(msg);
}

#define _update_corners(p) \
{ \
	if (p.x<bmn.x) \
		bmn.x = p.x; \
	if (p.y<bmn.y) \
		bmn.y = p.y; \
	if (p.z<bmn.z) \
		bmn.z = p.z; \
	if (p.x>bmx.x) \
		bmx.x = p.x; \
	if (p.y>bmx.y) \
		bmx.y = p.y; \
	if (p.z>bmx.z) \
		bmx.z = p.z; \
}

BBox3D TransformNode3D::getBBox()
{
	if (children.size()==0)
		return BBox3D();
	
	BBox3D tmp, box;
	Vector3D mn, mx, bmn, bmx, p;
	Matrix4D mat = Matrix4D(matrix.a);
	mat.transpose();
	bmn=Vector3D(100000.0f,100000.0f,100000.0f);
	bmx=Vector3D(-100000.0f,-100000.0f,-100000.0f);
	
	for (unsigned int i=0; i<children.size();i++)
	{
		tmp = children.at(i)->getBBox();
		mn =  tmp.getMin();
		mx =  tmp.getMax();

		p = Vector3D(mn.x,mn.y,mn.z);
		p.xformPt(mat);
		_update_corners(p);

		p = Vector3D(mn.x,mn.y,mx.z);
		p.xformPt(mat);
		_update_corners(p);

		p = Vector3D(mn.x,mx.y,mn.z);
		p.xformPt(mat);
		_update_corners(p);

		p = Vector3D(mn.x,mx.y,mx.z);
		p.xformPt(mat);
		_update_corners(p);

		p = Vector3D(mx.x,mn.y,mn.z);
		p.xformPt(mat);
		_update_corners(p);

		p = Vector3D(mx.x,mn.y,mx.z);
		p.xformPt(mat);
		_update_corners(p);

		p = Vector3D(mx.x,mx.y,mn.z);
		p.xformPt(mat);
		_update_corners(p);

		p = Vector3D(mx.x,mx.y,mx.z);
		p.xformPt(mat);
		_update_corners(p);
	}
	box.set(bmn,bmx);

	return box;
}

#undef _update_corners
