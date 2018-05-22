
#include <stdlib.h>
#include <limits.h>
#include <float.h>

#include "Ray3D.h"
#include "Primitive3D.h"

unsigned long RaySignature3D::_ray_counter_h = 0;
unsigned long RaySignature3D::_ray_counter_l = 0;

RaySignature3D::RaySignature3D(unsigned int ti)
{
	if (_ray_counter_l==ULONG_MAX)
	{
		_ray_counter_l = 0;
		if (_ray_counter_h==ULONG_MAX)
			_ray_counter_h = 0;
		else
			_ray_counter_h++;
	}
	else
		_ray_counter_l++;
		
	signature_l = _ray_counter_l;
	signature_h = _ray_counter_h;
	threadID = ti;
}

IsectPoint3D::IsectPoint3D()
{
	t = FLT_MAX;
	inside = false;
	p_isect = Vector3D(0,0,0);
	n_isect = Vector3D(0,0,1);
	pData = NULL;
	primitive = NULL;
}

Ray3D::Ray3D(Vector3D start, Vector3D direction, unsigned int thread_id)
{
	sig = RaySignature3D(thread_id);
	t = FLT_MAX;
	strength = 1;
	depth = 0;
	hit = false;
	p_isect = Vector3D(0,0,0);
	n_isect = Vector3D(0,0,1);
	primitive = NULL;
	origin = start;
	dir = direction;
	inside = false;
}

Ray3D::Ray3D(void)
{
	sig = RaySignature3D(0);
	t = FLT_MAX;
	strength = 1;
	hit = false;
	inside = false;
	p_isect = Vector3D(0,0,0);
	n_isect = Vector3D(0,0,1);
	primitive = NULL;
	pData = NULL;
	depth = 0;
	origin = Vector3D(0,0,0);
	dir = Vector3D(0,0,1);
}

Ray3D::~Ray3D(void)
{
}

void Ray3D::xform(Matrix4D m)
{
	origin.xformPt(m);
	dir.xformVec(m);
	dir.normalize();
	Vector3D v = Vector3D(0,0,t);
	v.xformPt(m);
	Vector3D base = Vector3D(0,0,0);
	base.xformPt(m);
	t = base.distance(v);
	n_isect.xformVec(m);
	n_isect.normalize();
	p_isect.xformPt(m);
}

bool RaySignature3D::operator==(RaySignature3D other)
{
	if ( signature_h==other.signature_h &&
		 signature_l==other.signature_l &&
		 threadID==other.threadID )
		 return true;
	else
		return false;

}

Ray3D Ray3D::operator= (Ray3D r)
{
	origin = r.origin;
	dir = r.dir;
	hit = r.hit;
	p_isect = r.p_isect;
	n_isect = r.n_isect;
	pData = r.pData;
	primitive = r.primitive;
	strength = r.strength;
	t = r.t;
	inside = r.inside;
	depth = r.depth;
	return *this;
}