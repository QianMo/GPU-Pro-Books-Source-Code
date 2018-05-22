
#include "Triangle3D.h"

Triangle3D::Triangle3D(void)
{
	vertIdx[0] = vertIdx[1] = vertIdx[2] = -1;
	normIdx[0] = normIdx[1] = normIdx[2] = -1;
	texcIdx[0] = texcIdx[1] = texcIdx[2] = -1; 
}

Triangle3D::~Triangle3D(void)
{
}

bool Triangle3D::intersect(Ray3D &r, Vector3D * vertexlist, Vector3D * normallist)
{
	return FindIntersection(*this, r, vertexlist, normallist);
}

bool Triangle3D::intersect(Ray3D &r, Vector3D * vertexlist){
	Vector3D e1, e2, p, s, q;
	DBL u,v,t_ch, tmp, e=0.00001f;
	Vector3D v0,v1,v2;
	Vector3D origin, dir;
	origin = r.origin;
	dir = r.dir;

	v0 = vertexlist[ this->vertIdx[0]];
	v1 = vertexlist[ this->vertIdx[1]];
	v2 = vertexlist[ this->vertIdx[2]];

	e1 = v1-v0;
	e2 = v2-v0;
	p=Vector3D::cross(dir, e2);
	tmp = p.dot(e1);

	if (tmp > -e && tmp < e)
		return false;

	tmp = 1.0f/tmp;
	s = origin-v0;
	u = tmp * s.dot(p);
	if (u<0 || u>1)
		return false;

	q=Vector3D::cross(s, e1);
	v = tmp * dir.dot(q);
	if (v<0 || v>1)
		return false;

	if ( u+v >1)
		return false;

	t_ch = tmp * e2.dot(q);

	//if(t_ch<e ) return false;
	
	r.t = t_ch;


	r.p_isect.x = dir[0] * (r.t) + origin[0];
	r.p_isect.y = dir[1] * (r.t) + origin[1];
	r.p_isect.z = dir[2] * (r.t) + origin[2];
	return true;

}

//   This is a generic method to compute an intersection between a semi-infinite line beginning at v1 and passing
//   through v2 and a single triangle. If the intersection between the line and the triangle is located before v1,
//   then it is discarded.
//
//   Returns: true if an intersection is found, false otherwise
//
//   triangle: A single triangle that is to be tested against a semi-infinite line
//   v1: The first point of the semi-infinite line.
//   v2: The second point on the line
//   new_normal: The returned normal at the intersection. This is calculated by interpolating the normals at the
//   triangle vertices. Valid, only if the method returns true.
//   intersection_point: The calculated intersection point. Valid, only if the method returns true.

bool FindIntersection(Triangle3D triangle, Ray3D &r, Vector3D * vertexlist, Vector3D * normallist)
{
#ifndef BUFFER_OBJECT
	Vector3D e1, e2, p, s, q;
	Vector3D bcoords;
	DBL u,v,t_ch, tmp, e=0.00001f;
	Vector3D v0,v1,v2;
	Vector3D origin, dir,barycentric;
	origin = r.origin;
	dir = r.dir;

	v0 = vertexlist[ triangle.vertIdx[0]];
	v1 = vertexlist[ triangle.vertIdx[1]];
	v2 = vertexlist[ triangle.vertIdx[2]];

	e1 = v1-v0;
	e2 = v2-v0;
	p=Vector3D::cross(dir, e2);
	tmp = p.dot(e1);

	if (tmp > -e && tmp < e)
		return false;

	tmp = 1.0f/tmp;
	s = origin-v0;
	u = tmp * s.dot(p);
	if (u<0 || u>1)
		return false;

	q=Vector3D::cross(s, e1);
	v = tmp * dir.dot(q);
	if (v<0 || v>1)
		return false;

	if ( u+v >1)
		return false;

	t_ch = tmp * e2.dot(q);

	//if(t_ch<e ) return false;
	
	r.t = t_ch;

	barycentric.y = u;
	barycentric.z = v;
	barycentric.x = 1 - u - v;

	r.n_isect.x = barycentric[0] * normallist[ triangle.normIdx[0]].x +
				  barycentric[1] * normallist[ triangle.normIdx[1]].x +
				  barycentric[2] * normallist[ triangle.normIdx[2]].x;
	r.n_isect.y = barycentric[0] * normallist[ triangle.normIdx[0]].y +
				  barycentric[1] * normallist[ triangle.normIdx[1]].y +
				  barycentric[2] * normallist[ triangle.normIdx[2]].y;
	r.n_isect.z = barycentric[0] * normallist[ triangle.normIdx[0]].z +
				  barycentric[1] * normallist[ triangle.normIdx[1]].z +
				  barycentric[2] * normallist[ triangle.normIdx[2]].z;
	
	r.n_isect.normalize();

	r.p_isect.x = dir[0] * (r.t) + origin[0];
	r.p_isect.y = dir[1] * (r.t) + origin[1];
	r.p_isect.z = dir[2] * (r.t) + origin[2];
#endif
	return true;
}

