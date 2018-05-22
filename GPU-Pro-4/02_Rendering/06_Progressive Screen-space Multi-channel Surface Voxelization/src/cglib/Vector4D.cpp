
#include <stdlib.h>
#include <math.h>

#include "Vector4D.h"
#include "Matrix4D.h"

Vector4D::Vector4D(void)
{
	x = y = z = w = 0.0f;
}

Vector4D::~Vector4D(void)
{
}

Vector4D::Vector4D(DBL a, DBL b, DBL c, DBL d)
{
	x = a; y = b; z = c; w = d;
}

DBL Vector4D::minimum()
{
	return MIN4 (x, y, z, w);
}

DBL Vector4D::maximum()
{
	return MAX4 (x, y, z, w);
}

Vector4D::Vector4D(DBL * v)
{
	x = v[0]; y = v[1]; z = v[2]; w = v[3];
}

DBL Vector4D::dot(Vector4D v)
{
	return x*v.x + y*v.y + z*v.z + w*v.w;
}

DBL Vector4D::dot(Vector4D a, Vector4D b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

DBL Vector4D::length()
{
	return sqrt(x*x + y*y + z*z + w*w);
}

DBL Vector4D::distance(Vector4D v)
{
	DBL dx, dy, dz, dw;
	dx = x-v.x; dy = y-v.y; dz = z-v.z; dw = w-v.w;

	return (DBL) sqrt (dx*dx + dy*dy + dz*dz + dw*dw);
}

DBL Vector4D::distance(Vector4D v1, Vector4D v2)
{
	DBL dx, dy, dz, dw;
	dx = v1.x-v2.x; dy = v1.y-v2.y; dz = v1.z-v2.z; dw = v1.w-v2.w;

	return (DBL) sqrt (dx*dx + dy*dy + dz*dz + dw*dw);
}

void Vector4D::normalize()
{
	DBL a = (DBL) sqrt (x*x + y*y + z*z + w*w);
	if (a == 0)
		return;
	x/=a; y/=a; z/=a; w/=a;
}

DBL Vector4D::operator[] (int i)
{
	switch(i)
	{
	case 0: return x; break;
	case 1: return y; break;
	case 2: return z; break;
	case 3: return w; break;
	default: return 0;
	}
}

Vector4D Vector4D::operator+ (Vector4D v)
{
	Vector4D a;
	a.x = x + v.x;
	a.y = y + v.y;
	a.z = z + v.z;
	a.w = w + v.w;
	return a;
}

Vector4D Vector4D::operator- (Vector4D v)
{
	Vector4D a;
	a.x = x - v.x;
	a.y = y - v.y;
	a.z = z - v.z;
	a.w = w - v.w;
	return a;
}

Vector4D Vector4D::operator- ()
{
	Vector4D a;
	a.x = -x;
	a.y = -y;
	a.z = -z;
	a.w = -w;
	return a;
}

Vector4D Vector4D::operator* (Vector4D v)
{
	Vector4D a;
	a.x = x * v.x;
	a.y = y * v.y;
	a.z = z * v.z;
	a.w = w * v.w;
	return a;
}

Vector4D Vector4D::operator* (DBL s)
{
	Vector4D a;
	a.x = x * s;
	a.y = y * s;
	a.z = z * s;
	a.w = w * s;
	return a;
}

// postMult
Vector4D Vector4D::operator* (Matrix4D m)
{
	Vector4D a;
    a.x = m(0,0)*x + m(1,0)*y + m(2,0)*z + m(3,0)*w;
    a.y = m(0,1)*x + m(1,1)*y + m(2,1)*z + m(3,1)*w;
    a.z = m(0,2)*x + m(1,2)*y + m(2,2)*z + m(3,2)*w;
    a.w = m(0,3)*x + m(1,3)*y + m(2,3)*z + m(3,3)*w;
    return a;
}

Vector4D Vector4D::operator/ (Vector4D v)
{
	Vector4D a;
	a.x = x / v.x;
	a.y = y / v.y;
	a.z = z / v.z;
	a.w = w / v.w;
	return a;
}

Vector4D Vector4D::operator/ (DBL s)
{
	Vector4D a;
	a.x = x / s;
	a.y = y / s;
	a.z = z / s;
	a.w = w / s;
	return a;
}

Vector4D Vector4D::operator= (Vector4D v)
{
	x=v.x; y=v.y; z=v.z; w=v.w;
	return *this;
}

Vector4D & Vector4D::operator+= (Vector4D v)
{
    x += v.x;
    y += v.y;
    z += v.z;
    w += v.w;
    return *this;
}

Vector4D & Vector4D::operator-= (Vector4D v)
{
    x -= v.x;
    y -= v.y;
    z -= v.z;
    w -= v.w;
    return *this;
}

Vector4D & Vector4D::operator*= (DBL s)
{
    x *= s;
    y *= s;
    z *= s;
    w *= s;
    return *this;
}

// postMult
Vector4D & Vector4D::operator*= (Matrix4D m)
{
	DBL dx, dy, dz, dw;
    dx = m(0,0)*x + m(1,0)*y + m(2,0)*z + m(3,0)*w;
    dy = m(0,1)*x + m(1,1)*y + m(2,1)*z + m(3,1)*w;
    dz = m(0,2)*x + m(1,2)*y + m(2,2)*z + m(3,2)*w;
    dw = m(0,3)*x + m(1,3)*y + m(2,3)*z + m(3,3)*w;
    x = dx;
    y = dy;
    z = dz;
    w = dw;
    return *this;
}

Vector4D & Vector4D::operator/= (DBL s)
{
    x /= s;
    y /= s;
    z /= s;
    w /= s;
    return *this;
}

bool Vector4D::operator== (Vector4D v)
{
	return (x==v.x && y==v.y && z==v.z && w==v.w) ? true : false;
}

bool Vector4D::operator!= (Vector4D v)
{
	return (x!=v.x && y!=v.y && z!=v.z && w!=v.w) ? true : false;
}

// preMult
void Vector4D::xform (Matrix4D m, Vector4D v)
{
	DBL vx,vy,vz,vw;
    vx = m(0,0)*v.x + m(0,1)*v.y + m(0,2)*v.z + m(0,3)*v.w;
    vy = m(1,0)*v.x + m(1,1)*v.y + m(1,2)*v.z + m(1,3)*v.w;
    vz = m(2,0)*v.x + m(2,1)*v.y + m(2,2)*v.z + m(2,3)*v.w;
    vw = m(3,0)*v.x + m(3,1)*v.y + m(3,2)*v.z + m(3,3)*v.w;
    x = vx;
    y = vy;
    z = vz;
    w = vw;
}

// postMult
void Vector4D::xform (Vector4D v, Matrix4D m)
{
	DBL vx,vy,vz,vw;
    vx = m(0,0)*v.x + m(1,0)*v.y + m(2,0)*v.z + m(3,0)*v.w;
    vy = m(0,1)*v.x + m(1,1)*v.y + m(2,1)*v.z + m(3,1)*v.w;
    vz = m(0,2)*v.x + m(1,2)*v.y + m(2,2)*v.z + m(3,2)*v.w;
    vw = m(0,3)*v.x + m(1,3)*v.y + m(2,3)*v.z + m(3,3)*v.w;
    x = vx;
    y = vy;
    z = vz;
    w = vw;
}

