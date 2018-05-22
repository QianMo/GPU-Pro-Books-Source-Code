
#include <stdlib.h>

#ifdef INTEL_COMPILER
	#include <dvec.h>
	#include <mathimf.h>
#else
	#include <math.h>
#endif

#include "Vector3D.h"
#include "Matrix4D.h"

static DBL _RandomMatrix[RAND_LENGTH];
static DBL _RandomVectorMatrix[RAND_LENGTH][3];
static int _HashTable[RAND_LENGTH];

Vector3D::Vector3D(void)
{
	x = y = z = 0.0f;
}

Vector3D::~Vector3D(void)
{
}

Vector3D::Vector3D(DBL a, DBL b, DBL c)
{
	x = a; y = b; z = c;
}

DBL Vector3D::minimum()
{
	return MIN3 (x, y, z);
}

DBL Vector3D::maximum()
{
	return MAX3 (x, y, z);
}

DBL Vector3D::X () { return x; }
DBL Vector3D::Y () { return y; }
DBL Vector3D::Z () { return z; }

Vector3D::Vector3D(DBL * v)
{
	x = v[0]; y = v[1]; z = v[2];
}

DBL Vector3D::dot(Vector3D v)
{
	return x*v.x + y*v.y + z*v.z;
}

DBL Vector3D::dot(Vector3D a, Vector3D b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z;
}

DBL Vector3D::length() const
{
	return (DBL) sqrt (x*x + y*y + z*z);
}

DBL Vector3D::length2() const
{
	return x*x + y*y + z*z;
}

Vector3D Vector3D::cross(Vector3D a)
{
	Vector3D p;
	p.x=y*a.z-z*a.y;
	p.y=z*a.x-x*a.z;
	p.z=x*a.y-y*a.x;
	return p;
}

Vector3D Vector3D::cross(Vector3D a, Vector3D b)
{
	Vector3D p;
	p.x=a.y*b.z-a.z*b.y;
	p.y=a.z*b.x-a.x*b.z;
	p.z=a.x*b.y-a.y*b.x;
	return p;
}

DBL Vector3D::distance(Vector3D v)
{
#ifdef INTEL_COMPILER
	F32vec4 a(x,y,z,0.0f);
	F32vec4 b(v.x,v.y,v.z,0.0f);
	F32vec4 d = a-b;
	F32vec4 dd = d*d;
	return sqrtf(add_horizontal(dd));
#else
	DBL dx, dy, dz;
	dx = x - v.x; dy = y - v.y; dz = z - v.z;
	return (DBL)sqrt(dx*dx+dy*dy+dz*dz);
#endif
}

DBL Vector3D::distance(Vector3D v1, Vector3D v2)
{
#ifdef INTEL_COMPILER
	F32vec4 a(v1.x,v1.y,v1.z,0.0f);
	F32vec4 b(v2.x,v2.y,v2.z,0.0f);
	F32vec4 d = a-b;
	F32vec4 dd = d*d;
	return sqrtf(add_horizontal(dd));
#else
	DBL dx, dy, dz;
	dx = v1.x - v2.x; dy = v1.y - v2.y; dz = v1.z - v2.z;
	return (DBL)sqrt(dx*dx+dy*dy+dz*dz);
#endif
}

DBL Vector3D::operator[] (int i) const
{
	switch(i)
	{
	case 0: return x; break;
	case 1: return y; break;
	case 2: return z; break;
	case 3: return 1.0f; break;
	default: return 0;
	}
}

void Vector3D::normalize()
{
#ifdef INTEL_COMPILER
	DBL a = (DBL) sqrtf (x*x + y*y + z*z);
#else
	DBL a = (DBL) sqrt (x*x + y*y + z*z);
#endif
	if (a == 0)
		return;
	x/=a; y/=a; z/=a;
}

Vector3D Vector3D::operator+ (Vector3D v)
{
	Vector3D a;
	a.x = x + v.x;
	a.y = y + v.y;
	a.z = z + v.z;
	return a;
}

Vector3D Vector3D::operator- (const Vector3D v) const
{
	Vector3D a;
	a.x = x - v.x;
	a.y = y - v.y;
	a.z = z - v.z;
	return a;
}

Vector3D Vector3D::operator* (Vector3D v)
{
	Vector3D a;
	a.x = x * v.x;
	a.y = y * v.y;
	a.z = z * v.z;
	return a;
}

Vector3D Vector3D::operator* (DBL s)
{
	Vector3D a;
	a.x = x * s;
	a.y = y * s;
	a.z = z * s;
	return a;
}

Vector3D & Vector3D::operator+= (Vector3D v)
{
	x += v.x;
	y += v.y;
	z += v.z;
	return *this;
}

Vector3D & Vector3D::operator-= (Vector3D v)
{
	x -= v.x;
	y -= v.y;
	z -= v.z;
	return *this;
}

Vector3D & Vector3D::operator*= (DBL s)
{
	x *= s;
	y *= s;
	z *= s;
	return *this;
}

Vector3D & Vector3D::operator/= (DBL s)
{
	x /= s;
	y /= s;
	z /= s;
	return *this;
}

// xform point: p = p * m
Vector3D & Vector3D::operator*= (Matrix4D m)
{
	DBL vx,vy,vz,vw;
	vx = m(0,0)*x + m(1,0)*y + m(2,0)*z + m(3,0);
	vy = m(0,1)*x + m(1,1)*y + m(2,1)*z + m(3,1);
	vz = m(0,2)*x + m(1,2)*y + m(2,2)*z + m(3,2);
	vw = m(0,3)*x + m(1,3)*y + m(2,3)*z + m(3,3);
	x = vx / vw;
	y = vy / vw;
	z = vz / vw;
	return *this;
}

// xform point: p = p * m
Vector3D Vector3D::operator* (Matrix4D m)
{
	Vector3D v;
	DBL vx,vy,vz,vw;
	vx = m(0,0)*x + m(1,0)*y + m(2,0)*z + m(3,0);
	vy = m(0,1)*x + m(1,1)*y + m(2,1)*z + m(3,1);
	vz = m(0,2)*x + m(1,2)*y + m(2,2)*z + m(3,2);
	vw = m(0,3)*x + m(1,3)*y + m(2,3)*z + m(3,3);
	v.x = vx / vw;
	v.y = vy / vw;
	v.z = vz / vw;
	return v;
}

Vector3D Vector3D::operator- ()
{
	Vector3D a;
	a.x = -x;
	a.y = -y;
	a.z = -z;
	return a;
}

Vector3D Vector3D::operator/ (Vector3D v)
{
	Vector3D a;
	a.x = x / v.x;
	a.y = y / v.y;
	a.z = z / v.z;
	return a;
}

Vector3D Vector3D::operator/ (DBL s)
{
	Vector3D a;
	a.x = x / s;
	a.y = y / s;
	a.z = z / s;
	return a;
}

void Vector3D::srand (unsigned int seed)
{
    long   i;
    long   t, k;

	::srand (seed);
    for (i = 0; i < RAND_LENGTH; i++)
    {
        _HashTable[i] = i;
    }
    for (i = 0; i < RAND_LENGTH; i++)
    {
        k = ((RAND_LENGTH - 1) * rand ()) / RAND_MAX;
        t = _HashTable[i];
        _HashTable[i] = _HashTable[k];
        _HashTable[k] = t;
    }
	DBL l;
    for (i = 0; i < RAND_LENGTH; i++)
    {
        _RandomMatrix[i] = rand () / (float) RAND_MAX;
        _RandomVectorMatrix[i][0] = (rand () / (float) RAND_MAX) - 0.5f;
        _RandomVectorMatrix[i][1] = (rand () / (float) RAND_MAX) - 0.5f;
        _RandomVectorMatrix[i][2] = (rand () / (float) RAND_MAX) - 0.5f;
        l = sqrt(_RandomVectorMatrix[i][0]*_RandomVectorMatrix[i][0]+
			     _RandomVectorMatrix[i][1]*_RandomVectorMatrix[i][1]+
				 _RandomVectorMatrix[i][2]*_RandomVectorMatrix[i][2]);
		_RandomVectorMatrix[i][0] /= l;
		_RandomVectorMatrix[i][1] /= l;
		_RandomVectorMatrix[i][2] /= l;
    }
}

Vector3D Vector3D::random(unsigned int seed)
{
	Vector3D v;
	extern DBL _RandomVectorMatrix[RAND_LENGTH][3];
    static long cur = 0;

    if (seed == 0)
        cur = (cur + 1) % RAND_LENGTH;

    else
        cur = seed % RAND_LENGTH;
    v.x = _RandomVectorMatrix[cur][0];
	v.y = _RandomVectorMatrix[cur][1];
	v.z = _RandomVectorMatrix[cur][2];
	return v;
}

bool Vector3D::operator== (Vector3D v)
{
	if (x==v.x && y==v.y && z==v.z)
		return true;
	else
		return false;
}

bool Vector3D::operator!= (Vector3D v)
{
	if (x!=v.x || y!=v.y || z!=v.z)
		return true;
	else
		return false;
}

Vector3D Vector3D::operator= (Vector3D v)
{
	x=v.x; y=v.y; z=v.z;
	return *this;
}

// xform point: p = m * p
void Vector3D::xformPt (Matrix4D m)
{
	DBL vx,vy,vz,vw;
	vx = m(0,0)*x + m(0,1)*y + m(0,2)*z + m(0,3);
	vy = m(1,0)*x + m(1,1)*y + m(1,2)*z + m(1,3);
	vz = m(2,0)*x + m(2,1)*y + m(2,2)*z + m(2,3);
	vw = m(3,0)*x + m(3,1)*y + m(3,2)*z + m(3,3);
	x = vx / vw;
	y = vy / vw;
	z = vz / vw;
}

// xform vector: v = v * m
void Vector3D::xformVec (Matrix4D m)
{
	DBL vx,vy,vz;
	vx = m(0,0)*x + m(0,1)*y + m(0,2)*z;
	vy = m(1,0)*x + m(1,1)*y + m(1,2)*z;
	vz = m(2,0)*x + m(2,1)*y + m(2,2)*z;
	x = vx;
	y = vy;
	z = vz;
}

