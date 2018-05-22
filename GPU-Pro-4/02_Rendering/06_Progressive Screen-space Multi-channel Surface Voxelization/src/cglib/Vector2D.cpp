
#include <stdlib.h>
#include <math.h>

#include "Vector2D.h"

Vector2D::Vector2D(void)
{
	x = y = 0.0f;
}

Vector2D::~Vector2D(void)
{
}

Vector2D::Vector2D(DBL a, DBL b)
{
	x = a; y = b;
}

DBL Vector2D::minimum()
{
	return MIN2 (x, y);
}

DBL Vector2D::maximum()
{
	return MAX2 (x, y);
}

Vector2D::Vector2D(DBL * v)
{
	x = v[0]; y = v[1];
}

DBL Vector2D::dot(Vector2D v)
{
	return x*v.x + y*v.y;
}

DBL Vector2D::dot(Vector2D a, Vector2D b)
{
	return a.x*b.x + a.y*b.y;
}

DBL Vector2D::length()
{
	return sqrt(x*x + y*y);
}

DBL Vector2D::distance(Vector2D v)
{
	DBL dx, dy;
	dx = x-v.x; dy = y-v.y;

	return (DBL) sqrt(dx*dx + dy*dy);
}

DBL Vector2D::distance(Vector2D v1, Vector2D v2)
{
	DBL dx, dy;
	dx = v1.x-v2.x; dy = v1.y-v2.y;

	return (DBL) sqrt(dx*dx + dy*dy);
}

DBL Vector2D::operator[] (int i)
{
	switch(i)
	{
	case 0: return x; break;
	case 1: return y; break;
	default: return 0;
	}
}

void Vector2D::normalize()
{
	DBL a = (DBL) sqrt (x*x + y*y);
	if (a == 0)
		return;
	x/=a; y/=a;
}

Vector2D Vector2D::operator+ (Vector2D v)
{
	Vector2D a;
	a.x = x + v.x;
	a.y = y + v.y;
	return a;
}

Vector2D Vector2D::operator- (Vector2D v)
{
	Vector2D a;
	a.x = x - v.x;
	a.y = y - v.y;
	return a;
}

Vector2D Vector2D::operator* (Vector2D v)
{
	Vector2D a;
	a.x = x * v.x;
	a.y = y * v.y;
	return a;
}

Vector2D Vector2D::operator* (DBL s)
{
	Vector2D a;
	a.x = x * s;
	a.y = y * s;
	return a;
}

Vector2D & Vector2D::operator+= (Vector2D v)
{
    x += v.x;
    y += v.y;
    return *this;
}

Vector2D & Vector2D::operator-= (Vector2D v)
{
    x -= v.x;
    y -= v.y;
    return *this;
}

Vector2D & Vector2D::operator*= (DBL s)
{
    x *= s;
    y *= s;
    return *this;
}

Vector2D & Vector2D::operator/= (DBL s)
{
    x /= s;
    y /= s;
    return *this;
}

Vector2D Vector2D::operator- ()
{
	Vector2D a;
	a.x = -x;
	a.y = -y;
	return a;
}

Vector2D Vector2D::operator/ (Vector2D v)
{
	Vector2D a;
	a.x = x / v.x;
	a.y = y / v.y;
	return a;
}

Vector2D Vector2D::operator/ (DBL s)
{
	Vector2D a;
	a.x = x / s;
	a.y = y / s;
	return a;
}

bool Vector2D::operator== (Vector2D v)
{
	return (x==v.x && y==v.y) ? true : false;
}

bool Vector2D::operator!= (Vector2D v)
{
	return (x!=v.x && y!=v.y) ? true : false;
}

Vector2D Vector2D::operator= (Vector2D v)
{
	x=v.x; y=v.y;
	return *this;
}

