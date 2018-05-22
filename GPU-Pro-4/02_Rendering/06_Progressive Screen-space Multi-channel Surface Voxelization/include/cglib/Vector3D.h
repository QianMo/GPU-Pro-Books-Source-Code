#pragma once

#include <math.h>

#include "cglibdefines.h"

#define ALMOST_EQUAL_VEC3(_v1, _v2, tol) \
    ((_v1)[0] >= (_v2)[0] - (tol) && (_v1)[0] <= (_v2)[0] + (tol) && \
     (_v1)[1] >= (_v2)[1] - (tol) && (_v1)[1] <= (_v2)[1] + (tol) && \
     (_v1)[2] >= (_v2)[2] - (tol) && (_v1)[2] <= (_v2)[2] + (tol))

#define SUB_SCALED_VEC3(_dst, _v1, _v2, _s2)      \
    (((_dst)[0] = ((_v1)[0] - (_v2)[0]) * (_s2)), \
     ((_dst)[1] = ((_v1)[1] - (_v2)[1]) * (_s2)), \
     ((_dst)[2] = ((_v1)[2] - (_v2)[2]) * (_s2)))

class Vector3D
{
public:
	Vector3D(void);
	Vector3D(DBL a, DBL b, DBL c);
	Vector3D(DBL *v);
	~Vector3D(void);

	DBL dot(Vector3D v);
	DBL length() const;
	DBL length2() const; // length squared
	DBL distance(Vector3D v);
	Vector3D cross(Vector3D a);
	static DBL dot(Vector3D a, Vector3D b);
	static Vector3D cross(Vector3D a, Vector3D b);	
	static Vector3D random(unsigned int seed=0);
	static DBL distance(Vector3D a, Vector3D b);
	DBL minimum();
	DBL maximum();
	DBL X (), Y (), Z ();
	void normalize();
	Vector3D operator= (Vector3D v);
	Vector3D operator+ (Vector3D v);
	Vector3D operator- (Vector3D v) const;
	Vector3D operator* (Vector3D v);
	Vector3D operator* (DBL s);
	Vector3D & operator+= (Vector3D v);
	Vector3D & operator-= (Vector3D v);
	Vector3D & operator*= (DBL s);
	Vector3D & operator*= (class Matrix4D m);
	Vector3D & operator/= (DBL s);
	Vector3D operator* (class Matrix4D m);
	Vector3D operator- ();
	Vector3D operator/ (Vector3D v);
	Vector3D operator/ (DBL s);
	DBL      operator[] (int i) const;
	bool     operator== (Vector3D v);
	bool     operator!= (Vector3D v);
	static void srand (unsigned int seed);
	void xformPt (class Matrix4D m);
	void xformVec (class Matrix4D m);

    void dump (void)
    {
        EAZD_PRINT ("(x, y, z) = (" << x << " " << y << " " << z << ")");
    }

	DBL x,y,z;
};

