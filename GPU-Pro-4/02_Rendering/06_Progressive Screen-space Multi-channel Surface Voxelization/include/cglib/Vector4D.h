#pragma once

#include "cglibdefines.h"
#include "Matrix4D.h"

class Vector4D
{
public:
	Vector4D (void);
	Vector4D (DBL a, DBL b, DBL c, DBL d);
	Vector4D (DBL *v);
	~Vector4D (void);
	
	DBL dot (Vector4D v);
	DBL length ();
	DBL distance (Vector4D v);
	static DBL dot (Vector4D a, Vector4D b);
	static DBL distance (Vector4D a, Vector4D b);
	DBL minimum ();
	DBL maximum ();
	void normalize ();
	
	Vector4D operator= (Vector4D v);
	Vector4D operator+ (Vector4D v);
	Vector4D operator- (Vector4D v);
	Vector4D operator- ();
	Vector4D operator* (Vector4D v);
	Vector4D operator* (DBL s);
	Vector4D operator* (Matrix4D m);
	Vector4D operator/ (Vector4D v);
	Vector4D operator/ (DBL s);
	DBL      operator[] (int i);
	bool     operator== (Vector4D v);
	bool     operator!= (Vector4D v);
	Vector4D & operator+= (Vector4D v);
	Vector4D & operator-= (Vector4D v);
	Vector4D & operator*= (DBL s);
	Vector4D & operator*= (Matrix4D m);
	Vector4D & operator/= (DBL s);
	
	void xform (class Matrix4D m, Vector4D v); // preMult
	void xform (Vector4D v, class Matrix4D m); // postMult
	
	DBL x,y,z,w;
};

