#pragma once

#include "cglibdefines.h"

class Vector2D
{
public:
	Vector2D(void);
	Vector2D(DBL a, DBL b);
	Vector2D(DBL *v);
	~Vector2D(void);

	DBL dot(Vector2D v);
	DBL length();
	DBL distance(Vector2D v);
	static DBL dot(Vector2D a, Vector2D b);
	static DBL distance(Vector2D a, Vector2D b);
	DBL minimum();
	DBL maximum();
	void normalize();
	Vector2D operator= (Vector2D v);
	Vector2D operator+ (Vector2D v);
	Vector2D operator- (Vector2D v);
	Vector2D operator* (Vector2D v);
    Vector2D & operator+= (Vector2D v);
    Vector2D & operator-= (Vector2D v);
    Vector2D & operator*= (DBL s);
    Vector2D & operator/= (DBL s);
	Vector2D operator* (DBL s);
	Vector2D operator- ();
	Vector2D operator/ (Vector2D v);
	Vector2D operator/ (DBL s);
	DBL      operator[] (int i);
	bool     operator== (Vector2D v);
	bool     operator!= (Vector2D v);

	DBL x,y;
};

