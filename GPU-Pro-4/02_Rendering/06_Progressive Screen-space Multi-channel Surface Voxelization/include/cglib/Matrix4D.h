#pragma once

#ifdef INTEL_COMPILER
	#include <dvec.h>
	#include <mathimf.h> 
#else
	#include "math.h"
#endif

#include "cglibdefines.h"

class Matrix4D
{
public:
	DBL a[16];
#ifdef INTEL_COMPILER
	F32vec4 row1, row2, row3, row4;
#endif

	Matrix4D(void);
	Matrix4D(DBL *data);
    Matrix4D(DBL a00, DBL a01, DBL a02, DBL a03,
             DBL a10, DBL a11, DBL a12, DBL a13,
             DBL a20, DBL a21, DBL a22, DBL a23,
             DBL a30, DBL a31, DBL a32, DBL a33);
	~Matrix4D(void);

	void setData(DBL *data);
	void transpose();
	void invert();
	void makeTranslate(class Vector3D t);
	void makeScale(class Vector3D s);
	void makeRotate(class Vector3D r, DBL theta);
	void makeOrtho(float l, float r, float b, float t, float n, float f);
	Matrix4D operator* (Matrix4D right);
	Matrix4D operator= (Matrix4D m);
    bool operator== (const Matrix4D &m);
    bool operator!= (const Matrix4D &m);
	class Vector3D operator* (class Vector3D vec);
	DBL operator[] (int);
	DBL operator() (int row, int col);
	DBL determinant();
	void dump(char * label, DBL *data);
	void dump(char * label);
	static Matrix4D identity();
	void sync();
};

