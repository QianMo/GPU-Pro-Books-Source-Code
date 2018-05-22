#ifndef maths_h 
#define maths_h

#ifndef M_PI
#define M_PI 3.14159265358979323846f 
#endif
#define TWOPI 6.28318530717958f
#define PIDIV2 1.57079632679489f
#define EPSILON 0.0001f

#define DEG2RAD(i) ((i)*M_PI/180)
#define RAD2DEG(i) ((i)*180/M_PI)
#define CLAMP(i,x,y) if((i)>(y)) (i)=(y); if((i)<(x)) (i)=(x);
#define IS_EQUAL(a,b) (((a)>=((b)-EPSILON))&&((a)<=((b)+EPSILON)))

#include <COLOR.h>
#include <VECTOR2D.h>
#include <VECTOR3D.h>
#include <VECTOR4D.h>
#include <MATRIX4X4.h>

#endif













