#ifndef MATHS_H
#define MATHS_H 

#ifndef M_PI
#define M_PI 3.14159265358979323846f 
#endif
#define TWOPI 6.28318530717958f
#define PIDIV2 1.57079632679489f
#define EPSILON 0.0001f

#define DEG2RAD(i) ((i) * M_PI / 180.0f)
#define RAD2DEG(i) ((i) * 180.0f / M_PI)
#define CLAMP(i, x, y) if((i) > (y)) (i) = (y); if((i) < (x)) (i) = (x);
#define IS_EQUAL(a, b) (((a) >= ((b) - EPSILON)) && ((a) <= ((b) + EPSILON)))
#define IS_POWEROF2(x) ((((x) & ((x) - 1)) == 0) && ((x) > 0))

#include <Vector2.h>
#include <Vector3.h>
#include <Vector4.h>
#include <Matrix4.h>
#include <Plane.h>
#include <Quat.h>
#include <Color.h>

#endif
