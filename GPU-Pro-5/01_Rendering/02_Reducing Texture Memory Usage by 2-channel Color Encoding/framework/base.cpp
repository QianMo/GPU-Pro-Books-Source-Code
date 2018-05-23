
// ******************************** cfg_parser.cpp ********************************

// ---- #include "base.h"
// ---> including base.h

#ifndef _BASE_H
#define _BASE_H


#include <vector>
#include <string>
#include <map>
#include <algorithm>


// ---- #include "base/math.h"
// ---> including math.h

#ifndef _BASE_MATH_H
#define _BASE_MATH_H


#include <math.h>

// TODO: remove this dependency
#include <d3d9.h>
#include <d3dx9.h>


//#pragma comment(lib, "d3d9.lib")
//#pragma comment(lib, "d3dx9.lib")


#ifndef M_PI
#define M_PI	3.14159265
#endif


#define _INLINE	__forceinline


namespace base {


template<class T>
struct vector3d {
	typedef vector3d<T> V;

	T x,y,z;


	_INLINE vector3d() {}
	_INLINE vector3d(T _x,T _y,T _z) : x(_x), y(_y), z(_z) {}

#if 1
    friend _INLINE V operator -(const V &a) { return V(-a.x,-a.y,-a.z); }
	friend _INLINE V operator +(const V &a,const V &b) { return V(a.x+b.x,a.y+b.y,a.z+b.z); }
	friend _INLINE V operator -(const V &a,const V &b) { return V(a.x-b.x,a.y-b.y,a.z-b.z); }
	friend _INLINE V operator *(const V &a,T s) { return V(a.x*s,a.y*s,a.z*s); }
	friend _INLINE V operator *(T s,const V &a) { return V(a.x*s,a.y*s,a.z*s); }
	friend _INLINE V operator /(const V &a,T s) { return (s!=0)?V(a.x/s,a.y/s,a.z/s):V(0,0,0); }
	friend _INLINE const V &operator +=(V &a,const V &b) { a.x+=b.x; a.y+=b.y; a.z+=b.z; return a; }
	friend _INLINE const V &operator -=(V &a,const V &b) { a.x-=b.x; a.y-=b.y; a.z-=b.z; return a; }
	friend _INLINE const V &operator *=(V &a,T s) { a.x*=s; a.y*=s; a.z*=s; return a; }
	friend _INLINE const V &operator /=(V &a,T s) { if(s!=0) { a.x/=s; a.y/=s; a.z/=s; } else { a.x=0; a.y=0; a.z=0; } return a; }
	friend _INLINE bool operator ==(const V &a,const V &b) { return (a.x==b.x)&&(a.y==b.y)&&(a.z==b.z); }
#else
	_INLINE _V operator -() const { return _V(-x,-y,-z); }
	_INLINE _V operator +(const _V &v) const { return _V(x+v.x,y+v.y,z+v.z); }
	_INLINE _V operator -(const _V &v) const { return _V(x-v.x,y-v.y,z-v.z); }
	_INLINE _V operator *(_T s) const { return _V(x*s,y*s,z*s); }
	_INLINE _V operator /(_T s) const { return (s!=0)?_V(x/s,y/s,z/s):_V(0,0,0); }
	_INLINE _V &operator +=(const _V &v) { x+=v.x; y+=v.y; z+=v.z; return *this; }
	_INLINE _V &operator -=(const _V &v) { x-=v.x; y-=v.y; z-=v.z; return *this; }
	_INLINE _V &operator *=(_T s) { x*=s; y*=s; z*=s; return *this; }
	_INLINE _V &operator /=(_T s) { if(s!=0) { x/=s; y/=s; z/=s; } else { x=0; y=0; z=0; } return *this; }
	_INLINE bool operator ==(const _V &v) const { return (x==v.x)&&(y==v.y)&&(z==v.z); }
#endif

	_INLINE bool all_lequal(const V &v) const { return (x<=v.x)&&(y<=v.y)&&(z<=v.z); }
	_INLINE bool all_gequal(const V &v) const { return (x>=v.x)&&(y>=v.y)&&(z>=v.z); }

	_INLINE T length()		const { return sqrt(dot(*this)); }
	_INLINE T length2()	const { return dot(*this); }

	_INLINE void normalize()
	{
		T len = length();
		if(!len) return;
		x/=len; y/=len; z/=len;
	}

	_INLINE V get_normalized() const
	{
		V v = *this;
		v.normalize();
		return v;
	}

	_INLINE T dot(const V &v) const { return x*v.x + y*v.y + z*v.z; }
	_INLINE V cross(const V &v) const { return V(y*v.z-z*v.y,z*v.x-x*v.z,x*v.y-y*v.x); }
	
	_INLINE void set_mins(const V &v)
	{
		if(v.x<x) x=v.x;
		if(v.y<y) y=v.y;
		if(v.z<z) z=v.z;
	}

	_INLINE void set_maxs(const V &v)
	{
		if(v.x>x) x=v.x;
		if(v.y>y) y=v.y;
		if(v.z>z) z=v.z;
	}

	_INLINE V transform3x3(const T *mtx)
	{
		return V(x*mtx[0]+y*mtx[1]+z*mtx[2],x*mtx[3]+y*mtx[4]+z*mtx[5],x*mtx[6]+y*mtx[7]+z*mtx[8]);
	}

	_INLINE void scale_xyz(const V &v) { x*=v.x; y*=v.y; z*=v.z; }
	_INLINE V get_scaled_xyz(const V &v) const { return V(x*v.x,y*v.y,z*v.z); }

	_INLINE void update_min_max(V &min,V &max) const
	{
		if(x<min.x) min.x=x;
		if(y<min.y) min.y=y;
		if(z<min.z) min.z=z;
		if(x>max.x) max.x=x;
		if(y>max.y) max.y=y;
		if(z>max.z) max.z=z;
	}

	_INLINE void saturate()
	{
		if(x<0) x = 0;
		if(x>1) x = 1;
		if(y<0) y = 0;
		if(y>1) y = 1;
		if(z<0) z = 0;
		if(z>1) z = 1;
	}

	_INLINE DWORD make_rgb() const			{ return (int(x*255)<<16) | (int(y*255)<<8) | int(z*255); }
	_INLINE DWORD make_rgba(float a) const	{ return (int(a*255)<<24) | (int(x*255)<<16) | (int(y*255)<<8) | int(z*255); }

	_INLINE D3DXVECTOR3 &d3dx() { return *(D3DXVECTOR3*)this; }
	_INLINE const D3DXVECTOR3 &d3dx() const { return *(const D3DXVECTOR3*)this; }

	_INLINE void unpack_rgb(DWORD rgb)
	{
		x = ((rgb>>16)&0xFF)*(1/255.0f);
		y = ((rgb>> 8)&0xFF)*(1/255.0f);
		z = ((rgb    )&0xFF)*(1/255.0f);
	}

	_INLINE float unpack_rgba(DWORD rgba)
	{
		unpack_rgb(rgba);
		return ((rgba>>24)&0xFF)*(1/255.0f);
	}

	static _INLINE V from_rgb(DWORD rgb)
	{
		return V(	((rgb>>16)&0xFF)*(1/255.0f),
					((rgb>> 8)&0xFF)*(1/255.0f),
					((rgb    )&0xFF)*(1/255.0f) );
	}
};

typedef vector3d<float> vec3;


template<class T>
struct vector2d {
	typedef vector2d<T> V;

	T x,y;
	

	_INLINE vector2d() {}
	_INLINE vector2d(T _x,T _y) : x(_x), y(_y) {}

#if 1
    friend _INLINE V operator -(const V &a) { return V(-a.x,-a.y); }
	friend _INLINE V operator +(const V &a,const V &b) { return V(a.x+b.x,a.y+b.y); }
	friend _INLINE V operator -(const V &a,const V &b) { return V(a.x-b.x,a.y-b.y); }
	friend _INLINE V operator *(const V &a,T s) { return V(a.x*s,a.y*s); }
	friend _INLINE V operator *(T s,const V &a) { return V(a.x*s,a.y*s); }
	friend _INLINE V operator /(const V &a,T s) { return (s!=0)?V(a.x/s,a.y/s):V(0,0,0); }
	friend _INLINE const V &operator +=(V &a,const V &b) { a.x+=b.x; a.y+=b.y; return a; }
	friend _INLINE const V &operator -=(V &a,const V &b) { a.x-=b.x; a.y-=b.y; return a; }
	friend _INLINE const V &operator *=(V &a,T s) { a.x*=s; a.y*=s; return a; }
	friend _INLINE const V &operator /=(V &a,T s) { if(s!=0) { a.x/=s; a.y/=s; } else { a.x=0; a.y=0; } return a; }
	friend _INLINE bool operator ==(const V &a,const V &b) { return (a.x==b.x)&&(a.y==b.y); }
#else
    _INLINE V operator -() const { return V(-x,-y); }
	_INLINE V operator +(const V &v) const { return V(x+v.x,y+v.y); }
	_INLINE V operator -(const V &v) const { return V(x-v.x,y-v.y); }
	_INLINE V operator *(T s) const { return V(x*s,y*s); }
	_INLINE V operator /(T s) const { return (s!=0)?V(x/s,y/s):V(0,0); }
	_INLINE void operator +=(const V &v) { x+=v.x; y+=v.y; }
	_INLINE void operator -=(const V &v) { x-=v.x; y-=v.y; }
	_INLINE void operator *=(T s) { x*=s; y*=s; }
	_INLINE void operator /=(T s) { if(s==0) x=0, y=0; else x/=s, y/=s; }
	_INLINE bool operator ==(const V &v) const { return (x==v.x)&&(y==v.y); }
#endif

	_INLINE bool all_lequal(const V &v) const { return (x<=v.x)&&(y<=v.y); }
	_INLINE bool all_gequal(const V &v) const { return (x>=v.x)&&(y>=v.y); }

	_INLINE T length()		const { return sqrt(dot(*this)); }
	_INLINE T length2()	const { return dot(*this); }

	_INLINE void normalize()
	{
		T len = length();
		if(!len) return;
		x/=len; y/=len;
	}

	_INLINE V get_normalized() const
	{
		V v = *this;
		v.normalize();
		return v;
	}

	_INLINE void rotate90() const
	{
		*this = V(y,-x);
	}

	_INLINE V get_rotated90() const
	{
		return V(y,-x);
	}

	_INLINE float dot(const V &v) const { return x*v.x+y*v.y; }
	_INLINE T cross(const V &v) const { return x*v.y-y*v.x; }

	_INLINE void set_mins(const V &v)
	{
		if(v.x<x) x=v.x;
		if(v.y<y) y=v.y;
	}

	_INLINE void set_maxs(const V &v)
	{
		if(v.x>x) x=v.x;
		if(v.y>y) y=v.y;
	}

	_INLINE void scale_xy(const V &v) { x*=v.x; y*=v.y; }
	_INLINE V get_scaled_xy(const V &v) const { return V(x*v.x,y*v.y); }

	_INLINE void update_min_max(V &min,V &max) const
	{
		if(x<min.x) min.x=x;
		if(y<min.y) min.y=y;
		if(x>max.x) max.x=x;
		if(y>max.y) max.y=y;
	}

	_INLINE void saturate()
	{
		if(x<0) x = 0;
		if(x>1) x = 1;
		if(y<0) y = 0;
		if(y>1) y = 1;
	}

};

typedef vector2d<float> vec2;


template<class T>
struct vector4d {
	typedef vector4d<T> V;

	T x,y,z,w;
	
	_INLINE vector4d() {}
	_INLINE vector4d(T _x,T _y,T _z,T _w) : x(_x), y(_y), z(_z), w(_w) {}

	// 2D mapping: pos' = pos*xy + wz
	void make_map_to_unit(const vector2d<T> &bmin,const vector2d<T> &bmax);
	void make_map_unit_to(const vector2d<T> &bmin,const vector2d<T> &bmax);
	void make_map_to_view(const vector2d<T> &bmin,const vector2d<T> &bmax);
	void make_map_view_to(const vector2d<T> &bmin,const vector2d<T> &bmax);
	void make_map_concat(const V &a,const V &b);
	void make_map_inverse(const V &a);
	void make_map_box_scale_fit(const vector2d<T> &box,const vector2d<T> &space,const vector2d<T> &align);
	vector2d<T>  map_apply(const vector2d<T> &p)		{ return vector2d<T>(p.x*x+w,p.y*y+z); }
	vector2d<T>  map_inv_apply(const vector2d<T> &p)	{ return vector2d<T>((p.x-w)/x,(p.y-z)/y); }
};

typedef vector4d<float> vec4;


template<class T> void vector4d<T>::make_map_to_unit(const vector2d<T> &bmin,const vector2d<T> &bmax)
{
	// P' = (P-bmin)/(bmax-bmin)
	x = 1.f/(bmax.x - bmin.x);
	y = 1.f/(bmax.y - bmin.y);
	w = -bmin.x*x;
	z = -bmin.y*y;
}

template<class T> void vector4d<T>::make_map_unit_to(const vector2d<T> &bmin,const vector2d<T> &bmax)
{
	// P' = P*(bmax-bmin) + bmin
	x = bmax.x - bmin.x;
	y = bmax.y - bmin.y;
	w = bmin.x;
	z = bmin.y;
}

template<class T> void vector4d<T>::make_map_to_view(const vector2d<T> &bmin,const vector2d<T> &bmax)
{
	// P' = ((P-bmin)/(bmax-bmin))*vector2d<T>(2,-2) + vector2d<T>(-1,1) = P/(bmax-bmin)*vector2d<T>(2,-2) - bmin/(bmax-bmin)*vector2d<T>(2,-2) + vector2d<T>(-1,1)
	x = 2.f/(bmax.x - bmin.x);
	y = 2.f/(bmin.y - bmax.y);
	w = -1.f - bmin.x*x;
	z =  1.f - bmin.y*y;
}

template<class T> void vector4d<T>::make_map_view_to(const vector2d<T> &bmin,const vector2d<T> &bmax)
{
	// P' = (P*vector2d<T>(.5,-.5)+vector2d<T>(.5,.5))*(bmax-bmin) + bmin = P*vector2d<T>(.5,-.5)*(bmax-bmin)+vector2d<T>(.5,.5)*(bmax-bmin) + bmin
	x = .5f*(bmax.x - bmin.x);
	y = .5f*(bmin.y - bmax.y);
	w = bmin.x + x;
	z = bmin.y - y;
}

template<class T> void vector4d<T>::make_map_concat(const vector4d<T> &a,const vector4d<T> &b)
{
	// P' = (P*a.xy + a.wz)*b.xy + b.wz = P*a.xy*b.xy + a.wz*b.xy + b.wz
	// target can't alias with b
	x = a.x*b.x;
	y = a.y*b.y;
	w = a.w*b.x + b.w;
	z = a.z*b.y + b.z;
}

template<class T> void vector4d<T>::make_map_inverse(const vector4d<T> &a)
{
	// P' = (P-a.wz)/a.xy
	x = 1.f/a.x;
	y = 1.f/a.y;
	w = -a.w*x;
	z = -a.z*y;
}

template<class T> void vector4d<T>::make_map_box_scale_fit(const vector2d<T> &box,const vector2d<T> &space,const vector2d<T> &align)
{
	x = space.x/box.x;
	y = space.y/box.y;
	if(x<y) x=y; else y=x;
	w = (space.x-box.x*x)*align.x;
	z = (space.y-box.y*y)*align.y;
}




struct quat4 : public D3DXQUATERNION {

	_INLINE quat4() {}
	_INLINE quat4(float _x,float _y,float _z,float _w) : D3DXQUATERNION(_x,_y,_z,_w) {}

	_INLINE void SetIdentity()				{ D3DXQuaternionIdentity(this); }
	_INLINE void SetRotationX(float ang)	{ D3DXQuaternionRotationYawPitchRoll(this,0,ang*float(M_PI/180.f),0); }
	_INLINE void SetRotationY(float ang)	{ D3DXQuaternionRotationYawPitchRoll(this,ang*float(M_PI/180.f),0,0); }
	_INLINE void SetRotationZ(float ang)	{ D3DXQuaternionRotationYawPitchRoll(this,0,0,ang*float(M_PI/180.f)); }

	_INLINE void SetRotationYPR(float yaw,float pitch,float roll)
	{
		quat4 y,p,r;
		D3DXQuaternionRotationAxis(&y,(const D3DXVECTOR3*)&vec3(0,0,1),yaw*float(M_PI/180.f));
		D3DXQuaternionRotationAxis(&p,(const D3DXVECTOR3*)&vec3(0,1,0),-pitch*float(M_PI/180.f));
		D3DXQuaternionRotationAxis(this,(const D3DXVECTOR3*)&vec3(1,0,0),roll*float(M_PI/180.f));
		*this *= p;
		*this *= y;
	}

	_INLINE vec3 Transform(const vec3 &v)
	{
		D3DXQUATERNION _v(v.x,v.y,v.z,0);
		_v = D3DXQUATERNION(-x,-y,-z,w) * _v;
		_v *= *this;
		return vec3(_v.x,_v.y,_v.z);
	}
};



struct matrix4x4 : public D3DXMATRIX {

	_INLINE vec3 Apply(const vec3 &v) const
	{
		float out[4];
		D3DXVec3Transform((D3DXVECTOR4*)out,(D3DXVECTOR3*)&v,this);
		return *(vec3*)out;
	}

	_INLINE vec3 Unapply(const vec3 &v) const
	{
		vec3 t, out;
//		out.x = v.x*_11 + v.y*_21 + v.z*_31 + _41;
//		out.y = v.x*_12 + v.y*_22 + v.z*_32 + _42;
//		out.z = v.x*_13 + v.y*_23 + v.z*_33 + _43;

		t.x = v.x - _41;
		t.y = v.y - _42;
		t.z = v.z - _43;

		out.x = t.x*_11 + t.y*_12 + t.z*_13;
		out.y = t.x*_21 + t.y*_22 + t.z*_23;
		out.z = t.x*_31 + t.y*_32 + t.z*_33;
		return out;
	}

	_INLINE void ApplyInPlaceVec4(float *v) const
	{
		D3DXVECTOR4 out;
		D3DXVec4Transform(&out,(D3DXVECTOR4*)v,this);
		*((D3DXVECTOR4*)v) = out;
	}

	_INLINE void SetIdentity()							{ D3DXMatrixIdentity(this); }
	_INLINE void SetRotationFromQuat(const quat4 &q)	{ D3DXMatrixRotationQuaternion(this,&q); }

	_INLINE void SetAffineTransform(const vec3 &trans,const quat4 &rot,const vec3 &scale)
	{
		D3DXMatrixTransformation(this,NULL,NULL,(D3DXVECTOR3*)&scale,NULL,&rot,(D3DXVECTOR3*)&trans);
	}

	_INLINE void Transpose()
	{
		matrix4x4 tmp;
		D3DXMatrixTranspose(&tmp,this);
		*this = tmp;
	}

	_INLINE void Inverse()
	{
		matrix4x4 tmp;
		D3DXMatrixInverse(&tmp,NULL,this);
		*this = tmp;
	}
};

struct plane4 {
	vec3	normal;
	float	offs;

	_INLINE void MakeFromPoints(const vec3 &p1,const vec3 &p2,const vec3 &p3)
	{
		normal = (p2-p1).cross(p3-p1);
		normal.normalize();
		offs = normal.dot(p1);
	}

	_INLINE void Normalize()
	{
		float len = normal.length();
		if(len!=0)
		{
			len = 1.f/len;
			normal *= len;
			offs *= len;
		}
	}
};



}



#undef _INLINE



#endif

// <--- back to base.h
// ---- #include "base/types.h"
// ---> including types.h

#ifndef _B_TYPES_H
#define _B_TYPES_H


// types defined outside namespace
typedef unsigned char	byte;
typedef unsigned short	word;
typedef unsigned int	dword;

typedef char				int8;
typedef short				int16;
typedef int					int32;
typedef __int64				int64;
typedef unsigned char		uint8;
typedef unsigned short		uint16;
typedef unsigned int		uint32;
typedef unsigned __int64	uint64;


#define FULL_PTR	((void*)(LONG_PTR)-1)


namespace base {


}


#endif

// <--- back to base.h
// ---- #include "base/strings.h"
// ---> including strings.h

#ifndef _BASE_STRINGS_H
#define _BASE_STRINGS_H


#include <stdio.h>
#include <stdarg.h>


namespace base
{


std::string format(const char *fmt, ...);
std::string vformat(const char *fmt, va_list arg);

void sprintf(std::string &out,const char *fmt, ...);
void vsprintf(std::string &out,const char *fmt, va_list arg);


}


#endif

// <--- back to base.h
// ---- #include "base/convert.h"
// ---> including convert.h

#ifndef _CONVERT_H
#define _CONVERT_H



namespace base
{

/*
const float EXP_UPPER_LIMIT = 1E6f;
const float EXP_LOWER_LIMIT = 1E-6f;

class Conversions {
public:
//	static const float EXP_UPPER_LIMIT = 1E6;
//	static const float EXP_LOWER_LIMIT = 1E-6;

// conversion rules:
// - functions return count of characters used (including leading whitespace)
// - if a function failed to convert a number it should return zero and keep return variable untouched
// - all leading whitespace characters (space, \t, \n, \r) are skipped
// - both decimal and hexadecimal (preceded with 0x) integers should be accepted
// - floats written in decimal (123.456) and scientific (123.456e+20) should be accepted
// - every number can be preceded with optional '+' (positive, default) or '-' (negative)
// - int->string conversion should use decimal format
// - float->string conversion should use standard format with enough precision
// - for very large and very small float values scientific format should be used automatically
// - ->string conversions should clear output string variable first
// - array is a set of values separated by optional comas
// - if value in an array is omited, zero should be used by default
// - examples
// - ""			-> { }			(zero elements)
// - "123"		-> { 123 }
// - "1 2 3"	-> { 1, 2, 3 }
// - "1, 2 3"	-> { 1, 2, 3 }
// - "1,,2,3"	-> { 1, 0, 2, 3 }


	static int	StringToInt(const char *s,int &out);
	static int	StringToFloat(const char *s,float &out);

	static int	StringToIntArray(const char *s,int *arr);
	static int	StringToFloatArray(const char *s,float *arr);

	static int	StringGetArraySize(const char *s);


	static void IntToString(int value,std::string &str);
	static void FloatToString(float value,std::string &str);

	static int	IntArrayToString	(int *data,int count,std::string &str);
	static int	FloatArrayToString	(float *data,int count,std::string &str);

};
*/

}




#endif

// <--- back to base.h
// ---- #include "base/convert2.h"
// ---> including convert2.h

#ifndef _BASE_CONVERT2_H
#define _BASE_CONVERT2_H


namespace base
{


void		ParseWhitespace(const char *&s);
int         ParseHex(const char *&s);
int			ParseInt(const char *&s);
float		ParseFloat(const char *&s);
void		ParseString(const char *&s,std::string &out);
std::string	ParseString(const char *&s);
void		ParseStringT(const char *&s,std::string &out,const char *terminators);
void        ParseHexBuffer(const char *&s,std::vector<byte> &buff);


inline void AppendHex(std::string &str,int value) { str += format("0x%08x",value); }
inline void AppendInt(std::string &str,int value) { str += format("%d",value); }
void        AppendFloat(std::string &str,float value);
void        AppendString(std::string &str,const char *v,const char *escape);
void        AppendHexBuffer(std::string &str,const void *v,int size);



}




#endif

// <--- back to base.h
// ---- #include "base/smartptr.h"
// ---> including smartptr.h

#ifndef _SMARTPTR_H
#define _SMARTPTR_H


#include <stdio.h>
#include <assert.h>


namespace base {

class Collectable;
template<class _T> class weak_ptr;
template<class _T> class shared_ptr;



template<class _T>
class traced_ptr {
public:


	traced_ptr()
	{
		next = prev = this;
		ptr = NULL;
	}
	
	traced_ptr(const traced_ptr<_T> &tp)
	{
		next = prev = this;
		BindTo(tp);
	}
	
	~traced_ptr()
	{
		Unbind();
	}

	_T *get() const { return ptr; }

	_T &operator *() const  { assert(ptr!=NULL); return *ptr; }
	_T *operator ->() const { assert(ptr!=NULL); return  ptr; }

	traced_ptr<_T> &operator =(const traced_ptr<_T> &tp)
	{
		BindTo(tp);
		return *this;
	}
	
	bool operator ==(const traced_ptr<_T> &tp) const { return (ptr==tp.ptr);	}
	bool operator ==(const _T *p) const { return (ptr==p); }

	bool operator !=(const traced_ptr<_T> &tp) const { return (ptr!=tp.ptr);	}
	bool operator !=(const _T *p) const { return (ptr!=p); }

	void set_all(_T *new_ptr)
	{
		SetRing(new_ptr);
	}
	
	void init_new(_T *new_ptr)
	{
		Unbind();
		ptr = new_ptr;
	}

	bool is_alone() const
	{
		return (next==this);
	}
	
	int ring_size() const
	{
		traced_ptr<_T> *p = (traced_ptr<_T>*)this;
		int cnt=0;
		do
		{
			cnt++;
			p = p->next;
		} while(p!=this);
		return cnt;
	}

	//operator _T*()
	//{
	//	return ptr;
	//}
private:
	template<class _T> friend class weak_ptr;
	_T				*ptr;
	traced_ptr<_T>	*next,*prev;


	void Unbind()
	{
		next->prev = prev;
		prev->next = next;
		next = prev = this;
	}
	
	void BindTo(const traced_ptr<_T> &tp)
	{
		Unbind();
		prev = (traced_ptr<_T>*)&tp;
		next = tp.next;
		prev->next = this;
		next->prev = this;
		ptr = tp.ptr;
	}
	
	void SetRing(_T *new_ptr)
	{
		traced_ptr<_T> *p = this;
		do
		{
			p->ptr = new_ptr;
			p = p->next;
		} while(p!=this);
	}

};


template<class _T>
class weak_ptr {
public:

	weak_ptr() {}
	weak_ptr(_T *p) { if(p) tracer = p->_this; }
	weak_ptr(const weak_ptr<_T> &wp) : tracer(wp.tracer) { }
	weak_ptr(const shared_ptr<_T> &sp) { if(sp!=NULL) tracer = sp->_this; }

	_T *get() const { return (_T*)(tracer.ptr); }

	_T &operator *() const  { assert(tracer.ptr!=NULL); return *(_T*)(tracer.ptr); }
	_T *operator ->() const { assert(tracer.ptr!=NULL); return  (_T*)(tracer.ptr); }


	weak_ptr<_T> &operator =(_T *p)
	{
		if(p)	tracer = p->_this;
		else	tracer = traced_ptr<Collectable>();
		return *this;
	}
	
	weak_ptr<_T> &operator =(const weak_ptr<_T> &wp)
	{
		tracer = wp.tracer;
		return *this;
	}
	
	weak_ptr<_T> &operator =(const shared_ptr<_T> &sp)
	{
		return operator =(sp.get());
	}
	
	bool operator ==(const weak_ptr<_T> &wp) const { return (tracer==wp.tracer);	}
	bool operator ==(const shared_ptr<_T> &sp) const { return (tracer==sp.get());	}
	bool operator ==(const _T *p) const { return (tracer==p); }

	bool operator !=(const weak_ptr<_T> &wp) const { return (tracer!=wp.tracer);	}
	bool operator !=(const shared_ptr<_T> &sp) const { return (tracer!=sp.get());	}
	bool operator !=(const _T *p) const { return (tracer!=p); }

	//operator _T*()
	//{
	//	return ptr;
	//}
private:
	template<class _T> friend class shared_ptr;
	traced_ptr<Collectable> tracer;

};

template<class _T>
class shared_ptr {
public:

	shared_ptr() : ptr(NULL) {}
	shared_ptr(_T *p) : ptr(NULL) { SetPointer(p); }
	shared_ptr(const shared_ptr<_T> &sp) : ptr(NULL) { SetPointer(sp.ptr); }
	shared_ptr(const weak_ptr<_T> &wp) : ptr(NULL) { SetPointer(wp.get()); }

	virtual ~shared_ptr() {SetPointer(NULL);} 


	_T *get() const { return ptr; }

	_T &operator *() const  { assert(ptr!=NULL); return *ptr; }
	_T *operator ->() const { assert(ptr!=NULL); return  ptr; }


	shared_ptr<_T> &operator =(_T *p)
	{
		SetPointer(p);
		return *this;
	}
	
	shared_ptr<_T> &operator =(const shared_ptr<_T> &sp)
	{
		SetPointer(sp.ptr);
		return *this;
	}
	
	shared_ptr<_T> &operator =(const weak_ptr<_T> &wp)
	{
		SetPointer(wp.get());
		return *this;
	}

	//operator _T*()
	//{
	//	return ptr;
	//}
	
	bool operator ==(const shared_ptr<_T> &sp) const { return (ptr==sp.ptr);	}
	bool operator ==(const weak_ptr<_T> &wp) const { return (ptr==wp.get());	}
	bool operator ==(const _T *p) const { return (ptr==p); }

	bool operator !=(const shared_ptr<_T> &sp) const { return (ptr!=sp.ptr);	}
	bool operator !=(const weak_ptr<_T> &wp) const { return (ptr!=wp.get());	}
	bool operator !=(const _T *p) const { return (ptr!=p); }


private:
	template<class _T> friend class weak_ptr;
	_T	*ptr;


	void SetPointer(_T *p)
	{
		if(p  ) p->_add_ref();
		if(ptr) ptr->_release();
		ptr = p;
	}
};


class Collectable {
private:
	template<class _T> friend class weak_ptr;
	template<class _T> friend class shared_ptr;
	int							_ref_count;
	traced_ptr<Collectable>		_this;

public:
	Collectable() : _ref_count(0)
	{
		_this.init_new(this);
	}
	
	virtual ~Collectable()
	{
		_this.set_all(NULL);
	}


	void _add_ref()
	{
		_ref_count++;
		assert(_ref_count>0);
	}

	void _release()
	{
		assert(_ref_count>0);
		_ref_count--;
		if(_ref_count<=0)
			delete this;
	}
};


}


#endif

// <--- back to base.h
// ---- #include "base/streams.h"
// ---> including streams.h

#ifndef _B_STREAMS_H
#define _B_STREAMS_H


#include <stdio.h>
#include <string>
#include <vector>



namespace base {


enum {
	STREAM_ERR_NONE			= 0,
	STREAM_ERR_EOF			= (1<<0),
	STREAM_ERR_NOT_OPEN		= (1<<1),
};


class InStream {
public:
	InStream() { error = 0; }

	virtual int ReadBytes(void *buffer,int size) { *(int*)0 = 0; return 0; }

	void Read(void *buffer,int size)
	{
		if(!error && size>0)
			if(ReadBytes(buffer,size)!=size)
				error |= STREAM_ERR_EOF;
	}
	bool WasError()		{ return (error!=0); }
	int  GetError()		{ return error; }
	void ClearError()	{ error &= ~STREAM_ERR_NOT_OPEN; }

	virtual ~InStream() {}

protected:
	int	error;
};

class OutStream {
public:
	OutStream() { error = 0; }

	virtual int WriteBytes(const void *buffer,int size) { *(int*)0 = 0; return 0; }
	int operator <<(InStream &input);

	void Write(const void *buffer,int size)
	{
		if(!error && size>0)
			if(WriteBytes(buffer,size)!=size)
				error |= STREAM_ERR_EOF;
	}
	bool WasError()		{ return (error!=0); }
	int  GetError()		{ return error; }
	void ClearError()	{ error &= ~STREAM_ERR_NOT_OPEN; }

	virtual ~OutStream() {}

protected:
	int error;
};


class FileReaderStream : public InStream {
public:
	FileReaderStream(const char *path,bool binary = true);
	virtual ~FileReaderStream();
	virtual int ReadBytes(void *buffer,int size);

private:
	FILE *fp;
};

class FileWriterStream : public OutStream {
public:
	FileWriterStream(const char *path,bool binary = true);
	virtual ~FileWriterStream();
	virtual int WriteBytes(const void *buffer,int size);

private:
	FILE *fp;
};


class StoredByteArrayReader : public InStream {
public:
	std::vector<byte>	data;

	StoredByteArrayReader() : pos(0) {}
	virtual int ReadBytes(void *buffer,int size);
	void Rewind() { pos = 0; }

private:
	int				pos;
};

class StoredByteArrayWriter : public OutStream {
public:
	std::vector<byte>	data;

	StoredByteArrayWriter() {}
	virtual int WriteBytes(const void *buffer,int size);

};


/*
class StringReaderStream : public InStream {
public:
	StringReaderStream(std::string &str);
	virtual ~StringReaderStream();
};

class StringWriterStream : public OutStream {
public:
	StringWriterStream(std::string &str);
	virtual ~StringWriterStream();
};
*/



bool GetStreamBytes(std::vector<byte> &out,InStream *in);
bool GetStreamLines(std::vector<std::string> &out,InStream *in);
bool GetStreamText(std::string &out,InStream *in);




}



#endif

// <--- back to base.h
// ---- #include "base/vfs.h"
// ---> including vfs.h

#ifndef _BASE_VFS_H
#define _BASE_VFS_H


#include <vector>


namespace base
{


class FileSystem {
public:
	virtual InStream *GetFileAsStream(const char *path) = 0;

	virtual bool GetFileBytes(const char *path,std::vector<byte> &data)
	{
		InStream *s = GetFileAsStream(path);
		bool ok = GetStreamBytes(data,s);
		if(s) delete s;
		return ok;
	}

	virtual bool GetFileLines(const char *path,std::vector<std::string> &data)
	{
		InStream *s = GetFileAsStream(path);
		bool ok = GetStreamLines(data,s);
		if(s) delete s;
		return ok;
	}

	virtual bool GetFileText(const char *path,std::string &data)
	{
		InStream *s = GetFileAsStream(path);
		bool ok = GetStreamText(data,s);
		if(s) delete s;
		return ok;
	}

	template <class _T>
	bool ReadRawVector(const char *path,std::vector<_T> &out);

	virtual bool GetSubdirList(const char *path,std::vector<std::string> &out) { out.clear(); return false; }
	virtual bool GetFileList(const char *path,std::vector<std::string> &out) { out.clear(); return false; }
	virtual int  GetFileSize(const char *path) { return 0; }

	virtual ~FileSystem() {}
};


template<class _T>
bool FileSystem::ReadRawVector(const char *path,std::vector<_T> &out)
{
	InStream *in = GetFileAsStream(path);
	if(!in || !sizeof(_T)) { out.resize(0); return false; }

	out.resize(0);

	// TODO: add error checking
	byte buffer[16*1024];
	int fill = 0;
	do {
		int len = in->ReadBytes(buffer,sizeof(buffer));
		if(len<=0) break;
		
		out.resize((fill+len+sizeof(_T)-1)/sizeof(_T));
		memcpy(((byte*)&out[0])+fill,buffer,len);
		fill += len;
	} while(true);

	out.resize(fill/sizeof(_T));

	return true;
}


class NativeFileSystem : public FileSystem {
public:
	virtual FileReaderStream *GetFileAsStream(const char *path);
	virtual bool GetSubdirList(const char *path,std::vector<std::string> &out);
	virtual bool GetFileList(const char *path,std::vector<std::string> &out);
	virtual int  GetFileSize(const char *path);

	bool DumpRaw(const char *path,void *data,int size);

	template <class _T>
	bool DumpRawVector(const char *path,std::vector<_T> &v)
	{ return DumpRaw(path,v.size()?&v[0]:NULL,sizeof(_T)*v.size()); }

	virtual ~NativeFileSystem() {};
};



std::string FilePathGetPart(const char *path,bool dir,bool name,bool ext);
std::string GetCurrentDir();
unsigned long long GetFileTime(const char *path);


extern NativeFileSystem NFS;


}



#endif

// <--- back to base.h
// ---- #include "base/treefile.h"
// ---> including treefile.h

#ifndef _B_TREEFILE_H
#define _B_TREEFILE_H


#include <string.h>

#include <vector>
#include <string>
#include <map>

// ---- #include "types.h"
// ---> including types.h
// <--- back to treefile.h
// ---- #include "streams.h"
// ---> including streams.h
// <--- back to treefile.h



namespace base {


class TreeFileNode {
private:
	friend class TreeFileRef;
	friend class TreeFileBuilder;

	enum {	T_VOID = 0,
			T_INT = 1,
			T_FLOAT = 2,
			T_STRING = 3,
			T_RAW = 4,
	};
	enum { SHORT_DATA_MAX = 4 };


	struct NameRef {
		const char	*name;
		int			clone_id;

		bool operator <(const NameRef &r) const
		{
			int cmp = _stricmp(name,r.name);
			if(cmp!=0) return cmp<0;
			return clone_id < r.clone_id;
		}

		bool operator ==(const NameRef &r) const
		{
			int cmp = _stricmp(name,r.name);
			if(cmp!=0) return false;
			return clone_id == r.clone_id;
		}
	};

public:
	typedef std::map<NameRef,TreeFileNode *>::iterator	children_iterator;

private:
	const char		*name;
	int				clone_id;
	int				type;
	unsigned char	sdata[SHORT_DATA_MAX];
	unsigned char	*ldata;
	int				size;
	bool			owns_name;
	bool			owns_data;
	bool			invalid;

	std::map<NameRef,TreeFileNode *>	children;


//public:	// class to be used only internally

	TreeFileNode(const char *_name,int _id,bool copy_name=true);
	~TreeFileNode();

	children_iterator begin() { return children.begin(); }
	children_iterator end() { return children.end(); }

	TreeFileNode *GetChild(const char *cname,int cid,bool create=false,bool copy_name=true);

	bool Validate();

	void			SetData(const unsigned char *new_data,int new_size,int new_type,bool copy=true);
	unsigned char	*GetData() { return (size<=SHORT_DATA_MAX && owns_data) ? sdata : ldata; }
	int				GetSize() { return size; }

	bool GetInt(int &out);
	bool GetFloat(float &out);
	bool GetString(std::string &out);

	void SetInt(int value)					{ SetData((unsigned char*)&value,sizeof(int),T_INT); }
	void SetFloat(float value)				{ SetData((unsigned char*)&value,sizeof(float),T_FLOAT); }
	void SetString(const char *str)			{ SetData((unsigned char*)str,(int)strlen(str),T_STRING); }

};



class TreeFileRef {
public:
	class iterator {
	public:

		iterator() : empty(true) {}

		TreeFileRef operator *()		{	return empty ? TreeFileRef() : TreeFileRef(iter->second,false);	}
		void operator ++()				{	if(!empty) ++iter;	}

		bool operator ==(const iterator &it) { return (empty || it.empty) ? (empty==it.empty) : (iter==it.iter); }
		bool operator !=(const iterator &it) { return !operator ==(it); }

	private:
		friend class TreeFileRef;

		TreeFileNode::children_iterator iter;
		bool empty;

		iterator(const TreeFileNode::children_iterator &it) : empty(false), iter(it) {}
	};


	TreeFileRef();
	TreeFileRef(const TreeFileRef &f);
	~TreeFileRef();

	bool IsValid()			{ return (node!=NULL); }
	const char *GetName()	{ return node ? node->name : ""; }
	int GetId()				{ return node ? node->clone_id : 0; }

	int GetCloneArraySize(const char *name);

	iterator begin()		{ return (!node || is_writing) ? iterator() : iterator(node->begin()); }
	iterator end()			{ return (!node || is_writing) ? iterator() : iterator(node->end()); }



	void SerBool(const char *name,bool &v,bool def,int id=0)	{ SerializeBasic(name,&v,sizeof(v),false,def,id); }
	void SerByte(const char *name,byte &v,byte def,int id=0)	{ SerializeBasic(name,&v,sizeof(v),false,def,id); }
	void SerWord(const char *name,word &v,word def,int id=0)	{ SerializeBasic(name,&v,sizeof(v),false,def,id); }
	void SerDword(const char *name,dword &v,dword def,int id=0)	{ SerializeBasic(name,&v,sizeof(v),false,def,id); }
	void SerChar(const char *name,char &v,char def,int id=0)	{ SerializeBasic(name,&v,sizeof(v),true,def,id); }
	void SerShort(const char *name,short &v,short def,int id=0)	{ SerializeBasic(name,&v,sizeof(v),true,def,id); }
	void SerInt(const char *name,int &v,int def,int id=0)		{ SerializeBasic(name,&v,sizeof(v),true,def,id); }

	void SerializeBasic(const char *name,void *v,int size,bool sign,int def,int id=0);
	void SerFloat(const char *name,float &v,float def,int id=0);
	void SerPChar(const char *name,char *str,int len,const char *def,int id=0);
	void SerString(const char *name,std::string &out,const char *def,int id=0);

	void		Write_SetRaw(const char *name,const void *data,int size,int id=0);
	int			Read_GetRawSize(const char *name,int id=0);
	const void *Read_GetRawData(const char *name,int id=0);

	TreeFileRef SerChild(const char *name,int id=0);

	template<class _T>
	void SerVector(const char *name,std::vector<_T> &v)
	{
		if(IsReading())
		{
			v.clear();
			v.resize(GetCloneArraySize(name));
		}
		for(int i=0;i<v.size();i++)
			v[i].Serialize(SerChild(name,i));
	}

	template<class _T>
	void SerVectorPtr(const char *name,std::vector<_T*> &v)
	{
		if(IsReading())
		{
			for(int i=0;i<v.size();i++)
				if(v[i])
					delete v[i];
			v.clear();
			v.resize(GetCloneArraySize(name));
		}
		for(int i=0;i<v.size();i++)
		{
			if(IsReading())
				v[i] = new _T();
			v[i]->Serialize(SerChild(name,i));
		}
	}

	bool IsWriting() { return is_writing; }
	bool IsReading() { return !is_writing; }


private:
	friend class TreeFileBuilder;

	TreeFileNode	*node;
	bool			is_writing;


	TreeFileRef(TreeFileNode *n,bool is_wr);

	TreeFileNode *GetChild(const char *name,int id);

};


class TreeFileBuilder {

public:

	TreeFileBuilder();
	~TreeFileBuilder();

	TreeFileRef GetRoot(bool write);
	void Clear();

	bool LoadTreeTxt(const char *path,FileSystem &fs);
	bool SaveTreeTxt(const char *path);
	bool LoadTreeBin(InStream *in);
	bool SaveTreeBin(OutStream *out);

private:
    struct TextFileContext {
        int                                 auto_index;
        std::map<std::string,std::string>   defines;
        std::string                         base_path;
        std::map<std::string,int>           *last_index;
        FileSystem                          *fs;
    };

	TreeFileNode				*root;
	std::map<std::string,int>	dictionary_hits;
	std::vector<std::string>	dictionary;

	bool ReadNode_Binary(TreeFileNode *parent, InStream &in);
	bool WriteNode_Binary(TreeFileRef tf, OutStream &out);

    bool ReadNode_Text(TreeFileNode *parent,const char *&s,TextFileContext &ctx);
    bool WriteNode_Text(TreeFileNode *n,FILE *file,int indent,std::map<std::string,int> *last_index);

	void BuildDictionary(TreeFileRef tf,bool root_level = true);

	bool ReadDictionary_Binary(InStream &in);
	bool WriteDictionary_Binary(OutStream &out);
};





}




#endif

// <--- back to base.h
// ---- #include "base/chunkfile.h"
// ---> including chunkfile.h

#ifndef _BASE_CHUNKFILE_H
#define _BASE_CHUNKFILE_H



namespace base {


/*

File format:
-  4 - magic
-  4 - version
-  4 - dir data offset

Directory (anywhere in the file):
-  1 - type ( 0 - dir end, 1 - chunk )
-  n - name (null-terminated)
-  4 - offset in file
-  4 - size
-  1 - extension (0 - old format, 1 - new format)
-  4 - struct size (only if extension == 1)

*/


class ChunkFileBuilder {
public:
	ChunkFileBuilder(DWORD magic=13,DWORD version=13);
	~ChunkFileBuilder() { }

	bool Save(const char *path);
	void AddChunk(const char *name,const byte *data,int size,int struct_size=0);
	
	template<class T>	void add(const char *name,const T &data)				{ AddChunk(name,&data,sizeof(T),sizeof(T)); }
	template<class T>	void add(const char *name,const T *data,int count=1)	{ AddChunk(name,data,count*sizeof(T),sizeof(T)); }
	
	template<class T>	void add(const char *name,const std::vector<T> &data)
	{ if(data.size()>0) AddChunk(name,(const byte*)&data[0],data.size()*sizeof(T),sizeof(T)); }

private:
	std::vector<byte>	file_data;
	std::vector<byte>	dir_data;
};


bool ChunkFileVerify(DWORD magic,DWORD version,const byte *data,int size);
const byte *ChunkFileGetChunk(const char *name,int *out_size,const byte *data,int *out_struct_size=0);



class ChunkFile {
public:
	ChunkFile(const char *path,DWORD magic=13,DWORD version=13)
	{
		if(!NFS.GetFileBytes(path,file) || file.size()<=0) { file.clear(); return; }
		if(!ChunkFileVerify(magic,version,&file[0],file.size())) { file.clear(); return; }
	}
	~ChunkFile() { }
	
	template<class T>
	void read(const char *name,T &data)
	{
		int size=0, ssize=0;
		const byte *src = file.size()>0 ? ChunkFileGetChunk(name,&size,&file[0],&ssize) : 0;
		memset(&data,0,sizeof(data));
		if(src)
		{
			if(sizeof(data)<size) size=sizeof(data);
			if(ssize && ssize<size) size=ssize;
			memcpy(&data,src,size);
		}
	}

	template<class T>
	void read(const char *name,std::vector<T> &data)
	{
		int size=0, ssize=0;
		const byte *src = file.size()>0 ? ChunkFileGetChunk(name,&size,&file[0],&ssize) : 0;
		data.clear();
		if(src && size>0)
		{
			if(!ssize) ssize=sizeof(T);
			data.resize(size/ssize);

			int csize = ssize;
			if(csize>sizeof(T)) csize = sizeof(T);

			for(int i=0;i<data.size();i++)
			{
				T &t = data[i];
				memset(&t,0,sizeof(T));
				memcpy(&t,src+i*ssize,csize);
			}
		}
	}

private:
	std::vector<byte>	file;
};



}

#endif

// <--- back to base.h
// ---- #include "base/cfg_parser.h"
// ---> including cfg_parser.h

#ifndef _CFG_PARSER_H
#define _CFG_PARSER_H


#include <vector>
#include <string>
#include <map>



namespace base {



class CfgPreprocessor {
public:
	CfgPreprocessor(const char *path);
	~CfgPreprocessor();

	char GetChar();

private:
	struct FilePos {
		FILE		*fp;
		std::string	buffer;
		int			bpos;
		int			line;
		std::string	filename;
	};

	std::vector<FilePos>	files;
	FilePos					*current;
	std::string				base_dir;
	const char				*read_pos;
	int						read_size;


	char GetChar(int offs=0);
	void Advance(int cnt=1);
	bool ReadFile();
	void Include(const char *path);
};



class CfgParser {

public:
	CfgParser() { auto_index = 0x40000000; };

	bool Load(const char *path,base::TreeFileRef &f,FileSystem &fs);

private:
	int										auto_index;
	std::map<std::string,std::string>		defines;
	std::string								base_dir;
	FileSystem								*filesystem;


	bool ReadName(const char *&p, std::string &name, int &index);
	bool ReadValue(const char *&p, std::string &value);
	void ParseBlock(const char *&p, TreeFileRef &tree);


	static bool IsAlnum		(char ch);
	static void SkipWhites	(const char *& p);
	static void SkipWhitesNL(const char *& p);
	static void SkipError	(const char *& p);
};




inline bool CfgParser::IsAlnum(char ch) {
	if((ch>='a' && ch<='z') || (ch>='A' && ch<='Z') || (ch>='0' && ch <='9') 
		|| ch=='_'
		|| ch=='"'
		|| ch=='-'
		) return true;
	else  return false;
		
}

inline void CfgParser::SkipWhitesNL(const char *& p) {
	while(*p==' ' || *p=='\t' || *p=='\n' || *p=='\r')
		p++;
}

inline void CfgParser::SkipWhites(const char *& p) {
	while(*p==' ' || *p=='\t' || *p=='\r')
		p++;
}

inline void CfgParser::SkipError(const char *& p) {
	while(*p && *p!='\n' && *p!=';')
		p++;
	if(*p==';') p++;
}


}



#endif

// <--- back to base.h
// ---- #include "base/config.h"
// ---> including config.h

#ifndef _BASE_CONFIG_H
#define _BASE_CONFIG_H


#include <map>



namespace base
{


class Config {
public:
	Config(const char *_path=NULL,bool _autosave=false);
	~Config();

    bool Load(const char *p);
    bool Save(const char *p);

	TreeFileRef	GetNode(const char *name,int id,bool write);
	int			GetInt(const char *name,int def,int id=0);
	float		GetFloat(const char *name,float def,int id=0);
	std::string	GetString(const char *name,const char *def,int id=0);

	TreeFileRef	GetNodeWrite(const char *name,int id=0);
	void SetInt(const char *name,int v,int id=0);
	void SetFloat(const char *name,float v,int id=0);
    void SetString(const char *name,const char *v,int id=0);

private:
	TreeFileBuilder		tree;
    std::string         path;
    bool                autosave;

};



}




#endif

// <--- back to base.h
// ---- #include "base/tracked.h"
// ---> including tracked.h

#ifndef _TRACKED_H
#define _TRACKED_H



namespace base {



// -------------------------------- Tracked --------------------------------

template<class T>
class Tracked {
public:

    class iterator {
    public:

        iterator() : _ptr(0) {}

        T &operator *()  const { return *_ptr; }
        T *operator ->() const { return _ptr; }

        iterator &operator ++() { _ptr = _ptr->_next; return *this; }
        iterator &operator --() { _ptr = (_ptr ? _ptr->_prev : T::Tail()); return *this; }

        iterator operator ++(int) { iterator pp(_ptr); _ptr = _ptr->_next; return pp; }
        iterator operator --(int) { iterator pp(_ptr); _ptr = (_ptr ? _ptr->_prev : T::Tail()); return pp; }

        bool operator ==(const iterator &i) const { return _ptr == i._ptr; }
        bool operator !=(const iterator &i) const { return _ptr != i._ptr; }

    private:
        friend class Tracked;

        T *_ptr;

        iterator(T *p) : _ptr(p) {}
    };

    static iterator begin() { return iterator(_head()); }
    static iterator end()   { return iterator(); }


protected:
    Tracked()                                       { _register(); }
    Tracked(const Tracked<T> &t)                   { _register(); }
    Tracked<T> &operator =(const Tracked<T> &t)   {}

    ~Tracked() { _unregister(); }

private:
    friend class Tracked::iterator;

    T  *_prev;
    T  *_next;

    void _register()
    {
        _prev = Tail();
        _next = 0;
        Tail() = (T*)this;
        if(_prev)   _prev->_next = (T*)this;
        else        _head() = (T*)this;
    }

    void _unregister()
    {
        if(_prev) _prev->_next = _next;
        else      _head() = _next;
        if(_next) _next->_prev = _prev;
        else      Tail() = _prev;
    }

    static T *&_head()
    {
        static T *head = 0;
        return head;
    }

    static T *&Tail()
    {
        static T *tail = 0;
        return tail;
    }
};


}



#endif

// <--- back to base.h


#ifndef _DEBUG
// ---- #pragma comment(lib, "base.lib")
#else
// ---- #pragma comment(lib, "base_d.lib")
#endif



#endif

// <--- back to cfg_parser.cpp


using namespace std;



namespace base
{



CfgPreprocessor::CfgPreprocessor(const char *path)
{
	const char *p = path;
	while(*p) p++;
	while(p>path && *p!='/' && *p!='\\') p--;
	if(*p=='/' || *p=='\\') p++;
	base_dir.assign(path,p-path);

	files.clear();
	current = NULL;
	read_pos = NULL;
	read_size = 0;

	Include(p);
}

CfgPreprocessor::~CfgPreprocessor()
{
}

char CfgPreprocessor::GetChar()
{
	if(!current) return 0;

	char ch0 = GetChar(0);
	char ch1 = GetChar(1);

	int comment = 0;
	if(ch0=='/' && ch1=='/') comment = 1;
	if(ch0=='/' && ch1=='*') comment = 2;
	while(comment)
	{
		Advance();
		ch0 = GetChar(0);
		ch1 = GetChar(1);
		if(ch0==0) break;
		if(comment==1 && ch0=='\n') break;
		if(comment==2 && ch0=='*' && ch1=='/') break;
	}

	if(ch0=='\n' && ch1=='#')
	{
	}

	Advance();
	return ch0;
}

char CfgPreprocessor::GetChar(int offs)
{
	bool cnt = true;
	while(true)
	{
		if(offs<read_size) return read_pos[offs];
		if(!cnt) break;
		cnt = ReadFile();
	}
	return 0;
}

void CfgPreprocessor::Advance(int cnt)
{
	while(cnt>0)
	{
		int adv = cnt;
		if(adv>read_size) adv = read_size;
		read_size -= adv;
		cnt -= adv;
		while(adv-->0)
		{
			if(*read_pos=='\n') current->line++;
			read_pos++;
		}

		if(read_size<=0) ReadFile();
		if(read_size<=0) break;
	}
}

bool CfgPreprocessor::ReadFile()
{
	if(!current->fp)
		return false;

	if(read_pos !=NULL)
		if( read_pos - &current->buffer[0] >= 4096)
		{
			current->buffer.erase(0,current->bpos);
			read_pos = &current->buffer[0];
			read_size = current->buffer.size();
		}

	int bpos = 0;
	if(read_pos != NULL)
		bpos = read_pos - &current->buffer[0];

	int bsize = current->buffer.size();
	current->buffer.resize(bsize+1024);
	int len = fread(&current->buffer[bsize],1,1024,current->fp);
	current->buffer.resize(bsize+len);

	read_pos = &current->buffer[0] + bpos;

	if(len<1024)
	{
		fclose(current->fp);
		current->fp = NULL;
		return false;
	}

	return true;
}

void CfgPreprocessor::Include(const char *path)
{
	string fpath = base_dir + path;

	FilePos fp;
	fp.filename = path;
	fp.fp = fopen(path,"rt");
	fp.bpos = 0;
	fp.line = 1;

	if(!fp.fp)
		return;

	files.push_back(fp);
	current = &files[files.size()-1];
}





bool CfgParser::Load(const char *path,TreeFileRef &f,FileSystem &fs)
{
	filesystem = &fs;

	{
		const char *p = path;
		while(*p) p++;
		while(p>path && *p!='/' && *p!='\\') p--;
		if(*p=='/' || *p=='\\') p++;
		base_dir.assign(path,p-path);
	}

	vector<unsigned char> data;
	if(!filesystem->GetFileBytes(path,data))
		return false;

	const char *p = (char*)&data[0];
	ParseBlock(p,f);

	return true;
}

bool CfgParser::ReadName(const char *&p, string &name, int &index)
{
	SkipWhitesNL(p);

	const char *n_begin, *n_end;
	int _index = 0;

	n_begin = p;
	while(IsAlnum(*p) || *p == '_' || *p=='@' || *p=='$')
		p++;
	n_end = p;

	SkipWhites(p);
	if(*p=='[')
	{
		p++;
		SkipWhites(p);

        _index = ParseInt(p);
		SkipWhites(p);
		if(*p!=']') return false;
		p++;
		SkipWhites(p);
	}
	else if(*p=='*')
	{
		p++;
		_index = auto_index++;
		SkipWhites(p);
	}

	if(*p==':' || *p=='=')
		p++;
	else
	{
		SkipWhitesNL(p);
		if(*p!='{') return false;
	}

	name.assign(n_begin, n_end-n_begin);
	index = _index;

	return true;
}

bool CfgParser::ReadValue(const char *&p, string &value)
{
	SkipWhites(p);

	bool quoted = (*p=='"');
	if(quoted) p++;

	value.clear();
	while(1)
	{
		if(quoted)
		{
			if(*p=='\n') return false;
			if(*p=='"')
			{
				p++;
				break;
			}
		}
		else
			if(*p==' ' || *p=='\t' || *p=='\r' || *p=='\n' || *p==';')
				break;

		if(*p=='\r' || *p=='\n')
			return false;

		if(*p!='\\')
			value.push_back(*p++);
		else
		{
			p++;
			if(*p=='\r') p++; // CRLF version

			if(*p=='\n') { } // do nothing 
			else if(*p=='n') value.push_back('\n');
			else if(*p=='t') value.push_back('\t');
			else if(*p=='"') value.push_back('"');
			else return false;
		}
	}

	SkipWhites(p);
	if(*p!=';') return false;
	p++;

	return true;
}

void CfgParser::ParseBlock(const char *&p,TreeFileRef &tree)
{
	string name, value;
	int index;
	while(1)
	{
		SkipWhitesNL(p);
		if(*p=='\0' || *p=='}')
			break;

		if(!ReadName(p,name,index))
		{
			SkipError(p);
			continue;
		}

		if(*p=='{')
		{
			p++;
			ParseBlock(p,tree.SerChild(name.c_str(),index));
		}
		else
		{
			if(!ReadValue(p,value))
				SkipError(p);
			else
			{
				map<string,string>::iterator f = defines.find(value);
				if(f!=defines.end())
					value = f->second;

				if(name.c_str()[0]=='@')
					defines[name.c_str()+1] = value;
				else if(name.c_str()[0]=='$')
				{
					if(name == "$include")
					{
						CfgParser included;
						string path = base_dir + value;
						included.auto_index = auto_index;
						included.defines = defines;
						included.Load(path.c_str(),tree,*filesystem);
						defines = included.defines;
						auto_index = included.auto_index;
					}
				}
				else
				{
					tree.SerString(name.c_str(),value,"",index);
				}
			}
		}
	}

	if(*p=='}')
		p++;
}


}

// ******************************** chunkfile.cpp ********************************

// ---- #include "base.h"
// ---> including base.h
// <--- back to chunkfile.cpp


namespace base {


ChunkFileBuilder::ChunkFileBuilder(DWORD magic,DWORD version)
{
	file_data.insert(file_data.end(),(byte*)&magic,((byte*)&magic)+4);
	file_data.insert(file_data.end(),(byte*)&version,((byte*)&version)+4);

	version = 0xFFFFFFFF;
	file_data.insert(file_data.end(),(byte*)&version,((byte*)&version)+4);	// dir offset
}

bool ChunkFileBuilder::Save(const char *path)
{
	FILE *fp = fopen(path,"wb");
	if(!fp) return false;

	((DWORD*)&file_data[0])[2] = file_data.size();

	fwrite(&file_data[0],1,file_data.size(),fp);
	if(dir_data.size()>0)
		fwrite(&dir_data[0],1,dir_data.size(),fp);
	int tmp = 0;
	fwrite(&tmp,1,1,fp);

	fclose(fp);
	return true;
}

void ChunkFileBuilder::AddChunk(const char *name,const byte *data,int size,int struct_size)
{
	dir_data.push_back(1);
	dir_data.insert(dir_data.end(),name,name+(strlen(name)+1));

	int tmp = file_data.size();
	dir_data.insert(dir_data.end(),(byte*)&tmp,((byte*)&tmp)+4);
	dir_data.insert(dir_data.end(),(byte*)&size,((byte*)&size)+4);
	if(struct_size>0)
	{
		dir_data.push_back(1);
		dir_data.insert(dir_data.end(),(byte*)&struct_size,((byte*)&struct_size)+4);
	}
	dir_data.push_back(0);

	file_data.insert(file_data.end(),data,data+size);
}


bool ChunkFileVerify(DWORD magic,DWORD version,const byte *data,int size)
{
	if(size<12) return false;
	if(((DWORD*)data)[0]!=magic) return false;
	if(((DWORD*)data)[1]!=version) return false;

	int dir_base = ((DWORD*)data)[2];
	if(dir_base<0 || dir_base>=size-1) return false;

	const byte *data_end = data + size;
	const byte *dir = data + dir_base;
	while(true)
	{
		if(dir>=data_end) return false;
		if(*dir==0) break;
		if(*dir!=1) return false;

		if(++dir>=data_end) return false;

		while(*dir)
			if(++dir>=data_end) return false;
		if(++dir>=data_end) return false;
		if(dir+8>=data_end) return false;
		int c_base = *(DWORD*)dir;
		dir += 4;
		int c_size = *(DWORD*)dir;
		dir += 4;

		if(c_base<0 || c_base+c_size>size || size<0)
			return false;

		if(*dir==1) dir += 5;
		if(*dir!=0) return false;
		if(++dir>=data_end) return false;
	}

	return true;
}

const byte *ChunkFileGetChunk(const char *name,int *out_size,const byte *data,int *out_struct_size)
{
	const byte *dir = data + ((DWORD*)data)[2];

	if(out_struct_size) *out_struct_size = 0;
	
	while(true)
	{
		if(*dir!=1) break;
		dir++;

		const byte *n = (const byte *)name;
		while(*dir && *dir==*n)
			dir++, n++;

		if(*dir==0 && *n==0)
		{
			// found
			dir++;
			const byte *ptr = data + *(DWORD*)dir;
			if(out_size) *out_size = ((DWORD*)dir)[1];
			dir += 8;
			if(out_struct_size && *dir==1)
			{
				dir++;
				*out_struct_size = *(DWORD*)dir;
			}
			return ptr;
		}

		while(*dir)
			dir++;
		dir++;			// terminating zero
		dir += 8;		// offs/size
		if(*dir==1)		// optional struct size
		{
			dir++;
			dir+=4;
		}
		dir++;			// reserved zero
	}

	if(out_size)
		*out_size = 0;

	return NULL;
}



}

// ******************************** config.cpp ********************************

// ---- #include "base.h"
// ---> including base.h
// <--- back to config.cpp


using namespace std;


namespace base
{



Config::Config(const char *_path,bool _autosave)
{
    autosave = _autosave;
    path = _path ? _path : "";
    if(_path)
        Load(_path);
}

Config::~Config()
{
    if(autosave)
        Save(path.c_str());
}

bool Config::Load(const char *p)
{
	return tree.LoadTreeTxt(p,NFS);
}

bool Config::Save(const char *p)
{
    return tree.SaveTreeTxt(p);
}

TreeFileRef	Config::GetNode(const char *name,int id,bool write)
{
	TreeFileRef n = tree.GetRoot(write);
	string tmp, ptmp;

	if(id!=0)
	{
		ptmp = format("%s[%d]",name,id);
		name = ptmp.c_str();
	}

	while(n.IsValid() && *name)
	{
		const char *e = name;
		while(*e && *e!='/') e++;
		tmp.assign(name,e);

		n = n.SerChild(tmp.c_str());

		name = e;
		if(*name=='/')
			name++;
	}

	return n;
}

int Config::GetInt(const char *name,int def,int id)
{
	TreeFileRef n = GetNode(name,id,false);
	n.SerInt(NULL,def,def);
	return def;
}

float Config::GetFloat(const char *name,float def,int id)
{
	TreeFileRef n = GetNode(name,id,false);
	n.SerFloat(NULL,def,def);
	return def;
}

string Config::GetString(const char *name,const char *def,int id)
{
	TreeFileRef n = GetNode(name,id,false);
	string out;
	n.SerString(NULL,out,def);
	return out;
}

void Config::SetInt(const char *name,int v,int id)
{
	TreeFileRef n = GetNode(name,id,true);
	n.SerInt(NULL,v,v);
}

void Config::SetFloat(const char *name,float v,int id)
{
	TreeFileRef n = GetNode(name,id,true);
	n.SerFloat(NULL,v,v);
}

void Config::SetString(const char *name,const char *v,int id)
{
	TreeFileRef n = GetNode(name,id,true);
    string s = v;
	n.SerString(NULL,s,v);
}



}

// ******************************** convert.cpp ********************************

// ---- #include "base.h"
// ---> including base.h
// <--- back to convert.cpp

using namespace std;



namespace base
{

int __convert_cpp_no_exports_warning_disable;


/*
#define EPSILON 10E-8
static int _skpwhts(const char *&str) { 
	const char *begin = str; 

	while(*str==' ' || *str=='\t' || *str=='\r' || *str=='\n') str++;  
	
	return int(str-begin);
}

static int _atoi(const char *str, int &out)
{
	const char *p = str;
	int val=0;
	while(*p>='0' && *p<='9')
	{
		val*=10; val+=*p-'0'; 
		p++;
	}
	out=val; return int(p-str);
}

static int _htoi(const char *str, int &out)
{
	const char *p = str;
	int val=0;
	while((*p>='0' && *p<='9') || (*p>='a' && *p<='f') || (*p>='A' && *p<='F'))
	{
		int v=0;
		if	   (*p>='A' && *p<='F')	 v=10+*p-'A';
		else if(*p>='a' && *p<='f')  v=10+*p-'a';
		else						 v=   *p-'0';
		
		val<<=4; val+=v; 
		p++;
	}
	out=val; return int(p-str);
}


#define CHK_END(ch) if(ch=='\0') return 0;

int	Conversions::StringToInt(const char *s,int &out)
{
	const char *p = s;
	int sign = 1;
	_skpwhts(p);

	CHK_END(*p)

	if		(*p == '+')			  ++p;
	else if (*p == '-') {sign=-1; ++p;}
	
	CHK_END(*p)

//	_skpwhts(p);

//	CHK_END(*p)
	

	int ret = 0;
	if(*p == '0' )
	{
		p++;
		if (*p != 'x' && *p != 'X') {out = 0; return p-s;}
		p++;

		ret=_htoi(p, out);
		if(ret == 0) return (p-s-1);
				
	} else
	{
		ret=_atoi(p, out);
		if(ret == 0) return 0;
	}
	
	out*=sign;
	return int(p-s+ret);
}

int	Conversions::StringToFloat(const char *s,float &out)
{
	int n=0;
	int val1,val2, dig=0, sign=1;

	const char *p = s;

	_skpwhts(p);

	CHK_END(*p)

	if		(*p == '+')			  ++p;
	else if (*p == '-') {sign=-1; ++p;}
	
	CHK_END(*p)

	_skpwhts(p);

	CHK_END(*p)
	
	n=_atoi(p,val1);
	if(n == 0) val1=0;
	p+=n;

	if(*p=='.')
	{
		p++;
		if(p=='\0')
		{
			out=val1*(float)sign;
			return int(p-s);
		}
		
		dig=_atoi(p, val2);
		if(dig!=0)
		{
			double f=1;
			for(int i=0; i<dig; i++) f*=0.1f;
			
			out=float(double(val1)+double(val2)*f);

			p+=dig;
		}

		float eval=10, f=1;
		int exp;
		if(*p=='E' || *p=='e')
		{
			p++;
			if		(*p == '+')	           ++p;
			else if (*p == '-') {eval=0.1f; ++p;}

			dig=_atoi(p, exp);

			if(dig!=0)
			{
				p+=dig;
				while(--exp >=0) f*=eval;
				out*=f;
			} else return int(p-s-1);
		}
	}
	else
		out = val1;
	out*=sign;
	return int(p-s);	
}

int	Conversions::StringToIntArray(const char *s,int *arr)
{
	const char *p = s;
	
	int val, n, num=0;
	do {
		n = StringToInt(p, val);
		if(n == 0) break;

		*arr++ = val;
		
		p+=n;
		num++;
		if(*p==',') p++; else break;
	}
	while(1);
	return num;
}

int	Conversions::StringToFloatArray(const char *s,float *arr)
{
	const char *p = s;
	
	int n, num=0;
	float val;
	do {
		n = StringToFloat(p, val);
		if(n == 0) break;

		*arr++ = val;
		
		p+=n;
		num++;
		if(*p==',') p++; else break;
	}
	while(1);
	return num;
}

int	Conversions::StringGetArraySize(const char *s)
{
	const char *p = s;

	_skpwhts(p);
	int num=0;
	while(*p!= '\0')
	{
		while(*p != '\0' && *p!=' ' && *p!='\n' && *p!='\r' && *p!='\t')
			p++;
		
		_skpwhts(p);

		num++;
		if(*p=='\0') return num;
		p++;
	
		_skpwhts(p);
	}
	return num;
}

void Conversions::IntToString(int value,string &str)
{
	//TODO: finish it
	bool positive = value > 0;
	if(value == 0) {str="0"; return;}
	if(!positive)  {value=-value; str='-';} else str="";

	while(value > 0) {str+=(char)(value%10+'0'); value/=10;};

	int i, j;
	for(j=(int)str.length()-1,i=positive?0:1; i < j; i++, j--)
	{
		char t = str[j];
		str[j] = str[i];
		str[i] = t;
	}
}

void Conversions::FloatToString(float value,string &str)
{
	//cutprec(1.1111100001);
	int exp=0;
	double _value = value;
	bool positive = _value > 0;
	if(!positive)  {_value=-_value; str='-';} else str="";
	
	if(_value==0.0f) {str="0.0"; return;}

	if(_value > EXP_UPPER_LIMIT) 
	{
		while(_value>=1.f) {_value/=10;exp++;}
		_value*=10;
		exp--;
	} else
	if(_value < EXP_LOWER_LIMIT) 
	{
		while(_value<1.f) {_value*=10;exp--;}
	}

	double frac, in;
	in = (int) _value;
	frac = _value-in;

	double temp=(float)frac;
	int nd=0;
	while(temp-int(temp) > EPSILON && nd<7) {temp*=10;nd++;}
	string str1, str2;

	IntToString((int)in, str1);
	IntToString((int)temp, str2);
	nd-= (int)str2.length();

	//remove trailing zeros
	int i;
	for(i=(int)str2.length()-1; i>=0 && str2[i]=='0'; i--);
	
	str2 = str2.substr(0, i+1);

	if(str2.size() >= 8) str2 = str2.substr(0, 8);

	str = str1+".";
	while(--nd >= 0) str+="0";

	str += str2;

	string strE;
	if(exp!=0) {IntToString(exp, strE);strE = "E"+strE;}

	str += strE;
	if(!positive) str = "-"+str;
	return;
}

int	Conversions::IntArrayToString(int *data,int count,string &str)
{
	str="";
	
	string temp;
	for(int i=0; i<count; i++) {IntToString(*data, temp); str+=temp+","; data++;}

	return 0;
}

int	Conversions::FloatArrayToString(float *data,int count,string &str)
{
	str="";
	
	string temp;
	for(int i=0; i<count; i++) {FloatToString(*data, temp);str+=temp+",";data++;}
	
	return 0;

}
*/

}

// ******************************** convert_append.cpp ********************************
// ---- #include "base.h"
// ---> including base.h
// <--- back to convert_append.cpp


using namespace std;


namespace base
{




void AppendFloat(string &str,float value)
{
    int sign = (*(DWORD*)&value >> 31) & 1;
    int exp  = (*(DWORD*)&value >> 23) & 0xFF;
    int vv   = (*(DWORD*)&value ) & 0x7FFFFF;

    if(exp==0xFF)
    {
        if(sign) str.push_back('-');
        if(!vv) str += "1.#Inf";
        else str += "1.#QNAN";
        return;
    }

    if(exp<0x7F || exp>0x7F+20)
        str += format("%.8e",value);
    else
        str += format("%.8f",value);
}

void AppendString(std::string &str,const char *v,const char *escape)
{
    DWORD esc[8] = {0xFFFFFFFF, (1<<(' '-32)) | (1<<('"'-32)), (1<<('\\'-64))};
    while(*escape)
    {
        esc[byte(*escape)/32] |= (1 << (byte(*escape)%32));
        escape++;
    }

    const char *s = v;
    int flag = 0;
    while(*s)
    {
        flag |= ( esc[byte(*s)>>5] >> (byte(*s)&31) );
        s++;
    }

    if(*v && !(flag&1))
    {
        str += v;
        return;
    }

    str.push_back('"');
    s = v;
    while(*s)
    {
        if( (*s>=0 && *s<32) || *s=='"' || *s=='\\')
        {
            str.push_back('\\');
            if(byte(*s)>=32) str.push_back(*s);
            else if(*s=='\n') str.push_back('n');
            else if(*s=='\r') str.push_back('r');
            else if(*s=='\t') str.push_back('t');
            else if(*s=='\b') str.push_back('b');
            else
            {
                str.push_back('0');
                str.push_back(*s/8+'0');
                str.push_back(*s%8+'0');
            }
        }
        else
            str.push_back(*s);
        s++;
    }
    str.push_back('"');
}

void AppendHexBuffer(std::string &str,const void *v,int size)
{
    const byte *ptr = *(const byte **)&v;
    while(size-->0)
    {
        str.push_back("0123456789ABCDEF"[*ptr>>4]);
        str.push_back("0123456789ABCDEF"[*ptr&15]);
        ptr++;
    }
}




}

// ******************************** convert_parse.cpp ********************************

// ---- #include "base.h"
// ---> including base.h
// <--- back to convert_parse.cpp


using namespace std;


namespace base
{




void ParseWhitespace(const char *&s)
{
	while(*s==' ' || *s=='\t' || *s=='\n' || *s=='\r')
		s++;
}

int ParseHexNoSkip(const char *&s)
{
    int v = 0;
    while(1)
    {
        if(*s>='0' && *s<='9') v = (v<<4) + (*s++ - '0');
        else if(*s>='a' && *s<='f') v = (v<<4) + (*s++ - 'a' + 10);
        else if(*s>='A' && *s<='F') v = (v<<4) + (*s++ - 'A' + 10);
        else break;
    }
    return v;
}

int ParseHex(const char *&s)
{
    ParseWhitespace(s);
    return ParseHexNoSkip(s);
}

int ParseInt(const char *&s)
{
    ParseWhitespace(s);

    int sign = 1;
    if(*s=='+') s++;
    else if(*s=='-') s++, sign = -1;

    if(s[0]=='0' && (s[1]=='x' || s[1]=='X'))
    {
        s += 2;
        return sign*ParseHexNoSkip(s);
    }

    int vv = 0;
    while(*s>='0' && *s<='9')
        vv = vv*10 + (*s++ - '0');

    return vv*sign;
}

float ParseFloat(const char *&s)
{
    ParseWhitespace(s);

    int sign = 0;
    if(*s=='+') s++;
    else if(*s=='-') s++, sign = 1;

    float vv = 0;
    const char *bs = s;
    while(*s>='0' && *s<='9') s++;
    if(*s=='.')
    {
        s++;
        while(*s>='0' && *s<='9') s++;
    }
    if(*s=='e')
    {
        s++;
        if(*s=='+' || *s=='-') s++;
        while(*s>='0' && *s<='9') s++;
    }

    if(*s=='#')
    {
        s++;
        if(_strnicmp(s,"Inf",3)==0)
        {
            s += 3;
            *(DWORD*)&vv = sign ? 0xFF800000 : 0x7F800000;
            return vv;
        }
        else if(_strnicmp(s,"QNAN",4)==0)
        {
            s += 4;
            *(DWORD*)&vv = sign ? 0xFFFFFFFF : 0x7FFFFFFF;
            return vv;
        }
    }

    sscanf(bs,"%f",&vv);

    if(sign) *(DWORD*)&vv |= 0x80000000;
    return vv;
}


static const DWORD TERM_PARSE_WORD[8] = {0xFFFFFFFF,(1<<(' '-32))};
static const DWORD TERM_BREAK_QUOTED[8] = {(1<<0) | (1<<'\n') | (1<<'\r'), (1<<('"'-32)), (1<<('\\'-64))};


static inline void _parse_string_raw(const char *&s,string &out,const DWORD term[8])
{
    const char *b = s;
    while(!( term[byte(*s)>>5] & (1<<(byte(*s)&31)) ) )
        s++;
    out.assign(b,s);
}

static inline void _parse_string_quoted_t(const char *&s,string &out,const DWORD term[8])
{
    out.clear();
    while(1)
    {
        const char *b = s;
        while(!( term[byte(*s)>>5] & (1<<(byte(*s)&31)) ) )
            s++;
        if(s!=b) out.append(b,s);

        if(*s!='\\') break;

        s++;
        if(!*s) break;

        if(*s=='n') out.push_back('\n');
        else if(*s=='r') out.push_back('\r');
        else if(*s=='t') out.push_back('\t');
        else if(*s=='b') out.push_back('\b');
        else if(*s>='0' && *s<='7')
        {
            int v = 0, n = 0;
            while(*s>='0' && *s<='7' && n<3)
            {
                v = v*8 + (*s++ - '0');
                n++;
            }
            out.push_back(v);
            s--;    // compensate for upcoming s++
        }
        else if(*s=='\n') {}
        else out.push_back(*s);
        s++;
    }
}

void ParseString(const char *&s,std::string &out)
{
    ParseWhitespace(s);
    if(*s=='"')
    {
        s++;
        _parse_string_quoted_t(s,out,TERM_BREAK_QUOTED);
        if(*s=='"')
            s++;
    }
    else
        _parse_string_raw(s,out,TERM_PARSE_WORD);
}

std::string	ParseString(const char *&s)
{
	string str;
	ParseString(s,str);
    return str;
}

void ParseStringT(const char *&s,std::string &out,const char *terminators)
{
    ParseWhitespace(s);
    if(*s=='"')
    {
        s++;
        _parse_string_quoted_t(s,out,TERM_BREAK_QUOTED);
        if(*s=='"')
            s++;
    }
    else
    {
        DWORD term[8]={TERM_PARSE_WORD[0],TERM_PARSE_WORD[1],TERM_PARSE_WORD[2],TERM_PARSE_WORD[3]};
        while(*terminators)
        {
            term[byte(*terminators)/32] |= (1<<(byte(*terminators)%32));
            terminators++;
        }

        _parse_string_raw(s,out,term);
    }
}

void ParseHexBuffer(const char *&s,vector<byte> &buff)
{
    buff.clear();

    while(1)
    {
        byte v;
             if(*s>='0' && *s<='9') v = (*s++ - '0')<<4;
        else if(*s>='a' && *s<='f') v = (*s++ - 'a' + 10)<<4;
        else if(*s>='A' && *s<='F') v = (*s++ - 'A' + 10)<<4;
        else break;

        if(*s>='0' && *s<='9') v |= (*s++ - '0');
        else if(*s>='a' && *s<='f') v |= (*s++ - 'a' + 10);
        else if(*s>='A' && *s<='F') v |= (*s++ - 'A' + 10);
        else break;

        buff.push_back(v);
    }
}



}

// ******************************** streams.cpp ********************************

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---- #include "base.h"
// ---> including base.h
// <--- back to streams.cpp

using namespace std;



namespace base {


// ---------------- In/Out Stream ----------------

int OutStream::operator <<(InStream &input)
{
	char buffer[4096];
	int rlen, wlen, rpos, total = 0;

	if(WasError())
		return 0;

	if(input.WasError())
	{
		error = 1;
		return 0;
	}

	while(1)
	{
		rlen = input.ReadBytes(buffer,sizeof(buffer));

		if(input.WasError())
		{
			error = 1;
			break;
		}

		if(rlen<=0)
			break;

		rpos = 0;
		while(rlen>0)
		{
			wlen = WriteBytes(buffer+rpos,rlen);
			if(wlen<=0) break;
			total += wlen;
			rpos += wlen;
			rlen -= wlen;
		}
	}

	if(rlen>0)
		error = 1;

	return total;
}



// ---------------- File Reader/Writer Stream ----------------

FileReaderStream::FileReaderStream(const char *path,bool binary)
{
	fp = fopen(path, binary ? "rb" : "rt" );
	if(!fp) error |= STREAM_ERR_NOT_OPEN;
}

FileReaderStream::~FileReaderStream()
{
	if(fp) fclose(fp);
}

int FileReaderStream::ReadBytes(void *buffer,int size)
{
	if(!fp) return 0;
	return fread(buffer,1,size,fp);
}


FileWriterStream::FileWriterStream(const char *path,bool binary)
{
	fp = fopen(path, binary ? "wb" : "wt" );
	if(!fp) error |= STREAM_ERR_NOT_OPEN;
}

FileWriterStream::~FileWriterStream()
{
	if(fp) fclose(fp);
}

int FileWriterStream::WriteBytes(const void *buffer,int size)
{
	if(!fp) return 0;
	return fwrite(buffer,1,size,fp);
}



// ---------------- Stored Byte Array Streams ----------------

int StoredByteArrayReader::ReadBytes(void *buffer,int size)
{
	if(pos>=(int)data.size())
	{
		pos = data.size();
		error |= STREAM_ERR_EOF;
		return 0;
	}

	int s1 = size;
	if(pos+s1>=(int)data.size())
	{
		error |= STREAM_ERR_EOF;
		s1 = data.size() - pos;
	}

	if(s1>0) memcpy(buffer,&data[pos],s1);
	pos += s1;

	return s1;
}


int StoredByteArrayWriter::WriteBytes(const void *buffer,int size)
{
	int psize = data.size();
	data.resize(psize+size);
	if(size) memcpy(&data[psize],buffer,size);
	return size;
}


// ---------------- helper functions ----------------



bool GetStreamBytes(vector<byte> &out,InStream *in)
{
	if(!in) { out.resize(0); return false; }

	out.resize(0);

	// TODO: add error checking
	byte buffer[16*1024];
	do {
		int len = in->ReadBytes(buffer,sizeof(buffer));
		if(len<=0) break;
		out.insert(out.end(),buffer+0,buffer+len);
	} while(true);

	return true;
}

bool GetStreamLines(vector<string> &out,InStream *in)
{
	if(!in) { out.resize(0); return false; }

	out.resize(1);
	out[0].clear();

	// TODO: add error checking
	char buffer[16*1024];
	do {
		int len = in->ReadBytes(buffer,sizeof(buffer));
		if(len<=0) break;

		char *s = buffer;
		char *e = buffer + len;
		while(s<e)
		{
			while(s<e && *s=='\r') s++;
			if(s<e && *s=='\n') { out.push_back(string()); s++; continue; }

			char *b = s;
			while(s<e && *s!='\n' && *s!='\r')
				s++;
			if(b<s)
				out.back().append(b,s);
		}
	} while(true);

	return true;
}


bool GetStreamText(string &out,InStream *in)
{
	out.clear();
	if(!in) return false;

	// TODO: add error checking
	char buffer[16*1024];
	do {
		int len = in->ReadBytes(buffer,sizeof(buffer));
		if(len<=0) break;

		char *s = buffer;
		char *e = buffer + len;

		while(s<e)
		{
			while(s<e && *s=='\r') s++;
			char *p = s;
			while(s<e && *s!='\r') s++;
			out.append(p,s);
		}
	} while(true);

	return true;
}



}

// ******************************** strings.cpp ********************************

// ---- #include "base.h"
// ---> including base.h
// <--- back to strings.cpp

using namespace std;




namespace base
{



string format(const char *fmt, ...)
{
	string tmp;
	va_list arg;
	va_start(arg,fmt);
	vsprintf(tmp,fmt,arg);
	va_end(arg);
	return tmp;
}


string vformat(const char *fmt, va_list arg)
{
	string tmp;
	vsprintf(tmp,fmt,arg);
	return tmp;
}


void sprintf(string &out,const char *fmt, ...)
{
	va_list arg;
	va_start(arg,fmt);
	vsprintf(out,fmt,arg);
	va_end(arg);
}


void vsprintf(string &out,const char *fmt, va_list arg)
{
	va_list start = arg;
	int len = vsnprintf(NULL,0,fmt,arg) + 1;
	out.resize(len);
	vsnprintf((char*)out.data(),len,fmt,start);
	out.resize(len-1);
}



}

// ******************************** treefile.cpp ********************************

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>

// ---- #include "base.h"
// ---> including base.h
// <--- back to treefile.cpp


using namespace std;



namespace base {


// ---------------- TreeFileNode ----------------


TreeFileNode::TreeFileNode(const char *_name,int _id,bool copy_name)
{
	owns_name = copy_name;
	if(copy_name)
	{
		int len = strlen(_name);
		name = new char[len+1];
		memcpy((char*)name,_name,len+1);
	}
	else
		name = _name;

	clone_id = _id;
	invalid = false;

	type = T_VOID;
	ldata = NULL;
	size = 0;
	owns_data = false;
}

TreeFileNode::~TreeFileNode()
{
	map<NameRef,TreeFileNode *>::iterator p,prev;

	p = children.begin();
	while(p!=children.end())
	{
		TreeFileNode *n = p->second;
		prev = p;
		++p;

		children.erase(prev);
		delete n;
	}

	if(name && owns_name) delete name;
	if(ldata && owns_data) delete ldata;
}

TreeFileNode *TreeFileNode::GetChild(const char *cname,int cid,bool create,bool copy_name)
{
	NameRef key;
	key.name = cname;
	key.clone_id = cid;

	map<NameRef,TreeFileNode *>::iterator child = children.find(key);

	if(child != children.end())
		return child->second;

	if(!create)
		return NULL;

	TreeFileNode *n = new TreeFileNode(cname,cid,copy_name);
	key.name = n->name;

	children[key] = n;

	return n;
}

bool TreeFileNode::Validate()
{
	if(invalid) return false;

	// call recursively
	map<NameRef,TreeFileNode *>::iterator p;

	for(p = children.begin();p!=children.end();++p)
		if(!p->second->Validate())
			return false;

	return true;
}

void TreeFileNode::SetData(const byte *new_data,int new_size,int new_type,bool copy)
{
	if(ldata && owns_data) delete ldata;
	ldata = NULL;
	owns_data = false;
    size = new_size;
    type = new_type;

	if(!copy)
	{
		ldata = (byte*)new_data;
		return;
	}

    byte *ptr = NULL;

    if(new_size<=SHORT_DATA_MAX)
        ptr = sdata;
    else
        ptr = ldata = (new_size>0) ? new byte[new_size] : NULL;

    owns_data = true;
    if(new_data)    memcpy(ptr,new_data,new_size);
    else            memset(ptr,0,new_size);
}

bool TreeFileNode::GetInt(int &out)
{
	if(type==T_INT)
	{
		out = *(int*)GetData();
		return true;
	}

	if(type==T_STRING)
	{
        string tmp((char*)GetData(),GetSize());
        const char *s = tmp.c_str();
        out = ParseInt(s);
        ParseWhitespace(s);
        return (*s==0);
	}

	return false;
}

bool TreeFileNode::GetFloat(float &out)
{
	if(type==T_FLOAT)
	{
		out = *(float*)GetData();
		return true;
	}

	if(type==T_STRING)
	{
		string tmp((char*)GetData(),GetSize());
        const char *s = tmp.c_str();
        out = ParseFloat(s);
        ParseWhitespace(s);
		return (*s==0);
	}

	return false;
}

bool TreeFileNode::GetString(std::string &out)
{
	if(type==T_STRING)
	{
		out.assign((char*)GetData(),GetSize());
		return true;
	}

	if(type==T_INT)
	{
        out.clear();
        AppendInt(out,*(int*)GetData());
		return true;
	}

	if(type==T_FLOAT)
	{
        out.clear();
        AppendFloat(out,*(float*)GetData());
		return true;
	}

    return false;
}





// ---------------- TreeFileRef ----------------


TreeFileRef::TreeFileRef()
{
	node = NULL;
	is_writing = false;
}

TreeFileRef::TreeFileRef(TreeFileNode *n,bool is_wr)
{
	node = n;
	is_writing = is_wr;
}

TreeFileRef::TreeFileRef(const TreeFileRef &f)
{
	node = f.node;
	is_writing = f.is_writing;
}

TreeFileRef::~TreeFileRef()
{
}

int TreeFileRef::GetCloneArraySize(const char *name)
{
	if(!node) return 0;

	int id = 0;
	while(1)
	{
		if(!node->GetChild(name,id,false))
			break;
		id++;
	}
	return id;
}

void TreeFileRef::SerializeBasic(const char *name,void *v,int size,bool sign,int def,int id)
{
	if(!is_writing)
		memcpy(v,&def,size);

	TreeFileNode *n = GetChild(name,id);
	if(!n) return;

	int i,m;

	if(is_writing)
	{
		i=0;
		memcpy(&i,v,size);

		if(sign)
		{
			m=0xFFFFFFFF<<(size*8-1);
			if(i&m)
				i|=m;
		}
		n->SetInt(i);
	}
	else
	{
		if(n->GetInt(i))
			memcpy(v,&i,size);
	}
}

void TreeFileRef::SerFloat(const char *name,float &v,float def,int id)
{
	if(!is_writing)
		v = def;

	TreeFileNode *n = GetChild(name,id);
	if(!n) return;

	if(is_writing)
		n->SetFloat(v);
	else
		n->GetFloat(v);
}

void TreeFileRef::SerPChar(const char *name,char *str,int len,const char *def,int id)
{
	TreeFileNode *n = GetChild(name,id);
	if(!n)
	{
		if(!is_writing)
		{
			strncpy(str,def,len-1);
			str[len-1]=0;
		}
		return;
	}

	if(is_writing)
		n->SetString(str);
	else
	{
		string s;
		const char *src = def;
		if(n->GetString(s))
			src = s.c_str();
		strncpy(str,src,len-1);
		str[len-1]=0;
	}
}

void TreeFileRef::SerString(const char *name,std::string &out,const char *def,int id)
{
	if(is_writing)
	{
		SerPChar(name,(char*)out.c_str(),out.length()+1,def,id);
		return;
	}

	TreeFileNode *n = GetChild(name,id);
	if(!n)
	{
		out = def;
		return;
	}

	if(!n->GetString(out))
		out = def;
}

void TreeFileRef::Write_SetRaw(const char *name,const void *data,int size,int id)
{
	if(!is_writing) return;
	if(size<0) size = 0;

	TreeFileNode *n = GetChild(name,id);
	if(!n) return;

	n->SetData((const unsigned char*)data,size,TreeFileNode::T_RAW);
}

int TreeFileRef::Read_GetRawSize(const char *name,int id)
{
	if(is_writing) return 0;

	TreeFileNode *n = GetChild(name,id);
	if(!n || n->type!=TreeFileNode::T_RAW) return 0;

	return n->GetSize();
}

const void *TreeFileRef::Read_GetRawData(const char *name,int id)
{
	if(is_writing) return 0;

	TreeFileNode *n = GetChild(name,id);
	if(!n || n->type!=TreeFileNode::T_RAW) return NULL;

	return n->GetData();
}



TreeFileRef TreeFileRef::SerChild(const char *name,int id)
{
	return TreeFileRef(GetChild(name,id),is_writing);
}

TreeFileNode *TreeFileRef::GetChild(const char *name,int id)
{
	if(!node)	return NULL;
	if(!name)	return node;

	const char *s = name;
	while(*s && *s!='[') s++;
	if(*s=='[')
	{
		s++;
		id += ParseInt(s);
	}

	return node->GetChild(name,id,is_writing);
}



// ---------------- TreeFileBuilder ----------------

TreeFileBuilder::TreeFileBuilder()
{
	root = NULL;
}

TreeFileBuilder::~TreeFileBuilder()
{
	Clear();
}


TreeFileRef TreeFileBuilder::GetRoot(bool write)
{
	if(write && root==NULL)
	{
		Clear();
		root = new TreeFileNode("",0);
	}

	return TreeFileRef(root,write);
}

void TreeFileBuilder::Clear()
{
	if(root)
		delete root;
	root = NULL;
}

bool TreeFileBuilder::LoadTreeTxt(const char *path,FileSystem &fs)
{
    Clear();
    TreeFileRef tf = GetRoot(true);

    if(!tf.IsValid()) return false;

    vector<byte> data;
    if(!fs.GetFileBytes(path,data))
        return false;

    data.push_back(0);

    const char *s = (const char*)&data[0];
    map<string,int> last_index;
    TextFileContext ctx;
    ctx.auto_index = 0x40000000;
    ctx.base_path = FilePathGetPart(path,true,false,false);
    ctx.fs = &fs;
    ctx.last_index = &last_index;

    while( *s )
    {
        const char *p = s;
        ReadNode_Text(tf.node,s,ctx);
        if(s<=p) break;
    }

    if(!root->Validate())
        return false;

    return true;
}

bool TreeFileBuilder::SaveTreeTxt(const char *path)
{
	FILE *file = fopen(path, "wt");

	if(!file) return false;
	
	TreeFileNode *n = root;
	map<TreeFileNode::NameRef,TreeFileNode *>::iterator p;
    map<string,int> li;
	for(p = n->children.begin();p!=n->children.end();++p)
		WriteNode_Text(p->second,file,0,&li);
	fclose(file);

	return true;
}

bool TreeFileBuilder::LoadTreeBin(InStream *in)
{
    Clear();
	TreeFileRef tf = GetRoot(true);

	dictionary_hits.clear();
	dictionary.clear();

	bool ok = true;
	if(ok) ok = tf.IsValid();
	if(ok) ok = ReadDictionary_Binary(*in);
	if(ok)
	{
		int cnt = 0;
		in->Read(&cnt,4);
		for(int i=0;ok && i<cnt;i++)
			ok = ReadNode_Binary(tf.node,*in);
	}
	if(ok) ok = root->Validate();

	dictionary_hits.clear();
	dictionary.clear();

	return ok;
}

bool TreeFileBuilder::SaveTreeBin(OutStream *out)
{
	// start
	TreeFileRef tf = GetRoot(false);
	if(!tf.IsValid()) return false;

	// build dictionary
	BuildDictionary(tf);

	// write data
	bool ok = true;
	if(ok) ok = WriteDictionary_Binary(*out);
	if(ok)
	{
		int cnt = tf.node->children.size();
		out->Write(&cnt,4);
		for(TreeFileRef::iterator p = tf.begin();ok && p!=tf.end();++p)
			ok = WriteNode_Binary(*p,*out);
	}

	// cleanup
	dictionary_hits.clear();
	dictionary.clear();

	return ok;
}

bool TreeFileBuilder::ReadNode_Binary(TreeFileNode *parent, InStream &in)
{
	int flags = 0;
	in.Read(&flags,2);

	if((flags&0xF000)!=0xB000)
		return false;

	// flags:	1011ccssiinntttt
	//	c - children info	(00: no children,	01: unsigned char, 10: unsigned short, 11: int)
	//	s - size info		(00: size = 0 or 4,	01: unsigned char, 10: unsigned short, 11: int)
	//	i - clone id info	(00: clone_id = 0,	01: unsigned char, 10: unsigned short, 11: int)
	//	n - name id info	(00: reserved,		01: unsigned char, 10: unsigned short, 11: int)
	//	t - type id

	static const int WIDTH[4] = { 0, 1, 2, 4 };
	int type = flags&0x000F;
	int n_children = 0;
	int data_size = (type==TreeFileNode::T_VOID) ? 0 : 4;
	int clone_id = 0;
	int name_id = 0;

	if(type>4) return false;
	if((flags&0x0030)==0) return false;

	in.Read(&name_id	,WIDTH[(flags>> 4)&3]);
	in.Read(&clone_id	,WIDTH[(flags>> 6)&3]);
	in.Read(&data_size	,WIDTH[(flags>> 8)&3]);
	in.Read(&n_children	,WIDTH[(flags>>10)&3]);

	if(name_id<0 || name_id>=(int)dictionary.size())
		return false;

	TreeFileNode *node = parent->GetChild(dictionary[name_id].c_str(),clone_id,true);
	node->SetData(NULL,data_size,type);
	if(node->GetSize()!=data_size)
		return false;

	in.Read(node->GetData(),node->GetSize());

	for(int i=0;i<n_children;i++)
		if(!ReadNode_Binary(node,in))
			return false;
	
	return !in.WasError();
}

bool TreeFileBuilder::WriteNode_Binary(TreeFileRef tf, OutStream &out)
{
	// flags:	1011ccssiinntttt
	//	c - children info	(00: no children,	01: unsigned char, 10: unsigned short, 11: int)
	//	s - size info		(00: size = 0 or 4,	01: unsigned char, 10: unsigned short, 11: int)
	//	i - clone id info	(00: clone_id = 0,	01: unsigned char, 10: unsigned short, 11: int)
	//	n - name id info	(00: reserved,		01: unsigned char, 10: unsigned short, 11: int)
	//	t - type id

	int flags = 0xB000;
	int type = tf.node->type;
	int n_children = tf.node->children.size();
	int data_size = tf.node->GetSize();
	int clone_id = tf.GetId();
	int name_id = dictionary_hits[tf.GetName()];
	int wc,ws,wi,wn;

	flags |= type;

	if(name_id<0x100)			flags |= 0x0010, wn = 1;
	else if(name_id<0x10000)	flags |= 0x0020, wn = 2;
	else						flags |= 0x0030, wn = 4;

	if(clone_id==0)				flags |= 0x0000, wi = 0;
	else if(clone_id<0x100)		flags |= 0x0040, wi = 1;
	else if(clone_id<0x10000)	flags |= 0x0080, wi = 2;
	else						flags |= 0x00C0, wi = 4;

	int ds = (type==TreeFileNode::T_VOID) ? 0 : 4;
	if(data_size==ds)			flags |= 0x0000, ws = 0;
	else if(data_size<0x100)	flags |= 0x0100, ws = 1;
	else if(data_size<0x10000)	flags |= 0x0200, ws = 2;
	else						flags |= 0x0300, ws = 4;

	if(n_children==0)			flags |= 0x0000, wc = 0;
	else if(n_children<0x100)	flags |= 0x0400, wc = 1;
	else if(n_children<0x10000)	flags |= 0x0800, wc = 2;
	else						flags |= 0x0C00, wc = 4;

	out.Write(&flags,2);
	out.Write(&name_id		,wn);
	out.Write(&clone_id		,wi);
	out.Write(&data_size	,ws);
	out.Write(&n_children	,wc);

	out.Write(tf.node->GetData(),data_size);

	for(TreeFileRef::iterator p = tf.begin();p!=tf.end();++p)
		if(!WriteNode_Binary(*p,out))
			return false;

	return !out.WasError();
}


#define TEXT_SPECIALS       ":;*@$[]{}"


bool TreeFileBuilder::ReadNode_Text(TreeFileNode *parent,const char *&s,TextFileContext &ctx)
{
    string name, value;
    int index = 0;

    ParseWhitespace(s);
    if(!*s || *s=='}')
        return false;

    if(*s=='@')
    {
        s++;
        ParseStringT(s,name,TEXT_SPECIALS);
        ParseWhitespace(s);
        if(*s==':') s++;
        ParseStringT(s,value,TEXT_SPECIALS);
        ParseWhitespace(s);
        if(*s==';') s++;
        ctx.defines[name] = value;
        return true;
    }

    if(*s=='$')
    {
        s++;
        ParseStringT(s,name,TEXT_SPECIALS);
        ParseWhitespace(s);
        if(*s==':') s++;
        ParseStringT(s,value,";");
        ParseWhitespace(s);
        if(*s==';') s++;
        if(name=="include")
        {
            string path = ctx.base_path + value;

            vector<byte> data;
            if(ctx.fs->GetFileBytes(path.c_str(),data))
            {
                data.push_back(0);

                const char *q = (const char*)&data[0];
                while( *q )
                {
                    const char *p = q;
                    ReadNode_Text(parent,q,ctx);
                    if(q<=p) break;
                }
            }
        }
        return true;
    }
    
    ParseStringT(s,name,TEXT_SPECIALS);

    ParseWhitespace(s);
    if(*s=='[')
    {
        s++;
        ParseWhitespace(s);
        if(*s==']')
        {
            index = ++(*ctx.last_index)[name];
            s++;
        }
        else
        {
            (*ctx.last_index)[name] = index = ParseInt(s);
            ParseWhitespace(s);
            if(*s==']') s++;
        }
    }
    else if(*s=='*')
    {
        s++;
        index = ctx.auto_index++;
    }

    TreeFileNode *node = parent ? parent->GetChild(name.c_str(),index,true) : NULL;

    if(*s==':')
    {
        s++;
        ParseWhitespace(s);
        if(*s=='$')
        {
            s++;

            vector<byte> buff;
            ParseHexBuffer(s,buff);
            if(buff.size()>0)   node->SetData(&buff[0],buff.size(),TreeFileNode::T_RAW);
            else                node->SetData(NULL,0,TreeFileNode::T_RAW);
        }
        else
        {
            ParseStringT(s,value,";{}");
            if(ctx.defines.find(value)!=ctx.defines.end())
                value = ctx.defines[value];
            if(node) node->SetString(value.c_str());
        }
    }
    ParseWhitespace(s);
    if(*s=='{')
    {
        s++;

        map<string,int> last_index, *_li;
        _li = ctx.last_index;
        ctx.last_index = &last_index;

        while(ReadNode_Text(node,s,ctx)) {}
        ParseWhitespace(s);
        if(*s=='}') s++;

        ctx.last_index = _li;
    }

    ParseWhitespace(s);
    if(*s==';') s++;

    return true;
}


#define WRITE_STR(str) fwrite(str.c_str(), 1, str.length(), file)

bool TreeFileBuilder::WriteNode_Text(TreeFileNode *n,FILE *file,int indent,std::map<std::string,int> *last_index)
{
	string out;
	
    for(int i=0; i<indent; i++)
        out.push_back('\t');

    AppendString(out,n->name,TEXT_SPECIALS);

    if(n->clone_id != 0) 
    {
        out.push_back('[');
        if(n->clone_id!=(*last_index)[n->name]+1)
            AppendInt(out,n->clone_id);
        out.push_back(']');
    }
    (*last_index)[n->name] = n->clone_id;

    string value;
    if(n->type==TreeFileNode::T_RAW)
    {
        out += ": $";
        AppendHexBuffer(out,n->GetData(),n->GetSize());
    }
    else if(n->GetString(value))
    {
        out += ": ";
        AppendString(out,value.c_str(),TEXT_SPECIALS);
    }

	if(n->children.size()>0)
	{
        out += " {\n";
		WRITE_STR(out);

		// call recursively
        map<string,int> li;

		map<TreeFileNode::NameRef,TreeFileNode *>::iterator p;
		for(p = n->children.begin();p!=n->children.end();++p)
			WriteNode_Text(p->second,file,indent+1,&li);

        out.clear();
        for(int i=0; i<indent; i++)
            out.push_back('\t');
		out += "}\n";
	}
    else
        out += ";\n";

    WRITE_STR(out);
	
	return true;
}
#undef WRITE_STR




void TreeFileBuilder::BuildDictionary(TreeFileRef tf,bool root_level)
{
	if(root_level)
	{
		dictionary_hits.clear();
		dictionary.clear();
	}

	for(TreeFileRef::iterator p = tf.begin();p!=tf.end();++p)
	{
		dictionary_hits[(*p).GetName()]++;
		BuildDictionary( TreeFileRef( (*p).node, false ), false );
	}

	if(root_level)
	{
		for(map<string,int>::iterator p=dictionary_hits.begin();p!=dictionary_hits.end();++p)
			dictionary.push_back(p->first);

		class FreqSorter {
		public:
			map<string,int>	*freq;
			bool operator ()(const string &a,const string &b) const
			{ return (*freq)[a] > (*freq)[b]; }
		} fs;
		fs.freq = &dictionary_hits;
		sort(dictionary.begin(),dictionary.end(),fs);

		for(int i=0;i<(int)dictionary.size();i++)
			dictionary_hits[dictionary[i]] = i;
	}
}

bool TreeFileBuilder::ReadDictionary_Binary(InStream &in)
{
	dictionary.clear();

	int magic = 0;
	in.Read(&magic,4);
	if(magic!=0xA0A2A1A3) return false;

	string buffer;
	int cnt = 0;
	in.Read(&cnt,4);
	for(int i=0;i<cnt;i++)
	{
		int len = 0;
		in.Read(&len,2);
		if((int)buffer.size()<len)
			buffer.resize(len);
		in.Read(&buffer[0],len);
		dictionary.push_back(string(buffer.data(),len));
	}

	return !in.WasError();
}

bool TreeFileBuilder::WriteDictionary_Binary(OutStream &out)
{
	int magic = 0xA0A2A1A3;
	out.Write(&magic,4);

	int cnt = dictionary.size();
	out.Write(&cnt,4);
	for(int i=0;i<cnt;i++)
	{
		int len = dictionary[i].size();
		if(len>0xFFFF) return false;
		out.Write(&len,2);
		out.Write(dictionary[i].c_str(),len);
	}

	return !out.WasError();
}



}

// ******************************** vfs.cpp ********************************

#include <windows.h>

// ---- #include "base.h"
// ---> including base.h
// <--- back to vfs.cpp

using namespace std;



namespace base
{


NativeFileSystem NFS;




FileReaderStream *NativeFileSystem::GetFileAsStream(const char *path)
{
	FileReaderStream *fr = new FileReaderStream(path);
	if(fr->WasError())
	{
		delete fr;
		return NULL;
	}
	return fr;
}

bool NativeFileSystem::GetSubdirList(const char *path,std::vector<std::string> &out)
{
	out.clear();
	
	WIN32_FIND_DATA fdata;
	HANDLE h = FindFirstFile(path,&fdata);
	if(h==INVALID_HANDLE_VALUE) return false;

	do {
		if(fdata.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			out.push_back(fdata.cFileName);
	} while( FindNextFile(h,&fdata) != 0 );

	FindClose(h);

	return true;
}

bool NativeFileSystem::GetFileList(const char *path,std::vector<std::string> &out)
{
	out.clear();
	
	WIN32_FIND_DATA fdata;
	HANDLE h = FindFirstFile(path,&fdata);
	if(h==INVALID_HANDLE_VALUE) return false;

	do {
		if(!(fdata.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
			out.push_back(fdata.cFileName);
	} while( FindNextFile(h,&fdata) != 0 );

	FindClose(h);

	return true;
}

int NativeFileSystem::GetFileSize(const char *path)
{
	HANDLE h = CreateFile(path, GENERIC_READ, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if(h == INVALID_HANDLE_VALUE) return 0;

	int s = ::GetFileSize(h,NULL);
	if(s<0) s = 0;

	CloseHandle(h);

	return s;
}

bool NativeFileSystem::DumpRaw(const char *path,void *data,int size)
{
	FILE *fp = fopen(path,"wb");
	if(!fp) return false;
	fwrite(data,1,size,fp);
	fclose(fp);
	return true;
}




string FilePathGetPart(const char *path,bool dir,bool name,bool ext)
{
	const char *p = path;
	while(*p) p++;
	p--;
	while(p>=path && *p!='/' && *p!='\\') p--;
	p++;

	const char *dir_e = p;
	const char *name_b = (*p=='/' || *p=='\\') ? p+1 : p;
	while(*p && *p!='.') p++;
	const char *ext_b = p;

	if(dir)	if(name)	if(ext)	return string(path);								// dir	name	ext
						else	return string(path,ext_b);							// dir	name
			else		if(ext)	{ }													// dir			ext
						else	return string(path,dir_e);							// dir
	else	if(name)	if(ext)	return string(name_b);								//		name	ext
						else	return string(name_b,ext_b);						//		name
			else		if(ext)	return string(ext_b);								//				ext
						else	return string();									//

	// dir + ext
	string out(path,dir_e);
	out.append(ext_b);
	return out;
}


string GetCurrentDir()
{
	vector<char> buff;
	int len = GetCurrentDirectory(0,NULL);
	buff.resize(len+1);
	GetCurrentDirectory(len+1,&buff[0]);
	return string(&buff[0]);
}


unsigned long long GetFileTime(const char *path)
{
    WIN32_FILE_ATTRIBUTE_DATA a;
    if(!GetFileAttributesEx(path,GetFileExInfoStandard,&a))
        return 0;

    return (((unsigned long long)a.ftLastWriteTime.dwHighDateTime)<<32) | a.ftLastWriteTime.dwLowDateTime;
}



}

