
// ******************************** base.h ********************************

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

