#pragma once

#include <d3dx9math.h>
#include <math.h>
#include <iostream>

/*!
\brief 3D vector class with overloaded operators.
Used for positions, directions, colors, etc. It has the same memory layout as D3DXVECTOR3.
*/
class ShortVector {
public:
	//constants
	static const ShortVector ZERO;
	static const ShortVector ONE;

	//data can be accessed through various aliases
	union{
		signed short v[3];
		struct{ signed short x; signed short y; signed short z; };
		struct{ signed short r; signed short g; signed short b; };
		};

	ShortVector(){}

	ShortVector(const D3DXVECTOR3& d)
	{
		v[0] = d.x * SHRT_MAX;
		v[1] = d.y * SHRT_MAX;
		v[2] = d.z * SHRT_MAX;
	}

	ShortVector(const signed short x, const signed short y, const signed short z)
	{
		v[0] = x; v[1] = y; v[2] = z;
	}

	void set(const signed short x, const signed short y, const signed short z)
	{
		v[0] = x; v[1] = y; v[2] = z;
	}

	void setScaled(float s, const ShortVector& a)
	{
		v[0] = s * a[0]; v[1] = s * a[1]; v[2] = s * a[2];
	}

	void addScaled(float s, const ShortVector& a)
	{
		v[0] += s * a[0]; v[1] += s * a[1]; v[2] += s * a[2];
	}

	void clear()
	{
		v[0] = v[1] = v[2] = 0;
	}

	void setDifference(const ShortVector& a, const ShortVector& b)
	{
		v[0] = a.v[0] - b.v[0];
		v[1] = a.v[1] - b.v[1];
		v[2] = a.v[2] - b.v[2];
	}

	ShortVector operator-() const
	{
		return ShortVector(-v[0], -v[1], -v[2]);
	}

	ShortVector operator+(const ShortVector& addOperand) const
	{
		return ShortVector (	v[0] + addOperand.v[0],
						v[1] + addOperand.v[1],
						v[2] + addOperand.v[2]);
	}

	ShortVector operator-(const ShortVector& substractOperand) const
	{
		return ShortVector(  v[0] - substractOperand.v[0],
						v[1] - substractOperand.v[1],
						v[2] - substractOperand.v[2]);
	}

	void operator-=(const ShortVector& a)
	{
		v[0] -= a[0];
		v[1] -= a[1];
		v[2] -= a[2];
	}

	void operator+=(const ShortVector& a)
	{
		v[0] += a[0];
		v[1] += a[1];
		v[2] += a[2];
	}

	//blend operator
	void operator%=(const ShortVector& a)
	{
		v[0] *= a[0];
		v[1] *= a[1];
		v[2] *= a[2];
	}

	//scale operator
	void operator*=(const signed short scale)
	{
		v[0] *= scale;
		v[1] *= scale;
		v[2] *= scale;
	}

	//blend operator
	ShortVector operator%(const ShortVector& blendOperand) const
	{
		return ShortVector(	v[0] * blendOperand.v[0],
						v[1] * blendOperand.v[1],
						v[2] * blendOperand.v[2]);
	}

	//scale operator
	ShortVector operator*(float scale) const
	{
		return ShortVector(scale * v[0], scale * v[1], scale * v[2]);
	}

	/*
	//dot product operator
	signed short operator*(const ShortVector& dotProductOperand) const
	{
		return	v[0] * dotProductOperand[0] + 
				v[1] * dotProductOperand[1] +
				v[2] * dotProductOperand[2];
	}

	//cross product operator
	ShortVector operator&&(const ShortVector& crossProductOperand) const
	{
		return ShortVector(
			v[1] * crossProductOperand[2] - v[2] * crossProductOperand[1],
			v[2] * crossProductOperand[0] - v[0] * crossProductOperand[2],
			v[0] * crossProductOperand[1] - v[1] * crossProductOperand[0]);
	}

	void setCrossProduct(const ShortVector& a, const ShortVector& b)
	{
		v[0] = a[1] * b[2] - a[2] * b[1];
		v[1] = a[2] * b[0] - a[0] * b[2];
		v[2] = a[0] * b[1] - a[1] * b[0];
	}

	signed short norm () const
	{
		return sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
	}

	signed short norm2 () const 
	{
		return v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
	}

	void normalize ()
	{
		signed short length = 1.0f / sqrtf (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
		v[0] *= length;
		v[1] *= length;
		v[2] *= length;
	}
*/
	signed short sum () const
	{
		return v[0] + v[1] + v[2];
	}

	const ShortVector& operator=(const ShortVector& other)
	{
		v[0] = other.v[0];
		v[1] = other.v[1];
		v[2] = other.v[2];
		return *this;
	}

	signed short operator[](const int index) const
	{
		return v[index%3];
	}

	//accumulate minimum operator
	ShortVector& operator<= ( const ShortVector& zsmall)
	{
		if(v[0] > zsmall[0]) v[0] = zsmall[0];
		if(v[1] > zsmall[1]) v[1] = zsmall[1];
		if(v[2] > zsmall[2]) v[2] = zsmall[2];
		return *this;
	}

	//accumulate maximum operator
	ShortVector& operator>= ( const ShortVector& large)
	{
		if(v[0] < large[0]) v[0] = large[0];
		if(v[1] < large[1]) v[1] = large[1];
		if(v[2] < large[2]) v[2] = large[2];
		return *this;
	}

	D3DXVECTOR3 asFloatVector()
	{
		D3DXVECTOR3 r(1, 1, 1);
		r.x *= (float)x;
		r.y *= (float)y;
		r.z *= (float)z;
		return r / SHRT_MAX;
	}

	D3DXVECTOR3 deunitize( const D3DXVECTOR3& lengthPerBit)
	{
		D3DXVECTOR3 r = lengthPerBit;
		r.x *= (float)x;
		r.y *= (float)y;
		r.z *= (float)z;
		return r;
	}
};
