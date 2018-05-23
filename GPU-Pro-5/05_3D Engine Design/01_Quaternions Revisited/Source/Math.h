/****************************************************************************

  GPU Pro 5 : Quaternions revisited - sample code
  All sample code written from scratch by Sergey Makeev specially for article.

  Copyright (c) 2013, Sergey Makeev

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software.

  2. If you use this software in a non-commercial product, an acknowledgment
     in the product documentation would be appreciated but is not required.

  3. If you use this software in a commercial product you requried to credit
     the author.

  4. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.

  5. This notice may not be removed or altered from any source distribution.
    

   Please let me know if you use the code in your products or have any questions or suggestions.

   e-mail: sergeymakeev@inbox.ru
   http://linkedin.com/in/sergeymakeev/

****************************************************************************/
#pragma once

#include "Utils.h"
#include "Assert.h"
#include <fbxsdk.h>
#include <D3DX9.h>



//////////////////////////////////////////////////////////////////////////
struct Vector2
{
	float x;
	float y;

	Vector2() {}

	explicit Vector2(const FbxVector2 & v)
	{
		x = (float)v[0];
		y = (float)v[1];
	}
};

inline Vector2 operator- (const Vector2 & lhs, const Vector2 & rhs)
{
	Vector2 v;
	v.x = lhs.x - rhs.x;
	v.y = lhs.y - rhs.y;
	return v;
}



//////////////////////////////////////////////////////////////////////////
struct Vector3
{
	float x;
	float y;
	float z;

	Vector3() {}

	explicit Vector3(float v)
	{
		x = v;
		y = v;
		z = v;
	}

	explicit Vector3(float _x, float _y, float _z)
	{
		x = _x;
		y = _y;
		z = _z;
	}

	explicit Vector3(const Vector2 & v, float _z)
	{
		x = v.x;
		y = v.y;
		z = _z;
	}

	explicit Vector3(const FbxVector4 & v)
	{
		x = (float)v[0];
		y = (float)v[1];
		z = (float)v[2];
	}

	void Max(const Vector3 & v)
	{
		x = Utils::Max(x, v.x);
		y = Utils::Max(y, v.y);
		z = Utils::Max(z, v.z);
	}

	void Min(const Vector3 & v)
	{
		x = Utils::Min(x, v.x);
		y = Utils::Min(y, v.y);
		z = Utils::Min(z, v.z);
	}

	float Normalize()
	{
		float len = sqrtf(x*x + y*y + z*z);
		if (len < 0.000001f)
			return 0.0f;

		float invLen = 1.0f / len;
		x *= invLen;
		y *= invLen;
		z *= invLen;
		return len;
	}


};


inline const Vector3 operator- ( const Vector3 & vec)
{
	Vector3 v;
	v.x = -vec.x;
	v.y = -vec.y;
	v.z = -vec.z;
	return v;
}

inline Vector3 operator- (const Vector3 & lhs, const Vector3 & rhs)
{
	Vector3 v;
	v.x = lhs.x - rhs.x;
	v.y = lhs.y - rhs.y;
	v.z = lhs.z - rhs.z;
	return v;
}

inline Vector3 operator+ (const Vector3 & lhs, const Vector3 & rhs)
{
	Vector3 v;
	v.x = lhs.x + rhs.x;
	v.y = lhs.y + rhs.y;
	v.z = lhs.z + rhs.z;
	return v;
}

inline Vector3 operator* (const Vector3 & lhs, const Vector3 & rhs)
{
	Vector3 v;
	v.x = lhs.x * rhs.x;
	v.y = lhs.y * rhs.y;
	v.z = lhs.z * rhs.z;
	return v;
}

inline Vector3 cross(const Vector3 & lhs, const Vector3 & rhs)
{
	Vector3 v;
	v.x = lhs.y * rhs.z - rhs.y * lhs.z;
	v.y = lhs.z * rhs.x - rhs.z * lhs.x;
	v.z = lhs.x * rhs.y - rhs.x * lhs.y;
	return v;
}

inline float dot(const Vector3 & lhs, const Vector3 & rhs)
{
	return ( lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z );
}

inline Vector3 lerp(const Vector3 & from, const Vector3 & to, float blendK)
{
	Vector3 r;
	r.x = from.x + (to.x - from.x)*blendK;
	r.y = from.y + (to.y - from.y)*blendK;
	r.z = from.z + (to.z - from.z)*blendK;
	return r;
}



//////////////////////////////////////////////////////////////////////////
struct Matrix4x3
{
	float m[4][3];

	Matrix4x3() {}

	explicit Matrix4x3(const FbxAMatrix & fbxMatrix )
	{
		FbxVector4 __checkPos = fbxMatrix.GetT();

		m[0][0] = (float)fbxMatrix.Get(0, 0);
		m[0][1] = (float)fbxMatrix.Get(0, 1);
		m[0][2] = (float)fbxMatrix.Get(0, 2);

		m[1][0] = (float)fbxMatrix.Get(1, 0);
		m[1][1] = (float)fbxMatrix.Get(1, 1);
		m[1][2] = (float)fbxMatrix.Get(1, 2);

		m[2][0] = (float)fbxMatrix.Get(2, 0);
		m[2][1] = (float)fbxMatrix.Get(2, 1);
		m[2][2] = (float)fbxMatrix.Get(2, 2);

		m[3][0] = (float)fbxMatrix.Get(3, 0);
		m[3][1] = (float)fbxMatrix.Get(3, 1);
		m[3][2] = (float)fbxMatrix.Get(3, 2);
	}

	Vector3 GetX() const
	{
		return Vector3(m[0][0], m[0][1], m[0][2]);
	}

	Vector3 GetY() const
	{
		return Vector3(m[1][0], m[1][1], m[1][2]);
	}

	Vector3 GetZ() const
	{
		return Vector3(m[2][0], m[2][1], m[2][2]);
	}

	Vector3 GetTranslate() const
	{
		return Vector3(m[3][0], m[3][1], m[3][2]);
	}

	void SetX(const Vector3 & v)
	{
		m[0][0] = v.x;
		m[0][1] = v.y;
		m[0][2] = v.z;
	}

	void SetY(const Vector3 & v)
	{
		m[1][0] = v.x;
		m[1][1] = v.y;
		m[1][2] = v.z;
	}

	void SetZ(const Vector3 & v)
	{
		m[2][0] = v.x;
		m[2][1] = v.y;
		m[2][2] = v.z;
	}

	void SetTranslate(const Vector3 & t)
	{
		m[3][0] = t.x;
		m[3][1] = t.y;
		m[3][2] = t.z;
	}

	Vector3 TrasnformVertex(const Vector3 & v) const
	{
		Vector3 r;
		r.x = m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z + m[3][0];
		r.y = m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z + m[3][1];
		r.z = m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z + m[3][2];
		return r;
	}

	Vector3 TrasnformNormal(const Vector3 & v) const
	{
		Vector3 r;
		r.x = m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z;
		r.y = m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z;
		r.z = m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z;
		return r;
	}

	D3DXMATRIX & ConvertToD3DMatrix(D3DXMATRIX & r) const
	{
		r.m[0][0] = m[0][0];
		r.m[0][1] = m[0][1];
		r.m[0][2] = m[0][2];
		r.m[0][3] = 0.0f;

		r.m[1][0] = m[1][0];
		r.m[1][1] = m[1][1];
		r.m[1][2] = m[1][2];
		r.m[1][3] = 0.0f;

		r.m[2][0] = m[2][0];
		r.m[2][1] = m[2][1];
		r.m[2][2] = m[2][2];
		r.m[2][3] = 0.0f;

		r.m[3][0] = m[3][0];
		r.m[3][1] = m[3][1];
		r.m[3][2] = m[3][2];
		r.m[3][3] = 1.0f;
		return r;
	}

	bool Invert(Matrix4x3 & res) const
	{
		const float m1223 = m[2][1] * m[3][2] - m[2][2] * m[3][1];
		const float m1123 = m[1][1] * m[3][2] - m[1][2] * m[3][1];
		const float m0223 = m[2][0] * m[3][2] - m[2][2] * m[3][0];
		const float m0123 = m[1][0] * m[3][2] - m[1][2] * m[3][0];
		const float m0213 = m[2][0] * m[3][1] - m[2][1] * m[3][0];
		const float m0113 = m[1][0] * m[3][1] - m[1][1] * m[3][0];

		// Adjoint Matrix
		res.m[0][0] =  m[1][1] * m[2][2] - m[1][2] * m[2][1];
		res.m[0][1] = -m[0][1] * m[2][2] + m[0][2] * m[2][1];
		res.m[0][2] =  m[0][1] * m[1][2] - m[0][2] * m[1][1];

		res.m[1][0] = -m[1][0] * m[2][2] + m[1][2] * m[2][0];
		res.m[1][1] =  m[0][0] * m[2][2] - m[0][2] * m[2][0];
		res.m[1][2] = -m[0][0] * m[1][2] + m[0][2] * m[1][0];

		res.m[2][0] =  m[1][0] * m[2][1] - m[1][1] * m[2][0];
		res.m[2][1] = -m[0][0] * m[2][1] + m[0][1] * m[2][0];
		res.m[2][2] =  m[0][0] * m[1][1] - m[0][1] * m[1][0];

		res.m[3][0] = -m[1][0] * m1223 + m[1][1] * m0223 - m[1][2] * m0213;
		res.m[3][1] =  m[0][0] * m1223 - m[0][1] * m0223 + m[0][2] * m0213;
		res.m[3][2] = -m[0][0] * m1123 + m[0][1] * m0123 - m[0][2] * m0113;

		// Division by determinant
		float fDet = m[0][0] * res.m[0][0] + m[0][1] * res.m[1][0] + m[0][2] * res.m[2][0];
		
		if ( fabsf( fDet ) < 1e-16 )
		{
			// Singular matrix found !
			return false;
		}

		fDet = 1.0f / fDet;
		res.m[0][0] *= fDet;   res.m[0][1] *= fDet;   res.m[0][2] *= fDet;
		res.m[1][0] *= fDet;   res.m[1][1] *= fDet;   res.m[1][2] *= fDet;
		res.m[2][0] *= fDet;   res.m[2][1] *= fDet;   res.m[2][2] *= fDet;
		
		res.m[3][0] *= fDet;   res.m[3][1] *= fDet;   res.m[3][2] *= fDet;

		return true;
	}

	static Matrix4x3 Identity()
	{
		Matrix4x3 m;
		m.m[0][0] = 1.0f;   m.m[0][1] = 0.0f;   m.m[0][2] = 0.0f;
		m.m[1][0] = 0.0f;   m.m[1][1] = 1.0f;   m.m[1][2] = 0.0f;
		m.m[2][0] = 0.0f;   m.m[2][1] = 0.0f;   m.m[2][2] = 1.0f;
		m.m[3][0] = 0.0f;   m.m[3][1] = 0.0f;   m.m[3][2] = 0.0f;
		return m;
	}
};


inline Matrix4x3 operator* (const Matrix4x3 & m1, const Matrix4x3 & m2)
{
	Matrix4x3 r;
	r.m[0][0] = m2.m[0][0]*m1.m[0][0] + m2.m[1][0]*m1.m[0][1] + m2.m[2][0]*m1.m[0][2];
	r.m[0][1] = m2.m[0][1]*m1.m[0][0] + m2.m[1][1]*m1.m[0][1] + m2.m[2][1]*m1.m[0][2];
	r.m[0][2] = m2.m[0][2]*m1.m[0][0] + m2.m[1][2]*m1.m[0][1] + m2.m[2][2]*m1.m[0][2];

	r.m[1][0] = m2.m[0][0]*m1.m[1][0] + m2.m[1][0]*m1.m[1][1] + m2.m[2][0]*m1.m[1][2];
	r.m[1][1] = m2.m[0][1]*m1.m[1][0] + m2.m[1][1]*m1.m[1][1] + m2.m[2][1]*m1.m[1][2];
	r.m[1][2] = m2.m[0][2]*m1.m[1][0] + m2.m[1][2]*m1.m[1][1] + m2.m[2][2]*m1.m[1][2];

	r.m[2][0] = m2.m[0][0]*m1.m[2][0] + m2.m[1][0]*m1.m[2][1] + m2.m[2][0]*m1.m[2][2];
	r.m[2][1] = m2.m[0][1]*m1.m[2][0] + m2.m[1][1]*m1.m[2][1] + m2.m[2][1]*m1.m[2][2];
	r.m[2][2] = m2.m[0][2]*m1.m[2][0] + m2.m[1][2]*m1.m[2][1] + m2.m[2][2]*m1.m[2][2];

	r.m[3][0] = m2.m[0][0]*m1.m[3][0] + m2.m[1][0]*m1.m[3][1] + m2.m[2][0]*m1.m[3][2] + m2.m[3][0];
	r.m[3][1] = m2.m[0][1]*m1.m[3][0] + m2.m[1][1]*m1.m[3][1] + m2.m[2][1]*m1.m[3][2] + m2.m[3][1];
	r.m[3][2] = m2.m[0][2]*m1.m[3][0] + m2.m[1][2]*m1.m[3][1] + m2.m[2][2]*m1.m[3][2] + m2.m[3][2];

	return r;
}



//////////////////////////////////////////////////////////////////////////
struct Quaternion
{
	enum CheckSourceResultFlag
	{
		SOURCE_BASIS_LEFT_HANDED = 0x1,
		SOURCE_BASIS_HAVE_SCALE = 0x2
	};

	static unsigned long CheckQuaternionSource(const Vector3 & x, const Vector3 & y, const Vector3 & z)
	{
		unsigned long flag = 0;
		float basisCheck = dot( cross( x, y ), z );
		if (basisCheck < 0.0f)
		{
			flag |= SOURCE_BASIS_LEFT_HANDED;
		}
		if (abs( 1.0f - abs( basisCheck ) ) >= 0.1f)
		{
			flag |= SOURCE_BASIS_HAVE_SCALE;
		}
		return flag;
	}


	float x;
	float y;
	float z;
	float w;

	Quaternion() {}

	explicit Quaternion(const Matrix4x3 & rotationMatrix)
	{
		ASSERT( CheckQuaternionSource( rotationMatrix.GetX(), rotationMatrix.GetY(), rotationMatrix.GetZ() ) == 0, "Source matrix invalid. Quaternions should be constructed from orthogonal, normalized, right-handed basis." );

		// First compute squared magnitudes of quaternion components - at least one
		// will be greater than 0 since quaternion is unit magnitude
		float qs2 = 0.25f * (rotationMatrix.m[0][0] + rotationMatrix.m[1][1] + rotationMatrix.m[2][2] + 1.0f);
		float qx2 = qs2 - 0.5f * (rotationMatrix.m[1][1] + rotationMatrix.m[2][2]);
		float qy2 = qs2 - 0.5f * (rotationMatrix.m[2][2] + rotationMatrix.m[0][0]);
		float qz2 = qs2 - 0.5f * (rotationMatrix.m[0][0] + rotationMatrix.m[1][1]);

		// Find maximum magnitude component
		int n = (qs2 > qx2 ) ?
			((qs2 > qy2) ? ((qs2 > qz2) ? 0 : 3) : ((qy2 > qz2) ? 2 : 3)) :
			((qx2 > qy2) ? ((qx2 > qz2) ? 1 : 3) : ((qy2 > qz2) ? 2 : 3));

		// Compute signed quaternion components using numerically stable method
		float tmp;
		switch ( n )
		{
		case 0:
			w = sqrtf( qs2 );
			tmp = 0.25f / w;
			x = ( rotationMatrix.m[1][2] - rotationMatrix.m[2][1] ) * tmp;
			y = ( rotationMatrix.m[2][0] - rotationMatrix.m[0][2] ) * tmp;
			z = ( rotationMatrix.m[0][1] - rotationMatrix.m[1][0] ) * tmp;
			break;
		case 1:
			x = sqrtf( qx2 );
			tmp = 0.25f / x;
			w = ( rotationMatrix.m[1][2] - rotationMatrix.m[2][1] ) * tmp;
			y = ( rotationMatrix.m[1][0] + rotationMatrix.m[0][1] ) * tmp;
			z = ( rotationMatrix.m[2][0] + rotationMatrix.m[0][2] ) * tmp;
			break;
		case 2:
			y = sqrtf( qy2 );
			tmp = 0.25f / y;
			w = ( rotationMatrix.m[2][0] - rotationMatrix.m[0][2] ) * tmp;
			z = ( rotationMatrix.m[2][1] + rotationMatrix.m[1][2] ) * tmp;
			x = ( rotationMatrix.m[0][1] + rotationMatrix.m[1][0] ) * tmp;
			break;
		case 3:
			z = sqrtf( qz2 );
			tmp = 0.25f / z;
			w = ( rotationMatrix.m[0][1] - rotationMatrix.m[1][0] ) * tmp;
			x = ( rotationMatrix.m[0][2] + rotationMatrix.m[2][0] ) * tmp;
			y = ( rotationMatrix.m[1][2] + rotationMatrix.m[2][1] ) * tmp;
			break;
		}

		//Make positive W
		if ( w < 0 )
		{
			x = -x;
			y = -y;
			z = -z;
			w = -w;
		}
		Normalize();
	}

	void Normalize()
	{
		float len = sqrtf(x*x + y*y + z*z + w*w);

		if (len < 0.000001f)
			return;

		float invLen = 1.0f / len;

		x *= invLen;
		y *= invLen;
		z *= invLen;
		w *= invLen;
	}

	Quaternion Conjugate() const
	{
		Quaternion r;
		r.x = -x;
		r.y = -y;
		r.z = -z;
		r.w = w;
		return r;
	}

	Matrix4x3 AsRotationMatrix() const
	{
		Matrix4x3 res;

		float tx  = 2.0f * x;
		float ty  = 2.0f * y;
		float tz  = 2.0f * z;

		float twx = tx*w;
		float twy = ty*w;
		float twz = tz*w;
		float txx = tx*x;
		float txy = ty*x;
		float txz = tz*x;
		float tyy = ty*y;
		float tyz = tz*y;
		float tzz = tz*z;

		res.m[0][0] = 1.0f - (tyy + tzz);      res.m[1][0] = txy - twz;              res.m[2][0] = txz + twy;
		res.m[0][1] = txy + twz;               res.m[1][1] = 1.0f - (txx + tzz);     res.m[2][1] = tyz - twx;
		res.m[0][2] = txz - twy;               res.m[1][2] = tyz + twx;	             res.m[2][2] = 1.0f - (txx + tyy);
		return res;
	}

	static Quaternion Identity()
	{
		Quaternion r;
		r.x = 0.0f;
		r.y = 0.0f;
		r.z = 0.0f;
		r.w = 1.0f;
		return r;
	}

};



inline Quaternion operator* ( const Quaternion &a, const Quaternion &b )
{
	Quaternion result;
	result.x = a.w*b.x + b.w*a.x + (a.y*b.z - a.z*b.y);
	result.y = a.w*b.y + b.w*a.y + (a.z*b.x - a.x*b.z);
	result.z = a.w*b.z + b.w*a.z + (a.x*b.y - a.y*b.x);
	result.w = a.w*b.w - (a.x*b.x + a.y*b.y + a.z*b.z);
	result.Normalize();
	return result;
}


inline Quaternion lerp(const Quaternion & from, const Quaternion & to, float blendK)
{
	Quaternion r;
	r.x = from.x + (to.x - from.x)*blendK;
	r.y = from.y + (to.y - from.y)*blendK;
	r.z = from.z + (to.z - from.z)*blendK;
	r.w = from.w + (to.w - from.w)*blendK;
	return r;
}

inline float dot(const Quaternion & lhs, const Quaternion & rhs)
{
	return ( lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z + lhs.w * rhs.w );
}

inline const Quaternion operator- ( const Quaternion & q)
{
	Quaternion r;
	r.x = -q.x;
	r.y = -q.y;
	r.z = -q.z;
	r.w = -q.w;
	return r;
}

inline Quaternion slerp(const Quaternion & from, const Quaternion & to, float blendK)
{
	float cosomega = from.x * to.x + from.y * to.y + from.z * to.z + from.w * to.w;
	float angle = acosf(cosomega);

	Quaternion r;
	if (fabsf(angle) > 0.0f)
	{
		float sn = sinf(angle);
		float invSn = 1.0f/sn;
		float tAngle = blendK*angle;
		float coeff0 = sinf(angle - tAngle)*invSn;
		float coeff1 = sinf(tAngle)*invSn;

		r.x = coeff0 * from.x + coeff1 * to.x;
		r.y = coeff0 * from.y + coeff1 * to.y;
		r.z = coeff0 * from.z + coeff1 * to.z;
		r.w = coeff0 * from.w + coeff1 * to.w;
	}
	else
	{
		r.x = from.x;
		r.y = from.y;
		r.z = from.z;
		r.w = from.w;
	}
	return r;
}


inline Vector3 operator* (const Quaternion &q, const Vector3 &v)
{
	Vector3 qxyz = Vector3(q.x, q.y, q.z);
	Vector3 t = Vector3(2.0f) * cross( qxyz, v );
	Vector3 r = Vector3(q.w) * t;
	Vector3 tmp = cross( qxyz, t );
	r = r + tmp + v;
	return r;
}

