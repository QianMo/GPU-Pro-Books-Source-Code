#include "stdafx.h"
#include "Quaternion.h"
#include "Math.h"


const Quaternion Quaternion::ZERO(0.0f, 0.0f, 0.0f, 0.0f);
const Quaternion Quaternion::IDENTITY(0.0f, 0.0f, 0.0f, 1.0f);

// -----------------------------------------------------------------------------
// --------------------------- Quaternion::Quaternion --------------------------
// -----------------------------------------------------------------------------
Quaternion::Quaternion() :
    x(0), y(0), z(0), w(0)
{
}

// -----------------------------------------------------------------------------
// --------------------------- Quaternion::Quaternion --------------------------
// -----------------------------------------------------------------------------
Quaternion::Quaternion(float _x, float _y, float _z, float _w) :
	x(_x), y(_y), z(_z), w(_w)
{
}

// -----------------------------------------------------------------------------
// --------------------------- Quaternion::Quaternion --------------------------
// -----------------------------------------------------------------------------
Quaternion::Quaternion(const Quaternion& other) : 
	x(other.x), y(other.y), z(other.z), w(other.w)
{
}

// -----------------------------------------------------------------------------
// --------------------------- Quaternion::Quaternion --------------------------
// -----------------------------------------------------------------------------
Quaternion::Quaternion(const Matrix3& other)
{
	// alternative version from:
	// http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
	// appears to more accurate
	// average error of this method is: 0.000001
	// average error of alternative method is: 0.000252
	//w = Math::Sqrt(Math::Max(0.0f, 1.0f + other[0][0] + other[1][1] + other[2][2])) / 2.0f; 
	//x = Math::Sqrt(Math::Max(0.0f, 1.0f + other[0][0] - other[1][1] - other[2][2])) / 2.0f; 
	//y = Math::Sqrt(Math::Max(0.0f, 1.0f - other[0][0] + other[1][1] - other[2][2])) / 2.0f; 
	//z = Math::Sqrt(Math::Max(0.0f, 1.0f - other[0][0] - other[1][1] + other[2][2])) / 2.0f; 
	//x = Math::CopySign(x, other[1][2] - other[2][1]);
	//y = Math::CopySign(y, other[2][0] - other[0][2]);
	//z = Math::CopySign(z, other[0][1] - other[1][0]);

	// old method from magic:
	float fTrace = other[0][0]+other[1][1]+other[2][2];
	float fRoot;

	if (Math::IsBigger(fTrace, 0.0f))
	{
		// |w| > 1/2, may as well choose w > 1/2
		fRoot = Math::Sqrt(fTrace + 1.0f);  // 2w
		w = 0.5f*fRoot;
		fRoot = 0.5f/fRoot;  // 1/(4w)
		x = (other[1][2]-other[2][1])*fRoot;
		y = (other[2][0]-other[0][2])*fRoot;
		z = (other[0][1]-other[1][0])*fRoot;
	}
	else
	{
		// |w| <= 1/2
		int i = 0;
		if ( other[1][1] > other[0][0] )
			i = 1;
		if ( other[2][2] > other.entry[i+i*3] )
			i = 2;
		int j = (i+1)%3;
		int k = (j+1)%3;

		fRoot = Math::Sqrt(other.entry[i+i*3]-other.entry[j+j*3]-other.entry[k+k*3] + 1.0f);
		x = y = z = 0.0f;
		float* apkQuat[3] = { &x, &y, &z };
		*apkQuat[i] = 0.5f*fRoot;
		fRoot = 0.5f/fRoot;
		w = (other.entry[k+j*3]-other.entry[j+k*3])*fRoot;
		*apkQuat[j] = (other.entry[j+i*3]+other.entry[i+j*3])*fRoot;
		*apkQuat[k] = (other.entry[k+i*3]+other.entry[i+k*3])*fRoot;
	}
}

// -----------------------------------------------------------------------------
// --------------------------- Quaternion::Quaternion --------------------------
// -----------------------------------------------------------------------------
Quaternion::Quaternion(const Vector3& axis, float angle)
{
	FromAxisAngle(axis, angle);
}

// -----------------------------------------------------------------------------
// --------------------------- Quaternion::operator= ---------------------------
// -----------------------------------------------------------------------------
const Quaternion& Quaternion::operator=(const Quaternion& other)
{
	if ((&other == this)||(other == *this))
		return (*this);

	x = other.x;
	y = other.y;
	z = other.z;
	w = other.w;
	return (*this);
}

// -----------------------------------------------------------------------------
// --------------------------- Quaternion::operator== --------------------------
// -----------------------------------------------------------------------------
bool Quaternion::operator==(const Quaternion& other) const
{
	return ((x == other.x) && (y == other.y) && (z == other.z));
}

// -----------------------------------------------------------------------------
// --------------------------- Quaternion::operator!= --------------------------
// -----------------------------------------------------------------------------
bool Quaternion::operator!=(const Quaternion& other) const
{
	return ((x != other.x) || (y != other.y) || (z != other.z));
}

// -----------------------------------------------------------------------------
// --------------------------- Quaternion::operator- ---------------------------
// -----------------------------------------------------------------------------
Quaternion Quaternion::operator-() const
{
	return Quaternion(-x, -y, -z, -w);
}

// -----------------------------------------------------------------------------
// --------------------------- Quaternion::operator+ ---------------------------
// -----------------------------------------------------------------------------
Quaternion Quaternion::operator+(const Quaternion& other) const
{
	return Quaternion(x + other.x, y + other.y, z + other.z, w + other.w);
}

// -----------------------------------------------------------------------------
// --------------------------- Quaternion::operator+= --------------------------
// -----------------------------------------------------------------------------
const Quaternion& Quaternion::operator+=(const Quaternion& other)
{
	x += other.x;
	y += other.y;
	z += other.z;
	w += other.w;
	return (*this);
}

// -----------------------------------------------------------------------------
// --------------------------- Quaternion::operator- ---------------------------
// -----------------------------------------------------------------------------
Quaternion Quaternion::operator-(const Quaternion& other) const
{
	return Quaternion(x - other.x, y - other.y, z - other.z, w - other.w);
}

// -----------------------------------------------------------------------------
// --------------------------- Quaternion::operator-= --------------------------
// -----------------------------------------------------------------------------
const Quaternion& Quaternion::operator-=(const Quaternion& other)
{
	x -= other.x;
	y -= other.y;
	z -= other.z;
	w -= other.w;
	return (*this);
}

// -----------------------------------------------------------------------------
// --------------------------- Quaternion::operator* ---------------------------
// -----------------------------------------------------------------------------
Quaternion Quaternion::operator*(const Quaternion& other) const
{
    return Quaternion
        (        
        w*other.x+x*other.w+y*other.z-z*other.y,
        w*other.y+y*other.w+z*other.x-x*other.z,
        w*other.z+z*other.w+x*other.y-y*other.x,
        w*other.w-x*other.x-y*other.y-z*other.z
        );	
}

// -----------------------------------------------------------------------------
// --------------------------- Quaternion::operator* ---------------------------
// -----------------------------------------------------------------------------
Quaternion Quaternion::operator*(const float scalar) const
{
	return Quaternion(x * scalar, y * scalar, z * scalar, w * scalar);
}

// -----------------------------------------------------------------------------
// --------------------------- Quaternion::operator*= --------------------------
// -----------------------------------------------------------------------------
const Quaternion& Quaternion::operator*=(const float scalar)
{
	x *= scalar;
	y *= scalar;
	z *= scalar;
	w *= scalar;
	return (*this);
}

// -----------------------------------------------------------------------------
// --------------------------- Quaternion::operator/ ---------------------------
// -----------------------------------------------------------------------------
Quaternion Quaternion::operator/(const float scalar) const
{
	float reciprocal = 1 / scalar;
	return Quaternion(x * reciprocal, y * reciprocal, z * reciprocal, w * reciprocal);
}

// -----------------------------------------------------------------------------
// --------------------------- Quaternion::operator/= --------------------------
// -----------------------------------------------------------------------------
const Quaternion& Quaternion::operator/=(const float scalar)
{
	float reciprocal = 1 / scalar;
	x *= reciprocal;
	y *= reciprocal;
	z *= reciprocal;
	w *= reciprocal;
	return (*this);
}

// -----------------------------------------------------------------------------
// --------------------------- Quaternion::operator* ---------------------------
// -----------------------------------------------------------------------------
Vector3 Quaternion::operator*(const Vector3& vec) const
{
	return Rotate(vec);
}

// -----------------------------------------------------------------------------
// --------------------------- Quaternion::GetLength ---------------------------
// -----------------------------------------------------------------------------
float Quaternion::GetLength() const
{
	return Math::Sqrt(GetSquaredLength());
}

// -----------------------------------------------------------------------------
// ------------------------ Quaternion::GetSquaredLength -----------------------
// -----------------------------------------------------------------------------
float Quaternion::GetSquaredLength() const
{
	return (x * x + y * y + z * z + w * w);
}

// -----------------------------------------------------------------------------
// --------------------------- Quaternion::DotProduct --------------------------
// -----------------------------------------------------------------------------
float Quaternion::DotProduct(const Quaternion& other) const
{
	return (x * other.x + y * other.y + z * other.z + w * other.w);
}

// -----------------------------------------------------------------------------
// --------------------------- Quaternion::Normalize ---------------------------
// -----------------------------------------------------------------------------
void Quaternion::Normalize()
{
	if (GetLength() != 0.0f)
    {
	    float reciprocalLength = (float)1.0 / GetLength();
	    x *= reciprocalLength;
	    y *= reciprocalLength;
	    z *= reciprocalLength;
	    w *= reciprocalLength;
    }
}

// -----------------------------------------------------------------------------
// ----------------------------- Quaternion::Negate ----------------------------
// -----------------------------------------------------------------------------
void Quaternion::Negate()
{

	x = -x;
	y = -y;
	z = -z;
	w = -w;
}

// -----------------------------------------------------------------------------
// ----------------------------- Quaternion::Rotate ----------------------------
// -----------------------------------------------------------------------------
Vector3 Quaternion::Rotate(const Vector3& vector3d) const
{
    Matrix3 kRot = BuildMatrix();
    return kRot*vector3d;	
}

// -----------------------------------------------------------------------------
// -------------------------- Quaternion::BuildMatrix --------------------------
// -----------------------------------------------------------------------------
Matrix3 Quaternion::BuildMatrix() const
{
    float fTx  = 2.0f*x;
    float fTy  = 2.0f*y;
    float fTz  = 2.0f*z;
    float fTwx = fTx*w;
    float fTwy = fTy*w;
    float fTwz = fTz*w;
    float fTxx = fTx*x;
    float fTxy = fTy*x;
    float fTxz = fTz*x;
    float fTyy = fTy*y;
    float fTyz = fTz*y;
    float fTzz = fTz*z;

    Matrix3 ret;
    ret[0][0] = 1.0f-(fTyy+fTzz);
    ret[1][0] = fTxy-fTwz;
    ret[2][0] = fTxz+fTwy;
    ret[0][1] = fTxy+fTwz;
    ret[1][1] = 1.0f-(fTxx+fTzz);
    ret[2][1] = fTyz-fTwx;
    ret[0][2] = fTxz-fTwy;
    ret[1][2] = fTyz+fTwx;
    ret[2][2] = 1.0f-(fTxx+fTyy);
	
	return ret;
}

// -----------------------------------------------------------------------------
// ----------------------------- Quaternion::nlerp -----------------------------
// -----------------------------------------------------------------------------
Quaternion Quaternion::Lerp(const Quaternion& other, const float factor) const
{
	Quaternion interpolated =  (*this)*(1.0-factor) + other*factor;
	interpolated.Normalize();

	return interpolated;

}

// -----------------------------------------------------------------------------
// ----------------------------- Quaternion::slerp -----------------------------
// -----------------------------------------------------------------------------
Quaternion Quaternion::Slerp(const Quaternion& other, const float factor) const
{
	float dot = this->DotProduct(other);

	//to near -> do nlerp
	if (dot > 0.999)
		return Lerp(other,factor);

	//clamp to -1,1
	if (dot>1)
		dot=1;
	if (dot<-1)
		dot=-1;

	//If negative dot: negate one quaternion to prevent large interpolation angle	
	Quaternion o = other;
	if (dot<0)
	{
		o.Negate();	
		dot = this->DotProduct(o);

	}

	float theta_0 = Math::ACos(dot);  
	float theta = theta_0*factor;   

	Quaternion v2 =  o - (*this)*dot;		
	v2.Normalize(); 

	return (*this)*Math::Cos(theta) + v2*Math::Sin(theta);
}

// -----------------------------------------------------------------------------
// ---------------------------- Quaternion::Inverse ----------------------------
// -----------------------------------------------------------------------------
Quaternion Quaternion::Inverse(void) const
{
    return Quaternion(-x, -y, -z, w);
}

// -----------------------------------------------------------------------------
// ------------------------- Quaternion::ToEulerAngles -------------------------
// -----------------------------------------------------------------------------
void Quaternion::ToEulerAngles(float& heading, float& bank, float &attitude) const
{   
	double sqw = w*w;
	double sqx = x*x;
	double sqy = y*y;
	double sqz = z*z;    
	heading = Math::ATan2(2.0 * (x*y + z*w),(sqx - sqy - sqz + sqw));    
	bank = Math::ATan2(2.0 * (y*z + x*w),(-sqx - sqy + sqz + sqw));    
	attitude = Math::ASin(-2.0 * (x*z - y*w));
}


// -----------------------------------------------------------------------------
// -------------------------- Quaternion::ToAxisAngle --------------------------
// -----------------------------------------------------------------------------
void Quaternion::ToAxisAngle(Vector3& axis, float& angle) const
{
	// The quaternion representing the rotation is
	//   q = cos(A/2)+sin(A/2)*(x*i+y*j+z*k)

	float fSqrLength = x*x+y*y+z*z;
	if ( fSqrLength > 0.0f )
	{
		angle = 2.0f*Math::ACos(w);
		float fInvLength = Math::InvSqrt(fSqrLength);
		axis.x = x*fInvLength;
		axis.y = y*fInvLength;
		axis.z = z*fInvLength;
	}
	else
	{
		// angle is 0 (mod 2*pi), so any axis will do
		angle = 0.0f;
		axis.x = 1.0f;
		axis.y = 0.0f;
		axis.z = 0.0f;
	}
}

// -----------------------------------------------------------------------------
// ------------------------- Quaternion::FromAxisAngle -------------------------
// -----------------------------------------------------------------------------
void Quaternion::FromAxisAngle(const Vector3& axis, const float& angle)
{
	float halfAngle = 0.5f * angle;
	float sinHalfAngle = Math::Sin(halfAngle);
    w = Math::Cos(halfAngle);
	x = sinHalfAngle * axis.x;
	y = sinHalfAngle * axis.y;
	z = sinHalfAngle * axis.z;	
}


// -----------------------------------------------------------------------------
// ----------------------------- Quaternion::Print -----------------------------
// -----------------------------------------------------------------------------
void Quaternion::Print(const char* name) const
{
	printf("%s: %f %f %f %f\n", name, x, y, z, w);
}


