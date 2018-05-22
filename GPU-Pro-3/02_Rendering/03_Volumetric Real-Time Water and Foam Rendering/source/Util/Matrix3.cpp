#include "Matrix3.h"
#include "Math.h"

#include <assert.h>

// template specialization of the static members for the f32 datatype
const Matrix3 Matrix3::ZERO(0.0f, 0.0f, 0.0f,
						0.0f, 0.0f, 0.0f,
						0.0f, 0.0f, 0.0f);
const Matrix3 Matrix3::IDENTITY(1.0f, 0.0f, 0.0f,
							0.0f, 1.0f, 0.0f,
							0.0f, 0.0f, 1.0f);


// -----------------------------------------------------------------------------
// ------------------------------ Matrix3::Matrix3 -----------------------------
// -----------------------------------------------------------------------------
Matrix3::Matrix3()
{
	for (int i=0; i<9; i++)
		entry[i] = IDENTITY.entry[i];
}


// -----------------------------------------------------------------------------
// ------------------------------ Matrix3::Matrix3 -----------------------------
// -----------------------------------------------------------------------------
Matrix3::Matrix3(float e11, float e12, float e13,
				 float e21, float e22, float e23,
				 float e31, float e32, float e33)
{
	entry[0] = e11;
	entry[1] = e12;
	entry[2] = e13;
	entry[3] = e21;
	entry[4] = e22;
	entry[5] = e23;
	entry[6] = e31;
	entry[7] = e32;
	entry[8] = e33;
}


// -----------------------------------------------------------------------------
// ------------------------------ Matrix3::Matrix3 -----------------------------
// -----------------------------------------------------------------------------
Matrix3::Matrix3(const Matrix3& other)
{
	memcpy(entry, other.entry, 9 * sizeof(float));
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix3::operator[] ----------------------------
// -----------------------------------------------------------------------------
const float* Matrix3::operator[](int row) const
{
	return &entry[3*row];
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix3::operator[] ----------------------------
// -----------------------------------------------------------------------------
float* Matrix3::operator[](int row)
{
	return &entry[3*row];
}


// -----------------------------------------------------------------------------
// ----------------------------- Matrix3::operator -----------------------------
// -----------------------------------------------------------------------------
float Matrix3::operator()(int row, int col) const
{
	assert((row>=0) && (row<3));
	assert((col>=0) && (col<3));
	return entry[3*row + col];
}


// -----------------------------------------------------------------------------
// ----------------------------- Matrix3::operator -----------------------------
// -----------------------------------------------------------------------------
float& Matrix3::operator()(int row, int col)
{
	assert((row>=0) && (row<3));
	assert((col>=0) && (col<3));
	return entry[3*row + col];
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix3::operator== ----------------------------
// -----------------------------------------------------------------------------
bool Matrix3::operator==(const Matrix3& other) const
{
	if (this==&other)
		return true;
	int i;
	for (i = 0; i<9; i++)
		if (Math::IsNotEqual(entry[i], other.entry[i]))
			return false;
	return true;
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix3::operator!= ----------------------------
// -----------------------------------------------------------------------------
bool Matrix3::operator!=(const Matrix3& other) const
{
	if (this==&other)
		return false;
	int i;
	for (i = 0; i<9; i++)
		if (Math::IsNotEqual(entry[i], other.entry[i]))
			return true;
	return false;
}


// -----------------------------------------------------------------------------
// ----------------------------- Matrix3::operator= ----------------------------
// -----------------------------------------------------------------------------
const Matrix3& Matrix3::operator=(const Matrix3& other)
{
	if ((&other == this)||(other == *this))
		return (*this);

	memcpy(entry, other.entry, 9 * sizeof(float));
	return (*this);
}


// -----------------------------------------------------------------------------
// ----------------------------- Matrix3::operator+ ----------------------------
// -----------------------------------------------------------------------------
const Matrix3 Matrix3::operator+(const Matrix3& other) const
{
	Matrix3 sum;
	for (int i = 0; i < 9; ++i)
		sum.entry[i] = entry[i] + other.entry[i];
	return sum;
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix3::operator+= ----------------------------
// -----------------------------------------------------------------------------
const Matrix3& Matrix3::operator+=(const Matrix3& other)
{
	for (int i = 0; i < 9; ++i)
		entry[i] += other.entry[i];
	return (*this);
}


// -----------------------------------------------------------------------------
// ----------------------------- Matrix3::operator- ----------------------------
// -----------------------------------------------------------------------------
const Matrix3 Matrix3::operator-(const Matrix3& other) const
{
	Matrix3 subtraction;
	for (int i = 0; i < 9; ++i)
		subtraction.entry[i] = entry[i] - other.entry[i];
	return subtraction;
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix3::operator-= ----------------------------
// -----------------------------------------------------------------------------
const Matrix3& Matrix3::operator-=(const Matrix3& other)
{
	for (int i = 0; i < 9; ++i)
		entry[i] -= other.entry[i];
	return (*this);
}


// -----------------------------------------------------------------------------
// ----------------------------- Matrix3::operator* ----------------------------
// -----------------------------------------------------------------------------
const Matrix3 Matrix3::operator*(float scalar) const
{
	Matrix3 product;
	for (int i = 0; i < 9; ++i)
		product.entry[i] = scalar * entry[i];
	return product;
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix3::operator*= ----------------------------
// -----------------------------------------------------------------------------
const Matrix3& Matrix3::operator*=(float scalar)
{
	for (int i = 0; i < 9; ++i)
		entry[i] *= scalar;
	return (*this);
}


// -----------------------------------------------------------------------------
// ----------------------------- Matrix3::operator* ----------------------------
// -----------------------------------------------------------------------------
const Matrix3 Matrix3::operator*(const Matrix3& other) const
{
	Matrix3 product;
	product.entry[0] = entry[0]*other.entry[0] + entry[1]*other.entry[3] + entry[2]*other.entry[6];
	product.entry[1] = entry[0]*other.entry[1] + entry[1]*other.entry[4] + entry[2]*other.entry[7];
	product.entry[2] = entry[0]*other.entry[2] + entry[1]*other.entry[5] + entry[2]*other.entry[8];

	product.entry[3] = entry[3]*other.entry[0] + entry[4]*other.entry[3] + entry[5]*other.entry[6];
	product.entry[4] = entry[3]*other.entry[1] + entry[4]*other.entry[4] + entry[5]*other.entry[7];
	product.entry[5] = entry[3]*other.entry[2] + entry[4]*other.entry[5] + entry[5]*other.entry[8];

	product.entry[6] = entry[6]*other.entry[0] + entry[7]*other.entry[3] + entry[8]*other.entry[6];
	product.entry[7] = entry[6]*other.entry[1] + entry[7]*other.entry[4] + entry[8]*other.entry[7];
	product.entry[8] = entry[6]*other.entry[2] + entry[7]*other.entry[5] + entry[8]*other.entry[8];

	return product;
}


// -----------------------------------------------------------------------------
// ----------------------------- Matrix3::operator* ----------------------------
// -----------------------------------------------------------------------------
Vector3 Matrix3::operator*(const Vector3& v) const
{
	//////////////////////////////////////////////////////////////////////////
	Vector3 product;
	product.comp[0] = entry[0]*v.comp[0] + entry[3]*v.comp[1] + entry[6]*v.comp[2];
	product.comp[1] = entry[1]*v.comp[0] + entry[4]*v.comp[1] + entry[7]*v.comp[2];
	product.comp[2] = entry[2]*v.comp[0] + entry[5]*v.comp[1] + entry[8]*v.comp[2];
	return product;
}


// -----------------------------------------------------------------------------
// ----------------------------- Matrix3::operator/ ----------------------------
// -----------------------------------------------------------------------------
const Matrix3 Matrix3::operator/(float scalar) const
{
	// TODO: if scalar to small, set values to MAX_REAL
	Matrix3 quotient;
	float reciprocal = 1 / scalar;
	for (int i = 0; i < 9; ++i)
		quotient.entry[i] = reciprocal * entry[i];
	return quotient;
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix3::operator/= ----------------------------
// -----------------------------------------------------------------------------
const Matrix3& Matrix3::operator/=(float scalar)
{
	// TODO: if scalar to small, set values to MAX_REAL
	float reciprocal = 1 / scalar;
	for (int i = 0; i < 9; ++i)
		entry[i] *= reciprocal;
	return (*this);
}


// -----------------------------------------------------------------------------
// ----------------------------- Matrix3::Transposed ---------------------------
// -----------------------------------------------------------------------------
Matrix3 Matrix3::Transposed(void) const
{
	Matrix3 transpose;
	transpose.entry[ 0] = entry[ 0];
	transpose.entry[ 1] = entry[ 3];
	transpose.entry[ 2] = entry[ 6];
	transpose.entry[ 3] = entry[ 1];
	transpose.entry[ 4] = entry[ 4];
	transpose.entry[ 5] = entry[ 7];
	transpose.entry[ 6] = entry[ 2];
	transpose.entry[ 7] = entry[ 5];
	transpose.entry[ 8] = entry[ 8];
	return transpose;
}


// -----------------------------------------------------------------------------
// ------------------------------ Matrix3::Inverse -----------------------------
// -----------------------------------------------------------------------------
Matrix3 Matrix3::Inverse(void) const
{
	// Invert a 3x3 using cofactors.  This is faster than using a generic
	// Gaussian elimination because of the loop overhead of such a method.

	Matrix3 inverse;

	inverse[0][0] = entry[4]*entry[8] - entry[5]*entry[7];
	inverse[0][1] = entry[2]*entry[7] - entry[1]*entry[8];
	inverse[0][2] = entry[1]*entry[5] - entry[2]*entry[4];
	inverse[1][0] = entry[5]*entry[6] - entry[3]*entry[8];
	inverse[1][1] = entry[0]*entry[8] - entry[2]*entry[6];
	inverse[1][2] = entry[2]*entry[3] - entry[0]*entry[5];
	inverse[2][0] = entry[3]*entry[7] - entry[4]*entry[6];
	inverse[2][1] = entry[1]*entry[6] - entry[0]*entry[7];
	inverse[2][2] = entry[0]*entry[4] - entry[1]*entry[3];

	float det = entry[0]*inverse[0][0] + entry[1]*inverse[1][0]+ entry[2]*inverse[2][0];

	if (fabs(det) <= Math::EPSILON_FLOAT)
		return ZERO;

	inverse /= det;
	return inverse;
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix3::BuildScale ----------------------------
// -----------------------------------------------------------------------------
void Matrix3::BuildScale(float x, float y, float z)
{
	entry[0] = x;
	entry[1] = 0.0;
	entry[2] = 0.0;

	entry[3] = 0.0;
	entry[4] = y;
	entry[5] = 0.0;

	entry[6] = 0.0;
	entry[7] = 0.0;
	entry[8] = z;
}


// -----------------------------------------------------------------------------
// ----------------------- Matrix3::BuildRotationXRadian -----------------------
// -----------------------------------------------------------------------------
void Matrix3::BuildRotationX(float angle)
{
	float sinAngle = Math::Sin(angle);
	float cosAngle = Math::Cos(angle);

	entry[0] = 1.0;
	entry[1] = 0.0;
	entry[2] = 0.0;

	entry[3] = 0.0;
	entry[4] = cosAngle;
	entry[5] = sinAngle;

	entry[6] = 0.0;
	entry[7] = -sinAngle;
	entry[8] = cosAngle;
}


// -----------------------------------------------------------------------------
// -------------------------- Matrix3::BuildRotationY --------------------------
// -----------------------------------------------------------------------------
void Matrix3::BuildRotationY(float angle)
{
	float sinAngle = Math::Sin(angle);
	float cosAngle = Math::Cos(angle);

	entry[0] = cosAngle;
	entry[1] = 0.0;
	entry[2] = -sinAngle;

	entry[3] = 0.0;
	entry[4] = 1.0;
	entry[5] = 0.0;

	entry[6] = sinAngle;
	entry[7] = 0.0;
	entry[8] = cosAngle;
}


// -----------------------------------------------------------------------------
// -------------------------- Matrix3::BuildRotationZ --------------------------
// -----------------------------------------------------------------------------
void Matrix3::BuildRotationZ(float angle)
{
	float sinAngle = Math::Sin(angle);
	float cosAngle = Math::Cos(angle);

	entry[0] = cosAngle;
	entry[1] = sinAngle;
	entry[2] = 0.0;

	entry[3] = -sinAngle;
	entry[4] = cosAngle;
	entry[5] = 0.0;

	entry[6] = 0.0;
	entry[7] = 0.0;
	entry[8] = 1.0;
}


// -----------------------------------------------------------------------------
// --------------------------- Matrix3::BuildRotation --------------------------
// -----------------------------------------------------------------------------
void Matrix3::BuildRotation(const Vector3& vec, float angle)
{
	float c = Math::Cos(angle);
	float s = Math::Sin(angle);
	float t = 1-c; 

	(*this)[0][0] = t*vec.x*vec.x + c;
	(*this)[0][1] = t*vec.x*vec.y - s*vec.z;
	(*this)[0][2] = t*vec.x*vec.z + s*vec.y;
	(*this)[1][0] = t*vec.x*vec.y + s*vec.z;
	(*this)[1][1] = t*vec.y*vec.y + c;
	(*this)[1][2] = t*vec.y*vec.z - s*vec.x;
	(*this)[2][0] = t*vec.x*vec.z - s*vec.y;
	(*this)[2][1] = t*vec.y*vec.z + s*vec.x;
	(*this)[2][2] = t*vec.z*vec.z + c;
}


// -----------------------------------------------------------------------------
// --------------------------- Matrix3::BuildRotation --------------------------
// -----------------------------------------------------------------------------
void Matrix3::BuildRotation(float yaw, float pitch, float roll)
{	
	float cs, sn;

	cs = Math::Cos(yaw);
	sn = Math::Sin(yaw);
	Matrix3 xMtx(
		1.0f,0.0f,0.0f,
		0.0f,cs,-sn,
		0.0f,sn,cs);

	cs = Math::Cos(pitch);
	sn = Math::Sin(pitch);
	Matrix3 yMtx(cs,0.0f,sn,
		0.0f,1.0f,0.0f,
		-sn,0.0f,cs);

	cs = Math::Cos(roll);
	sn = Math::Sin(roll);
	Matrix3 zMtx(cs,-sn,0.0f,
		sn,cs,0.0f,
		0.0f,0.0f,1.0f);

	*this = xMtx*(yMtx*zMtx);
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix3::ToAxisAngle ---------------------------
// -----------------------------------------------------------------------------
void Matrix3::ToAxisAngle(Vector3& axis, float& angle) const
{
	float trace = entry[0] + entry[4] + entry[8];
	float tempCos = (trace-1.0f)/2;
	angle = acos(tempCos);  // in [0,PI]

	if (angle > 0.0f)
	{
		if (angle < Math::PI)
		{
			axis.x = entry[7]-entry[5];
			axis.y = entry[2]-entry[6];
			axis.z = entry[3]-entry[1];
			axis.Normalize();
		}
		else
		{
			// angle is PI
			float halfInverse;
			if (entry[0] >= entry[4])
			{
				// r00 >= r11
				if (entry[0] >= entry[8])
				{
					// r00 is maximum diagonal term
					axis.x = (0.5f)*sqrt(entry[0] - entry[4] - entry[8] + 1.0f);
					halfInverse = (0.5f)/axis.x;
					axis.y = halfInverse*entry[1];
					axis.z = halfInverse*entry[2];
				}
				else
				{
					// r22 is maximum diagonal term
					axis.z = (0.5f)*sqrt(entry[8] - entry[0] - entry[4] + 1.0f);
					halfInverse = (0.5f)/axis.z;
					axis.x = halfInverse*entry[2];
					axis.y = halfInverse*entry[5];
				}
			}
			else
			{
				// r11 > r00
				if (entry[4] >= entry[8])
				{
					// r11 is maximum diagonal term
					axis.y = (0.5f)*sqrt(entry[4] - entry[0] - entry[8] + 1.0f);
					halfInverse  = (0.5f)/axis.y;
					axis.x = halfInverse*entry[1];
					axis.z = halfInverse*entry[5];
				}
				else
				{
					// r22 is maximum diagonal term
					axis.z = (0.5f)*sqrt(entry[8] - entry[0] - entry[4] + 1.0f);
					halfInverse = (0.5f)/axis.z;
					axis.x = halfInverse*entry[2];
					axis.y = halfInverse*entry[5];
				}
			}
		}
	}
	else
	{
		// The angle is 0 and the matrix is the identity.  Any axis will
		// work, so just use the x-axis.
		axis.x = 1.0f;
		axis.y = 0.0f;
		axis.z = 0.0f;
	}
}


// -----------------------------------------------------------------------------
// --------------------------- Matrix3::FromAxisAngle --------------------------
// -----------------------------------------------------------------------------
void Matrix3::FromAxisAngle(const Vector3& axis, float angle)
{
	float tempCos = cos(angle);
	float tempSin = sin(angle);
	float tempOneMinusCos = 1.0f-tempCos;
	float tempX2 = axis.x*axis.x;
	float tempY2 = axis.y*axis.y;
	float tempZ2 = axis.z*axis.z;
	float tempXYM = axis.x*axis.y*tempOneMinusCos;
	float tempXZM = axis.x*axis.z*tempOneMinusCos;
	float tempYZM = axis.y*axis.z*tempOneMinusCos;
	float tempXSin = axis.x*tempSin;
	float tempYSin = axis.y*tempSin;
	float tempZSin = axis.z*tempSin;

	(*this)[0][0] = tempX2*tempOneMinusCos + tempCos;
	(*this)[1][0] = tempXYM - tempZSin;
	(*this)[2][0] = tempXZM + tempYSin;
	(*this)[0][1] = tempXYM + tempZSin;
	(*this)[1][1] = tempY2*tempOneMinusCos + tempCos;
	(*this)[2][1] = tempYZM - tempXSin;
	(*this)[0][2] = tempXZM - tempYSin;
	(*this)[1][2] = tempYZM + tempXSin;
	(*this)[2][2] = tempZ2*tempOneMinusCos + tempCos;
}


// -----------------------------------------------------------------------------
// --------------------------- Matrix3::GetLookVector --------------------------
// -----------------------------------------------------------------------------
Vector3 Matrix3::GetLookVector(void) const
{
	return Vector3((*this)[2][0], (*this)[2][1], (*this)[2][2]);
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix3::GetUpVector ---------------------------
// -----------------------------------------------------------------------------
Vector3 Matrix3::GetRightVector(void) const
{
	return Vector3((*this)[0][0], (*this)[0][1], (*this)[0][2]);
}


// -----------------------------------------------------------------------------
// -------------------------- Matrix3::GetRightVector --------------------------
// -----------------------------------------------------------------------------
Vector3 Matrix3::GetUpVector(void) const
{
	return Vector3((*this)[1][0], (*this)[1][1], (*this)[1][2]);
}


// -----------------------------------------------------------------------------
// --------------------------- Matrix3::SetLookVector --------------------------
// -----------------------------------------------------------------------------
void Matrix3::SetLookVector(const Vector3& vec)
{
	(*this)(2,0) = vec.x;
	(*this)(2,1) = vec.y;
	(*this)(2,2) = vec.z;
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix3::SetUpVector ---------------------------
// -----------------------------------------------------------------------------
void Matrix3::SetUpVector(const Vector3& vec)
{
	(*this)(1,0) = vec.x;
	(*this)(1,1) = vec.y;
	(*this)(1,2) = vec.z;
}


// -----------------------------------------------------------------------------
// -------------------------- Matrix3::SetRightVector --------------------------
// -----------------------------------------------------------------------------
void Matrix3::SetRightVector(const Vector3& vec)
{
	(*this)(0,0) = vec.x;
	(*this)(0,1) = vec.y;
	(*this)(0,2) = vec.z;
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix3::CreateScale ---------------------------
// -----------------------------------------------------------------------------
Matrix3 Matrix3::CreateScale(float x, float y, float z)
{
	Matrix3 m;
	m.BuildScale(x, y, z);
	return m;
}


// -----------------------------------------------------------------------------
// -------------------------- Matrix3::CreateRotation --------------------------
// -----------------------------------------------------------------------------
Matrix3 Matrix3::CreateRotation(const Vector3& axis, float angle)
{
	Matrix3 m;
	m.BuildRotation(axis, angle);
	return m;
}


// -----------------------------------------------------------------------------
// -------------------------- Matrix3::CreateRotation --------------------------
// -----------------------------------------------------------------------------
Matrix3 Matrix3::CreateRotation(float yaw, float pitch, float roll)
{
	Matrix3 m;
	m.BuildRotation(yaw, pitch, roll);
	return m;
}


// -----------------------------------------------------------------------------
// -------------------------- Matrix3::CreateRotationX -------------------------
// -----------------------------------------------------------------------------
Matrix3 Matrix3::CreateRotationX(float angle)
{
	Matrix3 m;
	m.BuildRotationX(angle);
	return m;
}


// -----------------------------------------------------------------------------
// -------------------------- Matrix3::CreateRotationY -------------------------
// -----------------------------------------------------------------------------
Matrix3 Matrix3::CreateRotationY(float angle)
{
	Matrix3 m;
	m.BuildRotationY(angle);
	return m;
}


// -----------------------------------------------------------------------------
// -------------------------- Matrix3::CreateRotationZ -------------------------
// -----------------------------------------------------------------------------
Matrix3 Matrix3::CreateRotationZ(float angle)
{
	Matrix3 m;
	m.BuildRotationZ(angle);
	return m;
}
