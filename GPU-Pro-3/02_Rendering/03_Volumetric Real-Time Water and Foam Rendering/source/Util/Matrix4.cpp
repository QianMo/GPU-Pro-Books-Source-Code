#include <assert.h>
#include "Matrix4.h"
#include "Math.h"

// template specialization of the static members for the f32 datatype
const Matrix4 Matrix4::ZERO(0.0f, 0.0f, 0.0f, 0.0f,
					  	    0.0f, 0.0f, 0.0f, 0.0f,
							0.0f, 0.0f, 0.0f, 0.0f,
							0.0f, 0.0f, 0.0f, 0.0f);
const Matrix4 Matrix4::IDENTITY(1.0f, 0.0f, 0.0f, 0.0f,
					            0.0f, 1.0f, 0.0f, 0.0f,
								0.0f, 0.0f, 1.0f, 0.0f,
								0.0f, 0.0f, 0.0f, 1.0f);


// -----------------------------------------------------------------------------
// ------------------------------ Matrix4::Matrix4 -----------------------------
// -----------------------------------------------------------------------------
Matrix4::Matrix4()
{
	memcpy(entry, IDENTITY.entry, 16 * sizeof(float));
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix4::IsIdentity ----------------------------
// -----------------------------------------------------------------------------
bool Matrix4::IsIdentity(void) const
{
	for (int i=0; i<16; i++)
	{
		if (Math::IsNotEqual(entry[i],IDENTITY.entry[i]))
			return false;
	}
	return true;
}

// -----------------------------------------------------------------------------
// ------------------------------ Matrix4::Matrix4 -----------------------------
// -----------------------------------------------------------------------------
Matrix4::Matrix4(float e11, float e12, float e13, float e14,
				 float e21, float e22, float e23, float e24,
				 float e31, float e32, float e33, float e34,
				 float e41, float e42, float e43, float e44)
{
	entry[ 0] = e11;
	entry[ 1] = e12;
	entry[ 2] = e13;
	entry[ 3] = e14;
	entry[ 4] = e21;
	entry[ 5] = e22;
	entry[ 6] = e23;
	entry[ 7] = e24;
	entry[ 8] = e31;
	entry[ 9] = e32;
	entry[10] = e33;
	entry[11] = e34;
	entry[12] = e41;
	entry[13] = e42;
	entry[14] = e43;
	entry[15] = e44;
}


// -----------------------------------------------------------------------------
// ------------------------------ Matrix4::Matrix4 -----------------------------
// -----------------------------------------------------------------------------
Matrix4::Matrix4(const Matrix4& other)
{
	memcpy(entry, other.entry, 16 * sizeof(float));
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix4::operator[] ----------------------------
// -----------------------------------------------------------------------------
const float* Matrix4::operator[](int row) const
{
	return &entry[4*row];
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix4::operator[] ----------------------------
// -----------------------------------------------------------------------------
float* Matrix4::operator[](int row)
{
	return &entry[4*row];
}


// -----------------------------------------------------------------------------
// ----------------------------- Matrix4::operator -----------------------------
// -----------------------------------------------------------------------------
float Matrix4::operator()(int row, int col) const
{
	return entry[row*4+col];
}


// -----------------------------------------------------------------------------
// ----------------------------- Matrix4::operator -----------------------------
// -----------------------------------------------------------------------------
float& Matrix4::operator()(int row, int col)
{
	return entry[row*4+col];
}


// -----------------------------------------------------------------------------
// ----------------------------- Matrix4::operator= ----------------------------
// -----------------------------------------------------------------------------
const Matrix4& Matrix4::operator=(const Matrix4& other)
{
	memcpy(entry, other.entry, 16 * sizeof(float));
	return (*this);
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix4::operator== ----------------------------
// -----------------------------------------------------------------------------
bool Matrix4::operator==(const Matrix4& other) const
{
	if (this==&other)
		return true;
	int i;
	for (i = 0; i<16; i++)
		if (Math::IsNotEqual(entry[i], other.entry[i]))
			return false;
	return true;
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix4::operator!= ----------------------------
// -----------------------------------------------------------------------------
bool Matrix4::operator!=(const Matrix4& other) const
{
	if (this==&other)
		return false;
	int i;
	for (i = 0; i<16; i++)
		if (Math::IsNotEqual(entry[i], other.entry[i]))
			return true;
	return false;
}


// -----------------------------------------------------------------------------
// ----------------------------- Matrix4::operator+ ----------------------------
// -----------------------------------------------------------------------------
Matrix4 Matrix4::operator+(const Matrix4& other) const
{
	Matrix4 sum;
	for (int i = 0; i < 16; ++i)
		sum.entry[i] = entry[i] + other.entry[i];
	return sum;
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix4::operator+= ----------------------------
// -----------------------------------------------------------------------------
const Matrix4& Matrix4::operator+=(const Matrix4& other)
{
	for (int i = 0; i < 16; ++i)
		entry[i] += other.entry[i];
	return (*this);
}


// -----------------------------------------------------------------------------
// ----------------------------- Matrix4::operator- ----------------------------
// -----------------------------------------------------------------------------
Matrix4 Matrix4::operator-(const Matrix4& other) const
{
	Matrix4 subtraction;
	for (int i = 0; i < 16; ++i)
		subtraction.entry[i] = entry[i] - other.entry[i];
	return subtraction;
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix4::operator-= ----------------------------
// -----------------------------------------------------------------------------
const Matrix4& Matrix4::operator-=(const Matrix4& other)
{
	for (int i = 0; i < 16; ++i)
		entry[i] -= other.entry[i];
	return (*this);
}


// -----------------------------------------------------------------------------
// ----------------------------- Matrix4::operator* ----------------------------
// -----------------------------------------------------------------------------
Matrix4 Matrix4::operator*(float scalar) const
{
	Matrix4 product;
	for (int i = 0; i < 16; ++i)
		product.entry[i] = scalar * entry[i];
	return product;
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix4::operator*= ----------------------------
// -----------------------------------------------------------------------------
const Matrix4& Matrix4::operator*=(float scalar)
{
	for (int i = 0; i < 16; ++i)
		entry[i] *= scalar;
	return (*this);
}


// -----------------------------------------------------------------------------
// ----------------------------- Matrix4::operator* ----------------------------
// -----------------------------------------------------------------------------
Matrix4 Matrix4::operator*(const Matrix4& other) const
{
	Matrix4 product;
	product.entry[ 0] = entry[ 0]*other.entry[ 0] + entry[ 1]*other.entry[ 4] + entry[ 2]*other.entry[ 8] + entry[ 3]*other.entry[12];
	product.entry[ 1] = entry[ 0]*other.entry[ 1] + entry[ 1]*other.entry[ 5] + entry[ 2]*other.entry[ 9] + entry[ 3]*other.entry[13];
	product.entry[ 2] = entry[ 0]*other.entry[ 2] + entry[ 1]*other.entry[ 6] + entry[ 2]*other.entry[10] + entry[ 3]*other.entry[14];
	product.entry[ 3] = entry[ 0]*other.entry[ 3] + entry[ 1]*other.entry[ 7] + entry[ 2]*other.entry[11] + entry[ 3]*other.entry[15];

	product.entry[ 4] = entry[ 4]*other.entry[ 0] + entry[ 5]*other.entry[ 4] + entry[ 6]*other.entry[ 8] + entry[ 7]*other.entry[12];
	product.entry[ 5] = entry[ 4]*other.entry[ 1] + entry[ 5]*other.entry[ 5] + entry[ 6]*other.entry[ 9] + entry[ 7]*other.entry[13];
	product.entry[ 6] = entry[ 4]*other.entry[ 2] + entry[ 5]*other.entry[ 6] + entry[ 6]*other.entry[10] + entry[ 7]*other.entry[14];
	product.entry[ 7] = entry[ 4]*other.entry[ 3] + entry[ 5]*other.entry[ 7] + entry[ 6]*other.entry[11] + entry[ 7]*other.entry[15];

	product.entry[ 8] = entry[ 8]*other.entry[ 0] + entry[ 9]*other.entry[ 4] + entry[10]*other.entry[ 8] + entry[11]*other.entry[12];
	product.entry[ 9] = entry[ 8]*other.entry[ 1] + entry[ 9]*other.entry[ 5] + entry[10]*other.entry[ 9] + entry[11]*other.entry[13];
	product.entry[10] = entry[ 8]*other.entry[ 2] + entry[ 9]*other.entry[ 6] + entry[10]*other.entry[10] + entry[11]*other.entry[14];
	product.entry[11] = entry[ 8]*other.entry[ 3] + entry[ 9]*other.entry[ 7] + entry[10]*other.entry[11] + entry[11]*other.entry[15];

	product.entry[12] = entry[12]*other.entry[ 0] + entry[13]*other.entry[ 4] + entry[14]*other.entry[ 8] + entry[15]*other.entry[12];
	product.entry[13] = entry[12]*other.entry[ 1] + entry[13]*other.entry[ 5] + entry[14]*other.entry[ 9] + entry[15]*other.entry[13];
	product.entry[14] = entry[12]*other.entry[ 2] + entry[13]*other.entry[ 6] + entry[14]*other.entry[10] + entry[15]*other.entry[14];
	product.entry[15] = entry[12]*other.entry[ 3] + entry[13]*other.entry[ 7] + entry[14]*other.entry[11] + entry[15]*other.entry[15];
	return product;
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix4::operator*= ----------------------------
// -----------------------------------------------------------------------------
const Matrix4& Matrix4::operator*=(const Matrix4& other)
{
	Matrix4 product = (*this)*other;
	(*this) = product;
	return (*this);
}


// -----------------------------------------------------------------------------
// ----------------------------- Matrix4::operator* ----------------------------
// -----------------------------------------------------------------------------
Vector3 Matrix4::operator*(const Vector3& v) const
{
	Vector3 product;
	product.comp[0] = entry[0]*v.comp[0] + entry[ 4]*v.comp[1] + entry[ 8]*v.comp[2] + entry[12];
	product.comp[1] = entry[1]*v.comp[0] + entry[ 5]*v.comp[1] + entry[ 9]*v.comp[2] + entry[13];
	product.comp[2] = entry[2]*v.comp[0] + entry[ 6]*v.comp[1] + entry[10]*v.comp[2] + entry[14];
	return product;
}


// -----------------------------------------------------------------------------
// ----------------------------- Matrix4::operator* ----------------------------
// -----------------------------------------------------------------------------
Vector4 Matrix4::operator*(const Vector4& v) const
{
	Vector4 product;
	product.comp[0] = entry[ 0]*v.comp[0] + entry[ 4]*v.comp[1] + entry[ 8]*v.comp[2] + entry[12]*v.comp[3];
	product.comp[1] = entry[ 1]*v.comp[0] + entry[ 5]*v.comp[1] + entry[ 9]*v.comp[2] + entry[13]*v.comp[3];
	product.comp[2] = entry[ 2]*v.comp[0] + entry[ 6]*v.comp[1] + entry[10]*v.comp[2] + entry[14]*v.comp[3];
	product.comp[3] = entry[ 3]*v.comp[0] + entry[ 7]*v.comp[1] + entry[11]*v.comp[2] + entry[15]*v.comp[3];
	return product;
}


// -----------------------------------------------------------------------------
// ----------------------------- Matrix4::operator/ ----------------------------
// -----------------------------------------------------------------------------
Matrix4 Matrix4::operator/(float scalar) const
{
	// TODO: if scalar to small, set values to MAX_REAL
	Matrix4 quotient;
	float reciprocal = 1 / scalar;
	for (int i = 0; i < 16; ++i)
		quotient.entry[i] = reciprocal * entry[i];
	return quotient;
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix4::operator/= ----------------------------
// -----------------------------------------------------------------------------
const Matrix4& Matrix4::operator/=(float scalar)
{
	// TODO: if scalar to small, set values to MAX_REAL
	float reciprocal = 1 / scalar;
	for (int i = 0; i < 16; ++i)
		entry[i] *= reciprocal;
	return (*this);
}


// -----------------------------------------------------------------------------
// ----------------------------- Matrix4::Transposed ---------------------------
// -----------------------------------------------------------------------------
const Matrix4 Matrix4::Transposed(void) const
{
	Matrix4 transpose;
	transpose.entry[ 0] = entry[ 0];
	transpose.entry[ 1] = entry[ 4];
	transpose.entry[ 2] = entry[ 8];
	transpose.entry[ 3] = entry[12];
	transpose.entry[ 4] = entry[ 1];
	transpose.entry[ 5] = entry[ 5];
	transpose.entry[ 6] = entry[ 9];
	transpose.entry[ 7] = entry[13];
	transpose.entry[ 8] = entry[ 2];
	transpose.entry[ 9] = entry[ 6];
	transpose.entry[10] = entry[10];
	transpose.entry[11] = entry[14];
	transpose.entry[12] = entry[ 3];
	transpose.entry[13] = entry[ 7];
	transpose.entry[14] = entry[11];
	transpose.entry[15] = entry[15];
	return transpose;
}


// -----------------------------------------------------------------------------
// ------------------------------ Matrix4::Inverse -----------------------------
// -----------------------------------------------------------------------------
const Matrix4 Matrix4::Inverse(void) const
{
	float a0 = entry[ 0]*entry[ 5] - entry[ 1]*entry[ 4];
	float a1 = entry[ 0]*entry[ 6] - entry[ 2]*entry[ 4];
	float a2 = entry[ 0]*entry[ 7] - entry[ 3]*entry[ 4];
	float a3 = entry[ 1]*entry[ 6] - entry[ 2]*entry[ 5];
	float a4 = entry[ 1]*entry[ 7] - entry[ 3]*entry[ 5];
	float a5 = entry[ 2]*entry[ 7] - entry[ 3]*entry[ 6];
	float b0 = entry[ 8]*entry[13] - entry[ 9]*entry[12];
	float b1 = entry[ 8]*entry[14] - entry[10]*entry[12];
	float b2 = entry[ 8]*entry[15] - entry[11]*entry[12];
	float b3 = entry[ 9]*entry[14] - entry[10]*entry[13];
	float b4 = entry[ 9]*entry[15] - entry[11]*entry[13];
	float b5 = entry[10]*entry[15] - entry[11]*entry[14];
	float determinant = a0*b5 - a1*b4 + a2*b3 + a3*b2 - a4*b1 + a5*b0;

	Matrix4 inverse;
	inverse.entry[ 0] = + entry[ 5]*b5 - entry[ 6]*b4 + entry[ 7]*b3;
	inverse.entry[ 4] = - entry[ 4]*b5 + entry[ 6]*b2 - entry[ 7]*b1;
	inverse.entry[ 8] = + entry[ 4]*b4 - entry[ 5]*b2 + entry[ 7]*b0;
	inverse.entry[12] = - entry[ 4]*b3 + entry[ 5]*b1 - entry[ 6]*b0;
	inverse.entry[ 1] = - entry[ 1]*b5 + entry[ 2]*b4 - entry[ 3]*b3;
	inverse.entry[ 5] = + entry[ 0]*b5 - entry[ 2]*b2 + entry[ 3]*b1;
	inverse.entry[ 9] = - entry[ 0]*b4 + entry[ 1]*b2 - entry[ 3]*b0;
	inverse.entry[13] = + entry[ 0]*b3 - entry[ 1]*b1 + entry[ 2]*b0;
	inverse.entry[ 2] = + entry[13]*a5 - entry[14]*a4 + entry[15]*a3;
	inverse.entry[ 6] = - entry[12]*a5 + entry[14]*a2 - entry[15]*a1;
	inverse.entry[10] = + entry[12]*a4 - entry[13]*a2 + entry[15]*a0;
	inverse.entry[14] = - entry[12]*a3 + entry[13]*a1 - entry[14]*a0;
	inverse.entry[ 3] = - entry[ 9]*a5 + entry[10]*a4 - entry[11]*a3;
	inverse.entry[ 7] = + entry[ 8]*a5 - entry[10]*a2 + entry[11]*a1;
	inverse.entry[11] = - entry[ 8]*a4 + entry[ 9]*a2 - entry[11]*a0;
	inverse.entry[15] = + entry[ 8]*a3 - entry[ 9]*a1 + entry[10]*a0;

	float reciprocalDeterminant = 1 / determinant;
	for (int i = 0; i < 16; ++i)
		inverse.entry[i] *= reciprocalDeterminant;

	return inverse;
}


// -----------------------------------------------------------------------------
// ------------------------------ Materix4::Invert -----------------------------
// -----------------------------------------------------------------------------
void Matrix4::Invert(void)
{
	*this = Inverse();
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix4::Determinant ---------------------------
// -----------------------------------------------------------------------------
float Matrix4::Determinant(void) const
{
	float a0 = entry[ 0]*entry[ 5] - entry[ 1]*entry[ 4];
	float a1 = entry[ 0]*entry[ 6] - entry[ 2]*entry[ 4];
	float a2 = entry[ 0]*entry[ 7] - entry[ 3]*entry[ 4];
	float a3 = entry[ 1]*entry[ 6] - entry[ 2]*entry[ 5];
	float a4 = entry[ 1]*entry[ 7] - entry[ 3]*entry[ 5];
	float a5 = entry[ 2]*entry[ 7] - entry[ 3]*entry[ 6];
	float b0 = entry[ 8]*entry[13] - entry[ 9]*entry[12];
	float b1 = entry[ 8]*entry[14] - entry[10]*entry[12];
	float b2 = entry[ 8]*entry[15] - entry[11]*entry[12];
	float b3 = entry[ 9]*entry[14] - entry[10]*entry[13];
	float b4 = entry[ 9]*entry[15] - entry[11]*entry[13];
	float b5 = entry[10]*entry[15] - entry[11]*entry[14];
	float determinant = a0*b5 - a1*b4 + a2*b3 + a3*b2 - a4*b1 + a5*b0;
	return determinant;
}


// -----------------------------------------------------------------------------
// ------------------------- Matrix4::BuildTranslation -------------------------
// -----------------------------------------------------------------------------
void Matrix4::BuildTranslation(const Vector3& translation)
{
	entry[ 0] = 1.0;
	entry[ 1] = 0.0;
	entry[ 2] = 0.0;
	entry[ 3] = 0.0;

	entry[ 4] = 0.0;
	entry[ 5] = 1.0;
	entry[ 6] = 0.0;
	entry[ 7] = 0.0;

	entry[ 8] = 0.0;
	entry[ 9] = 0.0;
	entry[10] = 1.0;
	entry[11] = 0.0;

	entry[12] = translation.x;
	entry[13] = translation.y;
	entry[14] = translation.z;
	entry[15] = 1.0;
}


// -----------------------------------------------------------------------------
// ------------------------- Matrix4::BuildTranslation -------------------------
// -----------------------------------------------------------------------------
void Matrix4::BuildTranslation(float x, float y, float z)
{
	BuildTranslation(Vector3(x, y, z));
}


// -----------------------------------------------------------------------------
// -------------------------- Matrix4::SetTranslation --------------------------
// -----------------------------------------------------------------------------
void Matrix4::SetTranslation(float x, float y, float z)
{
	entry[12] = x;
	entry[13] = y;
	entry[14] = z;
}


// -----------------------------------------------------------------------------
// ----------------------------- Matrix4::Translate ----------------------------
// -----------------------------------------------------------------------------
void Matrix4::Translate(const Vector3& translation)
{
	entry[12] += translation.x;
	entry[13] += translation.y;
	entry[14] += translation.z;
}


// -----------------------------------------------------------------------------
// -------------------------- Matrix4::GetTranslation --------------------------
// -----------------------------------------------------------------------------
Vector3 Matrix4::GetTranslation(void) const
{
	return Vector3(entry[12], entry[13], entry[14]);
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix4::BuildScale ----------------------------
// -----------------------------------------------------------------------------
void Matrix4::BuildScale(float x, float y, float z)
{
	entry[ 0] = x;
	entry[ 1] = 0.0;
	entry[ 2] = 0.0;
	entry[ 3] = 0.0;

	entry[ 4] = 0.0;
	entry[ 5] = y;
	entry[ 6] = 0.0;
	entry[ 7] = 0.0;

	entry[ 8] = 0.0;
	entry[ 9] = 0.0;
	entry[10] = z;
	entry[11] = 0.0;

	entry[12] = 0.0;
	entry[13] = 0.0;
	entry[14] = 0.0;
	entry[15] = 1.0;
}


// -----------------------------------------------------------------------------
// -------------------------- Matrix4::OrthoNormalize --------------------------
// -----------------------------------------------------------------------------
void Matrix4::OrthoNormalize(void)
{
	Vector3 x = GetRight();
	Vector3 y = GetUp();
	Vector3 z = GetDir();

	x.Normalize();	
	y -= x * x.DotProduct(y);
	y.Normalize();
	z = x.CrossProduct(y);

	SetRight(x);
	SetUp(y);
	SetDir(z);
}


// -----------------------------------------------------------------------------
// ------------------------------- Matrix4::Scale ------------------------------
// -----------------------------------------------------------------------------
void Matrix4::Scale(float x, float y, float z)
{
	entry[ 0] *= x;
	entry[ 4] *= x;
	entry[ 8] *= x;

	entry[ 1] *= y;
	entry[ 5] *= y;
	entry[ 9] *= y;

	entry[ 2] *= z;
	entry[ 6] *= z;
	entry[10] *= z;
}


// -----------------------------------------------------------------------------
// ----------------------------- Matrix4::SetScale -----------------------------
// -----------------------------------------------------------------------------
void Matrix4::SetScale(float x, float y, float z)
{
	Vector3 currentScale = GetScale();

	float fX = x / currentScale.x;
	float fY = y / currentScale.y;
	float fZ = z / currentScale.z;

	Scale(fX, fY, fZ);
}


// -----------------------------------------------------------------------------
// ----------------------------- Matrix4::GetScale -----------------------------
// -----------------------------------------------------------------------------
Vector3 Matrix4::GetScale(void) const
{
	float sx = GetRight().Length();
	float sy = GetUp().Length();
	float sz = GetDir().Length();
	return Vector3(sx, sy, sz);
}


// -----------------------------------------------------------------------------
// ----------------------------- Matrix4::GetRight -----------------------------
// -----------------------------------------------------------------------------
Vector3 Matrix4::GetRight(void) const
{
	return Vector3(entry[0], entry[1], entry[2]);
}


// -----------------------------------------------------------------------------
// ------------------------------- Matrix4::GetUp ------------------------------
// -----------------------------------------------------------------------------
Vector3 Matrix4::GetUp(void) const
{
	return Vector3(entry[4], entry[5], entry[6]);
}


// -----------------------------------------------------------------------------
// ------------------------------ Matrix4::GetDir ------------------------------
// -----------------------------------------------------------------------------
Vector3 Matrix4::GetDir(void) const
{
	return Vector3(entry[8], entry[9], entry[10]);
}


// -----------------------------------------------------------------------------
// ---------------------------------- SetRight ---------------------------------
// -----------------------------------------------------------------------------
void Matrix4::SetRight(const Vector3& right)
{
	entry[0] = right.x;
	entry[1] = right.y;
	entry[2] = right.z;
}


// -----------------------------------------------------------------------------
// ----------------------------------- SetUp -----------------------------------
// -----------------------------------------------------------------------------
void Matrix4::SetUp(const Vector3& up)
{
	entry[4] = up.x;
	entry[5] = up.y;
	entry[6] = up.z;
}


// -----------------------------------------------------------------------------
// ----------------------------------- SetDir ----------------------------------
// -----------------------------------------------------------------------------
void Matrix4::SetDir(const Vector3& dir)
{
	entry[8] = dir.x;
	entry[9] = dir.y;
	entry[10] = dir.z;
}


// -----------------------------------------------------------------------------
// -------------------------- Matrix4::BuildRotationX --------------------------
// -----------------------------------------------------------------------------
void Matrix4::BuildRotationX(float angle)
{
	float sinAngle = Math::Sin(angle);
	float cosAngle = Math::Cos(angle);

	entry[ 0] = 1.0;
	entry[ 1] = 0.0;
	entry[ 2] = 0.0;
	entry[ 3] = 0.0;

	entry[ 4] = 0.0;
	entry[ 5] = cosAngle;
	entry[ 6] = sinAngle;
	entry[ 7] = 0.0;

	entry[ 8] = 0.0;
	entry[ 9] = -sinAngle;
	entry[10] = cosAngle;
	entry[11] = 0.0;

	entry[12] = 0.0;
	entry[13] = 0.0;
	entry[14] = 0.0;
	entry[15] = 1.0;
}


// -----------------------------------------------------------------------------
// -------------------------- Matrix4::BuildRotationY --------------------------
// -----------------------------------------------------------------------------
void Matrix4::BuildRotationY(float angle)
{
	float sinAngle = Math::Sin(angle);
	float cosAngle = Math::Cos(angle);

	entry[ 0] = cosAngle;
	entry[ 1] = 0.0;
	entry[ 2] = -sinAngle;
	entry[ 3] = 0.0;

	entry[ 4] = 0.0;
	entry[ 5] = 1.0;
	entry[ 6] = 0.0;
	entry[ 7] = 0.0;

	entry[ 8] = sinAngle;
	entry[ 9] = 0.0;
	entry[10] = cosAngle;
	entry[11] = 0.0;

	entry[12] = 0.0;
	entry[13] = 0.0;
	entry[14] = 0.0;
	entry[15] = 1.0;
}


// -----------------------------------------------------------------------------
// -------------------------- Matrix4::BuildRotationZ --------------------------
// -----------------------------------------------------------------------------
void Matrix4::BuildRotationZ(float angle)
{
	float sinAngle = Math::Sin(angle);
	float cosAngle = Math::Cos(angle);

	entry[ 0] = cosAngle;
	entry[ 1] = sinAngle;
	entry[ 2] = 0.0;
	entry[ 3] = 0.0;

	entry[ 4] = -sinAngle;
	entry[ 5] = cosAngle;
	entry[ 6] = 0.0;
	entry[ 7] = 0.0;

	entry[ 8] = 0.0;
	entry[ 9] = 0.0;
	entry[10] = 1.0;
	entry[11] = 0.0;

	entry[12] = 0.0;
	entry[13] = 0.0;
	entry[14] = 0.0;
	entry[15] = 1.0;
}


// -----------------------------------------------------------------------------
// --------------------------- Matrix4::BuildRotation --------------------------
// -----------------------------------------------------------------------------
void Matrix4::BuildRotation(float yaw, float pitch, float roll)
{
	Matrix4 temp = Matrix4::IDENTITY;
	if (yaw != 0)
	{
		Matrix4 rotationX;
		rotationX.BuildRotationX(yaw);
		temp = temp * rotationX;
	}
	if (pitch != 0)
	{
		Matrix4 rotationY;
		rotationY.BuildRotationY(pitch);
		temp = temp * rotationY;
	}
	if (roll != 0)
	{
		Matrix4 rotationZ;
		rotationZ.BuildRotationZ(roll);
		temp = temp * rotationZ;
	}
	*this = temp;
}

// -----------------------------------------------------------------------------
// ---------------------------- Matrix4::GetRotation ---------------------------
// -----------------------------------------------------------------------------
Matrix3 Matrix4::GetRotation(void) const
{
	Matrix3 mat;
	mat[0][0] = (*this)[0][0];
	mat[0][1] = (*this)[0][1];
	mat[0][2] = (*this)[0][2];

	mat[1][0] = (*this)[1][0];
	mat[1][1] = (*this)[1][1];
	mat[1][2] = (*this)[1][2];

	mat[2][0] = (*this)[2][0];
	mat[2][1] = (*this)[2][1];
	mat[2][2] = (*this)[2][2];
	return mat;
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix4::SetRotation ---------------------------
// -----------------------------------------------------------------------------
void Matrix4::SetRotation(const Matrix3& mat)
{
	(*this)[0][0] = mat[0][0];
	(*this)[0][1] = mat[0][1];
	(*this)[0][2] = mat[0][2];

	(*this)[1][0] = mat[1][0];
	(*this)[1][1] = mat[1][1];
	(*this)[1][2] = mat[1][2];

	(*this)[2][0] = mat[2][0];
	(*this)[2][1] = mat[2][1];
	(*this)[2][2] = mat[2][2];
}


// -----------------------------------------------------------------------------
// ------------------------- Matrix4::BuildPerspective -------------------------
// -----------------------------------------------------------------------------
void Matrix4::BuildPerspective(float fieldOfView, 
							   float aspectRatio,
							   float nearPlaneZ, 
							   float farPlaneZ,
							   bool leftHanded)
{
	float f = Math::Cot(fieldOfView / (float)2.0);
	float reciprocalDenumerator = (float)1.0 / (nearPlaneZ-farPlaneZ);

	float sign = 1.0f;
	if (!leftHanded)
		sign = -1.0f;

	entry[ 0] = f / aspectRatio;
	entry[ 1] = 0.0;
	entry[ 2] = 0.0;
	entry[ 3] = 0.0;

	entry[ 4] = 0.0;
	entry[ 5] = f;
	entry[ 6] = 0.0;
	entry[ 7] = 0.0;

	entry[ 8] = 0.0;
	entry[ 9] = 0.0;

	entry[10] = -farPlaneZ * reciprocalDenumerator*sign;
	entry[11] = sign;

	entry[12] = 0.0;
	entry[13] = 0.0;
	entry[14] = /*2 **/ nearPlaneZ * farPlaneZ * reciprocalDenumerator;
	entry[15] = 0.0;
}


// -----------------------------------------------------------------------------
// ----------------------------- Matrix4::BuildLook ----------------------------
// -----------------------------------------------------------------------------
void Matrix4::BuildLookDir(const Vector3& position, 
						 const Vector3& direction,
						 const Vector3& up)
{
	Vector3 directionNormalized = direction;
	directionNormalized.Normalize();

	Vector3 upNormalized = up;
	upNormalized.Normalize();

	Vector3 cross = directionNormalized.UnitCrossProduct(upNormalized);
	upNormalized = cross.UnitCrossProduct(directionNormalized);
	
	entry[ 0] = -cross.x;
	entry[ 1] = upNormalized.x;
	entry[ 2] = directionNormalized.x;
	entry[ 3] = 0.0;

	entry[ 4] = -cross.y;
	entry[ 5] = upNormalized.y;
	entry[ 6] = directionNormalized.y;
	entry[ 7] = 0.0;

	entry[ 8] = -cross.z;
	entry[ 9] = upNormalized.z;
	entry[10] = directionNormalized.z;
	entry[11] = 0.0;

	entry[12] = cross.DotProduct(position);
	entry[13] = -upNormalized.DotProduct(position);
	entry[14] = -directionNormalized.DotProduct(position);
	entry[15] = 1.0;
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix4::BuildLookAt ---------------------------
// -----------------------------------------------------------------------------
void Matrix4::BuildLookAt(const Vector3& position, const Vector3& at, const Vector3& up)
{
	Vector3 forward = at - position;
	Vector3 upVec = up;

	forward.Normalize();

	// side = forward x up
	Vector3 side = forward.CrossProduct(upVec);
	side.Normalize();

	// recompute up as: up = side x forward
	upVec = side.CrossProduct(forward);

	entry[0] = side.x;
	entry[1] = upVec.x;
	entry[2] = -forward.x;
	entry[3] = 0;

	entry[4] = side.y;
	entry[5] = upVec.y;
	entry[6] = -forward.y;
	entry[7] = 0;

	entry[8] = side.z;
	entry[9] = upVec.z;
	entry[10] = -forward.z;
	entry[11] = 0;

	entry[12] = side.DotProduct(position);
	entry[13] = upVec.DotProduct(position);
	entry[14] = forward.DotProduct(position);
	entry[15] = 1;
}


// -----------------------------------------------------------------------------
// ---------------------- Matrix4::BuildPlaneMirrorMatrix ----------------------
// -----------------------------------------------------------------------------
void Matrix4::BuildPlaneMirrorMatrix(const Vector4& planeEquation)
{
	entry[ 0] = 1-2*planeEquation.x*planeEquation.x;
	entry[ 1] = -2*planeEquation.y*planeEquation.x;
	entry[ 2] = -2*planeEquation.z*planeEquation.x;
	entry[ 3] = 0.0;

	entry[ 4] =  -2*planeEquation.x*planeEquation.y;
	entry[ 5] = 1-2*planeEquation.y*planeEquation.y;
	entry[ 6] = -2*planeEquation.z*planeEquation.y;
	entry[ 7] = 0.0;

	entry[ 8] = -2*planeEquation.x*planeEquation.z;
	entry[ 9] =  -2*planeEquation.y*planeEquation.z;
	entry[10] = 1-2*planeEquation.z*planeEquation.z;
	entry[11] = 0.0;

	entry[12] = -2*planeEquation.x*planeEquation.w;
	entry[13] = -2*planeEquation.y*planeEquation.w;
	entry[14] = -2*planeEquation.z*planeEquation.w;
	entry[15] = 1.0;
}


// -----------------------------------------------------------------------------
// -------------------------- Matrix4::BuildOrthogonal -------------------------
// -----------------------------------------------------------------------------
void Matrix4::BuildOrthogonal(float width, float height, float zNear, float zFar)
{
	entry[ 0] = 2/width;
	entry[ 1] = 0.0;
	entry[ 2] = 0.0;
	entry[ 3] = 0.0;

	entry[ 4] = 0.0;
	entry[ 5] = 2/height;
	entry[ 6] = 0.0;
	entry[ 7] = 0.0;

	entry[ 8] = 0.0;
	entry[ 9] = 0.0;

	entry[10] = 1/(zFar-zNear);

	entry[11] = 0.0;

	entry[12] = 0.0;
	entry[13] = 0.0;

	entry[14] = zNear/(zNear-zFar);

	entry[15] = 1.0;
}


// -----------------------------------------------------------------------------
// ------------------------- Matrix4::Matrix4Translation -----------------------
// -----------------------------------------------------------------------------
Matrix4 Matrix4::Matrix4Translation(const Vector3& translation)
{
	Matrix4 m;
	m.BuildTranslation(translation);
	return m;
}


// -----------------------------------------------------------------------------
// ------------------------- Matrix4::Matrix4Translation -----------------------
// -----------------------------------------------------------------------------
Matrix4 Matrix4::Matrix4Translation(float x, float y, float z)
{
	Matrix4 m;
	m.BuildTranslation(x, y, z);
	return m;
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix4::Matrix4Scale --------------------------
// -----------------------------------------------------------------------------
Matrix4 Matrix4::Matrix4Scale(float x, float y, float z)
{
	Matrix4 m;
	m.BuildScale(x,y,z);
	return m;
}


// -----------------------------------------------------------------------------
// ---------------------------- Matrix4::Matrix4Scale --------------------------
// -----------------------------------------------------------------------------
Matrix4 Matrix4::Matrix4Scale(const Vector3& scale)
{
	Matrix4 m;
	m.BuildScale(scale.x, scale.y, scale.z);
	return m;
}


// -----------------------------------------------------------------------------
// -------------------------- Matrix4::Matrix4Rotation -------------------------
// -----------------------------------------------------------------------------
Matrix4 Matrix4::Matrix4Rotation(float yaw, float pitch, float roll)
{
	Matrix4 m;
	m.BuildRotation(yaw, pitch, roll);
	return m;
}


// -----------------------------------------------------------------------------
// -------------------------- Matrix4::Matrix4RotationX ------------------------
// -----------------------------------------------------------------------------
Matrix4 Matrix4::Matrix4RotationX(float angle)
{
	Matrix4 m;
	m.BuildRotationX(angle);
	return m;
}


// -----------------------------------------------------------------------------
// -------------------------- Matrix4::Matrix4RotationY ------------------------
// -----------------------------------------------------------------------------
Matrix4 Matrix4::Matrix4RotationY(float angle)
{
	Matrix4 m;
	m.BuildRotationY(angle);
	return m;
}


// -----------------------------------------------------------------------------
// -------------------------- Matrix4::Matrix4RotationZ ------------------------
// -----------------------------------------------------------------------------
Matrix4 Matrix4::Matrix4RotationZ(float angle)
{
	Matrix4 m;
	m.BuildRotationZ(angle);
	return m;
}


// -----------------------------------------------------------------------------
// ------------------------- Matrix4::Matrix4Perspective -----------------------
// -----------------------------------------------------------------------------
Matrix4 Matrix4::Matrix4Perspective(float fieldOfView, float aspectRatio, float nearPlaneZ, float farPlaneZ, bool leftHanded /*= true*/)
{
	Matrix4 m;
	m.BuildPerspective(fieldOfView, aspectRatio, nearPlaneZ, farPlaneZ, leftHanded);
	return m;
}


// -----------------------------------------------------------------------------
// --------------------------- Matrix4::Matrix4LookDir -------------------------
// -----------------------------------------------------------------------------
Matrix4 Matrix4::Matrix4LookDir(const Vector3& position, const Vector3& direction, const Vector3& up)
{
	Matrix4 m;
	m.BuildLookDir(position, direction, up);
	return m;
}


// -----------------------------------------------------------------------------
// --------------------------- Matrix4::Matrix4LookAt --------------------------
// -----------------------------------------------------------------------------
Matrix4 Matrix4::Matrix4LookAt(const Vector3& position, const Vector3& at, const Vector3& up)
{
	Matrix4 m;
	m.BuildLookAt(position, at, up);
	return m;
}


// -----------------------------------------------------------------------------
// ---------------------- Matrix4::Matrix4PlaneMirrorMatrix --------------------
// -----------------------------------------------------------------------------
Matrix4 Matrix4::Matrix4PlaneMirrorMatrix(const Vector4& planeEquation)
{
	Matrix4 m;
	m.BuildPlaneMirrorMatrix(planeEquation);
	return m;
}


// -----------------------------------------------------------------------------
// ------------------------- Matrix4::Matrix4Orthogonal ------------------------
// -----------------------------------------------------------------------------
Matrix4 Matrix4::Matrix4Orthogonal(float width, float height, float zNear, float zFar)
{
	Matrix4 m;
	m.BuildOrthogonal(width, height, zNear, zFar);
	return m;
}
