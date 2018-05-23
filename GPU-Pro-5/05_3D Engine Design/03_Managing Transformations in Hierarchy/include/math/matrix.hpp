#ifndef MATH_MATRIX_HPP
#define MATH_MATRIX_HPP

#include <iomanip>

#include <math/vector.hpp>

#include <essentials/string.hpp>



class Quaternion;



class Matrix
{
	friend Matrix operator * (float f, const Matrix& m);

public:
	struct ZRange { enum TYPE { ZeroToOne, MinusOneToPlusOne }; };

public:
	static Matrix Identity()
	{
		Matrix temp;
		temp.LoadIdentity();
		return temp;
	}

	static Matrix Zeroes()
	{
		Matrix temp;
		temp.LoadZeroes();
		return temp;
	}

	static Matrix LookAtLH(const Vector3& eye, const Vector3& at, const Vector3& up)
	{
		Matrix temp;
		temp.LoadLookAtLH(eye, at, up);
		return temp;
	}

	static Matrix LookAtRH(const Vector3& eye, const Vector3& at, const Vector3& up)
	{
		Matrix temp;
		temp.LoadLookAtRH(eye, at, up);
		return temp;
	}

	static Matrix PerspectiveFovLH(float fovY, float aspectRatio, float zNear, float zFar)
	{
		Matrix temp;
		temp.LoadPerspectiveFovLH(fovY, aspectRatio, zNear, zFar);
		return temp;
	}

	static Matrix PerspectiveFovRH(float fovY, float aspectRatio, float zNear, float zFar)
	{
		Matrix temp;
		temp.LoadPerspectiveFovRH(fovY, aspectRatio, zNear, zFar);
		return temp;
	}

	static Matrix OrthoOffCenterLH(float left, float right, float bottom, float top, float zNear, float zFar)
	{
		Matrix temp;
		temp.LoadOrthoOffCenterLH(left, right, bottom, top, zNear, zFar);
		return temp;
	}

	static Matrix OrthoOffCenterRH(float left, float right, float bottom, float top, float zNear, float zFar)
	{
		Matrix temp;
		temp.LoadOrthoOffCenterRH(left, right, bottom, top, zNear, zFar);
		return temp;
	}

	static Matrix OrthoLH(float width, float height, float zNear, float zFar)
	{
		Matrix temp;
		temp.LoadOrthoLH(width, height, zNear, zFar);
		return temp;
	}

	static Matrix OrthoRH(float width, float height, float zNear, float zFar)
	{
		Matrix temp;
		temp.LoadOrthoRH(width, height, zNear, zFar);
		return temp;
	}

	static Matrix Translate(float tx, float ty, float tz)
	{
		Matrix temp;
		temp.LoadTranslate(tx, ty, tz);
		return temp;
	}

	static Matrix Translate(const Vector3& v)
	{
		return Translate(v.x, v.y, v.z);
	}

	static Matrix Rotate(float rx, float ry, float rz, float angle)
	{
		Matrix temp;
		temp.LoadRotate(rx, ry, rz, angle);
		return temp;
	}

	static Matrix Rotate(const Vector3& v, float angle)
	{
		return Rotate(v.x, v.y, v.z, angle);
	}

	static Matrix RotateX(float angle)
	{
		Matrix temp;
		temp.LoadRotateX(angle);
		return temp;
	}

	static Matrix RotateY(float angle)
	{
		Matrix temp;
		temp.LoadRotateY(angle);
		return temp;
	}

	static Matrix RotateZ(float angle)
	{
		Matrix temp;
		temp.LoadRotateZ(angle);
		return temp;
	}

	static Matrix Scale(float sx, float sy, float sz)
	{
		Matrix temp;
		temp.LoadScale(sx, sy, sz);
		return temp;
	}

	static Matrix Scale(const Vector3& v)
	{
		return Scale(v.x, v.y, v.z);
	}

	static Matrix Reflect(const Vector3& planePoint, Vector3 planeNormal)
	{
		Matrix temp;
		temp.LoadReflect(planePoint, planeNormal);
		return temp;
	}

public:
	static ZRange::TYPE zRange;

public:
	Matrix() {}
	Matrix(float _00_11_22_33);
	Matrix(float _00, float _01, float _02, float _03,
		   float _10, float _11, float _12, float _13,
		   float _20, float _21, float _22, float _23,
		   float _30, float _31, float _32, float _33);
	Matrix(const Vector4& row1, const Vector4& row2, const Vector4& row3, const Vector4& row4);
	Matrix(const Matrix& m);

	bool IsOrthonormal() const;

	void Orthogonalize3x3();
	Matrix GetOrthogonalized3x3() const;

	void Transpose();
	Matrix GetTransposed() const;

	void Inverse();
	Matrix GetInversed() const;

	Quaternion GetQuaternion();

	void LoadIdentity();
	void LoadZeroes();

	void LoadLookAtLH(const Vector3& eye, const Vector3& at, const Vector3& up);
	void LoadLookAtRH(const Vector3& eye, const Vector3& at, const Vector3& up);
	void LoadPerspectiveFovLH(float fovY, float aspectRatio, float zNear, float zFar);
	void LoadPerspectiveFovRH(float fovY, float aspectRatio, float zNear, float zFar);
	void LoadOrthoOffCenterLH(float left, float right, float bottom, float top, float zNear, float zFar);
	void LoadOrthoOffCenterRH(float left, float right, float bottom, float top, float zNear, float zFar);
	void LoadOrthoLH(float width, float height, float zNear, float zFar);
	void LoadOrthoRH(float width, float height, float zNear, float zFar);

	void LoadTranslate(float tx, float ty, float tz);
	void LoadTranslate(const Vector3& v) { LoadTranslate(v.x, v.y, v.z); }
	void LoadRotate(float rx, float ry, float rz, float angle);
	void LoadRotate(const Vector3& v, float angle) { LoadRotate(v.x, v.y, v.z, angle); }
	void LoadRotateX(float angle);
	void LoadRotateY(float angle);
	void LoadRotateZ(float angle);
	void LoadScale(float sx, float sy, float sz);
	void LoadScale(const Vector3& v) { LoadScale(v.x, v.y, v.z); }
	void LoadReflect(const Vector3& planePoint, const Vector3& planeNormal);

	float operator () (int row, int col) const { return _[row][col]; }
	float& operator () (int row, int col) { return _[row][col]; }

	Matrix& operator = (const Matrix& m);
	bool operator == (const Matrix& m) const;
	bool operator != (const Matrix& m) const;

	Matrix operator + (const Matrix& m) const;
	Matrix& operator += (const Matrix& m);

	Matrix operator * (const Matrix& m) const;
	Matrix& operator *= (const Matrix& m);

	Vector3 operator * (const Vector3& v) const;
	Vector4 operator * (const Vector4& v) const;

	Matrix operator * (float f) const;
	Matrix& operator *= (float f);

public:
	float _[4][4];
};



Matrix operator * (float f, const Matrix& m);



inline ostream & operator << (ostream &stream, const Matrix& m)
{
	int longestLength = 0;

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			int l = ToString(m(i, j)).length();

			if (l > longestLength)
				longestLength = l;
		}
	}

	int n = longestLength;

	stream << "[ " << left << setw(n) << m(0, 0) << "  " << left << setw(n) << m(0, 1) << "  " << left << setw(n) << m(0, 2) << "  " << m(0, 3) << " ]" << endl;
	stream << "[ " << left << setw(n) << m(1, 0) << "  " << left << setw(n) << m(1, 1) << "  " << left << setw(n) << m(1, 2) << "  " << m(1, 3) << " ]" << endl;
	stream << "[ " << left << setw(n) << m(2, 0) << "  " << left << setw(n) << m(2, 1) << "  " << left << setw(n) << m(2, 2) << "  " << m(2, 3) << " ]" << endl;
	stream << "[ " << left << setw(n) << m(3, 0) << "  " << left << setw(n) << m(3, 1) << "  " << left << setw(n) << m(3, 2) << "  " << m(3, 3) << " ]";

	return stream;
}



typedef Matrix mtx;



#endif
