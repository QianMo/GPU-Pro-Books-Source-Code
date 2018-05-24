#include <math/matrix.h>
#include <math/types.h>
#include <math/constants.h>
#include <math/vector.h>


NMath::Matrix NMath::MatrixCustom(
	float m00, float m01, float m02, float m03,
	float m10, float m11, float m12, float m13,
	float m20, float m21, float m22, float m23,
	float m30, float m31, float m32, float m33)
{
	Matrix temp;

	temp.m[0][0] = m00;
	temp.m[0][1] = m01;
	temp.m[0][2] = m02;
	temp.m[0][3] = m03;

	temp.m[1][0] = m10;
	temp.m[1][1] = m11;
	temp.m[1][2] = m12;
	temp.m[1][3] = m13;

	temp.m[2][0] = m20;
	temp.m[2][1] = m21;
	temp.m[2][2] = m22;
	temp.m[2][3] = m23;

	temp.m[3][0] = m30;
	temp.m[3][1] = m31;
	temp.m[3][2] = m32;
	temp.m[3][3] = m33;

	return temp;
}


NMath::Matrix NMath::MatrixCopy(const NMath::Matrix& m)
{
	return MatrixCustom(
		m.m[0][0], m.m[0][1], m.m[0][2], m.m[0][3],
		m.m[1][0], m.m[1][1], m.m[1][2], m.m[1][3],
		m.m[2][0], m.m[2][1], m.m[2][2], m.m[2][3],
		m.m[3][0], m.m[3][1], m.m[3][2], m.m[3][3]);
}


NMath::Matrix NMath::Add(const Matrix& m1, const Matrix& m2)
{
	Matrix temp;

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			temp.m[i][j] = m1.m[i][j] + m2.m[i][j];
		}
	}

	return temp;
}


NMath::Matrix NMath::Sub(const Matrix& m1, const Matrix& m2)
{
	Matrix temp;

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			temp.m[i][j] = m1.m[i][j] - m2.m[i][j];
		}
	}

	return temp;
}


NMath::Matrix NMath::Mul(const Matrix& m1, const Matrix& m2)
{
	Matrix temp;

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			temp.m[i][j] =
				m1.m[i][0] * m2.m[0][j] +
				m1.m[i][1] * m2.m[1][j] +
				m1.m[i][2] * m2.m[2][j] +
				m1.m[i][3] * m2.m[3][j];
		}
	}

	return temp;
}


NMath::Matrix NMath::Transpose(const Matrix& m)
{
	Matrix temp = m;
	TransposeIn(temp);
	return temp;
}


NMath::Matrix NMath::Invert(const Matrix& m)
{
	Matrix temp = m;
	InvertIn(temp);
	return temp;
}


NMath::Matrix NMath::Orthogonalize3x3(const Matrix& m)
{
	Matrix temp = m;
	Orthogonalize3x3In(temp);
	return temp;
}


void NMath::MulIn(Matrix& m1, const Matrix& m2)
{
	m1 = Mul(m1, m2);
}


void NMath::TransposeIn(Matrix& m)
{
	Matrix temp = m;

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			m.m[i][j] = temp.m[j][i];
		}
	}
}


void NMath::InvertIn(Matrix& m)
{
	float determinant =
		+m.m[0][0] * Determinant(
			m.m[1][1], m.m[1][2], m.m[1][3],
			m.m[2][1], m.m[2][2], m.m[2][3],
			m.m[3][1], m.m[3][2], m.m[3][3])
		-m.m[0][1] * Determinant(
			m.m[1][0], m.m[1][2], m.m[1][3],
			m.m[2][0], m.m[2][2], m.m[2][3],
			m.m[3][0], m.m[3][2], m.m[3][3])
		+m.m[0][2] * Determinant(
			m.m[1][0], m.m[1][1], m.m[1][3],
			m.m[2][0], m.m[2][1], m.m[2][3],
			m.m[3][0], m.m[3][1], m.m[3][3])
		-m.m[0][3] * Determinant(
			m.m[1][0], m.m[1][1], m.m[1][2],
			m.m[2][0], m.m[2][1], m.m[2][2],
			m.m[3][0], m.m[3][1], m.m[3][2]);

	if ( !(fabs(determinant) < Epsilon4) )
	{
		float adj[4][4];

		adj[0][0] = +Determinant(
			m.m[1][1], m.m[1][2], m.m[1][3],
			m.m[2][1], m.m[2][2], m.m[2][3],
			m.m[3][1], m.m[3][2], m.m[3][3]);
		adj[0][1] = -Determinant(
			m.m[1][0], m.m[1][2], m.m[1][3],
			m.m[2][0], m.m[2][2], m.m[2][3],
			m.m[3][0], m.m[3][2], m.m[3][3]);
		adj[0][2] = +Determinant(
			m.m[1][0], m.m[1][1], m.m[1][3],
			m.m[2][0], m.m[2][1], m.m[2][3],
			m.m[3][0], m.m[3][1], m.m[3][3]);
		adj[0][3] = -Determinant(
			m.m[1][0], m.m[1][1], m.m[1][2],
			m.m[2][0], m.m[2][1], m.m[2][2],
			m.m[3][0], m.m[3][1], m.m[3][2]);

		adj[1][0] = -Determinant(
			m.m[0][1], m.m[0][2], m.m[0][3],
			m.m[2][1], m.m[2][2], m.m[2][3],
			m.m[3][1], m.m[3][2], m.m[3][3]);
		adj[1][1] = +Determinant(
			m.m[0][0], m.m[0][2], m.m[0][3],
			m.m[2][0], m.m[2][2], m.m[2][3],
			m.m[3][0], m.m[3][2], m.m[3][3]);
		adj[1][2] = -Determinant(
			m.m[0][0], m.m[0][1], m.m[0][3],
			m.m[2][0], m.m[2][1], m.m[2][3],
			m.m[3][0], m.m[3][1], m.m[3][3]);
		adj[1][3] = +Determinant(
			m.m[0][0], m.m[0][1], m.m[0][2],
			m.m[2][0], m.m[2][1], m.m[2][2],
			m.m[3][0], m.m[3][1], m.m[3][2]);

		adj[2][0] = +Determinant(
			m.m[0][1], m.m[0][2], m.m[0][3],
			m.m[1][1], m.m[1][2], m.m[1][3],
			m.m[3][1], m.m[3][2], m.m[3][3]);
		adj[2][1] = -Determinant(
			m.m[0][0], m.m[0][2], m.m[0][3],
			m.m[1][0], m.m[1][2], m.m[1][3],
			m.m[3][0], m.m[3][2], m.m[3][3]);
		adj[2][2] = +Determinant(
			m.m[0][0], m.m[0][1], m.m[0][3],
			m.m[1][0], m.m[1][1], m.m[1][3],
			m.m[3][0], m.m[3][1], m.m[3][3]);
		adj[2][3] = -Determinant(
			m.m[0][0], m.m[0][1], m.m[0][2],
			m.m[1][0], m.m[1][1], m.m[1][2],
			m.m[3][0], m.m[3][1], m.m[3][2]);

		adj[3][0] = -Determinant(
			m.m[0][1], m.m[0][2], m.m[0][3],
			m.m[1][1], m.m[1][2], m.m[1][3],
			m.m[2][1], m.m[2][2], m.m[2][3]);
		adj[3][1] = +Determinant(
			m.m[0][0], m.m[0][2], m.m[0][3],
			m.m[1][0], m.m[1][2], m.m[1][3],
			m.m[2][0], m.m[2][2], m.m[2][3]);
		adj[3][2] = -Determinant(
			m.m[0][0], m.m[0][1], m.m[0][3],
			m.m[1][0], m.m[1][1], m.m[1][3],
			m.m[2][0], m.m[2][1], m.m[2][3]);
		adj[3][3] = +Determinant(
			m.m[0][0], m.m[0][1], m.m[0][2],
			m.m[1][0], m.m[1][1], m.m[1][2],
			m.m[2][0], m.m[2][1], m.m[2][2]);

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				m.m[i][j] = (1.0f / determinant) * adj[j][i];
			}
		}
	}
}


void NMath::Orthogonalize3x3In(Matrix& m)
{
	Vector3 v1 = VectorCustom(m.m[0][0], m.m[0][1], m.m[0][2]);
	Vector3 v2 = VectorCustom(m.m[1][0], m.m[1][1], m.m[1][2]);
	Vector3 v3 = VectorCustom(m.m[2][0], m.m[2][1], m.m[2][2]);

	Vector3 u1 = v1;
	Vector3 u2 = v2 - (Dot(v2, u1)/Dot(u1, u1))*u1;
	Vector3 u3 = v3 - (Dot(v3, u1)/Dot(u1, u1))*u1 - (Dot(v3, u2)/Dot(u2, u2))*u2;

	m.m[0][0] = u1.x;
	m.m[0][1] = u1.y;
	m.m[0][2] = u1.z;

	m.m[1][0] = u2.x;
	m.m[1][1] = u2.y;
	m.m[1][2] = u2.z;

	m.m[2][0] = u3.x;
	m.m[2][1] = u3.y;
	m.m[2][2] = u3.z;
}


NMath::Matrix NMath::operator + (const Matrix& m1, const Matrix& m2)
{
	return Add(m1, m2);
}


NMath::Matrix NMath::operator - (const Matrix& m1, const Matrix& m2)
{
	return Sub(m1, m2);
}


NMath::Matrix NMath::operator * (const Matrix& m1, const Matrix& m2)
{
	return Mul(m1, m2);
}


void NMath::SetZeros(Matrix& m)
{
	m.m[0][0] = 0.0f;
	m.m[0][1] = 0.0f;
	m.m[0][2] = 0.0f;
	m.m[0][3] = 0.0f;

	m.m[1][0] = 0.0f;
	m.m[1][1] = 0.0f;
	m.m[1][2] = 0.0f;
	m.m[1][3] = 0.0f;

	m.m[2][0] = 0.0f;
	m.m[2][1] = 0.0f;
	m.m[2][2] = 0.0f;
	m.m[2][3] = 0.0f;

	m.m[3][0] = 0.0f;
	m.m[3][1] = 0.0f;
	m.m[3][2] = 0.0f;
	m.m[3][3] = 0.0f;
}


void NMath::SetIdentity(Matrix& m)
{
	m.m[0][0] = 1.0f;
	m.m[0][1] = 0.0f;
	m.m[0][2] = 0.0f;
	m.m[0][3] = 0.0f;

	m.m[1][0] = 0.0f;
	m.m[1][1] = 1.0f;
	m.m[1][2] = 0.0f;
	m.m[1][3] = 0.0f;

	m.m[2][0] = 0.0f;
	m.m[2][1] = 0.0f;
	m.m[2][2] = 1.0f;
	m.m[2][3] = 0.0f;

	m.m[3][0] = 0.0f;
	m.m[3][1] = 0.0f;
	m.m[3][2] = 0.0f;
	m.m[3][3] = 1.0f;
}


void NMath::SetTranslate(Matrix& m, float x, float y, float z)
{
	m.m[0][0] = 1.0f;
	m.m[0][1] = 0.0f;
	m.m[0][2] = 0.0f;
	m.m[0][3] = 0.0f;

	m.m[1][0] = 0.0f;
	m.m[1][1] = 1.0f;
	m.m[1][2] = 0.0f;
	m.m[1][3] = 0.0f;

	m.m[2][0] = 0.0f;
	m.m[2][1] = 0.0f;
	m.m[2][2] = 1.0f;
	m.m[2][3] = 0.0f;

	m.m[3][0] = x;
	m.m[3][1] = y;
	m.m[3][2] = z;
	m.m[3][3] = 1.0f;
}


void NMath::SetTranslate(Matrix& m, const Vector3& v)
{
	SetTranslate(m, v.x, v.y, v.z);
}


void NMath::SetRotate(Matrix& m, float x, float y, float z, float angle)
{
	float s = sinf(angle);
	float c = cosf(angle);

	m.m[0][0] = c + x*x*(1-c);
	m.m[0][1] = x*y*(1-c) + z*s;
	m.m[0][2] = x*z*(1-c) - y*s;
	m.m[0][3] = 0.0f;

	m.m[1][0] = x*y*(1-c) - z*s;
	m.m[1][1] = c + y*y*(1-c);
	m.m[1][2] = y*z*(1-c) + x*s;
	m.m[1][3] = 0.0f;

	m.m[2][0] = x*z*(1-c) + y*s;
	m.m[2][1] = y*z*(1-c) - x*s;
	m.m[2][2] = c + z*z*(1-c);
	m.m[2][3] = 0.0f;

	m.m[3][0] = 0.0f;
	m.m[3][1] = 0.0f;
	m.m[3][2] = 0.0f;
	m.m[3][3] = 1.0f;
}


void NMath::SetRotate(Matrix& m, const Vector3& axis, float angle)
{
	SetRotate(m, axis.x, axis.y, axis.z, angle);
}


void NMath::SetRotateX(Matrix& m, float angle)
{
	m.m[0][0] = 1.0f;
	m.m[0][1] = 0.0f;
	m.m[0][2] = 0.0f;
	m.m[0][3] = 0.0f;

	m.m[1][0] = 0.0f;
	m.m[1][1] = cosf(angle);
	m.m[1][2] = sin(angle);
	m.m[1][3] = 0.0f;

	m.m[2][0] = 0.0f;
	m.m[2][1] = -sin(angle);
	m.m[2][2] = cos(angle);
	m.m[2][3] = 0.0f;

	m.m[3][0] = 0.0f;
	m.m[3][1] = 0.0f;
	m.m[3][2] = 0.0f;
	m.m[3][3] = 1.0f;
}


void NMath::SetRotateY(Matrix& m, float angle)
{
	m.m[0][0] = cos(angle);
	m.m[0][1] = 0.0f;
	m.m[0][2] = -sin(angle);
	m.m[0][3] = 0.0f;

	m.m[1][0] = 0.0f;
	m.m[1][1] = 1.0f;
	m.m[1][2] = 0.0f;
	m.m[1][3] = 0.0f;

	m.m[2][0] = sin(angle);
	m.m[2][1] = 0.0f;
	m.m[2][2] = cos(angle);
	m.m[2][3] = 0.0f;

	m.m[3][0] = 0.0f;
	m.m[3][1] = 0.0f;
	m.m[3][2] = 0.0f;
	m.m[3][3] = 1.0f;
}


void NMath::SetRotateZ(Matrix& m, float angle)
{
	m.m[0][0] = cos(angle);
	m.m[0][1] = sin(angle);
	m.m[0][2] = 0.0f;
	m.m[0][3] = 0.0f;

	m.m[1][0] = -sin(angle);
	m.m[1][1] = cos(angle);
	m.m[1][2] = 0.0f;
	m.m[1][3] = 0.0f;

	m.m[2][0] = 0.0f;
	m.m[2][1] = 0.0f;
	m.m[2][2] = 1.0f;
	m.m[2][3] = 0.0f;

	m.m[3][0] = 0.0f;
	m.m[3][1] = 0.0f;
	m.m[3][2] = 0.0f;
	m.m[3][3] = 1.0f;
}


void NMath::SetScale(Matrix& m, float x, float y, float z)
{
	m.m[0][0] = x;
	m.m[0][1] = 0.0f;
	m.m[0][2] = 0.0f;
	m.m[0][3] = 0.0f;

	m.m[1][0] = 0.0f;
	m.m[1][1] = y;
	m.m[1][2] = 0.0f;
	m.m[1][3] = 0.0f;

	m.m[2][0] = 0.0f;
	m.m[2][1] = 0.0f;
	m.m[2][2] = z;
	m.m[2][3] = 0.0f;

	m.m[3][0] = 0.0f;
	m.m[3][1] = 0.0f;
	m.m[3][2] = 0.0f;
	m.m[3][3] = 1.0f;
}


void NMath::SetScale(Matrix& m, const Vector3& s)
{
	SetScale(m, s.x, s.y, s.z);
}


void NMath::SetReflect(Matrix& m, const Plane& plane)
{
	m.m[0][0] = -2.0f*plane.a*plane.a + 1.0f;
	m.m[0][1] = -2.0f*plane.b*plane.a;
	m.m[0][2] = -2.0f*plane.c*plane.a;
	m.m[0][3] = 0.0f;

	m.m[1][0] = -2.0f*plane.a*plane.b;
	m.m[1][1] = -2.0f*plane.b*plane.b + 1.0f;
	m.m[1][2] = -2.0f*plane.c*plane.b;
	m.m[1][3] = 0.0f;

	m.m[2][0] = -2.0f*plane.a*plane.c;
	m.m[2][1] = -2.0f*plane.b*plane.c;
	m.m[2][2] = -2.0f*plane.c*plane.c + 1.0f;
	m.m[2][3] = 0.0f;

	m.m[3][0] = -2.0f*plane.a*plane.d;
	m.m[3][1] = -2.0f*plane.b*plane.d;
	m.m[3][2] = -2.0f*plane.c*plane.d;
	m.m[3][3] = 1.0f;
}


void NMath::SetLookAtLH(Matrix& m, const Vector3& eye, const Vector3& at, const Vector3& up)
{
	Vector3 zAxis = Normalize(at - eye);
	Vector3 xAxis = Normalize(Cross(up, zAxis));
	Vector3 yAxis = Cross(zAxis, xAxis);

	m.m[0][0] = xAxis.x;
	m.m[0][1] = yAxis.x;
	m.m[0][2] = zAxis.x;
	m.m[0][3] = 0.0f;

	m.m[1][0] = xAxis.y;
	m.m[1][1] = yAxis.y;
	m.m[1][2] = zAxis.y;
	m.m[1][3] = 0.0f;

	m.m[2][0] = xAxis.z;
	m.m[2][1] = yAxis.z;
	m.m[2][2] = zAxis.z;
	m.m[2][3] = 0.0f;

	m.m[3][0] = -Dot(xAxis, eye);
	m.m[3][1] = -Dot(yAxis, eye);
	m.m[3][2] = -Dot(zAxis, eye);
	m.m[3][3] = 1.0f;
}


void NMath::SetLookAtRH(Matrix& m, const Vector3& eye, const Vector3& at, const Vector3& up)
{
	Vector3 zAxis = Normalize(eye - at);
	Vector3 xAxis = Normalize(Cross(up, zAxis));
	Vector3 yAxis = Cross(zAxis, xAxis);

	m.m[0][0] = xAxis.x;
	m.m[0][1] = yAxis.x;
	m.m[0][2] = zAxis.x;
	m.m[0][3] = 0.0f;

	m.m[1][0] = xAxis.y;
	m.m[1][1] = yAxis.y;
	m.m[1][2] = zAxis.y;
	m.m[1][3] = 0.0f;

	m.m[2][0] = xAxis.z;
	m.m[2][1] = yAxis.z;
	m.m[2][2] = zAxis.z;
	m.m[2][3] = 0.0f;

	m.m[3][0] = -Dot(xAxis, eye);
	m.m[3][1] = -Dot(yAxis, eye);
	m.m[3][2] = -Dot(zAxis, eye);
	m.m[3][3] = 1.0f;
}


void NMath::SetPerspectiveFovLH(Matrix& m, ZRange zRange, float fovY, float aspectRatio, float zNear, float zFar)
{
	float yScale = 1.0f / Tan(fovY / 2.0f);
	float xScale = yScale / aspectRatio;

	m.m[0][0] = xScale;
	m.m[0][1] = 0.0f;
	m.m[0][2] = 0.0f;
	m.m[0][3] = 0.0f;

	m.m[1][0] = 0.0f;
	m.m[1][1] = yScale;
	m.m[1][2] = 0.0f;
	m.m[1][3] = 0.0f;

	m.m[2][0] = 0.0f;
	m.m[2][1] = 0.0f;
	if (zRange == ZRange::MinusOneToPlusOne)
		m.m[2][2] = (zNear + zFar) / (zFar - zNear);
	else if (zRange == ZRange::ZeroToOne)
		m.m[2][2] = zFar / (zFar - zNear);
	m.m[2][3] = 1.0f;

	m.m[3][0] = 0.0f;
	m.m[3][1] = 0.0f;
	if (zRange == ZRange::MinusOneToPlusOne)
		m.m[3][2] = -2.0f * zNear * zFar / (zFar - zNear);
	else if (zRange == ZRange::ZeroToOne)
		m.m[3][2] = -zNear * zFar / (zFar - zNear);
	m.m[3][3] = 0.0f;
}


void NMath::SetPerspectiveFovRH(Matrix& m, ZRange zRange, float fovY, float aspectRatio, float zNear, float zFar)
{
	float yScale = 1.0f / Tan(fovY / 2.0f);
	float xScale = yScale / aspectRatio;

	m.m[0][0] = xScale;
	m.m[0][1] = 0.0f;
	m.m[0][2] = 0.0f;
	m.m[0][3] = 0.0f;

	m.m[1][0] = 0.0f;
	m.m[1][1] = yScale;
	m.m[1][2] = 0.0f;
	m.m[1][3] = 0.0f;

	m.m[2][0] = 0.0f;
	m.m[2][1] = 0.0f;
	if (zRange == ZRange::MinusOneToPlusOne)
		m.m[2][2] = (zNear + zFar) / (zNear - zFar);
	else if (zRange == ZRange::ZeroToOne)
		m.m[2][2] = zFar / (zNear - zFar);
	m.m[2][3] = -1.0f;

	m.m[3][0] = 0.0f;
	m.m[3][1] = 0.0f;
	if (zRange == ZRange::MinusOneToPlusOne)
		m.m[3][2] = 2.0f * zNear * zFar / (zNear - zFar);
	else if (zRange == ZRange::ZeroToOne)
		m.m[3][2] = zNear * zFar / (zNear - zFar);
	m.m[3][3] = 0.0f;
}


void NMath::SetOrthoOffCenterLH(Matrix& m, ZRange zRange, float left, float right, float bottom, float top, float zNear, float zFar)
{
	m.m[0][0] = 2.0f / (right - left);
	m.m[0][1] = 0.0f;
	m.m[0][2] = 0.0f;
	m.m[0][3] = 0.0f;

	m.m[1][0] = 0.0f;
	m.m[1][1] = 2.0f / (top - bottom);
	m.m[1][2] = 0.0f;
	m.m[1][3] = 0.0f;

	m.m[2][0] = 0.0f;
	m.m[2][1] = 0.0f;
	if (zRange == ZRange::MinusOneToPlusOne)
		m.m[2][2] = -2.0f / (zNear - zFar);
	else if (zRange == ZRange::ZeroToOne)
		m.m[2][2] = 1.0f / (zFar - zNear);
	m.m[2][3] = 0.0f;

	m.m[3][0] = (1.0f + right) / (1.0f - right);
	m.m[3][1] = (top + bottom) / (bottom - top);
	if (zRange == ZRange::MinusOneToPlusOne)
		m.m[3][2] = (zNear + zFar) / (zNear - zFar);
	else if (zRange == ZRange::ZeroToOne)
		m.m[3][2] = -zNear / (zFar - zNear);
	m.m[3][3] = 1.0f;
}


void NMath::SetOrthoOffCenterRH(Matrix& m, ZRange zRange, float left, float right, float bottom, float top, float zNear, float zFar)
{
	m.m[0][0] = 2.0f / (right - left);
	m.m[0][1] = 0.0f;
	m.m[0][2] = 0.0f;
	m.m[0][3] = 0.0f;

	m.m[1][0] = 0.0f;
	m.m[1][1] = 2.0f / (top - bottom);
	m.m[1][2] = 0.0f;
	m.m[1][3] = 0.0f;

	m.m[2][0] = 0.0f;
	m.m[2][1] = 0.0f;
	if (zRange == ZRange::MinusOneToPlusOne)
		m.m[2][2] = 2.0f / (zNear - zFar);
	else if (zRange == ZRange::ZeroToOne)
		m.m[2][2] = 1.0f / (zNear - zFar);
	m.m[2][3] = 0.0f;

	m.m[3][0] = (1.0f + right) / (1.0f - right);
	m.m[3][1] = (top + bottom) / (bottom - top);
	if (zRange == ZRange::MinusOneToPlusOne)
		m.m[3][2] = (zNear + zFar) / (zNear - zFar);
	else if (zRange == ZRange::ZeroToOne)
		m.m[3][2] = zNear / (zNear - zFar);
	m.m[3][3] = 1.0f;
}


void NMath::SetOrthoLH(Matrix& m, ZRange zRange, float width, float height, float zNear, float zFar)
{
	m.m[0][0] = 2.0f / width;
	m.m[0][1] = 0.0f;
	m.m[0][2] = 0.0f;
	m.m[0][3] = 0.0f;

	m.m[1][0] = 0.0f;
	m.m[1][1] = 2.0f / height;
	m.m[1][2] = 0.0f;
	m.m[1][3] = 0.0f;

	m.m[2][0] = 0.0f;
	m.m[2][1] = 0.0f;
	if (zRange == ZRange::MinusOneToPlusOne)
		m.m[2][2] = -2.0f / (zNear - zFar);
	else if (zRange == ZRange::ZeroToOne)
		m.m[2][2] = 1.0f / (zFar - zNear);
	m.m[2][3] = 0.0f;

	m.m[3][0] = 0.0f;
	m.m[3][1] = 0.0f;
	if (zRange == ZRange::MinusOneToPlusOne)
		m.m[3][2] = (zNear + zFar) / (zNear - zFar);
	else if (zRange == ZRange::ZeroToOne)
		m.m[3][2] = -zNear / (zFar - zNear);
	m.m[3][3] = 1.0f;
}


void NMath::SetOrthoRH(Matrix& m, ZRange zRange, float width, float height, float zNear, float zFar)
{
	m.m[0][0] = 2.0f / width;
	m.m[0][1] = 0.0f;
	m.m[0][2] = 0.0f;
	m.m[0][3] = 0.0f;

	m.m[1][0] = 0.0f;
	m.m[1][1] = 2.0f / height;
	m.m[1][2] = 0.0f;
	m.m[1][3] = 0.0f;

	m.m[2][0] = 0.0f;
	m.m[2][1] = 0.0f;
	if (zRange == ZRange::MinusOneToPlusOne)
		m.m[2][2] = 2.0f / (zNear - zFar);
	else if (zRange == ZRange::ZeroToOne)
		m.m[2][2] = 1.0f / (zNear - zFar);
	m.m[2][3] = 0.0f;

	m.m[3][0] = 0.0f;
	m.m[3][1] = 0.0f;
	if (zRange == ZRange::MinusOneToPlusOne)
		m.m[3][2] = (zNear + zFar) / (zNear - zFar);
	else if (zRange == ZRange::ZeroToOne)
		m.m[3][2] = zNear / (zNear - zFar);
	m.m[3][3] = 1.0f;
}


NMath::Matrix NMath::MatrixZeros()
{
	Matrix temp;
	SetZeros(temp);
	return temp;
}


NMath::Matrix NMath::MatrixIdentity()
{
	Matrix temp;
	SetIdentity(temp);
	return temp;
}


NMath::Matrix NMath::MatrixTranslate(float x, float y, float z)
{
	Matrix temp;
	SetTranslate(temp, x, y, z);
	return temp;
}


NMath::Matrix NMath::MatrixTranslate(const Vector3& v)
{
	return MatrixTranslate(v.x, v.y, v.z);
}


NMath::Matrix NMath::MatrixRotate(float x, float y, float z, float angle)
{
	Matrix temp;
	SetRotate(temp, x, y, z, angle);
	return temp;
}


NMath::Matrix NMath::MatrixRotate(const Vector3& axis, float angle)
{
	return MatrixRotate(axis.x, axis.y, axis.z, angle);
}


NMath::Matrix NMath::MatrixRotateX(float angle)
{
	Matrix temp;
	SetRotateX(temp, angle);
	return temp;
}


NMath::Matrix NMath::MatrixRotateY(float angle)
{
	Matrix temp;
	SetRotateY(temp, angle);
	return temp;
}


NMath::Matrix NMath::MatrixRotateZ(float angle)
{
	Matrix temp;
	SetRotateZ(temp, angle);
	return temp;
}


NMath::Matrix NMath::MatrixScale(float x, float y, float z)
{
	Matrix temp;
	SetScale(temp, x, y, z);
	return temp;
}


NMath::Matrix NMath::MatrixScale(const Vector3& s)
{
	return MatrixScale(s.x, s.y, s.z);
}


NMath::Matrix NMath::MatrixReflect(const Plane& plane)
{
	Matrix temp;
	SetReflect(temp, plane);
	return temp;
}


NMath::Matrix NMath::MatrixLookAtLH(const Vector3& eye, const Vector3& at, const Vector3& up)
{
	Matrix temp;
	SetLookAtLH(temp, eye, at, up);
	return temp;
}


NMath::Matrix NMath::MatrixLookAtRH(const Vector3& eye, const Vector3& at, const Vector3& up)
{
	Matrix temp;
	SetLookAtRH(temp, eye, at, up);
	return temp;
}


NMath::Matrix NMath::MatrixPerspectiveFovLH(ZRange zRange, float fovY, float aspectRatio, float zNear, float zFar)
{
	Matrix temp;
	SetPerspectiveFovLH(temp, zRange, fovY, aspectRatio, zNear, zFar);
	return temp;
}


NMath::Matrix NMath::MatrixPerspectiveFovRH(ZRange zRange, float fovY, float aspectRatio, float zNear, float zFar)
{
	Matrix temp;
	SetPerspectiveFovRH(temp, zRange, fovY, aspectRatio, zNear, zFar);
	return temp;
}


NMath::Matrix NMath::MatrixOrthoOffCenterLH(ZRange zRange, float left, float right, float bottom, float top, float zNear, float zFar)
{
	Matrix temp;
	SetOrthoOffCenterLH(temp, zRange, left, right, bottom, top, zNear, zFar);
	return temp;
}


NMath::Matrix NMath::MatrixOrthoOffCenterRH(ZRange zRange, float left, float right, float bottom, float top, float zNear, float zFar)
{
	Matrix temp;
	SetOrthoOffCenterRH(temp, zRange, left, right, bottom, top, zNear, zFar);
	return temp;
}


NMath::Matrix NMath::MatrixOrthoLH(ZRange zRange, float width, float height, float zNear, float zFar)
{
	Matrix temp;
	SetOrthoLH(temp, zRange, width, height, zNear, zFar);
	return temp;
}


NMath::Matrix NMath::MatrixOrthoRH(ZRange zRange, float width, float height, float zNear, float zFar)
{
	Matrix temp;
	SetOrthoRH(temp, zRange, width, height, zNear, zFar);
	return temp;
}
