#include <cstring>
#include <cmath>

#include <math/matrix.hpp>
#include <math/quaternion.hpp>



Matrix::ZRange::TYPE Matrix::zRange = Matrix::ZRange::ZeroToOne;



Matrix::Matrix(float _00_11_22_33)
{
	_[0][0] = _00_11_22_33;	_[0][1] = 0.0f;			_[0][2] = 0.0f;			_[0][3] = 0.0f;
	_[1][0] = 0.0f;			_[1][1] = _00_11_22_33;	_[1][2] = 0.0f;			_[1][3] = 0.0f;
	_[2][0] = 0.0f;			_[2][1] = 0.0f;			_[2][2] = _00_11_22_33;	_[2][3] = 0.0f;
	_[3][0] = 0.0f;			_[3][1] = 0.0f;			_[3][2] = 0.0f;			_[3][3] = _00_11_22_33;
}



Matrix::Matrix(float _00, float _01, float _02, float _03,
			   float _10, float _11, float _12, float _13,
			   float _20, float _21, float _22, float _23,
			   float _30, float _31, float _32, float _33)
{
	_[0][0] = _00;	_[0][1] = _01;	_[0][2] = _02;	_[0][3] = _03;
	_[1][0] = _10;	_[1][1] = _11;	_[1][2] = _12;	_[1][3] = _13;
	_[2][0] = _20;	_[2][1] = _21;	_[2][2] = _22;	_[2][3] = _23;
	_[3][0] = _30;	_[3][1] = _31;	_[3][2] = _32;	_[3][3] = _33;
}



Matrix::Matrix(const Vector4& row1, const Vector4& row2, const Vector4& row3, const Vector4& row4)
{
	_[0][0] = row1.x;	_[0][1] = row1.y;	_[0][2] = row1.z;	_[0][3] = row1.w;
	_[1][0] = row2.x;	_[1][1] = row2.y;	_[1][2] = row2.z;	_[1][3] = row2.w;
	_[2][0] = row3.x;	_[2][1] = row3.y;	_[2][2] = row3.z;	_[2][3] = row3.w;
	_[3][0] = row4.x;	_[3][1] = row4.y;	_[3][2] = row4.z;	_[3][3] = row4.w;
}



Matrix::Matrix(const Matrix& m)
{
	memcpy(this, &m, 16*sizeof(float));
}



bool Matrix::IsOrthonormal() const
{
	Vector4 row1(_[0][0], _[0][1], _[0][2], _[0][3]);
	Vector4 row2(_[1][0], _[1][1], _[1][2], _[1][3]);
	Vector4 row3(_[2][0], _[2][1], _[2][2], _[2][3]);
	Vector4 row4(_[3][0], _[3][1], _[3][2], _[3][3]);

	if ( (row1 % row2 < epsilon4) &&
		 (row1 % row3 < epsilon4) &&
		 (row1 % row4 < epsilon4) &&
		 (row2 % row3 < epsilon4) &&
		 (row2 % row4 < epsilon4) &&
		 (row3 % row4 < epsilon4) )
	{
		return true;
	}
	else
	{
		return false;
	}
}



void Matrix::Orthogonalize3x3()
{
	Vector3 v1 = Vector3(_[0][0], _[0][1], _[0][2]);
	Vector3 v2 = Vector3(_[1][0], _[1][1], _[1][2]);
	Vector3 v3 = Vector3(_[2][0], _[2][1], _[2][2]);

	Vector3 u1 = v1;
	Vector3 u2 = v2 - ((v2%u1)/(u1%u1))*u1;
	Vector3 u3 = v3 - ((v3%u1)/(u1%u1))*u1 - ((v3%u2)/(u2%u2))*u2;

	_[0][0] = u1.x;		_[0][1] = u1.y;		_[0][2] = u1.z;
	_[1][0] = u2.x;		_[1][1] = u2.y;		_[1][2] = u2.z;
	_[2][0] = u3.x;		_[2][1] = u3.y;		_[2][2] = u3.z;
}



Matrix Matrix::GetOrthogonalized3x3() const
{
	Matrix temp = *this;
	temp.Orthogonalize3x3();
	return temp;
}



void Matrix::Transpose()
{
	Matrix temp = *this;

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			_[i][j] = temp._[j][i];
		}
	}
}



Matrix Matrix::GetTransposed() const
{
	Matrix temp = *this;
	temp.Transpose();
	return temp;
}



// Laplace's expansion
void Matrix::Inverse()
{
	float determinant = + _[0][0]*GetDeterminant(_[1][1], _[1][2], _[1][3],
												 _[2][1], _[2][2], _[2][3],
												 _[3][1], _[3][2], _[3][3])
						- _[0][1]*GetDeterminant(_[1][0], _[1][2], _[1][3],
												 _[2][0], _[2][2], _[2][3],
												 _[3][0], _[3][2], _[3][3])
						+ _[0][2]*GetDeterminant(_[1][0], _[1][1], _[1][3],
												 _[2][0], _[2][1], _[2][3],
												 _[3][0], _[3][1], _[3][3])
						- _[0][3]*GetDeterminant(_[1][0], _[1][1], _[1][2],
												 _[2][0], _[2][1], _[2][2],
												 _[3][0], _[3][1], _[3][2]);

	if ( !(fabs(determinant) < epsilon4) )
	{
		float adj_[4][4];

		adj_[0][0] = + GetDeterminant(_[1][1], _[1][2], _[1][3],
									  _[2][1], _[2][2], _[2][3],
									  _[3][1], _[3][2], _[3][3]);
		adj_[0][1] = - GetDeterminant(_[1][0], _[1][2], _[1][3],
									  _[2][0], _[2][2], _[2][3],
									  _[3][0], _[3][2], _[3][3]);
		adj_[0][2] = + GetDeterminant(_[1][0], _[1][1], _[1][3],
									  _[2][0], _[2][1], _[2][3],
									  _[3][0], _[3][1], _[3][3]);
		adj_[0][3] = - GetDeterminant(_[1][0], _[1][1], _[1][2],
									  _[2][0], _[2][1], _[2][2],
									  _[3][0], _[3][1], _[3][2]);

		adj_[1][0] = - GetDeterminant(_[0][1], _[0][2], _[0][3],
									  _[2][1], _[2][2], _[2][3],
									  _[3][1], _[3][2], _[3][3]);
		adj_[1][1] = + GetDeterminant(_[0][0], _[0][2], _[0][3],
									  _[2][0], _[2][2], _[2][3],
									  _[3][0], _[3][2], _[3][3]);
		adj_[1][2] = - GetDeterminant(_[0][0], _[0][1], _[0][3],
									  _[2][0], _[2][1], _[2][3],
									  _[3][0], _[3][1], _[3][3]);
		adj_[1][3] = + GetDeterminant(_[0][0], _[0][1], _[0][2],
									  _[2][0], _[2][1], _[2][2],
									  _[3][0], _[3][1], _[3][2]);

		adj_[2][0] = + GetDeterminant(_[0][1], _[0][2], _[0][3],
									 _[1][1], _[1][2], _[1][3],
									 _[3][1], _[3][2], _[3][3]);
		adj_[2][1] = - GetDeterminant(_[0][0], _[0][2], _[0][3],
									 _[1][0], _[1][2], _[1][3],
									 _[3][0], _[3][2], _[3][3]);
		adj_[2][2] = + GetDeterminant(_[0][0], _[0][1], _[0][3],
									 _[1][0], _[1][1], _[1][3],
									 _[3][0], _[3][1], _[3][3]);
		adj_[2][3] = - GetDeterminant(_[0][0], _[0][1], _[0][2],
									 _[1][0], _[1][1], _[1][2],
									 _[3][0], _[3][1], _[3][2]);

		adj_[3][0] = - GetDeterminant(_[0][1], _[0][2], _[0][3],
									 _[1][1], _[1][2], _[1][3],
									 _[2][1], _[2][2], _[2][3]);
		adj_[3][1] = + GetDeterminant(_[0][0], _[0][2], _[0][3],
									 _[1][0], _[1][2], _[1][3],
									 _[2][0], _[2][2], _[2][3]);
		adj_[3][2] = - GetDeterminant(_[0][0], _[0][1], _[0][3],
									 _[1][0], _[1][1], _[1][3],
									 _[2][0], _[2][1], _[2][3]);
		adj_[3][3] = + GetDeterminant(_[0][0], _[0][1], _[0][2],
									 _[1][0], _[1][1], _[1][2],
									 _[2][0], _[2][1], _[2][2]);

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				_[i][j] = (1.0f / determinant) * adj_[j][i];
			}
		}
	}
}



Matrix Matrix::GetInversed() const
{
	Matrix temp = *this;
	temp.Inverse();
	return temp;
}



Quaternion Matrix::GetQuaternion()
{
	Quaternion q;

	float fourWSqrMinusOne = _[0][0] + _[1][1] + _[2][2];
	float fourXSqrMinusOne = _[0][0] - _[1][1] - _[2][2];
	float fourYSqrMinusOne = _[1][1] - _[0][0] - _[2][2];
	float fourZSqrMinusOne = _[2][2] - _[0][0] - _[1][1];

	int biggestIndex = 0;
	float fourBiggestSqrMinusOne = fourWSqrMinusOne;
	if (fourXSqrMinusOne > fourBiggestSqrMinusOne)
	{
		fourBiggestSqrMinusOne = fourXSqrMinusOne;
		biggestIndex = 1;
	}
	if (fourYSqrMinusOne > fourBiggestSqrMinusOne)
	{
		fourBiggestSqrMinusOne = fourYSqrMinusOne;
		biggestIndex = 2;
	}
	if (fourZSqrMinusOne > fourBiggestSqrMinusOne)
	{
		fourBiggestSqrMinusOne = fourZSqrMinusOne;
		biggestIndex = 3;
	}

	float biggestValue = 0.5f * sqrtf(fourBiggestSqrMinusOne + 1.0f);
	float mult = 0.25f / biggestValue;

	switch (biggestIndex)
	{
	case 0:
		q.w = biggestValue;
		q.x = (_[1][2] - _[2][1]) * mult;
		q.y = (_[2][0] - _[0][2]) * mult;
		q.z = (_[0][1] - _[1][0]) * mult;
		break;

	case 1:
		q.x = biggestValue;
		q.w = (_[1][2] - _[2][1]) * mult;
		q.y = (_[0][1] + _[1][0]) * mult;
		q.z = (_[2][0] + _[0][2]) * mult;
		break;

	case 2:
		q.y = biggestValue;
		q.w = (_[2][0] - _[0][2]) * mult;
		q.x = (_[0][1] + _[1][0]) * mult;
		q.z = (_[1][2] + _[2][1]) * mult;
		break;

	case 3:
		q.z = biggestValue;
		q.w = (_[0][1] - _[1][0]) * mult;
		q.x = (_[2][0] + _[0][2]) * mult;
		q.y = (_[1][2] + _[2][1]) * mult;
		break;
	}

	return q;
}



void Matrix::LoadIdentity()
{
	_[0][0] = 1.0f;	_[0][1] = 0.0f;	_[0][2] = 0.0f;	_[0][3] = 0.0f;
	_[1][0] = 0.0f;	_[1][1] = 1.0f;	_[1][2] = 0.0f;	_[1][3] = 0.0f;
	_[2][0] = 0.0f;	_[2][1] = 0.0f;	_[2][2] = 1.0f;	_[2][3] = 0.0f;
	_[3][0] = 0.0f;	_[3][1] = 0.0f;	_[3][2] = 0.0f;	_[3][3] = 1.0f;
}



void Matrix::LoadZeroes()
{
	_[0][0] = 0.0f;	_[0][1] = 0.0f;	_[0][2] = 0.0f;	_[0][3] = 0.0f;
	_[1][0] = 0.0f;	_[1][1] = 0.0f;	_[1][2] = 0.0f;	_[1][3] = 0.0f;
	_[2][0] = 0.0f;	_[2][1] = 0.0f;	_[2][2] = 0.0f;	_[2][3] = 0.0f;
	_[3][0] = 0.0f;	_[3][1] = 0.0f;	_[3][2] = 0.0f;	_[3][3] = 0.0f;
}



void Matrix::LoadLookAtLH(const Vector3& eye, const Vector3& at, const Vector3& up)
{
	Vector3 zAxis = (at - eye).GetNormalized();
	Vector3 xAxis = (up ^ zAxis).GetNormalized();
	Vector3 yAxis = (zAxis ^ xAxis);

	_[0][0] = xAxis.x;			_[0][1] = yAxis.x;			_[0][2] = zAxis.x;			_[0][3] = 0.0f;
	_[1][0] = xAxis.y;			_[1][1] = yAxis.y;			_[1][2] = zAxis.y;			_[1][3] = 0.0f;
	_[2][0] = xAxis.z;			_[2][1] = yAxis.z;			_[2][2] = zAxis.z;			_[2][3] = 0.0f;
	_[3][0] = -(xAxis % eye);	_[3][1] = -(yAxis % eye);	_[3][2] = -(zAxis % eye);	_[3][3] = 1.0f;
}



void Matrix::LoadLookAtRH(const Vector3& eye, const Vector3& at, const Vector3& up)
{
	Vector3 zAxis = (eye - at).GetNormalized();
	Vector3 xAxis = (up ^ zAxis).GetNormalized();
	Vector3 yAxis = (zAxis ^ xAxis);

	_[0][0] = xAxis.x;			_[0][1] = yAxis.x;			_[0][2] = zAxis.x;			_[0][3] = 0.0f;
	_[1][0] = xAxis.y;			_[1][1] = yAxis.y;			_[1][2] = zAxis.y;			_[1][3] = 0.0f;
	_[2][0] = xAxis.z;			_[2][1] = yAxis.z;			_[2][2] = zAxis.z;			_[2][3] = 0.0f;
	_[3][0] = -(xAxis % eye);	_[3][1] = -(yAxis % eye);	_[3][2] = -(zAxis % eye);	_[3][3] = 1.0f;
}



void Matrix::LoadPerspectiveFovLH(float fovY, float aspectRatio, float zNear, float zFar)
{
	float yScale = 1.0f / tanf(fovY / 2.0f);
	float xScale = yScale / aspectRatio;

	_[0][0] = xScale;	_[0][1] = 0.0f;		_[0][2] = 0.0f;								_[0][3] = 0.0f;
	_[1][0] = 0.0f;		_[1][1] = yScale;	_[1][2] = 0.0f;								_[1][3] = 0.0f;
	_[2][0] = 0.0f;		_[2][1] = 0.0f;													_[2][3] = 1.0f;
	_[3][0] = 0.0f;		_[3][1] = 0.0f;													_[3][3] = 0.0f;

	if (zRange == ZRange::MinusOneToPlusOne)
	{
		_[2][2] = (zNear + zFar) / (zFar - zNear);
		_[3][2] = -2.0f * zNear * zFar / (zFar - zNear);
	}
	else if (zRange == ZRange::ZeroToOne)
	{
		_[2][2] = zFar / (zFar - zNear);
		_[3][2] = -zNear * zFar / (zFar - zNear);
	}
}



void Matrix::LoadPerspectiveFovRH(float fovY, float aspectRatio, float zNear, float zFar)
{
	float yScale = 1.0f / tanf(fovY / 2.0f);
	float xScale = yScale / aspectRatio;

	_[0][0] = xScale;	_[0][1] = 0.0f;		_[0][2] = 0.0f;								_[0][3] = 0.0f;
	_[1][0] = 0.0f;		_[1][1] = yScale;	_[1][2] = 0.0f;								_[1][3] = 0.0f;
	_[2][0] = 0.0f;		_[2][1] = 0.0f;													_[2][3] = -1.0f;
	_[3][0] = 0.0f;		_[3][1] = 0.0f;													_[3][3] = 0.0f;

	if (zRange == ZRange::MinusOneToPlusOne)
	{
		_[2][2] = (zNear + zFar) / (zNear - zFar);
		_[3][2] = 2.0f * zNear * zFar / (zNear - zFar);
	}
	else if (zRange == ZRange::ZeroToOne)
	{
		_[2][2] = zFar / (zNear - zFar);
		_[3][2] = zNear * zFar / (zNear - zFar);
	}
}



void Matrix::LoadOrthoOffCenterLH(float left, float right, float bottom, float top, float zNear, float zFar)
{
	_[0][0] = 2.0f / (right - left);			_[0][1] = 0.0f;								_[0][2] = 0.0f;						_[0][3] = 0.0f;
	_[1][0] = 0.0f;								_[1][1] = 2.0f / (top - bottom);			_[1][2] = 0.0f;						_[1][3] = 0.0f;
	_[2][0] = 0.0f;								_[2][1] = 0.0f;																	_[2][3] = 0.0f;
	_[3][0] = (1.0f + right)/(1.0f - right);	_[3][1] = (top + bottom)/(bottom - top);										_[3][3] = 1.0f;

	if (zRange == ZRange::MinusOneToPlusOne)
	{
		_[2][2] = -2.0f / (zNear - zFar);
		_[3][2] = (zNear + zFar) / (zNear - zFar);
	}
	else if (zRange == ZRange::ZeroToOne)
	{
		_[2][2] = 1.0f / (zFar - zNear);
		_[3][2] = -zNear / (zFar - zNear);
	}
}



void Matrix::LoadOrthoOffCenterRH(float left, float right, float bottom, float top, float zNear, float zFar)
{
	_[0][0] = 2.0f / (right - left);			_[0][1] = 0.0f;								_[0][2] = 0.0f;						_[0][3] = 0.0f;
	_[1][0] = 0.0f;								_[1][1] = 2.0f / (top - bottom);			_[1][2] = 0.0f;						_[1][3] = 0.0f;
	_[2][0] = 0.0f;								_[2][1] = 0.0f;																	_[2][3] = 0.0f;
	_[3][0] = (1.0f + right)/(1.0f - right);	_[3][1] = (top + bottom)/(bottom - top);										_[3][3] = 1.0f;

	if (zRange == ZRange::MinusOneToPlusOne)
	{
		_[2][2] = 2.0f / (zNear - zFar);
		_[3][2] = (zNear + zFar) / (zNear - zFar);
	}
	else if (zRange == ZRange::ZeroToOne)
	{
		_[2][2] = 1.0f / (zNear - zFar);
		_[3][2] = zNear / (zNear - zFar);
	}
}



void Matrix::LoadOrthoLH(float width, float height, float zNear, float zFar)
{
	_[0][0] = 2.0f / width;	_[0][1] = 0.0f;				_[0][2] = 0.0f;						_[0][3] = 0.0f;
	_[1][0] = 0.0f;			_[1][1] = 2.0f / height;	_[1][2] = 0.0f;						_[1][3] = 0.0f;
	_[2][0] = 0.0f;			_[2][1] = 0.0f;													_[2][3] = 0.0f;
	_[3][0] = 0.0f;			_[3][1] = 0.0f;													_[3][3] = 1.0f;

	if (zRange == ZRange::MinusOneToPlusOne)
	{
		_[2][2] = -2.0f / (zNear - zFar);
		_[3][2] = (zNear + zFar) / (zNear - zFar);
	}
	else if (zRange == ZRange::ZeroToOne)
	{
		_[2][2] = 1.0f / (zFar - zNear);
		_[3][2] = -zNear / (zFar - zNear);
	}
}



void Matrix::LoadOrthoRH(float width, float height, float zNear, float zFar)
{
	_[0][0] = 2.0f / width;	_[0][1] = 0.0f;				_[0][2] = 0.0f;						_[0][3] = 0.0f;
	_[1][0] = 0.0f;			_[1][1] = 2.0f / height;	_[1][2] = 0.0f;						_[1][3] = 0.0f;
	_[2][0] = 0.0f;			_[2][1] = 0.0f;													_[2][3] = 0.0f;
	_[3][0] = 0.0f;			_[3][1] = 0.0f;													_[3][3] = 1.0f;

	if (zRange == ZRange::MinusOneToPlusOne)
	{
		_[2][2] = 2.0f / (zNear - zFar);
		_[3][2] = (zNear + zFar) / (zNear - zFar);
	}
	else if (zRange == ZRange::ZeroToOne)
	{
		_[2][2] = 1.0f / (zNear - zFar);
		_[3][2] = zNear / (zNear - zFar);
	}
}



void Matrix::LoadTranslate(float tx, float ty, float tz)
{
	_[0][0] = 1.0f;	_[0][1] = 0.0f;	_[0][2] = 0.0f;	_[0][3] = 0.0f;
	_[1][0] = 0.0f;	_[1][1] = 1.0f;	_[1][2] = 0.0f;	_[1][3] = 0.0f;
	_[2][0] = 0.0f;	_[2][1] = 0.0f;	_[2][2] = 1.0f;	_[2][3] = 0.0f;
	_[3][0] = tx;	_[3][1] = ty;	_[3][2] = tz;	_[3][3] = 1.0f;
}



void Matrix::LoadRotate(float rx, float ry, float rz, float angle)
{
	float s = sinf(angle);
	float c = cosf(angle);

	float length = sqrtf(rx*rx + ry*ry + rz*rz);

	rx /= length;
	ry /= length;
	rz /= length;

	_[0][0] = c+rx*rx*(1-c);	_[0][1] = rx*ry*(1-c)+rz*s;	_[0][2] = rx*rz*(1-c)-ry*s;	_[0][3] = 0.0f;
	_[1][0] = rx*ry*(1-c)-rz*s;	_[1][1] = c+ry*ry*(1-c);	_[1][2] = ry*rz*(1-c)+rx*s;	_[1][3] = 0.0f;
	_[2][0] = rx*rz*(1-c)+ry*s;	_[2][1] = ry*rz*(1-c)-rx*s;	_[2][2] = c+rz*rz*(1-c);	_[2][3] = 0.0f;
	_[3][0] = 0.0f;				_[3][1] = 0.0f;				_[3][2] = 0.0f;				_[3][3] = 1.0f;
}



void Matrix::LoadRotateX(float angle)
{
	_[0][0] = 1.0f;	_[0][1] = 0.0f;			_[0][2] = 0.0f;			_[0][3] = 0.0f;
	_[1][0] = 0.0f;	_[1][1] = cosf(angle);	_[1][2] = sin(angle);	_[1][3] = 0.0f;
	_[2][0] = 0.0f;	_[2][1] = -sin(angle);	_[2][2] = cos(angle);	_[2][3] = 0.0f;
	_[3][0] = 0.0f;	_[3][1] = 0.0f;			_[3][2] = 0.0f;			_[3][3] = 1.0f;
}



void Matrix::LoadRotateY(float angle)
{
	_[0][0] = cos(angle);	_[0][1] = 0.0f;	_[0][2] = -sin(angle);	_[0][3] = 0.0f;
	_[1][0] = 0.0f;			_[1][1] = 1.0f;	_[1][2] = 0.0f;			_[1][3] = 0.0f;
	_[2][0] = sin(angle);	_[2][1] = 0.0f;	_[2][2] = cos(angle);	_[2][3] = 0.0f;
	_[3][0] = 0.0f;			_[3][1] = 0.0f;	_[3][2] = 0.0f;			_[3][3] = 1.0f;
}



void Matrix::LoadRotateZ(float angle)
{
	_[0][0] = cos(angle);	_[0][1] = sin(angle);	_[0][2] = 0.0f;	_[0][3] = 0.0f;
	_[1][0] = -sin(angle);	_[1][1] = cos(angle);	_[1][2] = 0.0f;	_[1][3] = 0.0f;
	_[2][0] = 0.0f;			_[2][1] = 0.0f;			_[2][2] = 1.0f;	_[2][3] = 0.0f;
	_[3][0] = 0.0f;			_[3][1] = 0.0f;			_[3][2] = 0.0f;	_[3][3] = 1.0f;
}



void Matrix::LoadScale(float sx, float sy, float sz)
{
	_[0][0] = sx;	_[0][1] = 0.0f;	_[0][2] = 0.0f;	_[0][3] = 0.0f;
	_[1][0] = 0.0f;	_[1][1] = sy;	_[1][2] = 0.0f;	_[1][3] = 0.0f;
	_[2][0] = 0.0f;	_[2][1] = 0.0f;	_[2][2] = sz;	_[2][3] = 0.0f;
	_[3][0] = 0.0f;	_[3][1] = 0.0f;	_[3][2] = 0.0f;	_[3][3] = 1.0f;
}



void Matrix::LoadReflect(const Vector3& planePoint, const Vector3& planeNormal)
{
	float planeD = -(planePoint % planeNormal);

	_[0][0] = -2.0f*planeNormal.x*planeNormal.x + 1.0f;	_[0][1] = -2.0f*planeNormal.y*planeNormal.x;		_[0][2] = -2.0f*planeNormal.z*planeNormal.x;		_[0][3] = 0.0f;
	_[1][0] = -2.0f*planeNormal.x*planeNormal.y;		_[1][1] = -2.0f*planeNormal.y*planeNormal.y + 1.0f;	_[1][2] = -2.0f*planeNormal.z*planeNormal.y;		_[1][3] = 0.0f;
	_[2][0] = -2.0f*planeNormal.x*planeNormal.z;		_[2][1] = -2.0f*planeNormal.y*planeNormal.z;		_[2][2] = -2.0f*planeNormal.z*planeNormal.z + 1.0f;	_[2][3] = 0.0f;
	_[3][0] = -2.0f*planeNormal.x*planeD;				_[3][1] = -2.0f*planeNormal.y*planeD;				_[3][2] = -2.0f*planeNormal.z*planeD;				_[3][3] = 1.0f;
}



Matrix& Matrix::operator = (const Matrix& m)
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			_[i][j] = m._[i][j];
		}
	}

	return *this;
}



bool Matrix::operator == (const Matrix& m) const
{
	bool equals = true;

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if ( fabs(_[i][j] - m._[i][j]) > epsilon4 )
				equals = false;
		}
	}

	return equals;
}



bool Matrix::operator != (const Matrix& m) const
{
	return !(*this == m);
}



Matrix Matrix::operator + (const Matrix& m) const
{
	Matrix temp;

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			temp._[i][j] = _[i][j] + m._[i][j];
		}
	}

	return temp;
}



Matrix& Matrix::operator += (const Matrix& m)
{
	*this = *this + m;
	return *this;
}



Matrix Matrix::operator * (const Matrix& m) const
{
	Matrix temp;

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			temp._[i][j] =
				_[i][0] * m._[0][j] +
				_[i][1] * m._[1][j] +
				_[i][2] * m._[2][j] +
				_[i][3] * m._[3][j];
		}
	}

	return temp;
}



Matrix& Matrix::operator *= (const Matrix& m)
{
	*this = *this * m;
	return *this;
}



Vector3 Matrix::operator * (const Vector3& v) const
{
	Vector3 temp;

	temp.x = _[0][0]*v.x + _[0][1]*v.y + _[0][2]*v.z;
	temp.y = _[1][0]*v.x + _[1][1]*v.y + _[1][2]*v.z;
	temp.z = _[2][0]*v.x + _[2][1]*v.y + _[2][2]*v.z;

	return temp;
}



Vector4 Matrix::operator * (const Vector4& v) const
{
	Vector4 temp;

	temp.x = _[0][0]*v.x + _[0][1]*v.y + _[0][2]*v.z + _[0][3]*v.w;
	temp.y = _[1][0]*v.x + _[1][1]*v.y + _[1][2]*v.z + _[1][3]*v.w;
	temp.z = _[2][0]*v.x + _[2][1]*v.y + _[2][2]*v.z + _[2][3]*v.w;
	temp.w = _[3][0]*v.x + _[3][1]*v.y + _[3][2]*v.z + _[3][3]*v.w;

	return temp;
}



Matrix Matrix::operator * (float f) const
{
	Matrix temp;

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			temp._[i][j] = _[i][j] * f;
		}
	}

	return temp;
}



Matrix& Matrix::operator *= (float f)
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			_[i][j] *= f;
		}
	}

	return *this;
}



Matrix operator * (float f, const Matrix& m)
{
	return m * f;
}
