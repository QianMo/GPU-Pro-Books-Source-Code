/******************************************************************/
/* Matrix4x4.cpp                                                  */
/* -----------------------                                        */
/*                                                                */
/* The file defines a basic matrix class that implments most of   */
/*     operations you need to perform on matrices, and interacts  */
/*     correctly with point and vector classes.                   */
/*                                                                */
/* Chris Wyman (02/23/2007)                                       */
/******************************************************************/

#include <math.h>
#include "Point.h"
#include "Vector.h"
#include "Matrix4x4.h"

float Matrix4x4::identityArray[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
float Matrix4x4::zeroArray[16] =     {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};


Matrix4x4& Matrix4x4::operator+=(const Matrix4x4& v) 
{
    mat[0]=mat[0]+v.mat[0];
    mat[1]=mat[1]+v.mat[1];
    mat[2]=mat[2]+v.mat[2];
	mat[3]=mat[3]+v.mat[3];
	mat[4]=mat[4]+v.mat[4];
    mat[5]=mat[5]+v.mat[5];
    mat[6]=mat[6]+v.mat[6];
	mat[7]=mat[7]+v.mat[7];
	mat[8]=mat[8]+v.mat[8];
    mat[9]=mat[9]+v.mat[9];
    mat[10]=mat[10]+v.mat[10];
	mat[11]=mat[11]+v.mat[11];
	mat[12]=mat[12]+v.mat[12];
    mat[13]=mat[13]+v.mat[13];
    mat[14]=mat[14]+v.mat[14];
	mat[15]=mat[15]+v.mat[15];
    return *this;
}

Matrix4x4& Matrix4x4::operator-=(const Matrix4x4& v) 
{
    mat[0]=mat[0]-v.mat[0];
    mat[1]=mat[1]-v.mat[1];
    mat[2]=mat[2]-v.mat[2];
	mat[3]=mat[3]-v.mat[3];
	mat[4]=mat[4]-v.mat[4];
    mat[5]=mat[5]-v.mat[5];
    mat[6]=mat[6]-v.mat[6];
	mat[7]=mat[7]-v.mat[7];
	mat[8]=mat[8]-v.mat[8];
    mat[9]=mat[9]-v.mat[9];
    mat[10]=mat[10]-v.mat[10];
	mat[11]=mat[11]-v.mat[11];
	mat[12]=mat[12]-v.mat[12];
    mat[13]=mat[13]-v.mat[13];
    mat[14]=mat[14]-v.mat[14];
	mat[15]=mat[15]-v.mat[15];
    return *this;
}

Matrix4x4 Matrix4x4::operator+(const Matrix4x4& v) const
{
	float tmp[16];
    tmp[0]=mat[0]+v.mat[0];
    tmp[1]=mat[1]+v.mat[1];
    tmp[2]=mat[2]+v.mat[2];
	tmp[3]=mat[3]+v.mat[3];
	tmp[4]=mat[4]+v.mat[4];
    tmp[5]=mat[5]+v.mat[5];
    tmp[6]=mat[6]+v.mat[6];
	tmp[7]=mat[7]+v.mat[7];
	tmp[8]=mat[8]+v.mat[8];
    tmp[9]=mat[9]+v.mat[9];
    tmp[10]=mat[10]+v.mat[10];
	tmp[11]=mat[11]+v.mat[11];
	tmp[12]=mat[12]+v.mat[12];
    tmp[13]=mat[13]+v.mat[13];
    tmp[14]=mat[14]+v.mat[14];
	tmp[15]=mat[15]+v.mat[15];
    return Matrix4x4( tmp );
}

Matrix4x4 Matrix4x4::operator-(const Matrix4x4& v) const
{
	float tmp[16];
    tmp[0]=mat[0]-v.mat[0];
    tmp[1]=mat[1]-v.mat[1];
    tmp[2]=mat[2]-v.mat[2];
	tmp[3]=mat[3]-v.mat[3];
	tmp[4]=mat[4]-v.mat[4];
    tmp[5]=mat[5]-v.mat[5];
    tmp[6]=mat[6]-v.mat[6];
	tmp[7]=mat[7]-v.mat[7];
	tmp[8]=mat[8]-v.mat[8];
    tmp[9]=mat[9]-v.mat[9];
    tmp[10]=mat[10]-v.mat[10];
	tmp[11]=mat[11]-v.mat[11];
	tmp[12]=mat[12]-v.mat[12];
    tmp[13]=mat[13]-v.mat[13];
    tmp[14]=mat[14]-v.mat[14];
	tmp[15]=mat[15]-v.mat[15];
    return Matrix4x4( tmp );
}

Matrix4x4& Matrix4x4::operator*=(const float s) 
{
    mat[0]=mat[0]*s;
    mat[1]=mat[1]*s;
    mat[2]=mat[2]*s;
	mat[3]=mat[3]*s;
	mat[4]=mat[4]*s;
    mat[5]=mat[5]*s;
    mat[6]=mat[6]*s;
	mat[7]=mat[7]*s;
	mat[8]=mat[8]*s;
    mat[9]=mat[9]*s;
    mat[10]=mat[10]*s;
	mat[11]=mat[11]*s;
	mat[12]=mat[12]*s;
    mat[13]=mat[13]*s;
    mat[14]=mat[14]*s;
	mat[15]=mat[15]*s;
    return *this;
}

Matrix4x4 Matrix4x4::operator*(const float s) const
{
	float tmp[16];
    tmp[0]=mat[0]*s;
    tmp[1]=mat[1]*s;
    tmp[2]=mat[2]*s;
	tmp[3]=mat[3]*s;
	tmp[4]=mat[4]*s;
    tmp[5]=mat[5]*s;
    tmp[6]=mat[6]*s;
	tmp[7]=mat[7]*s;
	tmp[8]=mat[8]*s;
    tmp[9]=mat[9]*s;
    tmp[10]=mat[10]*s;
	tmp[11]=mat[11]*s;
	tmp[12]=mat[12]*s;
    tmp[13]=mat[13]*s;
    tmp[14]=mat[14]*s;
	tmp[15]=mat[15]*s;
    return Matrix4x4( tmp );
}

Matrix4x4 operator*(const float s, const Matrix4x4& v)
{
	float tmp[16];
    tmp[0]=v.mat[0]*s;
    tmp[1]=v.mat[1]*s;
    tmp[2]=v.mat[2]*s;
	tmp[3]=v.mat[3]*s;
	tmp[4]=v.mat[4]*s;
    tmp[5]=v.mat[5]*s;
    tmp[6]=v.mat[6]*s;
	tmp[7]=v.mat[7]*s;
	tmp[8]=v.mat[8]*s;
    tmp[9]=v.mat[9]*s;
    tmp[10]=v.mat[10]*s;
	tmp[11]=v.mat[11]*s;
	tmp[12]=v.mat[12]*s;
    tmp[13]=v.mat[13]*s;
    tmp[14]=v.mat[14]*s;
	tmp[15]=v.mat[15]*s;
    return Matrix4x4( tmp );
}


Matrix4x4& Matrix4x4::operator*=(const Matrix4x4& v) 
{
	float tmp[16];
    tmp[0] = mat[0]*v.mat[0] + mat[4]*v.mat[1] + mat[8]*v.mat[2] + mat[12]*v.mat[3];
	tmp[1] = mat[1]*v.mat[0] + mat[5]*v.mat[1] + mat[9]*v.mat[2] + mat[13]*v.mat[3];
	tmp[2] = mat[2]*v.mat[0] + mat[6]*v.mat[1] + mat[10]*v.mat[2] + mat[14]*v.mat[3];
	tmp[3] = mat[3]*v.mat[0] + mat[7]*v.mat[1] + mat[11]*v.mat[2] + mat[15]*v.mat[3];

	tmp[4] = mat[0]*v.mat[4] + mat[4]*v.mat[5] + mat[8]*v.mat[6] + mat[12]*v.mat[6];
	tmp[5] = mat[1]*v.mat[4] + mat[5]*v.mat[5] + mat[9]*v.mat[6] + mat[13]*v.mat[6];
	tmp[6] = mat[2]*v.mat[4] + mat[6]*v.mat[5] + mat[10]*v.mat[6] + mat[14]*v.mat[6];
	tmp[7] = mat[3]*v.mat[4] + mat[7]*v.mat[5] + mat[11]*v.mat[6] + mat[15]*v.mat[6];

	tmp[8] = mat[0]*v.mat[8] + mat[4]*v.mat[9] + mat[8]*v.mat[10] + mat[12]*v.mat[11];
	tmp[9] = mat[1]*v.mat[8] + mat[5]*v.mat[9] + mat[9]*v.mat[10] + mat[13]*v.mat[11];
	tmp[10] = mat[2]*v.mat[8] + mat[6]*v.mat[9] + mat[10]*v.mat[10] + mat[14]*v.mat[11];
	tmp[11] = mat[3]*v.mat[8] + mat[7]*v.mat[9] + mat[11]*v.mat[10] + mat[15]*v.mat[11];

	tmp[12] = mat[0]*v.mat[12] + mat[4]*v.mat[13] + mat[8]*v.mat[14] + mat[12]*v.mat[15];
	tmp[13] = mat[1]*v.mat[12] + mat[5]*v.mat[13] + mat[9]*v.mat[14] + mat[13]*v.mat[15];
	tmp[14] = mat[2]*v.mat[12] + mat[6]*v.mat[13] + mat[10]*v.mat[14] + mat[14]*v.mat[15];
	tmp[15] = mat[3]*v.mat[12] + mat[7]*v.mat[13] + mat[11]*v.mat[14] + mat[15]*v.mat[15];
	memcpy( mat, tmp, 16*sizeof(float) );
	return *this;
}

Matrix4x4 Matrix4x4::operator*(const Matrix4x4& v) const
{
	float tmp[16];
    tmp[0] = mat[0]*v.mat[0] + mat[4]*v.mat[1] + mat[8]*v.mat[2] + mat[12]*v.mat[3];
	tmp[1] = mat[1]*v.mat[0] + mat[5]*v.mat[1] + mat[9]*v.mat[2] + mat[13]*v.mat[3];
	tmp[2] = mat[2]*v.mat[0] + mat[6]*v.mat[1] + mat[10]*v.mat[2] + mat[14]*v.mat[3];
	tmp[3] = mat[3]*v.mat[0] + mat[7]*v.mat[1] + mat[11]*v.mat[2] + mat[15]*v.mat[3];

	tmp[4] = mat[0]*v.mat[4] + mat[4]*v.mat[5] + mat[8]*v.mat[6] + mat[12]*v.mat[6];
	tmp[5] = mat[1]*v.mat[4] + mat[5]*v.mat[5] + mat[9]*v.mat[6] + mat[13]*v.mat[6];
	tmp[6] = mat[2]*v.mat[4] + mat[6]*v.mat[5] + mat[10]*v.mat[6] + mat[14]*v.mat[6];
	tmp[7] = mat[3]*v.mat[4] + mat[7]*v.mat[5] + mat[11]*v.mat[6] + mat[15]*v.mat[6];

	tmp[8] = mat[0]*v.mat[8] + mat[4]*v.mat[9] + mat[8]*v.mat[10] + mat[12]*v.mat[11];
	tmp[9] = mat[1]*v.mat[8] + mat[5]*v.mat[9] + mat[9]*v.mat[10] + mat[13]*v.mat[11];
	tmp[10] = mat[2]*v.mat[8] + mat[6]*v.mat[9] + mat[10]*v.mat[10] + mat[14]*v.mat[11];
	tmp[11] = mat[3]*v.mat[8] + mat[7]*v.mat[9] + mat[11]*v.mat[10] + mat[15]*v.mat[11];

	tmp[12] = mat[0]*v.mat[12] + mat[4]*v.mat[13] + mat[8]*v.mat[14] + mat[12]*v.mat[15];
	tmp[13] = mat[1]*v.mat[12] + mat[5]*v.mat[13] + mat[9]*v.mat[14] + mat[13]*v.mat[15];
	tmp[14] = mat[2]*v.mat[12] + mat[6]*v.mat[13] + mat[10]*v.mat[14] + mat[14]*v.mat[15];
	tmp[15] = mat[3]*v.mat[12] + mat[7]*v.mat[13] + mat[11]*v.mat[14] + mat[15]*v.mat[15];
	return Matrix4x4( tmp );
}

Matrix4x4 Matrix4x4::operator()(const Matrix4x4& v) const
{
	float tmp[16];
    tmp[0] = mat[0]*v.mat[0] + mat[4]*v.mat[1] + mat[8]*v.mat[2] + mat[12]*v.mat[3];
	tmp[1] = mat[1]*v.mat[0] + mat[5]*v.mat[1] + mat[9]*v.mat[2] + mat[13]*v.mat[3];
	tmp[2] = mat[2]*v.mat[0] + mat[6]*v.mat[1] + mat[10]*v.mat[2] + mat[14]*v.mat[3];
	tmp[3] = mat[3]*v.mat[0] + mat[7]*v.mat[1] + mat[11]*v.mat[2] + mat[15]*v.mat[3];

	tmp[4] = mat[0]*v.mat[4] + mat[4]*v.mat[5] + mat[8]*v.mat[6] + mat[12]*v.mat[6];
	tmp[5] = mat[1]*v.mat[4] + mat[5]*v.mat[5] + mat[9]*v.mat[6] + mat[13]*v.mat[6];
	tmp[6] = mat[2]*v.mat[4] + mat[6]*v.mat[5] + mat[10]*v.mat[6] + mat[14]*v.mat[6];
	tmp[7] = mat[3]*v.mat[4] + mat[7]*v.mat[5] + mat[11]*v.mat[6] + mat[15]*v.mat[6];

	tmp[8] = mat[0]*v.mat[8] + mat[4]*v.mat[9] + mat[8]*v.mat[10] + mat[12]*v.mat[11];
	tmp[9] = mat[1]*v.mat[8] + mat[5]*v.mat[9] + mat[9]*v.mat[10] + mat[13]*v.mat[11];
	tmp[10] = mat[2]*v.mat[8] + mat[6]*v.mat[9] + mat[10]*v.mat[10] + mat[14]*v.mat[11];
	tmp[11] = mat[3]*v.mat[8] + mat[7]*v.mat[9] + mat[11]*v.mat[10] + mat[15]*v.mat[11];

	tmp[12] = mat[0]*v.mat[12] + mat[4]*v.mat[13] + mat[8]*v.mat[14] + mat[12]*v.mat[15];
	tmp[13] = mat[1]*v.mat[12] + mat[5]*v.mat[13] + mat[9]*v.mat[14] + mat[13]*v.mat[15];
	tmp[14] = mat[2]*v.mat[12] + mat[6]*v.mat[13] + mat[10]*v.mat[14] + mat[14]*v.mat[15];
	tmp[15] = mat[3]*v.mat[12] + mat[7]*v.mat[13] + mat[11]*v.mat[14] + mat[15]*v.mat[15];
	return Matrix4x4( tmp );
}

Vector Matrix4x4::operator*(const Vector& v) const
{
	float tmp[3];
	tmp[0] = mat[0]*v.d[0] + mat[4]*v.d[1] + mat[8]*v.d[2];
	tmp[1] = mat[1]*v.d[0] + mat[5]*v.d[1] + mat[9]*v.d[2];
	tmp[2] = mat[2]*v.d[0] + mat[6]*v.d[1] + mat[10]*v.d[2];
    return Vector( tmp );
}

Vector operator*(const Vector& v, const Matrix4x4& m)
{
	float tmp[3];
	tmp[0] = m.mat[0]*v.X() + m.mat[1]*v.Y() + m.mat[2]*v.Z();
	tmp[1] = m.mat[4]*v.X() + m.mat[5]*v.Y() + m.mat[6]*v.Z();
	tmp[2] = m.mat[8]*v.X() + m.mat[9]*v.Y() + m.mat[10]*v.Z();
    return Vector( tmp );
}

Vector Matrix4x4::operator()(const Vector& v) const
{
	float tmp[3];
	tmp[0] = mat[0]*v.d[0] + mat[4]*v.d[1] + mat[8]*v.d[2];
	tmp[1] = mat[1]*v.d[0] + mat[5]*v.d[1] + mat[9]*v.d[2];
	tmp[2] = mat[2]*v.d[0] + mat[6]*v.d[1] + mat[10]*v.d[2];
    return Vector( tmp );
}

Point Matrix4x4::operator*(const Point& v) const
{
	float tmp[4];
	tmp[0] = mat[0]*v.d[0] + mat[4]*v.d[1] + mat[8]*v.d[2] + mat[12];
	tmp[1] = mat[1]*v.d[0] + mat[5]*v.d[1] + mat[9]*v.d[2] + mat[13];
	tmp[2] = mat[2]*v.d[0] + mat[6]*v.d[1] + mat[10]*v.d[2] + mat[14];
	//tmp[3] = mat[3]*v.d[0] + mat[7]*v.d[1] + mat[11]*v.d[2] + mat[15];
    //return Point( tmp[0]/tmp[3], tmp[1]/tmp[3], tmp[2]/tmp[3] );
    return Point( tmp[0], tmp[1], tmp[2] );
}

Point Matrix4x4::operator()(const Point& v) const
{
	float tmp[4];
	tmp[0] = mat[0]*v.d[0] + mat[4]*v.d[1] + mat[8]*v.d[2] + mat[12];
	tmp[1] = mat[1]*v.d[0] + mat[5]*v.d[1] + mat[9]*v.d[2] + mat[13];
	tmp[2] = mat[2]*v.d[0] + mat[6]*v.d[1] + mat[10]*v.d[2] + mat[14];
	//tmp[3] = mat[3]*v.d[0] + mat[7]*v.d[1] + mat[11]*v.d[2] + mat[15];
    //return Point( tmp[0]/tmp[3], tmp[1]/tmp[3], tmp[2]/tmp[3] );
	return Point( tmp[0], tmp[1], tmp[2] );
}

Matrix4x4 Matrix4x4::Transpose( void ) const
{
	float tmp[16];
	tmp[0] = mat[0]; tmp[5] = mat[5]; tmp[10] = mat[10]; tmp[15] = mat[15];
	tmp[1] = mat[4]; tmp[2] = mat[8]; tmp[3] = mat[12];
	tmp[4] = mat[1]; tmp[6] = mat[9]; tmp[7] = mat[13];
	tmp[8] = mat[2]; tmp[9] = mat[6]; tmp[11] = mat[14];
	tmp[12] = mat[3]; tmp[13] = mat[7]; tmp[14] = mat[11];
	return Matrix4x4( tmp );
}

Matrix4x4 Matrix4x4::Invert( void ) const
{
	float inverse[16];
	memcpy( inverse, identityArray, 16*sizeof(float) );

    float t;
    int i, j, k, swap;
    float tmp[4][4];

	// This can probably be done with 'memcpy( tmp, mat, 16*sizeof(float) );'
    for (i = 0; i < 4; i++)
		for (j = 0; j < 4; j++)
			tmp[i][j] = mat[i*4+j];

    for (i = 0; i < 4; i++) {
        /* look for largest element in column. */
        swap = i;
        for (j = i + 1; j < 4; j++)
            if (fabs(tmp[j][i]) > fabs(tmp[i][i]))
                swap = j;

        if (swap != i) {
            /* swap rows. */
            for (k = 0; k < 4; k++) {
                t = tmp[i][k];
                tmp[i][k] = tmp[swap][k];
                tmp[swap][k] = t;

                t = inverse[i*4+k];
                inverse[i*4+k] = inverse[swap*4+k];
                inverse[swap*4+k] = t;
            }
        }

        if (tmp[i][i] == 0) 
			return Matrix4x4::Zero();

        t = tmp[i][i];
        for (k = 0; k < 4; k++) {
            tmp[i][k] /= t;
            inverse[i*4+k] /= t;
        }
        for (j = 0; j < 4; j++)
            if (j != i) {
                t = tmp[j][i];
                for (k = 0; k < 4; k++) {
                    tmp[j][k] -= tmp[i][k]*t;
                    inverse[j*4+k] -= inverse[i*4+k]*t;
                }
            }
    }

    return Matrix4x4( inverse );
}


float Matrix4x4::Determinant( void ) const
{
	printf("Matrix4x4::Determinant is not yet implemented!\n");
	return 0;
}


// The following four functions (for Translate, Scale, Rotate) have not been
//     optimized for speed, but rather for readability.
Matrix4x4 Matrix4x4::Translate( float x, float y, float z )
{
	Matrix4x4 tmp;  // Default constructor gives the identity matrix
	tmp[12] = x;
	tmp[13] = y;
	tmp[14] = z;
	return tmp;
}

Matrix4x4 Matrix4x4::Scale    ( float x, float y, float z )
{
	Matrix4x4 tmp;  // Default constructor gives the identity matrix
	tmp[0] = x;
	tmp[5] = y;
	tmp[10] = z;
	return tmp;
}

Matrix4x4 Matrix4x4::Scale    ( float constantScale )
{
	Matrix4x4 tmp;  // Default constructor gives the identity matrix
	tmp[0] = constantScale;
	tmp[5] = constantScale;
	tmp[10] = constantScale;
	return tmp;
}

Matrix4x4 Matrix4x4::Rotate   ( float angle, const Vector& axis )
{
	Vector normAxis( axis );
	normAxis.Normalize();
	float sMatrix[16] = { 0, normAxis.Z(), -normAxis.Y(), 0, 
		                  -normAxis.Z(), 0, normAxis.X(), 0,
						  normAxis.Y(), -normAxis.X(), 0, 0,
						  0, 0, 0, 0 };
	Matrix4x4 sMat( sMatrix );
	Matrix4x4 uuMat = Matrix4x4::OuterProduct( normAxis );
	Matrix4x4 ret = uuMat + cos( angle*M_PI/180.0f )*( Matrix4x4::Identity() - uuMat ) + sin( angle*M_PI/180.0f )*sMat;
	ret[15] = 1;
	return ret;
}


Matrix4x4 Matrix4x4::Perspective  ( float fovy, float aspect, float zNear, float zFar )
{
	float cot = 1.0f / tan( fovy*float(M_PI)/360.0f );
	float denom = zNear-zFar;
	float pMatrix[16] = { cot / aspect, 0, 0, 0,  
		                  0, cot, 0, 0, 
						  0, 0, (zFar+zNear)/denom, -1,
						  0, 0, 2*zFar*zNear/denom, 0 };
	return Matrix4x4( pMatrix );
}

Matrix4x4 Matrix4x4::LookAt( const Point &eye, const Point &at, const Vector &up )
{
	Vector F(at - eye);
	F.Normalize();
	Vector Up(up);
	Up.Normalize();
	Vector s = F.Cross(Up);
	s.Normalize();
	Vector u = s.Cross(F);
	float mMatrix[16] = {s.X(), u.X(), -F.X(), 0,
						 s.Y(), u.Y(), -F.Y(), 0,
						 s.Z(), u.Z(), -F.Z(), 0,
						 0, 0, 0, 1};
	return Matrix4x4(mMatrix)*Translate( -eye.X(), -eye.Y(), -eye.Z() );
}


Matrix4x4 Matrix4x4::OuterProduct ( const Vector& u )
{
	float tmp[16] = { u.X()*u.X(), u.X()*u.Y(), u.X()*u.Z(), 0 /* u.X()*u.W() */,
		              u.Y()*u.X(), u.Y()*u.Y(), u.Y()*u.Z(), 0 /* u.Y()*u.W() */,
					  u.Z()*u.X(), u.Z()*u.Y(), u.Z()*u.Z(), 0 /* u.Z()*u.W() */,
					  0, 0, 0, 0 }; /* u.W()*u.X(), u.W()*u.Y(), u.W()*u.Z(), u.W()*u.W() */
	return Matrix4x4( tmp );
}

Matrix4x4 Matrix4x4::OuterProduct ( const Vector& u, const Vector& v )
{
	float tmp[16] = { v.X()*u.X(), v.X()*u.Y(), v.X()*u.Z(), 0 /* v.X()*u.W() */,
		              v.Y()*u.X(), v.Y()*u.Y(), v.Y()*u.Z(), 0 /* v.Y()*u.W() */,
					  v.Z()*u.X(), v.Z()*u.Y(), v.Z()*u.Z(), 0 /* v.Z()*u.W() */,
					  0, 0, 0, 0 }; /* v.W()*u.X(), v.W()*u.Y(), v.W()*u.Z(), v.W()*u.W() */
	return Matrix4x4( tmp );
}









Matrix4x4::Matrix4x4( FILE *f, char *restOfLine )
{
	char buf[ MAXLINELENGTH ], token[256], *ptr;

	if (restOfLine) // Check if this is a special one-line matrix
	{
		ptr = StripLeadingTokenToBuffer( restOfLine, token );
		MakeLower( token );

		if (!strcmp(token, "identity"))
		{
			memcpy( mat, Matrix4x4::identityArray, 16*sizeof(float) );
			return;
		}
		else if (!strcmp(token, "scale"))
		{
			Vector v( ptr );
			memcpy( mat, Matrix4x4::identityArray, 16*sizeof(float) );
			mat[0] = v.X();
			mat[5] = v.Y();
			mat[10] = v.Z();
			return;
		}
		else if (!strcmp(token, "translate"))
		{
			Vector v( ptr );
			memcpy( mat, Matrix4x4::identityArray, 16*sizeof(float) );
			mat[12] = v.X();
			mat[13] = v.Y();
			mat[14] = v.Z();
			return;
		}
		else if (!strcmp(token, "rotate"))
		{
			float angle;
			ptr = StripLeadingNumber( ptr, &angle );
			Vector v( ptr );
			Matrix4x4 tmp = Matrix4x4::Rotate( angle, v );
			memcpy( mat, tmp.mat, 16*sizeof(float) );
			return;
		}
	}

	// Set some defaults... (set to the identity matrix)
	memcpy( mat, Matrix4x4::identityArray, 16*sizeof(float) );
	
	// Search the scene file.
	while( fgets(buf, MAXLINELENGTH, f) != NULL )  
	{
		// Is this line a comment?
		ptr = StripLeadingWhiteSpace( buf );
		if (ptr[0] == '#') continue;

		// Nope.  So find out what the command is...
		//   The "command" is either "end" or a wavelength for the bin.
		ptr = StripLeadingTokenToBuffer( ptr, token );
		MakeLower( token );
	
		// Take different measures, depending on the command.
		if (!strcmp(token,"end")) break;
		else if (!strcmp(token, "row0")) 
		{
			ptr = StripLeadingNumber( ptr, &mat[0] );
			ptr = StripLeadingNumber( ptr, &mat[4] );
			ptr = StripLeadingNumber( ptr, &mat[8] );
			ptr = StripLeadingNumber( ptr, &mat[12] );
		}
		else if (!strcmp(token, "row1")) 
		{
			ptr = StripLeadingNumber( ptr, &mat[1] );
			ptr = StripLeadingNumber( ptr, &mat[5] );
			ptr = StripLeadingNumber( ptr, &mat[9] );
			ptr = StripLeadingNumber( ptr, &mat[13] );
		}
		else if (!strcmp(token, "row2")) 
		{
			ptr = StripLeadingNumber( ptr, &mat[2] );
			ptr = StripLeadingNumber( ptr, &mat[6] );
			ptr = StripLeadingNumber( ptr, &mat[10] );
			ptr = StripLeadingNumber( ptr, &mat[14] );
		}
		else if (!strcmp(token, "row3")) 
		{
			ptr = StripLeadingNumber( ptr, &mat[3] );
			ptr = StripLeadingNumber( ptr, &mat[7] );
			ptr = StripLeadingNumber( ptr, &mat[11] );
			ptr = StripLeadingNumber( ptr, &mat[15] );
		}
		else if (!strcmp(token, "col0")) 
		{
			ptr = StripLeadingNumber( ptr, &mat[0] );
			ptr = StripLeadingNumber( ptr, &mat[1] );
			ptr = StripLeadingNumber( ptr, &mat[2] );
			ptr = StripLeadingNumber( ptr, &mat[3] );
		}
		else if (!strcmp(token, "col1")) 
		{
			ptr = StripLeadingNumber( ptr, &mat[4] );
			ptr = StripLeadingNumber( ptr, &mat[5] );
			ptr = StripLeadingNumber( ptr, &mat[6] );
			ptr = StripLeadingNumber( ptr, &mat[7] );
		}
		else if (!strcmp(token, "col2")) 
		{
			ptr = StripLeadingNumber( ptr, &mat[8] );
			ptr = StripLeadingNumber( ptr, &mat[9] );
			ptr = StripLeadingNumber( ptr, &mat[10] );
			ptr = StripLeadingNumber( ptr, &mat[11] );
		}
		else if (!strcmp(token, "col3")) 
		{
			ptr = StripLeadingNumber( ptr, &mat[12] );
			ptr = StripLeadingNumber( ptr, &mat[13] );
			ptr = StripLeadingNumber( ptr, &mat[14] );
			ptr = StripLeadingNumber( ptr, &mat[15] );
		}
		else
		{
			printf("Error: Unknown token '%s' when reading matrix!\n", token );
		}
	}
}



