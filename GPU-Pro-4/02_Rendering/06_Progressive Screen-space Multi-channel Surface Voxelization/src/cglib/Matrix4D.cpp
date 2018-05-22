
#include <memory.h>
#include <math.h>
#include <string.h>

#include "Matrix4D.h"
#include "Vector3D.h"

// Cramer's rule
void Invert(float *mat, float *dst)
{
	DBL tmp[12]; /* temp array for pairs */
	DBL src[16]; /* array of transpose source matrix */
	DBL det; /* determinant */
	/* transpose matrix */
	for (int i = 0; i < 4; i++) {
	src[i] = mat[i*4];
	src[i + 4] = mat[i*4 + 1];
	src[i + 8] = mat[i*4 + 2];
	src[i + 12] = mat[i*4 + 3];
	}
	/* calculate pairs for first 8 elements (cofactors) */
	tmp[0] = src[10] * src[15];
	tmp[1] = src[11] * src[14];
	tmp[2] = src[9] * src[15];
	tmp[3] = src[11] * src[13];
	tmp[4] = src[9] * src[14];
	tmp[5] = src[10] * src[13];
	tmp[6] = src[8] * src[15];
	tmp[7] = src[11] * src[12];
	tmp[8] = src[8] * src[14];
	tmp[9] = src[10] * src[12];
	tmp[10] = src[8] * src[13];
	tmp[11] = src[9] * src[12];
	/* calculate first 8 elements (cofactors) */
	dst[0]  = tmp[0]*src[5] + tmp[3]*src[6] + tmp[4]*src[7];
	dst[0] -= tmp[1]*src[5] + tmp[2]*src[6] + tmp[5]*src[7];
	dst[1]  = tmp[1]*src[4] + tmp[6]*src[6] + tmp[9]*src[7];
	dst[1] -= tmp[0]*src[4] + tmp[7]*src[6] + tmp[8]*src[7];
	dst[2]  = tmp[2]*src[4] + tmp[7]*src[5] + tmp[10]*src[7];
	dst[2] -= tmp[3]*src[4] + tmp[6]*src[5] + tmp[11]*src[7];
	dst[3]  = tmp[5]*src[4] + tmp[8]*src[5] + tmp[11]*src[6];
	dst[3] -= tmp[4]*src[4] + tmp[9]*src[5] + tmp[10]*src[6];
	dst[4]  = tmp[1]*src[1] + tmp[2]*src[2] + tmp[5]*src[3];
	dst[4] -= tmp[0]*src[1] + tmp[3]*src[2] + tmp[4]*src[3];
	dst[5]  = tmp[0]*src[0] + tmp[7]*src[2] + tmp[8]*src[3];
	dst[5] -= tmp[1]*src[0] + tmp[6]*src[2] + tmp[9]*src[3];
	dst[6]  = tmp[3]*src[0] + tmp[6]*src[1] + tmp[11]*src[3];
	dst[6] -= tmp[2]*src[0] + tmp[7]*src[1] + tmp[10]*src[3];
	dst[7]  = tmp[4]*src[0] + tmp[9]*src[1] + tmp[10]*src[2];
	dst[7] -= tmp[5]*src[0] + tmp[8]*src[1] + tmp[11]*src[2];
	/* calculate pairs for second 8 elements (cofactors) */
	tmp[0] = src[2]*src[7];
	tmp[1] = src[3]*src[6];
	tmp[2] = src[1]*src[7];
	tmp[3] = src[3]*src[5];
	tmp[4] = src[1]*src[6];
	tmp[5] = src[2]*src[5];
	tmp[6] = src[0]*src[7];
	tmp[7] = src[3]*src[4];
	tmp[8] = src[0]*src[6];
	tmp[9] = src[2]*src[4];
	tmp[10] = src[0]*src[5];
	tmp[11] = src[1]*src[4];
	/* calculate second 8 elements (cofactors) */
	dst[8]  = tmp[0]*src[13] + tmp[3]*src[14] + tmp[4]*src[15];
	dst[8] -= tmp[1]*src[13] + tmp[2]*src[14] + tmp[5]*src[15];
	dst[9]  = tmp[1]*src[12] + tmp[6]*src[14] + tmp[9]*src[15];
	dst[9] -= tmp[0]*src[12] + tmp[7]*src[14] + tmp[8]*src[15];
	dst[10] = tmp[2]*src[12] + tmp[7]*src[13] + tmp[10]*src[15];
	dst[10]-= tmp[3]*src[12] + tmp[6]*src[13] + tmp[11]*src[15];
	dst[11] = tmp[5]*src[12] + tmp[8]*src[13] + tmp[11]*src[14];
	dst[11]-= tmp[4]*src[12] + tmp[9]*src[13] + tmp[10]*src[14];
	dst[12] = tmp[2]*src[10] + tmp[5]*src[11] + tmp[1]*src[9];
	dst[12]-= tmp[4]*src[11] + tmp[0]*src[9] + tmp[3]*src[10];
	dst[13] = tmp[8]*src[11] + tmp[0]*src[8] + tmp[7]*src[10];
	dst[13]-= tmp[6]*src[10] + tmp[9]*src[11] + tmp[1]*src[8];
	dst[14] = tmp[6]*src[9] + tmp[11]*src[11] + tmp[3]*src[8];
	dst[14]-= tmp[10]*src[11] + tmp[2]*src[8] + tmp[7]*src[9];
	dst[15] = tmp[10]*src[10] + tmp[4]*src[8] + tmp[9]*src[9];
	dst[15]-= tmp[8]*src[9] + tmp[11]*src[10] + tmp[5]*src[8];
	/* calculate determinant */
	det=src[0]*dst[0]+src[1]*dst[1]+src[2]*dst[2]+src[3]*dst[3];
	/* calculate matrix inverse */
	det = 1/det;
	for (int j = 0; j < 16; j++)
	dst[j] *= det;
}

void Invert2(DBL * Min, DBL * Mout)
{
#define SWAP_ROWS(a, b) { double *_tmp = a; (a)=(b); (b)=_tmp; }
#define MAT(m,r,c) (m)[(c)*4+(r)]

   double m[16], out[16];
   int i;
   double wtmp[4][8];
   double m0, m1, m2, m3, s;
   double *r0, *r1, *r2, *r3;
   for (i=0;i<16;i++)
	   m[i]=Min[i];

   r0 = wtmp[0], r1 = wtmp[1], r2 = wtmp[2], r3 = wtmp[3];

   r0[0] = MAT(m, 0, 0), r0[1] = MAT(m, 0, 1),
      r0[2] = MAT(m, 0, 2), r0[3] = MAT(m, 0, 3),
      r0[4] = 1.0, r0[5] = r0[6] = r0[7] = 0.0,
      r1[0] = MAT(m, 1, 0), r1[1] = MAT(m, 1, 1),
      r1[2] = MAT(m, 1, 2), r1[3] = MAT(m, 1, 3),
      r1[5] = 1.0, r1[4] = r1[6] = r1[7] = 0.0,
      r2[0] = MAT(m, 2, 0), r2[1] = MAT(m, 2, 1),
      r2[2] = MAT(m, 2, 2), r2[3] = MAT(m, 2, 3),
      r2[6] = 1.0, r2[4] = r2[5] = r2[7] = 0.0,
      r3[0] = MAT(m, 3, 0), r3[1] = MAT(m, 3, 1),
      r3[2] = MAT(m, 3, 2), r3[3] = MAT(m, 3, 3),
      r3[7] = 1.0, r3[4] = r3[5] = r3[6] = 0.0;

   /* choose pivot - or die */
   if (fabs(r3[0]) > fabs(r2[0]))
      SWAP_ROWS(r3, r2);
   if (fabs(r2[0]) > fabs(r1[0]))
      SWAP_ROWS(r2, r1);
   if (fabs(r1[0]) > fabs(r0[0]))
      SWAP_ROWS(r1, r0);
   if (0.0 == r0[0])
      return;

   /* eliminate first variable     */
   m1 = r1[0] / r0[0];
   m2 = r2[0] / r0[0];
   m3 = r3[0] / r0[0];
   s = r0[1];
   r1[1] -= m1 * s;
   r2[1] -= m2 * s;
   r3[1] -= m3 * s;
   s = r0[2];
   r1[2] -= m1 * s;
   r2[2] -= m2 * s;
   r3[2] -= m3 * s;
   s = r0[3];
   r1[3] -= m1 * s;
   r2[3] -= m2 * s;
   r3[3] -= m3 * s;
   s = r0[4];
   if (s != 0.0) {
      r1[4] -= m1 * s;
      r2[4] -= m2 * s;
      r3[4] -= m3 * s;
   }
   s = r0[5];
   if (s != 0.0) {
      r1[5] -= m1 * s;
      r2[5] -= m2 * s;
      r3[5] -= m3 * s;
   }
   s = r0[6];
   if (s != 0.0) {
      r1[6] -= m1 * s;
      r2[6] -= m2 * s;
      r3[6] -= m3 * s;
   }
   s = r0[7];
   if (s != 0.0) {
      r1[7] -= m1 * s;
      r2[7] -= m2 * s;
      r3[7] -= m3 * s;
   }

   /* choose pivot - or die */
   if (fabs(r3[1]) > fabs(r2[1]))
      SWAP_ROWS(r3, r2);
   if (fabs(r2[1]) > fabs(r1[1]))
      SWAP_ROWS(r2, r1);
   if (0.0 == r1[1])
      return;

   /* eliminate second variable */
   m2 = r2[1] / r1[1];
   m3 = r3[1] / r1[1];
   r2[2] -= m2 * r1[2];
   r3[2] -= m3 * r1[2];
   r2[3] -= m2 * r1[3];
   r3[3] -= m3 * r1[3];
   s = r1[4];
   if (0.0 != s) {
      r2[4] -= m2 * s;
      r3[4] -= m3 * s;
   }
   s = r1[5];
   if (0.0 != s) {
      r2[5] -= m2 * s;
      r3[5] -= m3 * s;
   }
   s = r1[6];
   if (0.0 != s) {
      r2[6] -= m2 * s;
      r3[6] -= m3 * s;
   }
   s = r1[7];
   if (0.0 != s) {
      r2[7] -= m2 * s;
      r3[7] -= m3 * s;
   }

   /* choose pivot - or die */
   if (fabs(r3[2]) > fabs(r2[2]))
      SWAP_ROWS(r3, r2);
   if (0.0 == r2[2])
      return;

   /* eliminate third variable */
   m3 = r3[2] / r2[2];
   r3[3] -= m3 * r2[3], r3[4] -= m3 * r2[4],
      r3[5] -= m3 * r2[5], r3[6] -= m3 * r2[6], r3[7] -= m3 * r2[7];

   /* last check */
   if (0.0 == r3[3])
      return;

   s = 1.0 / r3[3];		/* now back substitute row 3 */
   r3[4] *= s;
   r3[5] *= s;
   r3[6] *= s;
   r3[7] *= s;

   m2 = r2[3];			/* now back substitute row 2 */
   s = 1.0 / r2[2];
   r2[4] = s * (r2[4] - r3[4] * m2), r2[5] = s * (r2[5] - r3[5] * m2),
      r2[6] = s * (r2[6] - r3[6] * m2), r2[7] = s * (r2[7] - r3[7] * m2);
   m1 = r1[3];
   r1[4] -= r3[4] * m1, r1[5] -= r3[5] * m1,
      r1[6] -= r3[6] * m1, r1[7] -= r3[7] * m1;
   m0 = r0[3];
   r0[4] -= r3[4] * m0, r0[5] -= r3[5] * m0,
      r0[6] -= r3[6] * m0, r0[7] -= r3[7] * m0;

   m1 = r1[2];			/* now back substitute row 1 */
   s = 1.0 / r1[1];
   r1[4] = s * (r1[4] - r2[4] * m1), r1[5] = s * (r1[5] - r2[5] * m1),
      r1[6] = s * (r1[6] - r2[6] * m1), r1[7] = s * (r1[7] - r2[7] * m1);
   m0 = r0[2];
   r0[4] -= r2[4] * m0, r0[5] -= r2[5] * m0,
      r0[6] -= r2[6] * m0, r0[7] -= r2[7] * m0;

   m0 = r0[1];			/* now back substitute row 0 */
   s = 1.0 / r0[0];
   r0[4] = s * (r0[4] - r1[4] * m0), r0[5] = s * (r0[5] - r1[5] * m0),
      r0[6] = s * (r0[6] - r1[6] * m0), r0[7] = s * (r0[7] - r1[7] * m0);

   MAT(out, 0, 0) = r0[4];
   MAT(out, 0, 1) = r0[5], MAT(out, 0, 2) = r0[6];
   MAT(out, 0, 3) = r0[7], MAT(out, 1, 0) = r1[4];
   MAT(out, 1, 1) = r1[5], MAT(out, 1, 2) = r1[6];
   MAT(out, 1, 3) = r1[7], MAT(out, 2, 0) = r2[4];
   MAT(out, 2, 1) = r2[5], MAT(out, 2, 2) = r2[6];
   MAT(out, 2, 3) = r2[7], MAT(out, 3, 0) = r3[4];
   MAT(out, 3, 1) = r3[5], MAT(out, 3, 2) = r3[6];
   MAT(out, 3, 3) = r3[7];

   for (i=0;i<16;i++)
	   Mout[i]=(float)out[i];

   return;

#undef MAT
#undef SWAP_ROWS
}

Matrix4D::Matrix4D(void)
{
	memset(a,0,sizeof(DBL)*16);
	a[15] = 1;
	sync();
}

Matrix4D::Matrix4D(DBL *data)
{
	memcpy(a,data,sizeof(DBL)*16);
	sync();
}

Matrix4D::Matrix4D(DBL a00, DBL a01, DBL a02, DBL a03,
                   DBL a10, DBL a11, DBL a12, DBL a13,
                   DBL a20, DBL a21, DBL a22, DBL a23,
                   DBL a30, DBL a31, DBL a32, DBL a33)
{
    a [0] = a00; a [1] = a01; a [2] = a02; a [3] = a03;
    a [4] = a10; a [5] = a11; a [6] = a12; a [7] = a13;
    a [8] = a20; a [9] = a21; a[10] = a22; a[11] = a23;
    a[12] = a30; a[13] = a31; a[14] = a32; a[15] = a33;
}

void Matrix4D::setData(DBL *data)
{
	memcpy(a,data,sizeof(DBL)*16);
	sync();
}

void Matrix4D::sync()
{
#ifdef INTEL_COMPILER
	row1 = F32vec4(a[0],a[1],a[2],a[3]);
	row2 = F32vec4(a[4],a[5],a[6],a[7]);
	row3 = F32vec4(a[8],a[9],a[10],a[11]);
	row4 = F32vec4(a[12],a[13],a[14],a[15]);
#endif
}

DBL Matrix4D::operator[](int i)
{
	return a[i];
}

DBL Matrix4D::operator()(int row, int col)
{
	return a[4*row+col];
}

Matrix4D Matrix4D::operator=(Matrix4D m)
{
	memcpy(a,m.a,16*sizeof(DBL));
	return *this;
}

Matrix4D Matrix4D::operator*(Matrix4D right)
{
	Matrix4D result;
#ifdef INTEL_COMPILER
	F32vec4 col1, col2, col3, col4;
	col1 = F32vec4(right(0,0),right(1,0),right(2,0),right(3,0));
	col2 = F32vec4(right(0,1),right(1,1),right(2,1),right(3,1));
	col2 = F32vec4(right(0,2),right(1,2),right(2,2),right(3,2));
	col2 = F32vec4(right(0,3),right(1,3),right(2,3),right(3,3));
	result.a[0]=add_horizontal(row1*col1);
	result.a[1]=add_horizontal(row1*col2);
	result.a[2]=add_horizontal(row1*col3);
	result.a[3]=add_horizontal(row1*col4);
	result.a[4]=add_horizontal(row2*col1);
	result.a[5]=add_horizontal(row2*col2);
	result.a[6]=add_horizontal(row2*col3);
	result.a[7]=add_horizontal(row2*col4);
	result.a[8]=add_horizontal(row3*col1);
	result.a[9]=add_horizontal(row3*col2);
	result.a[10]=add_horizontal(row3*col3);
	result.a[11]=add_horizontal(row3*col4);
	result.a[12]=add_horizontal(row4*col1);
	result.a[13]=add_horizontal(row4*col2);
	result.a[14]=add_horizontal(row4*col3);
	result.a[15]=add_horizontal(row4*col4);
#else
	int i,j;
	for (i=0;i<4;i++)
		for (j=0;j<4;j++)
			result.a[i*4+j] = a[i*4+0]*right.a[0*4+j] + a[i*4+1]*right.a[1*4+j] +
				              a[i*4+2]*right.a[2*4+j] + a[i*4+3]*right.a[3*4+j];
#endif
	result.sync();

	return result;
}

void Matrix4D::makeTranslate(Vector3D t)
{
	memset(a,0,sizeof(DBL)*16);
	a[0] = a[5] = a[10] = a[15] = 1;
	a[3] = t.x; a[7] = t.y; a[11] = t.z;
	sync();
}

void Matrix4D::makeScale(Vector3D s)
{
	memset(a,0,sizeof(DBL)*16);
	a[0] = s.x; a[5] = s.y; a[10] = s.z; a[15] = 1;
	sync();
}

void Matrix4D::makeRotate(class Vector3D r, DBL theta)
{
	int i, j;

	theta = theta * PI/180.0f;
	DBL s[9], uut[9], sec[9];
	memset(s,0,sizeof(DBL)*9);
	memset(uut,0,sizeof(DBL)*9);
	memset(sec,0,sizeof(DBL)*9);
	r.normalize();
	s[1] = -r.z;
	s[2] = r.y;
	s[3] = r.z;
	s[5] = -r.x;
	s[6] = -r.y;
	s[7] = r.x;

	for(i = 0; i<9; i++){
		s[i] *= sin(theta);
	}

	uut[0] = r.x * r.x;
	uut[1] = uut[3] = r.x * r.y;
	uut[2] = uut[6] = r.x * r.z;
	uut[4] = r.y * r.y;
	uut[5] = uut[7] = r.y * r.z;
	uut[8] = r.z * r.z;


	sec[0] = sec[4] = sec[8] = 1;

	for(i = 0; i<9; i++){
		sec[i] -= uut[i];
		sec[i] *= cos(theta);
	}
	for(i = 0; i<9; i++){
		sec[i] = sec[i] + uut[i] + s[i] ;
	}

	memset(a,0,sizeof(DBL)*16);	
	for(i=0; i<3; i++)
		for(j=0; j<3; j++)
			a[4*i + j] = sec[3*i + j];
	
	a[15] = 1;
}


Vector3D Matrix4D::operator*(Vector3D vec)
{
	Vector3D res;
#ifdef INTEL_COMPILER
	F32vec4 col;
	col = F32vec4(vec.x,vec.y,vec.z,1.0f);
	res.x=add_horizontal(row1*col);
	res.y=add_horizontal(row2*col);
	res.z=add_horizontal(row3*col);
#else
	res.x = a[0]*vec.x + a[1]*vec.y + a[2]*vec.z + a[3];
	res.y = a[4]*vec.x + a[5]*vec.y + a[6]*vec.z + a[7];
	res.z = a[8]*vec.x + a[9]*vec.y + a[10]*vec.z + a[11];
#endif
	return res;
}

bool Matrix4D::operator== (const Matrix4D &m)
{
    return memcmp (a, m.a, 16 * sizeof(DBL)) == 0;
}

bool Matrix4D::operator!= (const Matrix4D &m)
{
    return memcmp (a, m.a, 16 * sizeof(DBL)) != 0;
}

void Matrix4D::invert()
{
	DBL b[16];
	Invert2(a,b);
	memcpy(a,b,16*sizeof(DBL));
	sync();
}

void Matrix4D::transpose()
{
	int i,j;
	DBL s;
	for (i=0; i<4;i++)
		for(j=0;j<i;j++)
		{
			s=a[4*i+j];
			a[4*i+j]=a[4*j+i];
			a[4*j+i]=s;
		}
	sync();
}

void Matrix4D::makeOrtho(float left, float right, float bottom, float top, float znear, float zfar)
{
	memset(a,0,16*sizeof(DBL));
	
	a[0] = 2.0f/(right-left);
	a[5] = 2.0f/(top-bottom);
	a[10]= 2.0f/(zfar-znear);
	a[15]= 1.0f;

	a[3] = -(right+left)/(right-left);
	a[7] = -(top+bottom)/(top-bottom);
	a[11]= -(zfar+znear)/(zfar-znear);
}

Matrix4D Matrix4D::identity()
{
	float mat[16];
	memset(mat,0,16*sizeof(DBL));
	
	mat[0] = mat[5] = mat[10]= mat[15]= 1.0f;
	return Matrix4D(mat);
}

Matrix4D::~Matrix4D(void)
{

}

DBL Matrix4D::determinant()
{
	return a[0]*a[5]*a[10] - a[0]*a[6]*a[9] - a[1]*a[4]*a[10] +
		       a[1]*a[6]*a[8] + a[2]*a[4]*a[9] - a[2]*a[5]*a[8] ;
}

void Matrix4D::dump(char * label, DBL *data)
{
	int i,j;

	fprintf (stdout, "%s:\n", label);
	for (i=0;i<4;i++)
	{
		for (j=0;j<4;j++)
			fprintf (stdout, "% f ", data[i*4+j]);
		fprintf (stdout, "\n");
	}
    fflush (stdout);
}

void Matrix4D::dump(char * label)
{
	int i,j;

	fprintf (stdout, "%s:\n", label);
	for (i=0;i<4;i++)
	{
		for (j=0;j<4;j++)
			fprintf (stdout, "% f ", a[i*4+j]);
		fprintf (stdout, "\n");
	}
    fflush  (stdout);
}

