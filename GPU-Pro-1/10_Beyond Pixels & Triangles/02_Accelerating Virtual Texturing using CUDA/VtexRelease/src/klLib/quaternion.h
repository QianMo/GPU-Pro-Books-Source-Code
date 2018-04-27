/**
 *
 *  This software module was originally developed for research purposes,
 *  by Multimedia Lab at Ghent University (Belgium).
 *  Its performance may not be optimized for specific applications.
 *
 *  Those intending to use this software module in hardware or software products
 *  are advized that its use may infringe existing patents. The developers of 
 *  this software module, their companies, Ghent Universtity, nor Multimedia Lab 
 *  have any liability for use of this software module or modifications thereof.
 *
 *  Ghent University and Multimedia Lab (Belgium) retain full right to modify and
 *  use the code for their own purpose, assign or donate the code to a third
 *  party, and to inhibit third parties from using the code for their products. 
 *
 *  This copyright notice must be included in all copies or derivative works.
 *
 *  For information on its use, applications and associated permission for use,
 *  please contact Prof. Rik Van de Walle (rik.vandewalle@ugent.be). 
 *
 *  Detailed information on the activities of
 *  Ghent University Multimedia Lab can be found at
 *  http://multimedialab.elis.ugent.be/.
 *
 *  Copyright (c) Ghent University 2004-2009.
 *
 **/

#ifndef KL_QUATERNION_H
#define KL_QUATERNION_H

#include "vectors.h"
#include "matrix.h"

template <class Type> class klQuaternionTempl : public klVector4d<Type> {
public:
	inline klQuaternionTempl(void) {}

	inline klQuaternionTempl(Type x, Type y, Type z, Type w) {
		data[0] = x;
		data[1] = y;
		data[2] = z;
		data[3] = w;
	}

	inline klQuaternionTempl(klVector4d<Type> p) {
		data[0] = p[0];
		data[1] = p[1];
		data[2] = p[2];
		data[3] = p[3];
	}

	inline klQuaternionTempl(klVector3d<Type> axis, Type angle) {
		Type sina = sin(angle/2);
		Type cosa = cos(angle/2);

		data[0] = axis[0] * sina;
		data[1] = axis[1] * sina;
		data[2] = axis[2] * sina;
		data[3] = cosa;

		normalize();
	};

	inline void toAxisAngle(klVector3d<Type> &axis, Type &angle) {
		Type sina = sqrt((Type)1.0-data[3]*data[3]);
		angle = acos(data[3])*2;

		if (fabs(sina) < 0.000005) sina = 1;

		axis[0] = data[0] / sina;
		axis[1] = data[1] / sina;
		axis[2] = data[2] / sina;
	}

	inline void slerp(const klQuaternionTempl<Type>& from,const klQuaternionTempl<Type>& to,Type slerp, bool shortestPath) {
		Type cosa = from*to;
		Type angle = acos(cosa);

		if ( abs(angle) < 0.00005) {
			*this = from;
			return;
		}

		Type sina = sin(angle);
		Type sini = (Type)1.0/sina;
		Type i = sin(((Type)1.0-slerp)*angle)*sini;
		Type ii = sin(slerp*angle)*sini;
		if (cosa < 0.0f && shortestPath) {
			i = -i;
			*this = from*i + to*ii;
			normalize();
		} else {
			*this = from*i + to*ii;
		}
	}

	inline void identity(void) { 
		data[0] = data[1] = data[2] = 0;
		data[3] = 1;
	};

	inline klQuaternionTempl<Type> mult(const klQuaternionTempl<Type>& r) const { 
		klVec3 tv(data[0],data[1],data[2]);
		float tw = data[3];
		klVec3 rv(r[0],r[1],r[2]);
		float rw = r[3];

		klVec3 kv;
		kv.cross(tv, rv);
		kv = kv + tv*rw + rv*tw;
		float kw = tw*rw - tv.dot(rv);
		return klQuaternionTempl(kv[0],kv[1],kv[2],kw);
	};

	inline void negate(void) { 
		data[0] = -data[0];
		data[1] = -data[1];
		data[2] = -data[2];
	};

	inline void toMatrix(Type *mat) {
		Type xx = data[0] * data[0];
		Type xy = data[0] * data[1];
		Type xz = data[0] * data[2];
		Type xw = data[0] * data[3];

		Type yy = data[1] * data[1];
		Type yz = data[1] * data[2];
		Type yw = data[1] * data[3];

		Type zz = data[2] * data[2];
		Type zw = data[2] * data[3];

		mat[0] = 1 - 2 * ( yy + zz );
		mat[1] =     2 * ( xy - zw );
		mat[2] =     2 * ( xz + yw );

		mat[4] =     2 * ( xy + zw );
		mat[5] = 1 - 2 * ( xx + zz );
		mat[6] =     2 * ( yz - xw );

		mat[8] =     2 * ( xz - yw );
		mat[9] =     2 * ( yz + xw );
		mat[10]= 1 - 2 * ( xx + yy );

		mat[3] = mat[7] = mat[11] = mat[12] = mat[13] = mat[14] = 0;
		mat[15]= 1;
	};

	klMatrix4x4 toMatrix(void) {
		Type xx = data[0] * data[0];
		Type xy = data[0] * data[1];
		Type xz = data[0] * data[2];
		Type xw = data[0] * data[3];

		Type yy = data[1] * data[1];
		Type yz = data[1] * data[2];
		Type yw = data[1] * data[3];

		Type zz = data[2] * data[2];
		Type zw = data[2] * data[3];

		klVec3 line1(1 - 2 * ( yy + zz ), 2 * ( xy - zw ), 2 * ( xz + yw ));
		klVec3 line2(2 * ( xy + zw ), 1 - 2 * ( xx + zz ), 2 * ( yz - xw ));
		klVec3 line3(2 * ( xz - yw ), 2 * ( yz + xw ), 1 - 2 * ( xx + yy ));

		return klMatrix4x4(line1, line2, line3);
	};

};

typedef klQuaternionTempl<float> klQuaternion;
typedef klQuaternionTempl<double>  klQuaterniond;

#endif KL_QUATERNION_H