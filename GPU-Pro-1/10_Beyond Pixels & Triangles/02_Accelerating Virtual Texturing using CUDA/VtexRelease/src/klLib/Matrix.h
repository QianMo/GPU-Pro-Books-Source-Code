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

#ifndef __KL_MATRIX_H
#define __KL_MATRIX_H

#include "Maths.h"
#include "Vectors.h"

/**
    This class uses the "traditional" column major form of matrices. So what you see in a book
    you can write down "visually" in the code. (se the kl***Matrix functions for example)

    So a rigid body transform would be T*R*v
       * I.e First rotate the vertex locally then fransform it T*(R*v)

*/
class klMatrix3x3 {
    float data[9];

    inline float at(int i, int j) const {
        return data[i+(j*3)];
    }

    inline float& at(int i, int j) {
        return data[i+(j*3)];
    }
public:

    // Uninitialised memory by default!
    klMatrix3x3(void) {}

    klMatrix3x3(float f00, float f01, float f02,
                float f10, float f11, float f12,
                float f20, float f21, float f22)
    {
        at(0,0) = f00;
        at(1,0) = f10;
        at(2,0) = f20;

        at(0,1) = f01;
        at(1,1) = f11;
        at(2,1) = f21;

        at(0,2) = f02;
        at(1,2) = f12;
        at(2,2) = f22;
    }

    // Get element at row i column j
    inline float operator() (int i, int j) const {
        return data[i+(j*3)];
    }

    // Get element at row i column j
    inline float& operator() (int i, int j) {
        return data[i+(j*3)];
    }

    // Get X-axis of coordinate system
    klVec3 getXAxis(void) const {
        return klVec3(at(0,0),at(0,1),at(0,2));
    }

    // Get Y-axis of coordinate system
    klVec3 getYAxis(void) const {
        return klVec3(at(1,0),at(1,1),at(1,2));       
    }

    // Get Z-axis of coordinate system
    klVec3 getZAxis(void) const {
        return klVec3(at(2,0),at(2,1),at(2,2));        
    }

    void setIdentity() {
        data[0] = data[4] = data[8] = 1.0f; //majority independent
        data[1] = data[2] = data[3] =
        data[5] = data[6] = data[7] = 0.0f;
    }

    klMatrix3x3 &transposeSelf(void) {
	    for( int i=0; i<3; i++ ) {
		    for( int j=i+1; j<3; j++ ) {
			    float temp = at(i,j);
			    at(i,j) = at(j,i);
			    at(j,i) = temp;
            }
	    }
        return (*this);
    }

    klMatrix3x3 transpose(void) const {
	    klMatrix3x3	result;
   	    for( int i=0; i<3; i++ ) {
		    for( int j=0; j<3; j++ ) {
			    result.at(i,j) = at(j,i);
            }
	    }   
        return result;        
    }

    inline klMatrix3x3 operator+( const klMatrix3x3 &o ) const {
	    return klMatrix3x3( 
            o.at(0,0) + at(0,0),o.at(0,1) + at(0,1),o.at(0,2) + at(0,2),
            o.at(1,0) + at(1,0),o.at(1,1) + at(1,1),o.at(1,2) + at(1,2),
            o.at(2,0) + at(2,0),o.at(2,1) + at(2,1),o.at(2,2) + at(2,2));
    }

    inline klMatrix3x3 operator-( const klMatrix3x3 &o ) const {
	    return klMatrix3x3( 
            o.at(0,0) - at(0,0),o.at(0,1) - at(0,1),o.at(0,2) - at(0,2),
            o.at(1,0) - at(1,0),o.at(1,1) - at(1,1),o.at(1,2) - at(1,2),
            o.at(2,0) - at(2,0),o.at(2,1) - at(2,1),o.at(2,2) - at(2,2));
    }

    inline klMatrix3x3 operator*( const klMatrix3x3 &o ) const {
	    klMatrix3x3 r;
        for ( int i=0; i<3; i++ ) {
            for ( int j=0; j<3; j++ ) {
                float t = 0.0f;
                for ( int k=0; k<3; k++ ) {
                     t += at(i,k)*o.at(k,j);
                }
                r.at(i,j) = t;
            }
        }
        return r;
    }

    klVec3 operator*( const klVec3 &o ) const {
        return klVec3(at(0,0)*o[0] + at(0,1)*o[1] + at(0,2)*o[2],
                      at(1,0)*o[0] + at(1,1)*o[1] + at(1,2)*o[2],
                      at(2,0)*o[0] + at(2,1)*o[1] + at(2,2)*o[2]);
    }

};


/**
    This class hides the row major/minor order, all functions and interfaces use
    traditional mathematics indexes A(i,j)
*/
class klMatrix4x4 {
    float data[16];
    inline float at(int i, int j) const {
        return data[i+(j<<2)];
    }

    inline float& at(int i, int j) {
        return data[i+(j<<2)];
    }
public:

    // Uninitialised memory by default!
    klMatrix4x4(void) {}

    klMatrix4x4(float f00, float f01, float f02, float f03,
                float f10, float f11, float f12, float f13,
                float f20, float f21, float f22, float f23,
                float f30, float f31, float f32, float f33)
    {
        at(0,0) = f00; at(0,1) = f01; at(0,2) = f02; at(0,3) = f03;
        at(1,0) = f10; at(1,1) = f11; at(1,2) = f12; at(1,3) = f13;
        at(2,0) = f20; at(2,1) = f21; at(2,2) = f22; at(2,3) = f23;
        at(3,0) = f30; at(3,1) = f31; at(3,2) = f32; at(3,3) = f33;
    }

    klMatrix4x4(const klMatrix3x3 &rot) {
        at(0,0) = rot(0,0); at(0,1) = rot(0,1); at(0,2) = rot(0,2); at(0,3) = 0.0f;
        at(1,0) = rot(1,0); at(1,1) = rot(1,1); at(1,2) = rot(1,2); at(1,3) = 0.0f;
        at(2,0) = rot(2,0); at(2,1) = rot(2,1); at(2,2) = rot(2,2); at(2,3) = 0.0f;
        at(3,0) = 0.0;      at(3,1) = 0.0;      at(3,2) = 0.0;      at(3,3) = 1.0f;
    }

    float *toPtr(void) {
        return data;
    }

    const float *toCPtr(void) const {
        return data;
    }

    // Get element at row i column j
    inline float operator() (int i, int j) const {
        return data[i+(j<<2)];
    }

    // Get element at row i column j
    inline float& operator() (int i, int j) {
        return data[i+(j<<2)];
    }

    // Get X-axis of coordinate system
    klVec3 getXAxis(void) const {
        return klVec3(at(0,0),at(0,1),at(0,2));
    }

    // Get Y-axis of coordinate system
    klVec3 getYAxis(void) const {
        return klVec3(at(1,0),at(1,1),at(1,2));       
    }

    // Get Z-axis of coordinate system
    klVec3 getZAxis(void) const {
        return klVec3(at(2,0),at(2,1),at(2,2));        
    }

    // Get the translation of the coord system
    klVec3 getTranslation(void) const {
        return klVec3(at(0,3),at(1,3),at(2,3));        
    }

    // Get the rotation of the coord system
    klMatrix3x3 getRotation(void) const {
        return klMatrix3x3(at(0,0), at(0,1), at(0,2),
                           at(1,0), at(1,1), at(1,2),
                           at(2,0), at(2,1), at(2,2));     
    }


    void setIdentity() {
        data[0] = data[5] = data[10] = data[15] = 1.0f; //majority independent
        data[1] = data[2] = data[3] = data[4] =
        data[6] = data[7] = data[8] = data[9] =
        data[11] = data[12] = data[13] = data[14] = 0.0f;
    }

    klMatrix4x4 &transposeSelf(void) {
	    for( int i=0; i<4; i++ ) {
		    for( int j=i+1; j<4; j++ ) {
			    float temp = at(i,j);
			    at(i,j) = at(j,i);
			    at(j,i) = temp;
            }
	    }
        return (*this);
    }

    klMatrix4x4 transpose(void) const {
	    klMatrix4x4	result;
   	    for( int i=0; i<4; i++ ) {
		    for( int j=0; j<4; j++ ) {
			    result.at(i,j) = at(j,i);
            }
	    }   
        return result;        
    }

    klMatrix4x4 inverse(void) const;

    inline klMatrix4x4 operator+( const klMatrix4x4 &o ) const {
	    return klMatrix4x4( 
            o.at(0,0) + at(0,0),o.at(0,1) + at(0,1),o.at(0,2) + at(0,2),o.at(0,3) + at(0,3),
            o.at(1,0) + at(1,0),o.at(1,1) + at(1,1),o.at(1,2) + at(1,2),o.at(1,3) + at(1,3),
            o.at(2,0) + at(2,0),o.at(2,1) + at(2,1),o.at(2,2) + at(2,2),o.at(2,3) + at(2,3),
            o.at(3,0) + at(3,0),o.at(3,1) + at(3,1),o.at(3,2) + at(3,2),o.at(3,3) + at(3,3));
    }

    inline klMatrix4x4 operator-( const klMatrix4x4 &o ) const {
	    return klMatrix4x4( 
            o.at(0,0) - at(0,0),o.at(0,1) - at(0,1),o.at(0,2) - at(0,2),o.at(0,3) - at(0,3),
            o.at(1,0) - at(1,0),o.at(1,1) - at(1,1),o.at(1,2) - at(1,2),o.at(1,3) - at(1,3),
            o.at(2,0) - at(2,0),o.at(2,1) - at(2,1),o.at(2,2) - at(2,2),o.at(2,3) - at(2,3),
            o.at(3,0) - at(3,0),o.at(3,1) - at(3,1),o.at(3,2) - at(3,2),o.at(3,3) - at(3,3));
    }

    inline klMatrix4x4 operator*( const klMatrix4x4 &o ) const {
	    klMatrix4x4 r;
        for ( int i=0; i<4; i++ ) {
            for ( int j=0; j<4; j++ ) {
                float t = 0.0f;
                for ( int k=0; k<4; k++ ) {
                     t += at(i,k)*o.at(k,j);
                }
                r.at(i,j) = t;
            }
        }
        return r;
    }

    inline klVec3 operator*( const klVec3 &o ) const {
        return klVec3(  at(0,0)*o[0] + at(0,1)*o[1] + at(0,2)*o[2] + at(0,3),
                        at(1,0)*o[0] + at(1,1)*o[1] + at(1,2)*o[2] + at(1,3),
                        at(2,0)*o[0] + at(2,1)*o[1] + at(2,2)*o[2] + at(2,3));
    }

    inline klVec4 operator*( const klVec4 &o ) const {
        return klVec4(  at(0,0)*o[0] + at(0,1)*o[1] + at(0,2)*o[2] + at(0,3)*o[3],
                        at(1,0)*o[0] + at(1,1)*o[1] + at(1,2)*o[2] + at(1,3)*o[3],
                        at(2,0)*o[0] + at(2,1)*o[1] + at(2,2)*o[2] + at(2,3)*o[3],
                        at(3,0)*o[0] + at(3,1)*o[1] + at(3,2)*o[2] + at(3,3)*o[3]);
    }

    // Skip the translation
    inline klVec3 transformVector(const klVec3 &o) const {
        return klVec3(  at(0,0)*o[0] + at(0,1)*o[1] + at(0,2)*o[2],
                        at(1,0)*o[0] + at(1,1)*o[1] + at(1,2)*o[2],
                        at(2,0)*o[0] + at(2,1)*o[1] + at(2,2)*o[2]);
    }

    // Skip the translation and use transposed 3x3
    inline klVec3 transposedTransformVector(const klVec3 &o) const {
        return klVec3(  at(0,0)*o[0] + at(1,0)*o[1] + at(2,0)*o[2],
                        at(0,1)*o[0] + at(1,1)*o[1] + at(2,1)*o[2],
                        at(0,2)*o[0] + at(1,2)*o[1] + at(2,2)*o[2]);
    }

    float determinant(void) const {
      // see http://www.euclideanspace.com/maths/algebra/matrix/functions/determinant/fourD/index.htm
      return
      at(3,0) * at(2,1) * at(1,2) * at(0,3)-at(2,0) * at(3,1) * at(1,2) * at(0,3)-at(3,0) * at(1,1) * at(2,2) * at(0,3)+at(1,0) * at(3,1) * at(2,2) * at(0,3)+
      at(2,0) * at(1,1) * at(3,2) * at(0,3)-at(1,0) * at(2,1) * at(3,2) * at(0,3)-at(3,0) * at(2,1) * at(0,2) * at(1,3)+at(2,0) * at(3,1) * at(0,2) * at(1,3)+
      at(3,0) * at(0,1) * at(2,2) * at(1,3)-at(0,0) * at(3,1) * at(2,2) * at(1,3)-at(2,0) * at(0,1) * at(3,2) * at(1,3)+at(0,0) * at(2,1) * at(3,2) * at(1,3)+
      at(3,0) * at(1,1) * at(0,2) * at(2,3)-at(1,0) * at(3,1) * at(0,2) * at(2,3)-at(3,0) * at(0,1) * at(1,2) * at(2,3)+at(0,0) * at(3,1) * at(1,2) * at(2,3)+
      at(1,0) * at(0,1) * at(3,2) * at(2,3)-at(0,0) * at(1,1) * at(3,2) * at(2,3)-at(2,0) * at(1,1) * at(0,2) * at(3,3)+at(1,0) * at(2,1) * at(0,2) * at(3,3)+
      at(2,0) * at(0,1) * at(1,2) * at(3,3)-at(0,0) * at(2,1) * at(1,2) * at(3,3)-at(1,0) * at(0,1) * at(2,2) * at(3,3)+at(0,0) * at(1,1) * at(2,2) * at(3,3);
    }

    // Calculates the inverse (fast) assuming the matrix is a rigid body transform
    // i.e. only rotation and translation.
    klMatrix4x4 inverseRigid(void) const {
        // Just transpose the rotation and calculate the inverse translation
        klVec3 tr = transposedTransformVector(klVec3(at(0,3),at(1,3),at(2,3)));
        return klMatrix4x4(
            at(0,0),at(1,0),at(2,0),-tr[0],
            at(0,1),at(1,1),at(2,1),-tr[1],
            at(0,2),at(1,2),at(2,2),-tr[2],
               0.0f,   0.0f,   0.0f,  1.0f);        
    }

    // Just sets the translation part of the matrix
    // other parts are left AS IS
    void setTranslation(const klVec3 &pos ) {
        at(0,3) = pos[0];
        at(1,3) = pos[1];
        at(2,3) = pos[2];
    }

    // Just sets the rotation part of the matrix
    // other parts are left AS IS
    void setRotation(const klMatrix3x3 &rot ) {
        at(0,0) = rot(0,0); at(0,1) = rot(0,1); at(0,2) = rot(0,2);
        at(1,0) = rot(1,0); at(1,1) = rot(1,1); at(1,2) = rot(1,2);
        at(2,0) = rot(2,0); at(2,1) = rot(2,1); at(2,2) = rot(2,2);
    }
};

inline klMatrix4x4 klIdentityMatrix(void) {
    return (klMatrix4x4(
        1.0f,0.0f,0.0f,0.0f,
        0.0f,1.0f,0.0f,0.0f,
        0.0f,0.0f,1.0f,0.0f,
        0.0f,0.0f,0.0f,1.0f));
}


// Initialize with a translation matrix
inline klMatrix4x4 klTranslationMatrix(const klVec3 &pos) {
    return (klMatrix4x4(
        1.0f,0.0f,0.0f,pos[0],
        0.0f,1.0f,0.0f,pos[1],
        0.0f,0.0f,1.0f,pos[2],
        0.0f,0.0f,0.0f,1.0f));
}

// Initialize with a scaling matrix
inline klMatrix3x3 klScaleMatrix( const klVec3 &sc ) {
    return (klMatrix3x3(
        sc[0],0.0f ,0.0f ,
        0.0f ,sc[1],0.0f ,
        0.0f ,0.0f ,sc[2]));
}

// Initialize with a rotation matrix gives as three euler angles
// (heading,pitch,roll) in radians.
// makes a concatenated matrix Rz(ang[2])*Rx(ang[1])*Ry(ang[0])
inline klMatrix3x3 klEulerAnglesMatrix(const klVec3 &ang) {
    float cosr, sinr;
    float cosh, sinh;
    float cosp, sinp;
    sincos(ang[0], sinh, cosh);
    sincos(ang[1], sinp, cosp);
    sincos(ang[2], sinr, cosr);

    return (klMatrix3x3(
        cosr*cosh - sinr*sinp*sinh, -sinr*cosp, cosr*sinh + sinr*sinp*cosh,
        sinr*cosh + cosr*sinp*sinh,  cosr*cosp, sinr*sinh - cosr*sinp*cosh,
                        -cosp*sinh,       sinp,                  cosp*cosh));
}

// Initialize with a rotation matrix rotating angle RADIANS along axis
inline klMatrix3x3 klAxisAngleMatrix(const klVec3 &axis, float angle) {
    // Convert to quaternion and convert the resulting quaternion to a matrix.        
    float sinA = sin(angle/2.0f);
    float cosA = cos(angle/2.0f);

    float X = axis[0]*sinA;
    float Y = axis[1]*sinA;
    float Z = axis[2]*sinA;
    float W = cosA;

    return (klMatrix3x3(
        1 - 2*Y*Y  - 2*Z*Z,   2*X*Y - 2*Z*W     ,   2*X*Z + 2*Y*W        ,
        2*X*Y + 2*Z*W     ,   1 - 2*X*X  - 2*Z*Z,   2*Y*Z - 2*X*W        ,
        2*X*Z - 2*Y*W     ,   2*Y*Z + 2*X*W     ,   1 - 2*X*X  - 2*Y*Y   ));
}

// Initialize with a rotation mapping the first axis to the second with a
// "minimal" rotation
// Note: INPUT VECTORS NEED TO BE NORMALISED
// See "Real Time Rendering" pg. 52.
inline klMatrix3x3 klAxisToAxisMatrix(const klVec3 &from, const klVec3 &to) {
    klVec3 v;
    v.cross(from,to);
    float e = from.dot(to);
    float h = (1.0f - e) / (v.dot(v));
    
    return (klMatrix3x3(
           e + h*v[0]*v[0], h*v[0]*v[1] - v[2], h*v[0]*v[2] + v[1],
        h*v[0]*v[1] + v[2],    e + h*v[1]*v[1], h*v[1]*v[2] - v[0],
        h*v[0]*v[2] - v[1], h*v[1]*v[2] + v[0],    e + h*v[2]*v[2]));
}

// Initialize with a 3x3 rotation marix
inline klMatrix3x3 klAxisMatrix(const klVec3 &xAxis, const klVec3 &yAxis, const klVec3 &zAxis) {
    return (klMatrix3x3( xAxis[0], xAxis[1], xAxis[2],
                         yAxis[0], yAxis[1], yAxis[2],
                         zAxis[0], zAxis[1], zAxis[2]));
}

#endif //__KL_MATRIX_H