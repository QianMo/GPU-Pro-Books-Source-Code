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

#ifndef KLVECTOR_H
#define KLVECTOR_H

#undef min
#undef max

/**
Copyright (c) 2003-2008 Charles Hollemeersch

-Not for commercial use without written permission
-This code should not be redistributed or made public
-This code is distributed without any warranty

*/

#include <math.h>

template <class Type> class klVector2d
{
public:
	Type data[2];

	klVector2d(void) {};

	inline klVector2d(const klVector2d &a) {
		data[0] = a[0];
		data[1] = a[1];
	};

	inline klVector2d(Type a,Type b) {
		data[0] = a;
		data[1] = b;
	};

	inline klVector2d(const Type *a) {
		data[0] = a[0];
		data[1] = a[1];
	};

	inline void zero(void) {
		data[0] = data[1];
	};

	inline klVector2d operator + (const klVector2d& A) const {
		return klVector2d(data[0]+A[0], data[1]+A[1]);
	};

	inline klVector2d operator - (const klVector2d& A) const {
		return klVector2d(data[0]-A[0], data[1]-A[1]);
	};

	inline Type operator * (const klVector2d& A) const { 
		return data[0]*A[0]+data[1]*A[1];
	};

	inline klVector2d operator * (const Type sc) const { 
		return klVector2d(data[0]*sc, data[1]*sc);
	};

	inline klVector2d operator + (const Type sc) const { 
		return klVector2d(data[0]+sc, data[1]+sc);
	};


	inline klVector2d operator / (const Type s) const { 
		Type r = 1.0f / s;
		return klVector2d(data[0]*r, data[1]*r);
	};

	inline void operator += (const klVector2d A) {
		data[0]+=A[0]; data[1]+=A[1];
	};

	inline void operator -= (const klVector2d A) {
		data[0]-=A[0]; data[1]-=A[1];
	};

	inline void operator *= (const Type s) {
		data[0]*=s; data[1]*=s;
	};

	/*
	void operator *= (const Type s) {
	data[0]*=s; data[1]*=s; data[2]*=s;
	}*/

	inline klVector2d operator - (void) const {
		klVector2d Negated(-data[0], -data[1]);
		return(Negated);
	};

	inline bool operator==(const klVector2d<Type> &a) const {
		if ( a[0] == data[0] && a[1] == data[1]) return true;
		return false;
	};

	inline bool operator!=(const klVector2d<Type> &a) const {
		if ( a[0] != data[0] || a[1] != data[1]) return true;
		return false;
	};

	inline klVector2d& operator = (const klVector2d& A) {
		data[0]=A[0]; data[1]=A[1];
		return(*this);
	};

	inline Type operator [] (const int i) const {
		return data[i];
	};

	inline Type & operator [] (const int i) {
		return data[i];
	};

	void lerp(const klVector2d<Type>& from,const klVector2d<Type>& to,float slerp) {
		*this = to-from;
		*this = *this * slerp;
		*this+=from;
	};

	inline Type length(void) const {
		return Type(sqrt( data[0] * data[0] + data[1] * data[1]));
	};

	Type lengthSqr(void) const {
		return Type(data[0] * data[0] + data[1] * data[1]);
	};

	Type distance(const klVector2d<Type> &a) const {
		klVector2d<Type> d(a[0]-data[0],a[1]-data[1]);
		return d.length();
	}

	Type distanceSqr(const klVector2d<Type> &a) const {
		float dx = a[0] - data[0];
		float dy = a[1] - data[1];
		return dx*dx + dy*dy;
	};

	Type normalize(void) {
		Type l = length();
		if ( l != 0 )
		{
			data[0]/=l;
			data[1]/=l;
		}
		else
		{
			data[0] = data[1] = 0.0f;
		}
		return l;
	};

	inline Type getX(void) const { return data[0]; };
	inline Type getY(void) const { return data[1]; };

	inline void get(Type *dest) const { dest[0] = data[0]; dest[1] = data[1];};

	inline void setX(Type t) { data[0] = t; };
	inline void setY(Type t) { data[1] = t; };

	void set(Type a,Type b) {
		data[0] = a;
		data[1] = b;
	};

	inline Type *toPtr(void) {
		return &data[0];
	};

	inline const Type *toCPtr(void) const {
		return &data[0];
	};

	inline void min(const klVector2d<Type> &a,const klVector2d<Type> &b) {
		data[0] = (a.data[0] < b.data[0]) ? a.data[0] : b.data[0];
		data[1] = (a.data[1] < b.data[1]) ? a.data[1] : b.data[1];
	};

	inline void max(const klVector2d<Type> &a,const klVector2d<Type> &b) {
		data[0] = (a.data[0] > b.data[0]) ? a.data[0] : b.data[0];
		data[1] = (a.data[1] > b.data[1]) ? a.data[1] : b.data[1];
	};
};

template <class Type> class klVector3d
{
public:
	Type data[3];

	klVector3d(void) {};

	inline klVector3d(const klVector3d &a) {
		data[0] = a[0];
		data[1] = a[1];
		data[2] = a[2];
	};

	inline klVector3d(Type a,Type b,Type c) {
		data[0] = a;
		data[1] = b;
		data[2] = c;
	};

	inline klVector3d(const Type *a) {
		data[0] = a[0];
		data[1] = a[1];
		data[2] = a[2];
	};

	inline void zero(void) {
		data[0] = data[1] = data[2] = 0;
	};

	inline klVector3d operator + (const klVector3d& A) const {
		return klVector3d(data[0]+A[0], data[1]+A[1], data[2]+A[2]);
	};

	inline klVector3d operator - (const klVector3d& A) const {
		return klVector3d(data[0]-A[0], data[1]-A[1], data[2]-A[2]);
	};

	inline Type operator * (const klVector3d& A) const { 
		return data[0]*A[0]+data[1]*A[1]+data[2]*A[2];
	};

	inline Type dot(const klVector3d& A) const { 
		return data[0]*A[0]+data[1]*A[1]+data[2]*A[2];
	};

	inline klVector3d operator * (const Type sc) const { 
		return klVector3d(data[0]*sc, data[1]*sc, data[2]*sc);
	};

	inline klVector3d operator + (const Type sc) const { 
		return klVector3d(data[0]+sc, data[1]+sc, data[2]+sc);
	};

	inline klVector3d operator / (const Type s) const { 
		Type r = 1.0f / s;
		return klVector3d(data[0]*r, data[1]*r, data[2]*r);
	};

	inline void operator += (const klVector3d A) {
		data[0]+=A[0]; data[1]+=A[1]; data[2]+=A[2];
	};
	
	inline void operator -= (const klVector3d A) {
		data[0]-=A[0]; data[1]-=A[1]; data[2]-=A[2];
	};

	inline void operator *= (const Type s) {
		data[0]*=s; data[1]*=s; data[2]*=s;
	};
	/*
	void operator *= (const Type s) {
		data[0]*=s; data[1]*=s; data[2]*=s;
	}*/

	inline klVector3d operator - (void) const {
		klVector3d Negated(-data[0], -data[1], -data[2]);
		return(Negated);
	};

	inline bool operator==(const klVector3d<Type> &a) const {
		if ( a[0] == data[0] && a[1] == data[1] && a[2] == data[2] ) return true;
		return false;
	};

	inline bool operator!=(const klVector3d<Type> &a) const {
		if ( a[0] != data[0] || a[1] != data[1] || a[2] != data[2] ) return true;
		return false;
	};

	inline klVector3d& operator = (const klVector3d& A) {
		data[0]=A[0]; data[1]=A[1]; data[2]=A[2];
		return(*this);
	};

	inline Type operator [] (const int i) const {
		return data[i];
	};
	
	inline Type & operator [] (const int i) {
		return data[i];
	};
	
	void lerp(const klVector3d<Type>& from,const klVector3d<Type>& to,float slerp) {
		*this = to-from;
		*this = *this * slerp;
		*this+=from;
	};

	inline Type length(void) const {
		return Type(sqrt( data[0] * data[0] + data[1] * data[1] + data[2] * data[2] ));
	};

	Type lengthSqr(void) const {
		return Type(data[0] * data[0] + data[1] * data[1] + data[2] * data[2]);
	};

	Type distance(const klVector3d<Type> &a) const {
		klVector3d<Type> d(a[0]-data[0],a[1]-data[1],a[2]-data[2]);
		return d.length();
	}

	Type distanceSqr(const klVector3d<Type> &a) const {
		float dx = a[0] - data[0];
		float dy = a[1] - data[1];
		float dz = a[2] - data[2];
		return dx*dx + dy*dy + dz*dz;
	};

	Type normalize(void) {
		Type l = length();
		if ( l != 0 )
		{
			data[0]/=l;
			data[1]/=l;
			data[2]/=l;
		}
		else
		{
			data[0] = data[1] = data[2] = 0;
		}
		return l;
	};

	 inline klVector3d<Type> normalize(void) const {
		Type l = length();
		if ( l != 0 )
		{
            l = 1.0f / l;
            return klVector3d<Type>(data[0]*l,data[1]*l,data[2]*l);
        } else {
            return klVector3d<Type>(0.0,0.0,0.0);
		}
	};

	inline void cross(const klVector3d<Type> &a,const klVector3d<Type> &b) {
		data[0] = a[1]*b[2] - a[2]*b[1];
		data[1] = a[2]*b[0] - a[0]*b[2];
		data[2] = a[0]*b[1] - a[1]*b[0];
	};

	inline klVector3d<Type> cross(const klVector3d<Type> &other) const {
		return (klVector3d<Type>(data[1]*other[2] - data[2]*other[1],
		                         data[2]*other[0] - data[0]*other[2],
                                 data[0]*other[1] - data[1]*other[0]));
	};

	inline Type getX(void) const { return data[0]; };
	inline Type getY(void) const { return data[1]; };
	inline Type getZ(void) const { return data[2]; };

	inline void get(Type *dest) const { dest[0] = data[0]; dest[1] = data[1]; dest[2] = data[2];};

	inline void setX(Type t) { data[0] = t; };
	inline void setY(Type t) { data[1] = t; };
	inline void setZ(Type t) { data[2] = t; };

	void set(Type a,Type b,Type c) {
		data[0] = a;
		data[1] = b;
		data[2] = c;
	};

	inline Type *toPtr(void) {
		return &data[0];
	};

	inline const Type *toCPtr(void) const {
		return &data[0];
	};

	inline void min(const klVector3d<Type> &a,const klVector3d<Type> &b) {
		data[0] = (a.data[0] < b.data[0]) ? a.data[0] : b.data[0];
		data[1] = (a.data[1] < b.data[1]) ? a.data[1] : b.data[1];
		data[2] = (a.data[2] < b.data[2]) ? a.data[2] : b.data[2];
	};

	inline void max(const klVector3d<Type> &a,const klVector3d<Type> &b) {
		data[0] = (a.data[0] > b.data[0]) ? a.data[0] : b.data[0];
		data[1] = (a.data[1] > b.data[1]) ? a.data[1] : b.data[1];
		data[2] = (a.data[2] > b.data[2]) ? a.data[2] : b.data[2];
	};

	inline void swapHand() {
		klVector3d<Type> temp = *this;
		data[0] = -temp[1];
		data[1] = temp[2];
		data[2] = temp[0];
	}

    // Simple fast gamma correction using a power funciton
    // (use 1.0/power to do inverse correction)
    inline klVector3d<Type> gammaCorrect(float gamma) const {
        return klVector3d(pow(data[0],gamma),pow(data[1],gamma),pow(data[2],gamma));
    }

    // The exact sRGB correction function
    inline klVector3d<Type> srgbToLinear(void) const {
        return klVector3d(
            ( data[0] < 0.04045f ) ? data[0] / 12.92f : pow((data[0] + 0.055f)/1.055f,2.4f),
            ( data[1] < 0.04045f ) ? data[1] / 12.92f : pow((data[1] + 0.055f)/1.055f,2.4f),
            ( data[2] < 0.04045f ) ? data[2] / 12.92f : pow((data[2] + 0.055f)/1.055f,2.4f));
    }            

    // The exact sRGB correction function
    inline klVector3d<Type> linearToSrgb(void) const {
        return klVector3d(
            ( data[0] < 0.0031308f ) ? 12.92f * data[0] : 1.055f*pow(data[0],1.0f/2.4f)-0.055f,
            ( data[1] < 0.0031308f ) ? 12.92f * data[1] : 1.055f*pow(data[1],1.0f/2.4f)-0.055f,
            ( data[2] < 0.0031308f ) ? 12.92f * data[2] : 1.055f*pow(data[2],1.0f/2.4f)-0.055f);
    }

};

template<typename T>
klVector3d<T> operator *(T a, const klVector3d<T> &b) {
	return b * a;
}

template <class Type> class klVector4d
{
public:
	Type data[4];

	klVector4d(void) {};

	inline klVector4d(const klVector4d &a) {
		data[0] = a[0];
		data[1] = a[1];
		data[2] = a[2];
		data[3] = a[3];
	};

	inline klVector4d(Type a,Type b,Type c, Type d) {
		data[0] = a;
		data[1] = b;
		data[2] = c;
		data[3] = d;
	};

	inline klVector4d(const Type *a) {
		data[0] = a[0];
		data[1] = a[1];
		data[2] = a[2];
		data[3] = a[3];
	};

	inline void zero(void) {
		data[0] = data[1] = data[2] = data[3] = 0;
	};

	inline klVector4d operator + (const klVector4d& A) const {
		return klVector4d(data[0]+A[0], data[1]+A[1], data[2]+A[2], data[3]+A[3]);
	};

	inline klVector4d operator - (const klVector4d& A) const {
		return klVector4d(data[0]-A[0], data[1]-A[1], data[2]-A[2], data[3]-A[3]);
	};

	inline Type dot (const klVector4d& A) const { 
		return data[0]*A[0]+data[1]*A[1]+data[2]*A[2]+data[3]*A[3];
	};

	inline klVector4d operator * (const klVector4d& A) const { 
		return klVector4d(data[0]*A[0],data[1]*A[1],data[2]*A[2],data[3]*A[3]);
	};

	inline klVector4d operator * (const Type sc) const { 
		return klVector4d(data[0]*sc, data[1]*sc, data[2]*sc, data[3]*sc);
	};

	inline klVector4d operator + (const Type sc) const { 
		return klVector4d(data[0]+sc, data[1]+sc, data[2]+sc, data[3]+sc);
	};

	inline klVector4d operator / (const Type s) const { 
		Type r = 1.0f / s;
		return klVector4d(data[0]*r, data[1]*r, data[2]*r, data[3]*r);
	};

	inline void operator += (const klVector4d A) {
		data[0]+=A[0]; data[1]+=A[1]; data[2]+=A[2]; data[3]+=A[3];
	};

	inline void operator -= (const klVector4d A) {
		data[0]-=A[0]; data[1]-=A[1]; data[2]-=A[2]; data[3]-=A[3];
	};

	inline void operator *= (const Type s) {
		data[0]*=s; data[1]*=s; data[2]*=s; data[3]*=s;
	};
	/*
	void operator *= (const Type s) {
	data[0]*=s; data[1]*=s; data[2]*=s;
	}*/

	inline klVector4d operator - (void) const {
		klVector4d Negated(-data[0], -data[1], -data[2], -data[3]);
		return(Negated);
	};

	inline bool operator==(const klVector4d<Type> &a) const {
		if ( a[0] == data[0] && a[1] == data[1] && a[2] == data[2] && a[3] == data[3]) return true;
		return false;
	};

	inline bool operator!=(const klVector4d<Type> &a) const {
		if ( a[0] != data[0] || a[1] != data[1] || a[2] != data[2] && a[3] == data[3]) return true;
		return false;
	};

	inline klVector4d& operator = (const klVector4d& A) {
		data[0]=A[0]; data[1]=A[1]; data[2]=A[2]; data[3]=A[3];
		return(*this);
	};

	inline Type operator [] (const int i) const {
		return data[i];
	};

	inline Type & operator [] (const int i) {
		return data[i];
	};

	void lerp(const klVector4d<Type>& from,const klVector4d<Type>& to,float slerp) {
		*this = to-from;
		*this = *this * slerp;
		*this+=from;
	};

	inline Type length(void) const {
		return Type(sqrt( data[0] * data[0] + data[1] * data[1] + data[2] * data[2] + data[3] * data[3] ));
	};

	Type lengthSqr(void) const {
		return Type(data[0] * data[0] + data[1] * data[1] + data[2] * data[2] + data[3] * data[3]);
	};

	Type distance(const klVector4d<Type> &a) const {
		klVector4d<Type> d(a[0]-data[0],a[1]-data[1],a[2]-data[2],a[3]-data[3]);
		return d.length();
	}

	Type distanceSqr(const klVector4d<Type> &a) const {
		float dx = a[0] - data[0];
		float dy = a[1] - data[1];
		float dz = a[2] - data[2];
		float dw = a[3] - data[3];
		return dx*dx + dy*dy + dz*dz + dw*dw;
	};

	Type normalize(void) {
		Type l = length();
		if ( l != 0 )
		{
			data[0]/=l;
			data[1]/=l;
			data[2]/=l;
			data[3]/=l;
		}
		else
		{
			data[0] = data[1] = data[2] = data[3] = 0;
		}
		return l;
	};

	inline Type getX(void) const { return data[0]; };
	inline Type getY(void) const { return data[1]; };
	inline Type getZ(void) const { return data[2]; };
	inline Type getW(void) const { return data[3]; };

	inline void get(Type *dest) const { dest[0] = data[0]; dest[1] = data[1]; dest[2] = data[2]; dest[3] = data[3];};

	inline void setX(Type t) { data[0] = t; };
	inline void setY(Type t) { data[1] = t; };
	inline void setZ(Type t) { data[2] = t; };
	inline void setW(Type t) { data[3] = t; };

	void set(Type a,Type b,Type c, Type d) {
		data[0] = a;
		data[1] = b;
		data[2] = c;
		data[3] = d;
	};

	inline Type *toPtr(void) {
		return &data[0];
	}

	inline const Type *toCPtr(void) const {
		return &data[0];
	}

	inline const klVector3d<Type> &toVec3(void) const {
        return *((klVector3d<Type> *)this);
	}

	inline const klVector2d<Type> &toVec2(void) const {
        return *((klVector2d<Type> *)this);
	}

	inline klVector3d<Type> &toVec3(void) {
        return *((klVector3d<Type> *)this);
	}

	inline klVector2d<Type> &toVec2(void) {
        return *((klVector2d<Type> *)this);
	}

	inline void min(const klVector4d<Type> &a, const klVector4d<Type> &b) {
		data[0] = (a.data[0] < b.data[0]) ? a.data[0] : b.data[0];
		data[1] = (a.data[1] < b.data[1]) ? a.data[1] : b.data[1];
		data[2] = (a.data[2] < b.data[2]) ? a.data[2] : b.data[2];
		data[3] = (a.data[3] < b.data[3]) ? a.data[3] : b.data[3];
	}

	inline void max(const klVector4d<Type> &a, const klVector4d<Type> &b) {
		data[0] = (a.data[0] > b.data[0]) ? a.data[0] : b.data[0];
		data[1] = (a.data[1] > b.data[1]) ? a.data[1] : b.data[1];
		data[2] = (a.data[2] > b.data[2]) ? a.data[2] : b.data[2];
		data[3] = (a.data[3] > b.data[3]) ? a.data[3] : b.data[3];
	}
};

typedef klVector2d<float> klVec2;
typedef klVector2d<double> klVec2d;
typedef klVector2d<int> klVec2i;

typedef klVector3d<float> klVec3;
typedef klVector3d<double> klVec3d;
typedef klVector3d<int> klVec3i;

typedef klVector4d<float> klVec4;
typedef klVector4d<double> klVec4d;
typedef klVector4d<int> klVec4i;

#endif //KLVECTOR_H
