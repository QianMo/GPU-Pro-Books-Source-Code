#ifndef __QUATERNION_HPP__
#define __QUATERNION_HPP__

#include <Math/Vector/Vector.hpp>
#include <Math/Matrix/Matrix.hpp>

template< class T >
class Quaternion
{
public:

    typedef Matrix< T, 4, 1 > QuaternionType;
	typedef Matrix< T, 3, 1> VecType;
private:

    QuaternionType m_q;

public:

    Quaternion(){ *this = Identity(); }
    Quaternion( const QuaternionType& _q ) : m_q(_q){} 
	
	Quaternion(const VecType& _v) : m_q(_v[0], _v[1], _v[2], static_cast<T>(0) ){ }
    Quaternion(const T& _x, const T& _y, const T& _z, const T& _w) : m_q(_x, _y, _z, _w){}   

    inline const T& x() const { return m_q[0]; }
    inline const T& y() const { return m_q[1]; }
    inline const T& z() const { return m_q[2]; }
    inline const T& w() const { return m_q[3]; }

    inline  T& x()  { return m_q[0]; }
    inline  T& y()  { return m_q[1]; }
    inline  T& z()  { return m_q[2]; }
    inline  T& w()  { return m_q[3]; }

	T& operator[]							(const uint32 _i)	{ return m_q[_i]; }

    inline const size_t		Size			() const			{ return m_vec.size(); }

	const T					SquaredLength	() const			{return m_q.SquaredLength(); }	
	const T					Length			() const			{return m_q.Length(); }

    static Quaternion		Identity		()					{ return Quaternion( 0, 0, 0, 1 ); }
	
	Quaternion Half() const {  Quaternion q; q.m_q=m_q; q.m_q[3]=0.5f*q.m_q[3]; return q.Normalize(); }
	///<
	VecType					VecPart			() const			{ return VecType(m_q[0], m_q[1], m_q[2]); }

	///<
	static const Quaternion Slerp(const Quaternion& _q0, const Quaternion& _q1, float32 _t)  
    {	
		T dot = acos(DotProduct(_q0.m_q, _q1.m_q));

		T s0 = sin((1-_t)*dot)/ sin(dot);
		T s1 = sin(_t*dot)/sin(dot);

        return (s0*_q0+ s1*_q1).Normalize();
    }

	static float32 ASin(const float32 _s)
	{
		if(M::Abs(_s)< 0.001f)
			return 0.0f;
		else if(M::Abs(_s)> 0.999f)
			return -M::Pi*0.5f;
		else
			return asinf(_s);
	}

	inline static const Quaternion GenRotation(const T _a, const VecType& _axis) {
        T sinangle = sin(_a*0.5f);
		return Quaternion(_axis[0]*sinangle, _axis[1]*sinangle, _axis[2]*sinangle, cos(_a*0.5f)).Normalize();
    }


    inline const VecType Rotate(const VecType& _x) const {
        Quaternion result = (*this)*Quaternion(_x)*Inverse();
        return VecType(result.x(), result.y(), result.z());
    }

    inline const Quaternion Inverse() const {
        return Quaternion(-x(), -y(), -z(), w());
    }

	inline const Quaternion operator * (const T& _s) const {
        return Quaternion(_s*m_q);
    }

    inline const Quaternion& operator += (const Quaternion& _q) {
        *this = Quaternion(m_q + _q.m_q);
        return *this;
    }
    inline const Quaternion& operator -= (const Quaternion& _q) {
        *this = Quaternion(m_q - _q.m_q);
        return *this;
    }

    inline const Quaternion operator + (const Quaternion& _q) const {
        Quaternion q = *this;
		return q+=_q;
    }

    inline const Quaternion operator - (const Quaternion& _q) const {
         Quaternion q = *this;
		return q-=_q;    }

    inline const Quaternion operator* ( const Quaternion& rq) const {
		return Quaternion(	  w() * rq.x() + x() * rq.w() + y() * rq.z() - z() * rq.y(),
	            w() * rq.y() + y() * rq.w() + z() * rq.x() - x() * rq.z(),
	            w() * rq.z() + z() * rq.w() + x() * rq.y() - y() * rq.x(),
	            w() * rq.w() - x() * rq.x() - y() * rq.y() - z() * rq.z()
				);
                
    }

    inline const Quaternion Normalize() const
    {
        return Quaternion( m_q.Normalize() );
    }

    inline const Matrix< T, 4, 4 > ToMatrix() const {
        Matrix< T, 4, 4 >  result=Matrix<T,4,4>::Identity();
        result(0,0)= 1.0f - 2.0f *( y()*y() + z()*z() );
        result(0,1)= 2.0f * ( x()*y() - w()*z() );
        result(0,2)= 2.0f * ( x()*z()+w()*y() );
        result(1,0)= 2.0f * ( x()*y() + w()*z() );
        result(1,1)= 1.0f - 2.0f * ( x()*x() + z()*z() );
        result(1,2)= 2.0f * ( y()*z() - w()*x() );
        result(2,0)= 2.0f * ( x()*z() - w()*y() );
        result(2,1)= 2.0f * ( y()*z() + w()*x() );
        result(2,2)= 1.0f - 2.0f * ( x()*x() + y()*y() );
        return result;
    }


};

template<class T>
const Quaternion<T> operator * (const T& _s, const Quaternion<T>& _q) {
        return _q*_s;
    }

typedef Quaternion<float32> Quaternionf;


#endif