
#ifndef __VECTOR_HPP__
#define __VECTOR_HPP__

// System includes

#include <math.h>

#include <Common\System\Types.hpp>
#include <Common\System\Assert.hpp>

#include <algorithm>

/// Forward declaration of Matrix template
template< class Type, size_t M, size_t N > class Matrix;

template<class Type>
bool AbsGreater(const Type& _a, const Type& _b)
{
	return std::fabs(_a)>std::fabs(_b);
}


template < class Type, size_t Size >
class Vector
{
public:

    typedef size_t               size_type;
    typedef Type                 value_type;
    typedef Vector< Type, Size > ClassType;

    typedef Type*                iterator;
    typedef const Type*          const_iterator;

protected:

    Type m_vector[Size];

public:

    Vector();
	~Vector(){}

	Vector( const Matrix<Type, 1, Size>& );
    Vector( const Vector& );
	Vector( const Type& x )												{std::fill(Begin(), End(), static_cast<Type>(x));}
    Vector( const Type& x, const Type& y )								{STATIC_CHECK(Size==2); m_vector[0]=x; m_vector[1]=y;}
    Vector( const Type& x, const Type& y, const Type& z )				{STATIC_CHECK(Size==3); m_vector[0] = x; m_vector[1] = y; m_vector[2] = z; }
    Vector( const Type& x, const Type& y, const Type& z, const Type& w ){STATIC_CHECK(Size==4); m_vector[0] = x; m_vector[1] = y; m_vector[2] = z; m_vector[3] = w;}
	Vector( const Type& x, const Type& y, const Type& z, const Type& w, const Type& a ){STATIC_CHECK(Size==5); m_vector[0] = x; m_vector[1] = y; m_vector[2] = z; m_vector[3] = w; m_vector[4]=a; }
	Vector( const Type& x, const Type& y, const Type& z, const Type& w, const Type& a, const Type& b ){STATIC_CHECK(Size==6); m_vector[0] = x; m_vector[1] = y; m_vector[2] = z; m_vector[3] = w; m_vector[4]=a; m_vector[5]=b;}

	template<class OtherType>
	Vector(const Vector<OtherType, Size>& _v){for(size_t i=0; i<Size; ++i){m_vector[i]=static_cast<Type>(_v[i]);}}

	//operator float(){STATIC_CHECK(Size==1);return m_vector[0];}

	///< Functions :
	template<size_t OtherSize>
	Vector&				operator=		(const Vector<Type, OtherSize>& _v){STATIC_CHECK(OtherSize<=Size);	memcpy(this, &_v, sizeof(Vector<Type, OtherSize>));	return *this;}

	template<typename Type, size_t Size, typename TypeApparented>
	inline void			operator+=		(const Vector<TypeApparented, Size>& _vec) {     (*this) = (*this) + _vec;   }


	static Vector		VSign			(const Vector& _v);
	static Vector		VAbs			(const Vector& _v);
    Vector&				operator=		( const Vector& );
    inline Type&		operator[]		( const size_t& );
    inline const Type&	operator[]		( const size_t& ) const;

	static const size_t size			()	{ return StaticSize(); }
	static const size_t StaticSize		()	{ return Size; }
	
    inline const Type	Min				() const;
    inline const Type	Max				() const;

    inline const Type&	x				()const{ STATIC_CHECK(Size>0);return m_vector[0]; }
    inline const Type&	y				()const{ STATIC_CHECK(Size>1);return m_vector[1]; }
    inline const Type&	z				()const{ STATIC_CHECK(Size>2);return m_vector[2]; }
	inline const Type&	w				()const{ STATIC_CHECK(Size>3);return m_vector[3]; }

    inline const Type	Norm			() const;
	inline const Type	ZeroNorm		() const;
	inline const Type	InfinityNorm	() const;
	inline const Type	SquaredNorm		() const;

	inline Vector		Normalize		() const;
	inline value_type	Length			() const;
	static Vector		Null			(){ return Vector(); }

    inline ClassType	operator+		( const Type& ) const;
    inline ClassType	operator+		( const ClassType& ) const;
    inline void			operator+=		( const ClassType& );


    inline ClassType	operator-		( const Type& ) const;
    inline ClassType	operator-		( const ClassType& ) const;
    inline void			operator-=		( const ClassType& );

    inline ClassType	operator*		( const Type& ) const ;
    inline Type			operator*		( const ClassType& ) const ;

    inline ClassType	operator/		( const Type& ) const ;
    inline ClassType	operator/		( const ClassType& ) const;

	inline iterator         Begin		()			{ return m_vector; }
	inline iterator         End			()			{ return m_vector+Size; }
	inline const_iterator   Begin		() const	{ return m_vector; }
	inline const_iterator   End			() const	{ return m_vector+Size; }

    inline Matrix<Type, Size, Size> Outer	( const ClassType& ) const;    

    inline Matrix<Type, Size, 1>	ToColon	()const
    {
            Matrix<Type, Size, 1> col;
            for(size_t i=0; i<Size; ++i)
                col(i, 0)=m_vector[i];

            return col;
    }

    inline Matrix<Type, 1, Size> ToLine()const
    {
        Matrix<Type, 1, Size> line;
        for(size_t i=0; i<Size; ++i)
            line(0, i)=m_vector[i];

        return line;
    }



    Vector  CrossProduct( const Vector& v ) const
    {
        STATIC_CHECK(Size==3);

        Vector rVector;
    
        rVector[0] = m_vector[1]*v[2] - m_vector[2]*v[1];
        rVector[1] = m_vector[2]*v[0] - m_vector[0]*v[2];
        rVector[2] = m_vector[0]*v[1] - m_vector[1]*v[0];

        return rVector;
    }

    Type DotProduct( const Vector& v ) const
    {
        Type sum = static_cast<Type>(0);
        for( size_t i = 0; i < Size; ++i )
        {
            sum += m_vector[i]*v[i];
        }
        return sum;
    }

	Vector DirectProduct( const Vector& v ) const
	{
		Vector result;
		for( size_t i = 0; i < v.size(); ++i )
		{
			result[i] = (*this)[i]*v[i];
		}
		return result;
	}

    Vector InverseDirectProduct( const Vector& v  ) const
    {
        Vector result;
        for( size_t i = 0; i < size(); ++i )
        {
            result[i] = (*this)[i]/v[i];
        }
        return result;
    }

};




template< class Type, size_t Size >
bool operator<(const Vector<Type, Size>& _v1, const Vector<Type, Size>& _v2)
{
	return (_v1.Norm()<_v2.Norm());
}

template< class Type, size_t Size >
Vector< Type, Size > operator*( const Type& t, const Vector<Type, Size>& v )
{
    return v.operator*(t);
}

template< class Type, size_t Size >
bool operator==(const Vector<Type, Size>& _v1, const Vector<Type, Size>& _v2)
{
#pragma message("clean vector's operator==");
	bool bSame=true;
	for (size_t i=0; i<Size;++i)
	{
		bSame&=Compare<Type>(_v1[i], _v2[i]);
	}
	return bSame;
}

#include "Vector.inl"

// Vecteurs de base constants

typedef Vector< float32,  4 >			Vector4f;
typedef Vector< float64, 4 >			Vector4d;
typedef Vector< uint8,  4 >				Vector4uc;

typedef Vector< float32,  3 >			Vector3f;
typedef Vector< float64, 3 >			Vector3d;
typedef Vector< uint16, 3 >				Vector3us;
typedef Vector< int32,  3 >				Vector3i;

typedef Vector< float32,  2 >			Vector2f;
typedef Vector< float64, 2 >			Vector2d;
typedef Vector< int32,  2 >				Vector2i;
typedef Vector< uint16, 2 >				Vector2us;
typedef Vector< int16, 2 >				Vector2s;

typedef Vector< float32,  1 >			Vector1f;
typedef Vector< float64, 1 >			Vector1d;
typedef Vector< uint16, 1 >				Vector1us;


const Vector3f xAxis			= Vector3f( 1.0f, 0.0f, 0.0f );;
const Vector3f yAxis			= Vector3f( 0.0f, 1.0f, 0.0f );
const Vector3f zAxis			= Vector3f( 0.0f, 0.0f, 1.0f );

const Vector3f Vect3BaseX		= xAxis;
const Vector3f Vect3BaseY		= yAxis;
const Vector3f Vect3BaseZ		= zAxis;
const Vector3f Vect3BaseNull	= Vector3f::Null();

namespace Color{
	const Vector4uc Black		= Vector4uc( 0, 0, 0, 0 );
	const Vector4uc Red			= Vector4uc( 255, 0, 0, 0 );
	const Vector4uc Green		= Vector4uc( 0, 255, 0, 0 );
	const Vector4uc Blue		= Vector4uc( 0, 0, 255, 0 );
	const Vector4uc Purple		= Vector4uc( 255, 0, 255, 0 );
	const Vector4uc White		= Vector4uc( 255, 255, 255, 0 );
}




#endif


