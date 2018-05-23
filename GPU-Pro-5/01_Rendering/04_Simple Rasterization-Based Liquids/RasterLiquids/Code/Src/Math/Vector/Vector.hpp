
#ifndef __VECTOR_HPP__
#define __VECTOR_HPP__

#include <algorithm>

#include <Math\Math.hpp>
#include <Common\Common.hpp>

#include <Common\StaticAssert.hpp>

///<
template < class T, size_t N >
class Vector
{
public:

    typedef size_t              size_type;
    typedef T					value_type;

    typedef T*					iterator;
    typedef const T*			const_iterator;

	typedef Vector<T, N>		ClassT;
		
	inline iterator         Begin		()			{ return m_d; }
	inline iterator         End			()			{ return m_d+Size(); }
	inline const_iterator   Begin		() const	{ return m_d; }
	inline const_iterator   End			() const	{ return m_d+Size(); }

protected:

    T m_d[N];

public:

	~Vector(){}

	///<
	Vector(){memset( this, 0, sizeof(Vector) );	}
	Vector( const Vector& );
	
	template<class T2, size_t N2>
	Vector(const Vector<T2, N2>& _v)
	{ 
		memset(this,0,sizeof(ClassT)); 
		for (size_t i=0; i<M::Min(N2, N); ++i)
			m_d[i]=M::SCast<T>(_v[i]);
	}

	Vector&					operator=		( const Vector& );
	static const size_t		Size			()	{ return N; }

	///<
	Vector( const T& _x)											{  std::fill(Begin(), End(), _x); }
	Vector( const T&  _x, const T&  _y)								{STATIC_CHECK(N==2); m_d[0]= _x; m_d[1]= _y;}
	Vector( const T&  _x, const T&  _y, const T&  _z)				{STATIC_CHECK(N==3); m_d[0] =  _x; m_d[1] =  _y; m_d[2] =  _z; }
	Vector( const T&  _x, const T&  _y, const T&  _z, const T& _w)	{STATIC_CHECK(N==4); m_d[0] =  _x; m_d[1] =  _y; m_d[2] =  _z; m_d[3] =  _w;}
	Vector( const T&  _x, const T&  _y, const T&  _z, const T& _w, const T& _a)	{STATIC_CHECK(N==5); m_d[0] =  _x; m_d[1] =  _y; m_d[2] =  _z; m_d[3] =  _w; m_d[4] =  _a;}
	Vector( const T&  _x, const T&  _y, const T&  _z, const T& _w, const T& _a, const T& _b)	{STATIC_CHECK(N==6); m_d[0] =  _x; m_d[1] =  _y; m_d[2] =  _z; m_d[3] =  _w; m_d[4] =  _a;m_d[5]= _b; }
	Vector( const T&  _x, const T&  _y, const T&  _z, const T& _w, const T& _a, const T& _b, const T& _c)	{STATIC_CHECK(N==7); m_d[0] =  _x; m_d[1] =  _y; m_d[2] =  _z; m_d[3] =  _w; m_d[4] =  _a; m_d[5]= _b; m_d[6]= _c;}

	Vector( const T& _x, const T& _y, const T& _z, const T& _w, const T& _a, const T& _b, const T& _c, const T& _d )
	{STATIC_CHECK(N==8); m_d[0] = _x; m_d[1] = _y; m_d[2] = _z; m_d[3] = _w; m_d[4]= _a; m_d[5]= _b; m_d[6]= _c; m_d[7]= _d;}

	///<
    inline T&						operator[]		( const size_t& );
    inline const T&					operator[]		( const size_t& ) const;

	inline T&	x					(){ STATIC_CHECK(N>0);return m_d[0]; }
    inline T&	y					(){ STATIC_CHECK(N>1);return m_d[1]; }
    inline T&	z					(){ STATIC_CHECK(N>2);return m_d[2]; }
	inline T&	w					(){ STATIC_CHECK(N>3);return m_d[3]; }

	inline const T&	x				() const{ STATIC_CHECK(N>0); return m_d[0]; }
    inline const T&	y				() const{ STATIC_CHECK(N>1); return m_d[1]; }
    inline const T&	z				() const{ STATIC_CHECK(N>2); return m_d[2]; }
	inline const T&	w				() const{ STATIC_CHECK(N>3); return m_d[3]; }

	
    inline const T					Min				() const;
    inline const T					Max				() const;

	inline const T					ZeroNorm		() const;
	inline const T					InfinityNorm	() const;

	inline Vector					Normalize		() const;

	inline T						Length			() const;
    inline T						SquaredLength   () const;
	inline T						AbsLength		() const;	
	
	static Vector					Null			(){ return Vector(); }

	inline Vector					operator+=		(const T& _s);
    inline Vector					operator+=		(const Vector& );	
    inline Vector					operator-=		(const Vector& );
	inline Vector					operator*=		(const Vector& _v);
	inline Vector					operator*=		(const T& );
	inline Vector					operator/=		(const T& );
	inline Vector					operator/=		(const Vector& );
	
 	inline Vector					operator-		() const;

	inline Vector					operator/		(const T& ) const;
	inline Vector					operator+		(const T& ) const;
	inline Vector					operator*		(const T&) const ;
	inline Vector					operator+		(const Vector& ) const;
	inline Vector					operator*		(const Vector& ) const;
	inline Vector					operator-		(const Vector& ) const;
    inline Vector					operator/		(const Vector& ) const;
	
};
///< End def.


template< class T, size_t N >
bool operator < (const Vector<T, N>& _v1, const Vector<T, N>& _v2)
{
	return (_v1.Norm()<_v2.Norm());
}

template< class T, size_t N >
Vector< T, N > operator * (const T& _t, const Vector<T, N>& _v)
{
    return _v.operator*(_t);
}

template< class T, size_t N >
Vector< T, N > operator * (const Vector<T, N>& _v, const T& _t)
{
	return _v.operator*(_t);
}

template< class T, size_t N >
Vector< T, N > operator - (const T& _t, const Vector<T, N>& _v)
{
	return _v.operator-(_t);
}

#include "Vector.inl"
#include <Common/Incopiable.hpp>

template<class T>
class DVector
{
	T*      m_pD;
	uint32   m_d;

public:

	typedef T value_type;

	DVector(): m_pD(0), m_d(0) {}
	DVector(const uint32 _d) : m_pD(0), m_d(0)
	{ 
		Create(_d); 
	}
	DVector(const DVector& _v) : m_pD(0), m_d(0)
	{
		if(_v.Size()>0)
		{
			Create(_v.Size());
			Copy(_v);
		}
	}

	void Release()
	{ 
		if (m_pD!=NULL)
		{
			delete[] m_pD;
			m_pD=NULL;
			m_d=0;
		}
	}

	~DVector()
	{
		Release();
	}

	void Swap(DVector<T>& _v)
	{
		ASSERT(_v.Size()==Size(), "Not same sizes");
		std::swap(m_pD, _v.m_pD);
	}

	DVector<T>& operator=(const DVector<T>& _v)
	{
		if(Size()==0)
			Create(_v.Size());

		Copy(_v);

		return *this;
	}

	///<
	void Copy(const DVector<T>& _v)
	{
		ASSERT(_v.Size()==Size(), "Not same sizes");
		for(uint32 i=0; i<_v.Size(); ++i)
			m_pD[i] = _v[i];
	}
	
	///<
	void Create(const uint32 _d)
	{
		if (_d>0)
		{
			if(m_pD!=NULL)
				Release();

			m_d=_d;
			m_pD = new T[m_d]; 
			memset(m_pD,0,sizeof(T)*_d);			
		}
	}

	///<
	void		Resize	(const uint32 _size) {ASSERT(_size<m_d && _size>0, "Only manages smaller sizes !"); m_d=_size;}

	T&			First	(){ ASSERT(m_pD!=NULL, "Not initialized!"); return m_pD[0]; }
	T&			At		(const uint32 _i){return this->operator[](_i);}
	const T&	At		(const uint32 _i) const {return this->operator[](_i);}

	uint32		Size	()const{return m_d;}

	const	T*	Begin() const	{return m_pD;}
	T*			Begin()			{return m_pD;}
	T*			End()			{return m_pD+m_d;}

	T&          operator[](const uint32 _i)          {ASSERT(_i < m_d, "Indice over"); return m_pD[_i];}
	const T&    operator[](const uint32 _i) const    {ASSERT(_i < m_d, "Indice over"); return m_pD[_i];}

	///<
	T SquaredLength() const
	{
		T l=0;
		for (uint32 i=0; i<Size(); ++i)
		{
			l+=M::Squared(this->operator[](i));
		}

		return l;
	}
};


template<class V>
float32 GenericVectorSquaredLength(const V& _v)
{
	float32 l=0;
	for (uint32 i=0; i<_v.Size(); ++i)
	{
		l+=_v[i].SquaredLength();
	}
	return l;
}


#endif


