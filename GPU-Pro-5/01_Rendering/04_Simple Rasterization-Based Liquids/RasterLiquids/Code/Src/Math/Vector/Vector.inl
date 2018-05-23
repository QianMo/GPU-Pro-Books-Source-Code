

///<
template <typename T, size_t N>
Vector<T, N>::Vector( const Vector<T, N>& _v ) 
{
	memcpy(Begin(), _v.Begin(), sizeof(Vector));
}


///<
template < typename T, size_t N >
Vector<T, N>& Vector<T, N>::operator=(const Vector<T, N>& _v)
{
    if (this != &_v)
		memcpy(Begin(), _v.Begin(), sizeof(Vector));

    return (*this);
}

///<
template <typename T, size_t N>
T& Vector<T, N>::operator[]( const size_t& i ) 
{
    ASSERT( i >= 0 && i < N, "Wrong index");
    return m_d[i];
}

///<
template <typename T, size_t N>
const T& Vector<T, N>::operator[]( const size_t& i ) const 
{
    ASSERT( i>=0 && i<N, "Wrong index");
    return m_d[i];
}

///<
template <typename T, size_t N>
inline Vector<T,N> Vector<T, N>::operator+=(const T& _s)
{ 
	for (size_t i=0; i<N; ++i)
		m_d[i] = m_d[i] + _s;

	return *this;
}

///<
template <typename T, size_t N>
inline Vector<T, N> Vector<T, N>::operator+(const T& _s) const
{       
    Vector<T, N> rV = (*this);
	return rV+=_s;
}

///<
template <typename T, size_t N>
inline Vector<T, N> Vector<T, N>::operator+=( const Vector<T, N>& _v)
{     
	for (size_t i=0; i<N; ++i)
		m_d[i]+=_v[i];

	return *this;
}

///<
template <typename T, size_t N>
inline Vector<T, N> Vector<T, N>::operator+(const Vector<T, N>& _v) const
{      
    Vector<T, N> rV=(*this);
	return rV+=_v;
}

///<
template <typename T, size_t N>
inline Vector<T, N> Vector<T, N>::operator-() const
{     
	Vector<T, N> Rev=(*this);
	for (size_t i=0; i<N; ++i)
		Rev[i] = -m_d[i];
	
	return Rev;
}

///<
template <typename T, size_t N>
inline Vector<T, N> Vector<T, N>::operator-=(const Vector<T, N>& _v)
{     
	return (*this)+= -_v;
}

///<
template <typename T, size_t N>
inline Vector<T, N> Vector<T, N>::operator-(const Vector<T, N>& _v) const
{     
    Vector<T, N> rV=(*this);
	return rV-=_v;
}

///<
template < typename T, size_t N >
inline Vector<T, N> Vector<T, N>::operator*=(const Vector<T, N>& _v)
{     
	for (size_t i=0; i<N; ++i)
		m_d[i]=m_d[i]*_v[i];
	
	return *this;
}

///<
template < typename T, size_t N >
inline Vector<T, N> Vector<T, N>::operator*(const Vector<T, N>& _v)const
{     
	Vector<T, N> rV=*this;
	return rV*=_v;
}

///<
template <typename T, size_t N>
inline Vector<T, N> Vector<T, N>::operator*=(const T& _s) 
{
	for (size_t i=0; i<N; ++i)
		m_d[i] = m_d[i]*_s;	

	return *this;
}

///<
template <typename T, size_t N>
inline Vector<T, N> Vector<T, N>::operator*(const T& _s) const
{       
    Vector<T, N> rV=*this;
	return rV*=_s;
}

///<
template <typename T, size_t N>
inline Vector<T, N> Vector<T, N>::operator/(const T& _s) const
{       
	Vector<T, N> rVector;
	for (size_t i=0; i<N; ++i)
		rVector[i] = M::Divide(m_d[i],_s);

	return rVector;
}

///<
template <typename T, size_t N>
inline Vector<T, N> Vector<T, N>::operator/(const Vector<T, N>& _v) const
{       
    Vector<T, N> rV=*this;
	return rV/=_v;
}

///<
template <typename T, size_t N>
inline Vector<T, N> Vector<T, N>::operator/=(const Vector<T, N>& _v)
{       

	for (size_t i=0; i<N; ++i)
		m_d[i] = M::Divide(m_d[i],_v[i]);

	return *this;
}

///< Functions:
template< class T >
bool Greater(const T& _a, const T& _b)
{
	return _a<_b;
}

template <typename T, size_t N>
const T Vector<T, N>::Max() const
{
	return *std::max_element(Begin(), End());
}

///<
template <typename T, size_t N>
const T Vector<T, N>::Min() const
{
	return *std::min_element(Begin(), End());
}


template <typename T, size_t N>
const T Vector<T, N>::ZeroNorm() const
{
	T Sum=0;
	for (size_t i=0;i<N;++i)
		Sum+=M::Abs(m_d[i]);

	return Sum;
}

template <typename T, size_t N>
const T Vector<T, N>::InfinityNorm() const
{
	return *std::max_element(Begin(), End(), AbsGreater<T>);
}


template <typename T, size_t N>
T Vector<T, N>::AbsLength() const
{
	T rN = 0.0f;
	for( size_t i = 0; i < N; ++i )
	{
		rN+=M::Abs(m_d[i]);
	}

	return rN;
}

///<
template <typename T, size_t N>
T Vector<T, N>::SquaredLength() const
{
	T rN = 0.0f;
	for( size_t i = 0; i < N; ++i )
	{
		rN+=M::Squared(m_d[i]);
	}

	return rN;
}

///<
template <typename T, size_t N>
inline T Vector<T, N>::Length() const
{
    return M::Sqrt(SquaredLength());
}

///<
template <typename T, size_t N>
inline Vector<T, N> Vector<T, N>::Normalize() const
{
	return (*this)/Length();
}


/*
///<
template <typename T, size_t N>
Vector<T, N> Vector<T, N>::VSign(const Vector& _v) 
{
	Vector rV;
	for(size_t i=0; i<N; ++i)
		rV[i] = M::Sign(_v[i]);
	
	return rV;
}

///<
template <typename T, size_t N>
Vector<T, N> Vector<T, N>::VAbs(const Vector& _v) 
{
	Vector rV;
	for (size_t i=0; i<N; ++i)
		rV[i] = M::Abs(_v[i]);
	
	return rV;
}


///< Global


template<typename T, typename TApparented, size_t N>
inline Vector<T, N> operator+(const Vector<T, N>& _vecLeft, const Vector<TApparented, N>& _v) 
{       
	Vector<T, N> rV;
	for (size_t i=0; i<N; ++i)
		rV[i] = _vecLeft[i] + static_cast<T>(_v[i]);

	return rV;
}

*/