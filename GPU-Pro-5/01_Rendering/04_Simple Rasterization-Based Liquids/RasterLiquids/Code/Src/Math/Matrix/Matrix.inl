
///<
template < class T, uint32 M, uint32 N >
Matrix<T, M, N>::Matrix()
{
    memset(Begin(),0, M*N*sizeof(T));
}

///<
template < class T, uint32 M, uint32 N >
Matrix<T, M, N>::Matrix(const T& _t) 
{
    std::fill(Begin(), End(), _t);
}

///<
template < class T, uint32 M, uint32 N>
Matrix<T, M, N>::Matrix(const Matrix<T, M, N>& _m) 
{
	memcpy(Begin(), _m.Begin(), sizeof(Matrix));
}

///<
template <class T, uint32 M, uint32 N>
Matrix<T, M, N>& Matrix<T, M, N>::operator=(const Matrix<T, M, N>& _m)
{
	memcpy(Begin(), _m.Begin(), sizeof(Matrix));
    return (*this);
}

///<
template< class T, uint32 M, uint32 N >
inline T& Matrix<T, M, N>::operator[](const uint32 _i) 
{
    ASSERT(_i>=0 && _i<M*N, "Out of bounds m_d");
    return m_d[_i];
}

///<  
template< class T, uint32 M, uint32 N >
inline const T& Matrix<T, M, N>::operator[](const uint32 _i) const
{
    ASSERT(_i>=0 && _i<M*N, "Out of bounds m_d");
    return m_d[_i];
}

///<
template< class T, uint32 M, uint32 N >
inline Matrix<T, N, M> Matrix<T, M, N>::Transpose() const
{
    Matrix<T, N, M> t;

    for (uint32 i = 0; i < N; ++i)
        for (uint32 j = 0; j < M; ++j)
            t(i, j) = this->operator()(j, i);
          
    return t;
}

template < typename T, uint32 M, uint32 N >
inline Matrix<T, M, N> Matrix<T, M, N>::operator-() const
{
	Matrix<T, M, N> r;
	const Matrix<T, M, N>& m=*this;
    for (uint32 i=0; i<M; ++i)
        for (uint32 j = 0; j<N; ++j)
            r(i, j) = -m(i, j);

	return r;
}


template < typename T, uint32 M, uint32 N >
inline  Matrix<T, M, N> Matrix<T, M, N>::operator/(const Matrix<T, M, N>& _m) const
{
	Matrix<T, M, N> r;
	const Matrix<T, M, N>& m=*this;
    for (uint32 i=0; i<M; ++i)
        for (uint32 j = 0; j<N; ++j)
            r(i, j) = m(i, j)/ _m(i, j);    

	return r;
}


template < typename T, uint32 M, uint32 N >
inline  Matrix<T, M, N> Matrix<T, M, N>::operator+=(const Matrix<T, M, N>& _m)
{
	Matrix<T, M, N>& m=*this;
    for (uint32 i=0; i<M; ++i)
        for (uint32 j = 0; j<N; ++j)
            m(i, j) = m(i, j) + _m(i, j);    


	return m;
}

template < typename T, uint32 M, uint32 N >
inline  Matrix<T, M, N> Matrix<T, M, N>::operator-=(const Matrix<T, M, N>& _m)
{
	Matrix<T, M, N>& m=*this;
    for (uint32 i=0; i<M; ++i)
        for (uint32 j = 0; j<N; ++j)
            m(i, j) = m(i, j) - _m(i, j);    

	return m;
}

///<
template < typename T, uint32 M, uint32 N >
inline Matrix<T, M, N> Matrix<T, M, N>::operator+(const Matrix<T, M, N>& _m) const
{
    Matrix<T, M, N> r;
    for (uint32 i=0; i<M; ++i)
        for (uint32 j = 0; j<N; ++j)
            r(i, j) = (*this)(i, j) + _m(i, j);    

    return r;
}

///<
template < typename T, uint32 M, uint32 N >
inline Matrix< T, M, N > Matrix<T, M, N>::operator-(const Matrix<T, M, N>& _m) const
{
    Matrix< T, M, N > difference;
    for (uint32 i=0; i<M; ++i)
        for (uint32 j=0; j<N; ++j)
            difference(i, j) = (*this)(i, j) - _m(i, j);    

    return difference;
}

///<
template < typename T, uint32 M, uint32 N >
inline Matrix<T, M, N> Matrix<T, M, N>::operator / (const T& _s) const
{
    Matrix<T, M, N> r;
    for (uint32 i=0; i<M; ++i)
        for (uint32 j=0; j<N; ++j)
            r(i,j) = (*this)(i, j)/_s;     

    return r;
}


///<
template< class T, uint32 M, uint32 N >
inline T& Matrix<T, M, N>::operator()(const uint32 _i, const uint32 _j)
{
	ASSERT(_i>=0 && _j>=0 && _i<NumRows() && _j<NumColumns(), "m_d indexing error" );

	return m_d[_i*NumColumns() + _j];
}

///<
template< class T, uint32 M, uint32 N >
inline const T& Matrix<T, M, N>::operator()(const uint32 _i, const uint32 _j) const
{
	ASSERT(_i>=0 && _j>=0 && _i<NumRows() && _j<NumColumns(), "m_d indexing error" );

	return m_d[_i*NumColumns() + _j];
}


///<
template< class T, uint32 M, uint32 N >
Matrix<T, M, N> Matrix<T, M, N>::Zero()
{
    Matrix<T, M, N> result;
    memset(result.m_d, 0, sizeof(result.m_d));

    return result;
}

///<
template< class T, uint32 M, uint32 N >
Matrix<T, M, N> Matrix<T, M, N>::Identity()
{
    Matrix<T, M, N> r;
	for (uint32 i=0; i<M; ++i)
		r(i,i) = static_cast<T>(1);

    return r;
}

///<
template< class T, uint32 M, uint32 N >
bool Matrix<T, M, N>::operator==(const Matrix<T, M, N>& _m) const
{
	return std::equal(begin(), end(), _m.begin());
}

///<
template< class T, uint32 M, uint32 N >
T Matrix<T, M, N>::InfinityNorm() const
{ 
	T max=0;
	for(uint32 i=0; i<M; i++)
		for(uint32 j=0; j<N; ++j)
		{
			T abs_current=M::Abs((*this)(i,j));
			if (abs_current>max)
				max = abs_current;
		}

	return max;
	
}

template< class T, uint32 M, uint32 N >
T Matrix<T, M, N>::AbsLength() const
{
	T l=0;
	for(uint32 i=0; i<M; ++i)
		for(uint32 j=0; j<N; ++j)
			l+=M::Abs((*this)(i,j));

	return l;
}


template< class T, uint32 M, uint32 N >
T Matrix<T, M, N>::SquaredLength() const
{
	T l=0;
	for(uint32 i=0; i<M; ++i)
		for(uint32 j=0; j<N; ++j)
			l+=M::Squared((*this)(i,j));

	return l;
}

template< class T, uint32 M, uint32 N >
Matrix<T, M, N> Matrix<T, M, N>::Normalize() const
{
	return (*this)*M::Divide(1.0f,Length());
}
 
///<
template<class T, uint32 M, uint32 N, uint32 K>
class Volume
{

public:
	T m_d[M*N*K];

    Volume(){ memset(this,0,sizeof(Volume<T,M,N,K>)); }

	const uint32                  Width           () const        { return N; }
	const uint32                  Height          () const        { return M; } 
	const uint32                  Depth          () const        { return K; } 

    inline T&					operator()		(const Vector3i& _i){ASSERT(_i.x()*N*K + _i.y()*K + _i.z()<M*N*K, "bad indice!"); return m_d[_i.x()*N*K + _i.y()*K + _i.z()];}

	inline T&					operator()		(const uint32 _i, const uint32 _j, const uint32 _k){return m_d[_i*N*K + _j*K + _k];}
	inline const T&				operator()		(const uint32 _i, const uint32 _j, const uint32 _k) const {return m_d[_i*N*K + _j*K + _k];}
};

