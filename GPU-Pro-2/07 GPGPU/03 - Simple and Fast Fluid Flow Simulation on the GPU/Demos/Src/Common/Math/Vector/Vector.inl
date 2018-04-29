
/// Default constructor. It creates a vector of size zero.

template <typename Type, size_t Size>
Vector<Type, Size>::Vector()
{
	memset( this, 0, sizeof(Vector) );
}

/// Copy constructor. It creates a copy From a matrix. 
///
/// @param matrix
template <typename Type, size_t Size>
Vector<Type, Size>::Vector( const Matrix<Type, 1, Size>& _matrix )
{
	// Matrices define first line first.(not colomn)
	memcpy(m_vector, _matrix.begin(), sizeof(Vector));
}


/// Copy constructor. It creates a copy of an existing Vector. 
///
/// @param oldVector: Vector to be copied.

template <typename Type, size_t Size>
Vector<Type, Size>::Vector( const Vector<Type, Size>& oldVector ) 
{
    for( size_t i = 0; i < Size; ++i )
    {
        m_vector[i] = oldVector[i];
    }
}

template <typename Type, size_t Size>
Vector<Type, Size> Vector<Type, Size>::VSign(const Vector& _v) 
{
	Vector rV;
	for(size_t i=0; i<Size; ++i)
	{
		rV[i] = Sign(_v[i]);
	}
	return rV;
}

template <typename Type, size_t Size>
Vector<Type, Size> Vector<Type, Size>::VAbs(const Vector& _v) 
{
	Vector rV;
	for(size_t i=0; i<Size; ++i)
	{
		rV[i] = fabs(_v[i]);
	}
	return rV;
}

template < typename Type, size_t Size >
Vector<Type, Size>& Vector<Type, Size>::operator=(const Vector<Type, Size>& oldVector)
{
    if (this != &oldVector)
    {
        for ( size_t i = 0; i < Size; ++i )
        {
            m_vector[i] = oldVector[i];
        }
    }

    return(*this);
}



template <typename Type, size_t Size>
Type& Vector<Type, Size>::operator[]( const size_t& i ) 
{
    ASSERT( i >= 0 && i < Size, "Wrong index");

    return m_vector[i];
}


/// Reference operator. 

template <typename Type, size_t Size>
const Type& Vector<Type, Size>::operator[]( const size_t& i ) const 
{
    // Control sentence	
    ASSERT( i>=0 && i<Size, "Wrong index");
    return m_vector[i];
}


/// This method returns the largest element in the vector.

template< class Type >
bool bigger( const Type& a, const Type& b )
{
    return a<b;
}

template <typename Type, size_t Size>
const Type Vector<Type, Size>::Max() const
{
    return std::max_element( begin(), end() );
}

/// This method returns the smallest element in the vector.

template <typename Type, size_t Size>
const Type Vector<Type, Size>::Min() const
{
    return *std::min_element( begin(), end() );
}


template <typename Type, size_t Size>
const Type Vector<Type, Size>::ZeroNorm() const
{
	Type Sum=0;
	for (size_t i=0;i<Size;++i)
		Sum+=std::fabs(m_vector[i]);
	
	return Sum;
}

template <typename Type, size_t Size>
const Type Vector<Type, Size>::InfinityNorm() const
{
	return *std::max_element(begin(), end(), AbsGreater<Type>);
}


template <typename Type, size_t Size>
const Type Vector<Type, Size>::Norm() const
{
    return sqrt(SquaredNorm());
}

template <typename Type, size_t Size>
const Type Vector<Type, Size>::SquaredNorm() const
{
	Type norm = 0.0;

    for( size_t i = 0; i < Size; ++i )
    {
        norm += m_vector[i]*m_vector[i];
    }

	return norm;
}


// Vector<Type> operator + (Type) method 

/// Sum vector+scalar arithmetic operator. 

template <typename Type, size_t Size>
inline Vector<Type, Size> Vector<Type, Size>::operator + ( const Type& scalar ) const
{       
    Vector<Type, Size> sum;

    for(int i = 0; i < Size; i++)
    {
        sum[i] = m_vector[i] + scalar;
    }

    return(sum);
}


// Vector<Type> operator + (Vector<Type>)

/// Sum vector+vector arithmetic operator.

template <typename Type, size_t Size>
inline Vector<Type, Size> Vector<Type, Size>::operator + ( const Vector<Type, Size>& otherVector ) const
{      
    Vector<Type, Size> sum;
    for( size_t i = 0; i < Size; i++)
    {
        sum[i] = m_vector[i] + otherVector[i];
    }
    return(sum);
}

template<typename Type, typename TypeApparented, size_t Size>
inline Vector<Type, Size> operator+( const Vector<Type, Size>& _vecLeft, const Vector<TypeApparented, Size>& _vecRight ) 
{       
    Vector<Type, Size> sum;
    for( size_t i = 0; i < Size; i++)
    {
        sum[i] = _vecLeft[i] + static_cast<Type>(_vecRight[i]);
    }
    return sum;
}

template <typename Type, size_t Size>
inline void Vector<Type, Size>::operator+=( const Vector<Type, Size>& _vec)
{     
    (*this) = (*this) + _vec;
}


//Vector<Type> operator - (Type) method 

/// Difference vector-scalar arithmetic operator.

template <typename Type, size_t Size>
inline Vector<Type, Size> Vector<Type, Size>::operator - ( const Type& scalar ) const
{       
    Vector<Type, Size> difference;

    for(int i = 0; i < Size; i++)
    {
        difference[i] = m_vector[i] - scalar;
    }

    return(difference);
}


// Vector<Type> operator - (Vector<Type>)

/// Difference vector-vector arithmetic operator.

template <typename Type, size_t Size>
inline Vector<Type, Size> Vector<Type, Size>::operator - ( const Vector<Type, Size>& otherVector) const
{     
    Vector<Type, Size> difference;

    for( size_t i = 0; i < Size; i++)
    {
        difference[i] = (*this)[i] - otherVector[i];
    }

    return(difference);
}

template <typename Type, size_t Size>
inline void Vector<Type, Size>::operator-=( const Vector<Type, Size>& otherVector)
{     
    (*this) = (*this) - otherVector;
}


// Vector<Type> operator * (Type) method 

/// Product vector*scalar arithmetic operator.

template <typename Type, size_t Size>
inline Vector<Type, Size> Vector<Type, Size>::operator * ( const Type& scalar) const
{       
    Vector<Type, Size> product;

    for(int i = 0; i < Size; i++)
    {
        product[i] = m_vector[i]*scalar;
    }

    return(product);
}


// Type operator * (Vector<Type>)  

/// Scalar product vector*vector arithmetic operator.

template < typename Type, size_t Size >
inline Type Vector<Type, Size>::operator*(const Vector<Type, Size>& otherVector)const
{     
    Type product = 0;
    for(int i = 0; i < Size; i++)
    {
        product += m_vector[i]*otherVector[i];
    }
    return(product);
}


//Vector<Type> operator / (Type) method 

/// Cocient vector/scalar arithmetic operator.

template <typename Type, size_t Size>
inline Vector<Type, Size> Vector<Type, Size>::operator / ( const Type& scalar ) const
{       
    Vector<Type, Size> cocient;

    for(int i = 0; i < Size; i++)
    {
        cocient[i] = m_vector[i]/scalar;
    }

    return(cocient);
}


// Vector<Type> operator - (Vector<Type>)

/// Cocient vector/vector arithmetic operator.

template <typename Type, size_t Size>
inline Vector<Type, Size> Vector<Type, Size>::operator / ( const Vector<Type, Size>& otherVector) const
{       
    Vector<Type, Size> cocient;

    for(int i = 0; i < Size; i++)
    {
        cocient[i] = m_vector[i]/otherVector[i];
    }

    return(cocient);
}

template <typename Type, size_t Size>
inline Type Vector<Type, Size>::Length() const
{
    value_type sum = 0.0f;
    for( size_type i = 0 ; i < Size; ++i )
    {
        sum += m_vector[i]*m_vector[i];
    }

    return sqrt( sum );
}

template <typename Type, size_t Size>
inline Vector<Type, Size> Vector<Type, Size>::Normalize() const
{
    return (*this)/Length();
}


template <typename Type, size_t Size >
inline Matrix<Type, Size, Size> Vector<Type, Size>::Outer( const ClassType& otherVector ) const
{       
    Matrix<Type, Size, Size> outer;

    for( size_t i = 0;  i < Size; ++i )
    {
        for( size_t j = 0;  j < Size; ++j )
        {
            outer(i, j) = m_vector[i]*otherVector[j];
        }           
    }

    return outer;
}


// Output operator

/// This method re-writes the output operator << for the Vector template. 

/*

template< typename Type, size_t Size >
std::ostream& operator<<(std::ostream& os, Vector<Type, Size>& v)
{

    for(int i = 0; i < Size; i++)
    {
        os << v[i] << " ";
    }

    return(os);
}

*/