

#ifndef __MATRIX_HPP__
#define __MATRIX_HPP__

#include <math.h>
#include <Common\Assert.hpp>
#include <Common\StaticAssert.hpp>
#include <Math\Vector\Vector.hpp>


////////////////////////////////////////////////////////////////
///
/// -Vectors are Nx1 matrices.
///
////////////////////////////////////////////////////////////////

template <class T, uint32 M, uint32 N>
class Matrix 
{

protected:

   T m_d[M*N];

public:

    typedef T*			iterator;
    typedef const T*	const_iterator;
    typedef T			value_type;

	Matrix();
	Matrix(const T&);
	Matrix(const Matrix<T, M, N>&);

    template<class T2, size_t M2, size_t N2>
	Matrix(const Matrix<T2, M2, N2>& _om)
	{
		memset(this, 0, sizeof(Matrix<T, M, N>));
		for (size_t i=0; i<M::Min(M, M2); ++i)
			for (size_t j=0; j<M::Min(N, N2); ++j)
			{
				this->operator()(i,j)=M::SCast<T>(_om(i,j));
			}
	}

    ~Matrix<T, M, N>(){}

   Matrix<T, M, N>& operator= (const Matrix<T, M, N>& _m);
   
   uint32						Width           () const        { return N; }
   uint32						Height          () const        { return M; } 

   uint32						NumRows			() const		{ return M; }
   uint32						NumColumns			() const		{ return N; }

	inline iterator          	Begin			()				{ return &(m_d[0]); }
	inline const_iterator    	Begin			() const		{ return &(m_d[0]); }
	inline iterator          	End				()				{ return Begin() + M*N; }
	inline const_iterator    	End				() const		{ return Begin() + M*N; }
   
	inline Matrix<T, N, M>		Transpose		() const;
	inline Matrix<T, M, N>		Inverse			() const;

	static Matrix<T, M, N>		Identity		();
	static Matrix<T, M, N>		Zero			();

	inline T&					operator[]		(const uint32);
	inline const T&				operator[]		(const uint32) const;

	inline T&					operator()		(const Matrix<uint32, 2, 1>& _i){ return this->operator()(_i.x(),_i.y()); }
	inline const T&				operator()		(const Matrix<uint32, 2, 1>& _i) const{{ return this->operator()(_i.x(),_i.y()); }}

	inline T&					operator()		(const uint32 i, const uint32 j);
	inline const T&				operator()		(const uint32 i, const uint32 j) const;

	///< Math
	inline bool					operator==		(const Matrix<T, M, N>& _m) const;  

	inline Matrix				operator+		(const Matrix<T, M, N>&) const;
	
	inline Matrix				operator+=		(const Matrix<T, M, N>&);
	inline Matrix				operator-=		(const Matrix<T, M, N>&);

	inline Matrix				operator-		(const Matrix<T, M, N>&) const;
    inline Matrix				operator-		() const;

	inline Matrix				operator/		(const T&) const;
	inline Matrix				operator/		(const Matrix&) const;

	inline T					InfinityNorm	() const;
	inline T					SquaredLength   () const;
	inline T					Length			() const { return M::Sqrt(SquaredLength()); }

   	inline Matrix				Normalize		() const;
	inline T					AbsLength		() const;	

	///< Vector Interface
	///<
	Matrix( const T&  _x, const T&  _y)								{STATIC_CHECK( (M==2 && N==1) || (M==1 && N==2) ); m_d[0]= _x; m_d[1]= _y;}
	Matrix( const T&  _x, const T&  _y, const T&  _z)				{STATIC_CHECK( (M==3 && N==1) || (M==1 && N==3) ); m_d[0] =  _x; m_d[1] =  _y; m_d[2] =  _z; }
	Matrix( const T&  _x, const T&  _y, const T&  _z, const T& _w)	{STATIC_CHECK( (M==4 && N==1) || (M==1 && N==4) ); m_d[0] =  _x; m_d[1] =  _y; m_d[2] =  _z; m_d[3] =  _w;}
	Matrix( const T& _x, const T& _y, const T& _z, const T& _w, const T& _a, const T& _b, const T& _c, const T& _d )
	{STATIC_CHECK(M==8 && N==1); m_d[0] = _x; m_d[1] = _y; m_d[2] = _z; m_d[3] = _w; m_d[4]= _a; m_d[5]= _b; m_d[6]= _c; m_d[7]= _d;}

	inline size_t Size() const {return M*N;}
	
	inline const T&	x				() const{ STATIC_CHECK(M>0 && N==1);return m_d[0]; }
    inline const T&	y				() const{ STATIC_CHECK(M>1 && N==1);return m_d[1]; }
    inline const T&	z				() const{ STATIC_CHECK(M>2 && N==1);return m_d[2]; }
	inline const T&	w				() const{ STATIC_CHECK(M>3 && N==1);return m_d[3]; }
  
};

typedef Matrix< float32,  8, 1 >		Vector8f;
typedef Matrix< int32,  8, 1 >			Vector8i;
typedef Matrix< int32,  6, 1 >			Vector6i;

typedef Matrix< float32,  4, 1 >		Vector4f;
typedef Matrix< float64, 4, 1 >			Vector4d;
typedef Matrix< uint16, 4, 1 >			Vector4us;
typedef Matrix< uint32, 4, 1 >			Vector4ui;
typedef Matrix< int32,  4, 1 >			Vector4i;

typedef Matrix< uint8,  4, 1 >			Vector4uc;

typedef Matrix< float32, 3, 1 >			Vector3f;
typedef Matrix< float64, 3, 1 >			Vector3d;
typedef Matrix< uint16, 3, 1 >			Vector3us;
typedef Matrix< int32,  3, 1 >			Vector3i;
typedef Matrix< uint32,  3, 1 >			Vector3ui;
typedef Matrix< uint8,  3, 1 >			Vector3uc;

typedef Matrix< float32, 2, 1 >			Vector2f;
typedef Matrix< float64, 2, 1 >			Vector2d;
typedef Matrix< int32,  2, 1 >			Vector2i;
typedef Matrix< uint32,  2, 1 >			Vector2ui;

typedef Matrix< uint16, 2, 1 >			Vector2us;
typedef Matrix< int16, 2, 1 >			Vector2s;

typedef Matrix< float32, 1, 1 >			Vector1f;
typedef Matrix< float64, 1, 1 >			Vector1d;
typedef Matrix< uint16, 1, 1 >			Vector1us;

template<class T, size_t M>
T DotProduct(const Matrix<T, M, 1>& _v1, const Matrix<T, M, 1>& _v2) 
{
	T sum = static_cast<T>(0);
	for (size_t i = 0; i<M; ++i)
		sum += _v1[i]*_v2[i];
        
	return sum;
}

template<class T, uint32 M>
Matrix<T,M, 1> DirectProduct(const Matrix<T, M, 1>& _v1, const Matrix<T, M, 1>& _v2) 
{
	Matrix<T, M, 1> r;
	for (size_t i = 0; i<M; ++i)
		r[i] = _v1[i]*_v2[i];
		
	return r;
}

///<
template<class T>
Matrix<T, 3, 1> CrossProduct(const Matrix<T, 3, 1>& _v1, const Matrix<T, 3, 1>& _v2)
{
    Matrix<T, 3, 1> rVector;
    
    rVector[0] = _v1[1]*_v2[2] - _v1[2]*_v2[1];
    rVector[1] = _v1[2]*_v2[0] - _v1[0]*_v2[2];
    rVector[2] = _v1[0]*_v2[1] - _v1[1]*_v2[0];

    return rVector;
}


typedef Matrix<float32, 3, 1> Matrix31f;

typedef Matrix<float32, 2, 2> Matrix2f;
typedef Matrix<float32, 3, 3> Matrix3f;
typedef Matrix<float32, 4, 4> Matrix4f;


#include "Matrix.inl"

///<
template < class T, class K, uint32 M, uint32 N >
inline Matrix<T, M, N> operator*(const K& _s, const Matrix<T, M, N>& _m)
{
	Matrix<T, M, N> r;
	for (int32 i=0; i<M; ++i)
		for (int32 j=0; j<N; ++j)
			r(i,j) = _m(i,j)*_s;

	return r;
}

template < class T, class K, size_t M, size_t N >
inline Matrix<T, M, N> operator*(const Matrix<T, M, N>& _m, const K& _s)
{
	return _s*_m;
}

template<class T, class K, size_t M, size_t N, size_t A>
inline Matrix<K, M, A> operator*(const Matrix<T, M, N>& _mLeft, const Matrix<K, N, A>& _mRight)
{
	Matrix<K, M, A> r;
	
	for (size_t i=0; i<M;++i)	
		for(size_t k=0; k<A; ++k )
			for (size_t j=0; j<N;++j)			
				r(i,k) += _mLeft(i,j)*_mRight(j,k);

	return r;
}


///<
template<class T>
Matrix<T, 4, 4> AffineInverse(const Matrix<T, 4, 4>& _m)
{
	Matrix<T, 4, 4> inv = _m.Transpose();
	inv(3,0)=0;
	inv(3,1)=0;
	inv(3,2)=0;
	
	Matrix<T, 4, 1> b(_m(0,3), _m(1,3), _m(2,3), 0);
	
	Matrix<T, 4, 1> invB = -1.0f*inv*b;
	inv(0,3)=invB[0]; inv(1,3)=invB[1]; inv(2,3)=invB[2];

	return inv;
}

///<
template<class T>
class DMatrix : public Incopiable
{
	uint32	m_NumRows, m_NumColumns;
	T*		m_d;	
public:

	void Swap(DMatrix<T>& _d)
	{
		std::swap(m_d,_d.m_d);
	}

	typedef T value_type;
	DMatrix():m_NumRows(0),m_NumColumns(0),m_d(0){}

	DMatrix(const uint32 _NumRows, const uint32 _NumColumns) : m_NumRows(_NumRows),m_NumColumns(_NumColumns),m_d(0)
	{
		Create(_NumRows, _NumColumns);
	}

	///<
	void Create(const uint32 _NumRows, const uint32 _NumColumns)
	{
		if (m_d!=NULL)
			M::DeleteArray(&m_d);

		m_NumRows=_NumRows;
		m_NumColumns=_NumColumns;
		ASSERT(NumRows()*NumColumns()>0,"bad values !");

		m_d=new T[NumRows()*NumColumns()];

		memset(m_d,0,NumRows()*NumColumns()*sizeof(T));
	}

	DMatrix(const DMatrix& _d):m_NumRows(_d.m_NumRows),m_NumColumns(_d.m_NumColumns),m_d(0)
	{
		Create(_d.m_NumRows, _d.m_NumColumns);
	}

	~DMatrix()	{ 		M::DeleteArray(&m_d);	}

	inline const uint32		Size		() const { return NumColumns()*NumRows()*Depth(); }
	
	inline const uint32		NumColumns			() const { return m_NumColumns; }
	inline const uint32		NumRows				() const { return m_NumRows; }

	inline const uint32		Depth		() const { return 1; }
	inline const Vector2i	Dimensions	() const { return Vector2i(NumRows(),NumColumns()); }
	inline const Vector2f	Dimensionsf	() const { return Vector2f(static_cast<float32>(NumRows()),static_cast<float32>(NumColumns()) ); }

	void Copy(const DMatrix<T>& _m){ memcpy(m_d, _m.m_d, NumRows()*NumColumns()*sizeof(T)); }

	T SquaredLength() const 
	{
		T l=0;
		for(uint32 i=0; i<NumRows(); ++i)
		{
			for(uint32 j=0; j<NumColumns(); ++j)
			{
				l+= M::Squared(this->operator()(i,j));
			}
		}

		return l;
	}

	T&			operator()      (const Vector2ui& _i)        { return this->operator()(_i.x(), _i.y());}
	const T&	operator()		(const Vector2ui& _i) const  { return this->operator()(_i.x(), _i.y());}

	T&			At				(const uint32 _i, const uint32 _j){ return this->operator ()(_i,_j); }
	T&			At				(const Vector2ui& _i){ return this->operator ()(_i); }

	T&			operator()      (const uint32 _i, const uint32 _j)         { ASSERT(_i < NumRows() && _j < NumColumns(), "Matrix indices"); return m_d[_i*NumColumns() + _j];}
	const T&	operator()		(const uint32 _i, const uint32 _j) const   { ASSERT(_i < NumRows() && _j < NumColumns(), "Matrix indices"); return m_d[_i*NumColumns() + _j];}

	const T&	operator[]		(const uint32 _i) const	{ ASSERT(_i<NumRows()*NumColumns(),"To large"); return m_d[_i]; }
	T&			operator[]		(const uint32 _i)		{ ASSERT(_i<NumRows()*NumColumns(),"To large"); return m_d[_i]; }
};




namespace M{ ///< Move to .inl !

	
	template< class T >
	void SetAffineTranslation(const Matrix<T, 3, 1>& _x, Matrix<T, 4, 4>& _m)
	{
		for (size_t i = 0; i<_x.Size(); ++i)
			_m(i, 3) = _x[i];
	}
	///<
	template< class T >
	Matrix<T, 3, 1> GetAffineTranslation(const Matrix<T, 4, 4>& _m)
	{
		Matrix<T, 3, 1> rV;		
		for (size_t i = 0; i<rV.Size(); ++i)
			rV[i] = _m(i, 3);

		return rV;
	}

	///< Normal transformations (from right to left composition): 
	///<
	template< class T >
	void AffineTranslation(const Vector3f& _x, Matrix<T, 4, 4>& _m)
	{
		_m = Matrix<T, 4, 4>::Identity();
		SetAffineTranslation(_x, _m);
	}	

	///<
	template< class T >
	void AffineScale(const Vector3f& _v, Matrix<T, 4, 4>& _s)
	{
		_s = Matrix<T, 4, 4>::Identity();
		for (size_t i = 0; i<_v.Size(); ++i)
			_s(i, i) = _v[i];		
	}

	///<
	template<class T>
	void RotationZ(const T& _a,  Matrix<T, 4, 4>& _m)
	{
		_m = Matrix<T, 4, 4>::Identity();
		_m(0, 0) = cos(_a);   _m(0, 1) = -sin(_a);
		_m(1, 0) = sin(_a);   _m(1, 1) = cos(_a); 
	}

	///<
	template<class T>
	void RotationY(const T& _a,  Matrix<T, 4, 4>& _m)
	{
		_m = Matrix<T, 4, 4>::Identity();

		_m(0, 0) = cos(_a);   _m(0, 2) = sin(_a);
		_m(2, 0) = -sin(_a);   _m(2, 2) = cos(_a); 

	}

	///<
	template<class T>
	void RotationX(const T& _a,  Matrix<T, 4, 4>& _m)
	{
		_m = Matrix<T, 4, 4>::Identity();

		_m(1, 1) = cos(_a);   _m(1, 2) = -sin(_a);
		_m(2, 1) = sin(_a);   _m(2, 2) = cos(_a); 

	}

	template< class T >
	void AffineRotation(const Vector3f& _w, Matrix<T, 4, 4>& _m)
	{
		Matrix<T, 4, 4> RotX, RotY, RotZ;
		RotationX(_w.x(), RotX);
		RotationY(_w.y(), RotY);
		RotationZ(_w.z(), RotZ);

		_m = RotX*RotY*RotZ;
	}


	static const Vector3f ColorToGreyScale(0.2989f, 0.5870f, 0.1140f);
	static const Vector3f xAxis			= Vector3f( 1.0f, 0.0f, 0.0f );
	static const Vector3f yAxis			= Vector3f( 0.0f, 1.0f, 0.0f );
	static const Vector3f zAxis			= Vector3f( 0.0f, 0.0f, 1.0f );

	static const Vector<Vector3f, 3> XYZAxis = Vector<Vector3f, 3>(xAxis, yAxis, zAxis);

	static const Vector<Vector2i, 4> Cardinals( Vector2i(1, 0),Vector2i(-1, 0),Vector2i(0, -1),Vector2i(0, 1));
	static const Vector<Vector2f, 4> FCardinals( Vector2f(1, 0),Vector2f(-1, 0),Vector2f(0, -1),Vector2f(0, 1));

	static const Vector<Vector2i, 4> Corners( Vector2i(-1, -1),Vector2i(1, -1),Vector2i(-1, 1),Vector2i(1, 1));
	static const Vector<Vector2f, 4> FCorners( Vector2f(-1, -1),Vector2f(1, -1),Vector2f(-1, 1),Vector2f(1, 1));

	static const Vector<Vector3f, 8> F3DCorners(Vector3f(-1, -1, -1),Vector3f(1, -1, -1),Vector3f(-1, 1, -1),Vector3f(1, 1, -1),
												Vector3f(-1, -1, 1),Vector3f(1, -1, 1),Vector3f(-1, 1, 1),Vector3f(1, 1, 1));


}

#define FOR_EACH_MATRIX(_M) for (uint32 i=0; i<_M.NumRows(); ++i){ for (uint32 j=0; j<_M.NumColumns(); ++j) {

#define FOR_EACH(G) for (uint32 j=0; j<G.Y(); ++j){ for (uint32 i=0; i<G.X(); ++i) {
#define FOR_EACH_V(G) for (uint32 k=0; k<G.Z(); ++k){ for (uint32 j=0; j<G.Y(); ++j) { for (uint32 i=0; i<G.X(); ++i){ 

#define FOR_EACH_VECTOR3_COMPONENT(G) for (uint32 k=0; k<G.z(); ++k){ for (uint32 j=0; j<G.y(); ++j) { for (uint32 i=0; i<G.x(); ++i){ 
#define FOR_EACH_VECTOR2_COMPONENT(G) for (uint32 j=0; j<G.y(); ++j) { for (uint32 i=0; i<G.x(); ++i){ 


#define FOR_EACH_INBOUND(G) for (uint32 j=1; j<G.Y()-1; ++j) { for (uint32 i=1; i<G.X()-1; ++i){ 
#define FOR_EACH_INBOUND_V(G) for (uint32 k=1; k<G.Z()-1; ++k){ for (uint32 j=1; j<G.Y()-1; ++j) { for (uint32 i=1; i<G.X()-1; ++i){ 

#define END_FOR_EACH }}
#define END_FOR_EACH_V }}}

#endif

