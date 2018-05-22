///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2007-02-21
// Updated : 2007-03-01
// Licence : This source is under MIT License
// File    : glm/gtx/matx.h
///////////////////////////////////////////////////////////////////////////////////////////////////
// Dependency:
// - GLM core
// - GLM_GTX_vecx
// - GLM_GTX_matrix_selection
// - GLM_GTX_matrix_access
// - GLM_GTX_inverse_transpose
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_gtx_matx
#define glm_gtx_matx

// Dependency:
#include "../glm.hpp"
#include "../gtx/vecx.hpp"

namespace glm{
namespace detail{

    template <int N, typename T = float> 
	class _xmatxGTX
    {
    private:
        // Data
        _xvecxGTX<N, T> value[N];

    public:
        _xmatxGTX<N, T> _inverse() const;

    public:
		typedef T value_type;
		typedef int size_type;
		static const size_type value_size;

        // Constructors
        _xmatxGTX();
        explicit _xmatxGTX(const T x);

        // Accesses
        _xvecxGTX<N, T>& operator[](int i) {return value[i];}
        const _xvecxGTX<N, T> & operator[](int i) const {return value[i];}
        operator T*() {return &value[0][0];}
        operator const T*() const {return &value[0][0];}

        // Unary updatable operators
        _xmatxGTX<N, T>& operator=  (const _xmatxGTX<N, T>& m);
        _xmatxGTX<N, T>& operator+= (const T s);
        _xmatxGTX<N, T>& operator+= (const _xmatxGTX<N, T>& m);
        _xmatxGTX<N, T>& operator-= (const T s);
        _xmatxGTX<N, T>& operator-= (const _xmatxGTX<N, T>& m);
        _xmatxGTX<N, T>& operator*= (const T s);
        _xmatxGTX<N, T>& operator*= (const _xmatxGTX<N, T>& m);
        _xmatxGTX<N, T>& operator/= (const T s);
        _xmatxGTX<N, T>& operator/= (const _xmatxGTX<N, T>& m);
        _xmatxGTX<N, T>& operator++ ();
        _xmatxGTX<N, T>& operator-- ();
    };

	// Binary operators
    template <int N, typename T>
    _xmatxGTX<N, T> operator+ (const _xmatxGTX<N, T>& m, const T s);

    template <int N, typename T> 
    _xmatxGTX<N, T> operator+ (const T s, const _xmatxGTX<N, T>& m);

    template <int N, typename T>
    _xvecxGTX<N, T> operator+ (const _xmatxGTX<N, T>& m, const _xvecxGTX<N, T>& v);

    template <int N, typename T>
    _xvecxGTX<N, T> operator+ (const _xvecxGTX<N, T>& v, const _xmatxGTX<N, T>& m);
 
    template <int N, typename T> 
    _xmatxGTX<N, T> operator+ (const _xmatxGTX<N, T>& m1, const _xmatxGTX<N, T>& m2);
    
    template <int N, typename T> 
    _xmatxGTX<N, T> operator- (const _xmatxGTX<N, T>& m, const T s);

    template <int N, typename T> 
    _xmatxGTX<N, T> operator- (const T s, const _xmatxGTX<N, T>& m);

    template <int N, typename T> 
    _xvecxGTX<N, T> operator- (const _xmatxGTX<N, T>& m, const _xvecxGTX<N, T>& v);

    template <int N, typename T> 
    _xvecxGTX<N, T> operator- (const _xvecxGTX<N, T>& v, const _xmatxGTX<N, T>& m);

    template <int N, typename T>
    _xmatxGTX<N, T> operator- (const _xmatxGTX<N, T>& m1, const _xmatxGTX<N, T>& m2);

    template <int N, typename T> 
    _xmatxGTX<N, T> operator* (const _xmatxGTX<N, T>& m, const T s);

    template <int N, typename T>
    _xmatxGTX<N, T> operator* (const T s, const _xmatxGTX<N, T>& m);

    template <int N, typename T>
    _xvecxGTX<N, T> operator* (const _xmatxGTX<N, T>& m, const _xvecxGTX<N, T>& v);

    template <int N, typename T>
    _xvecxGTX<N, T> operator* (const _xvecxGTX<N, T>& v, const _xmatxGTX<N, T>& m);

    template <int N, typename T>
    _xmatxGTX<N, T> operator* (const _xmatxGTX<N, T>& m1, const _xmatxGTX<N, T>& m2);

    template <int N, typename T>
    _xmatxGTX<N, T> operator/ (const _xmatxGTX<N, T>& m, const T s);

    template <int N, typename T>
    _xmatxGTX<N, T> operator/ (const T s, const _xmatxGTX<N, T>& m);

    template <int N, typename T>
    _xvecxGTX<N, T> operator/ (const _xmatxGTX<N, T>& m, const _xvecxGTX<N, T>& v);

    template <int N, typename T>
    _xvecxGTX<N, T> operator/ (const _xvecxGTX<N, T>& v, const _xmatxGTX<N, T>& m);

    template <int N, typename T>
    _xmatxGTX<N, T> operator/ (const _xmatxGTX<N, T>& m1, const _xmatxGTX<N, T>& m2);

	// Unary constant operators
    template <int N, typename T>
    const _xmatxGTX<N, T> operator- (const _xmatxGTX<N, T>& m);

    template <int N, typename T>
    const _xmatxGTX<N, T> operator-- (const _xmatxGTX<N, T>& m, int);

    template <int N, typename T>
    const _xmatxGTX<N, T> operator++ (const _xmatxGTX<N, T>& m, int);

}//namespace detail

	// Extension functions
	template <int N, typename T> detail::_xmatxGTX<N, T> matrixCompMultGTX(const detail::_xmatxGTX<N, T>& x, const detail::_xmatxGTX<N, T>& y);
	template <int N, typename T> detail::_xmatxGTX<N, T> outerProductGTX(const detail::_xvecxGTX<N, T>& c, const detail::_xvecxGTX<N, T>& r);
	template <int N, typename T> detail::_xmatxGTX<N, T> transposeGTX(const detail::_xmatxGTX<N, T>& x);
	
	template <int N, typename T> T determinantGTX(const detail::_xmatxGTX<N, T>& m);
	template <int N, typename T> detail::_xmatxGTX<N, T> inverseTransposeGTX(const detail::_xmatxGTX<N, T> & m);

	template <int N, typename T> void columnGTX(detail::_xmatxGTX<N, T>& m, int ColIndex, const detail::_xvecxGTX<N, T>& v);
	template <int N, typename T> void rowGTX(detail::_xmatxGTX<N, T>& m, int RowIndex, const detail::_xvecxGTX<N, T>& v);

	template <int N, typename T> detail::_xvecxGTX<N, T> columnGTX(const detail::_xmatxGTX<N, T>& m, int ColIndex);
	template <int N, typename T> detail::_xvecxGTX<N, T> rowGTX(const detail::_xmatxGTX<N, T>& m, int RowIndex);

    namespace gtx
    {
		//! GLM_GTX_matx extension: - Work in progress - NxN matrix types.
        namespace matx
        {
	        // Matrix Functions
	        template <int N, typename T> inline detail::_xmatxGTX<N, T> matrixCompMult(const detail::_xmatxGTX<N, T>& x, const detail::_xmatxGTX<N, T>& y){return matrixCompMult(x, y);}
	        template <int N, typename T> inline detail::_xmatxGTX<N, T> outerProduct(const detail::_xvecxGTX<N, T>& c, const detail::_xvecxGTX<N, T>& r){return outerProductGTX(c, r);}
	        template <int N, typename T> inline detail::_xmatxGTX<N, T> transpose(const detail::_xmatxGTX<N, T>& x){return transposeGTX(x);}
        	
	        template <int N, typename T> inline T determinant(const detail::_xmatxGTX<N, T>& m){return determinantGTX(m);}
	        template <int N, typename T> inline detail::_xmatxGTX<N, T> inverseTranspose(const detail::_xmatxGTX<N, T>& m){return inverseTransposeGTX(m);}

	        template <int N, typename T> inline void column(detail::_xmatxGTX<N, T>& m, int ColIndex, const detail::_xvecxGTX<N, T>& v){setColumnGTX(m, v);}
	        template <int N, typename T> inline void row(detail::_xmatxGTX<N, T>& m, int RowIndex, const detail::_xvecxGTX<N, T>& v){setRowGTX(m, v);}

	        template <int N, typename T> inline detail::_xvecxGTX<N, T> column(const detail::_xmatxGTX<N, T>& m, int ColIndex){return column(m, ColIndex);}
	        template <int N, typename T> inline detail::_xvecxGTX<N, T> row(const detail::_xmatxGTX<N, T>& m, int RowIndex){return row(m, RowIndex);}
        }
    }
}

#define GLM_GTX_matx namespace gtx::matx
#ifndef GLM_GTX_GLOBAL
namespace glm {using GLM_GTX_matx;}
#endif//GLM_GTX_GLOBAL

#include "matx.inl"

#endif//glm_gtx_matx
