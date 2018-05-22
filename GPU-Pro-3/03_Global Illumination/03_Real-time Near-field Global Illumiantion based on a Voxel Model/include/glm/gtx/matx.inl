///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2007-02-21
// Updated : 2007-02-21
// Licence : This source is under MIT License
// File    : glm/gtx/matx.inl
///////////////////////////////////////////////////////////////////////////////////////////////////

#include <cassert>
#include <algorithm>

namespace glm{
namespace detail{

	template <int N, typename T> const typename _xmatxGTX<N, T>::size_type _xmatxGTX<N, T>::value_size = N;

    //////////////////////////////////////////////////////////////
    // _xmatxGTX constructors

    template <int N, typename T>
    inline _xmatxGTX<N, T>::_xmatxGTX()
    {
		for(int i = 0; i < N; ++i)
			this->value[i][i] = T(0);
    }

    template <int N, typename T>
    inline _xmatxGTX<N, T>::_xmatxGTX(const T f)
    {
		for(int i = 0; i < N; ++i)
			this->value[i][i] = f;
    }

    //////////////////////////////////////////////////////////////
    // _xmatxGTX operators

    // This function shouldn't required but it seems that VC7.1 have an optimisation bug if this operator wasn't declared
    template <int N, typename T>
    inline _xmatxGTX<N, T>& _xmatxGTX<N, T>::operator= (const _xmatxGTX<N, T>& m)
    {
        //memcpy could be faster
        //memcpy(&this->value, &m.value, 16 * sizeof(T));
		for(int i = 0; i < N; ++i)
			this->value[i] = m[i];
        return *this;
    }

    template <int N, typename T>
    inline _xmatxGTX<N, T>& _xmatxGTX<N, T>::operator+= (const T s)
    {
		for(int i = 0; i < N; ++i)
			this->value[i] += s;
        return *this;
    }

    template <int N, typename T>
    inline _xmatxGTX<N, T>& _xmatxGTX<N, T>::operator+= (const _xmatxGTX<N, T>& m)
    {
		for(int i = 0; i < N; ++i)
			this->value[i] += m[i];
        return *this;
    }

    template <int N, typename T>
    inline _xmatxGTX<N, T>& _xmatxGTX<N, T>::operator-= (const T s)
    {
		for(int i = 0; i < N; ++i)
			this->value[i] -= s;
        return *this;
    }

    template <int N, typename T>
    inline _xmatxGTX<N, T>& _xmatxGTX<N, T>::operator-= (const _xmatxGTX<N, T>& m)
    {
		for(int i = 0; i < N; ++i)
			this->value[i] -= m[i];
        return *this;
    }

    template <int N, typename T>
    inline _xmatxGTX<N, T>& _xmatxGTX<N, T>::operator*= (const T s)
    {
		for(int i = 0; i < N; ++i)
			this->value[i] *= s;
        return *this;
    }

    template <int N, typename T>
    inline _xmatxGTX<N, T>& _xmatxGTX<N, T>::operator*= (const _xmatxGTX<N, T>& m)
    {
        return (*this = *this * m);
    }

    template <int N, typename T>
    inline _xmatxGTX<N, T>& _xmatxGTX<N, T>::operator/= (const T s)
    {
		for(int i = 0; i < N; ++i)
			this->value[i] /= s;
        return *this;
    }

    template <int N, typename T>
    inline _xmatxGTX<N, T>& _xmatxGTX<N, T>::operator/= (const _xmatxGTX<N, T>& m)
    {
        return (*this = *this / m);
    }

    template <int N, typename T>
    inline _xmatxGTX<N, T>& _xmatxGTX<N, T>::operator-- ()
    {
		for(int i = 0; i < N; ++i)
			--this->value[i];
        return *this;
    }

    template <int N, typename T>
    inline _xmatxGTX<N, T>& _xmatxGTX<N, T>::operator++ ()
    {
		for(int i = 0; i < N; ++i)
			++this->value[i];
        return *this;
    }

    // Private functions
    template <int N, typename T>
    inline _xmatxGTX<N, T> _xmatxGTX<N, T>::_inverse() const
    {
		_xmatxGTX<N, T> Result = *this;

 		int ColIndex[N];
		int RowIndex[N];
		bool Pivoted[N];
		memset(ColIndex, 0, N * sizeof(int));
		memset(RowIndex, 0, N * sizeof(int));
		memset(Pivoted, 0, N * sizeof(bool));

		int iRow = 0, iCol = 0;

		// elimination by full pivoting
		for(int i0 = 0; i0 < N; i0++)
		{
			// search matrix (excluding pivoted rows) for maximum absolute entry
			T fMax = T(0);
			for(int i1 = 0; i1 < N; i1++)
			{
				if(Pivoted[i1])
					continue;

				for(int i2 = 0; i2 < N; i2++)
				{
					if(Pivoted[i2])
						continue;
						
					T Abs = abs(Result[i1][i2]);
					if(Abs > fMax)
					{
						fMax = Abs;
						iRow = i1;
						iCol = i2;
					}
				}
			}

			if(fMax == T(0))
			{
				return _xmatxGTX<N, T>(1.0f); // Error
			}

			Pivoted[iCol] = true;

			// swap rows so that A[iCol][iCol] contains the pivot entry
			if(iRow != iCol)
			{
				_xvecxGTX<N, T> Row = rowGTX(Result, iRow);
				_xvecxGTX<N, T> Col = rowGTX(Result, iCol);
				rowGTX(Result, iRow, Col);
				rowGTX(Result, iCol, Row);
			}

			// keep track of the permutations of the rows
			RowIndex[i0] = iRow;
			ColIndex[i0] = iCol;

			// scale the row so that the pivot entry is 1
			T fInv = T(1) / Result[iCol][iCol];
			Result[iCol][iCol] = T(1);
			for(int i2 = 0; i2 < N; i2++)
				Result[iCol][i2] *= fInv;

			// zero out the pivot column locations in the other rows
			for(int i1 = 0; i1 < N; ++i1)
			{
				if(i1 == iCol)
					continue;

				T Tmp = Result[i1][iCol];
				Result[i1][iCol] = T(0);
				for(int i2 = 0; i2 < N; i2++)
					Result[i1][i2] -= Result[iCol][i2] * Tmp;
			}
		}

		// reorder rows so that A[][] stores the inverse of the original matrix
		for(int i1 = N-1; i1 >= 0; --i1)
		{
			if(RowIndex[i1] == ColIndex[i1])
				continue;
			for(int i2 = 0; i2 < N; ++i2)
				std::swap(Result[i2][RowIndex[i1]], Result[i2][ColIndex[i1]]);
		}

		return Result;
    }

	// Binary operators
    template <int N, typename T>
    inline _xmatxGTX<N, T> operator+ (const _xmatxGTX<N, T>& m, const T s)
    {
		_xmatxGTX<N, T> result;
		for(int i = 0; i < N; ++i)
			result[i] = m[i] + s;
		return result;
    }

    template <int N, typename T>
    inline _xmatxGTX<N, T> operator+ (const T s, const _xmatxGTX<N, T>& m)
    {
		_xmatxGTX<N, T> result;
		for(int i = 0; i < N; ++i)
			result[i] = s + m[i];
		return result;
    }
/*
    template <int N, typename T>
    inline tvec4<T> operator+ (const _xmatxGTX<N, T>& m, const tvec4<T>& v)
    {

    }

    template <int N, typename T>
    inline tvec4<T> operator+ (const tvec4<T>& v, const _xmatxGTX<N, T>& m)
    {

    }
*/
    template <int N, typename T>
    inline _xmatxGTX<N, T> operator+ (const _xmatxGTX<N, T>& m1, const _xmatxGTX<N, T>& m2)
    {
		_xmatxGTX<N, T> result;
		for(int i = 0; i < N; ++i)
			result[i] = m1[i] + m2[i];
		return result;
    }
    
    template <int N, typename T>
    inline _xmatxGTX<N, T> operator- (const _xmatxGTX<N, T>& m, const T s)
    {
		_xmatxGTX<N, T> result;
		for(int i = 0; i < N; ++i)
			result[i] = m[i] - s;
		return result;
    }

    template <int N, typename T>
    inline _xmatxGTX<N, T> operator- (const T s, const _xmatxGTX<N, T>& m)
    {
		_xmatxGTX<N, T> result;
		for(int i = 0; i < N; ++i)
			result[i] = s - m[i];
		return result;
    }
/*
    template <int N, typename T>
    inline tvec4<T> operator- (const _xmatxGTX<N, T>& m, const tvec4<T>& v)
    {

    }

    template <int N, typename T>
    inline tvec4<T> operator- (const tvec4<T>& v, const _xmatxGTX<N, T>& m)
    {

    }
*/
    template <int N, typename T>
    inline _xmatxGTX<N, T> operator- (const _xmatxGTX<N, T>& m1, const _xmatxGTX<N, T>& m2)
    {
		_xmatxGTX<N, T> result;
		for(int i = 0; i < N; ++i)
			result[i] = m1[i] - m2[i];
		return result;
    }

    template <int N, typename T>
    inline _xmatxGTX<N, T> operator* (const _xmatxGTX<N, T>& m, const T s)
    {
		_xmatxGTX<N, T> result;
		for(int i = 0; i < N; ++i)
			result[i] = m[i] * s;
		return result;
    }

    template <int N, typename T>
    inline _xmatxGTX<N, T> operator* (const T s, const _xmatxGTX<N, T>& m)
    {
		_xmatxGTX<N, T> result;
		for(int i = 0; i < N; ++i)
			result[i] = s * m[i];
		return result;
    }

    template <int N, typename T>
    inline _xvecxGTX<N, T> operator* (const _xmatxGTX<N, T>& m, const _xvecxGTX<N, T>& v)
    {
		_xvecxGTX<N, T> result(T(0));
		for(int j = 0; j < N; ++j)
		for(int i = 0; i < N; ++i)
			result[j] += m[i][j] * v[i];
		return result;
    }

    template <int N, typename T>
    inline _xvecxGTX<N, T> operator* (const _xvecxGTX<N, T>& v, const _xmatxGTX<N, T>& m)
    {
		_xvecxGTX<N, T> result(T(0));
		for(int j = 0; j < N; ++j)
		for(int i = 0; i < N; ++i)
			result[j] += m[j][i] * v[i];
		return result;
    }

    template <int N, typename T>
    inline _xmatxGTX<N, T> operator* (const _xmatxGTX<N, T>& m1, const _xmatxGTX<N, T>& m2)
    {
        _xmatxGTX<N, T> Result(T(0));
		for(int k = 0; k < N; ++k)
		for(int j = 0; j < N; ++j)
		for(int i = 0; i < N; ++i)
			Result[k][j] += m1[i][j] * m2[k][i];
        return Result;
    }

    template <int N, typename T>
    inline _xmatxGTX<N, T> operator/ (const _xmatxGTX<N, T>& m, const T s)
    {
		_xmatxGTX<N, T> result;
		for(int i = 0; i < N; ++i)
			result[i] = m[i] / s;
		return result;
    }

    template <int N, typename T>
    inline _xmatxGTX<N, T> operator/ (const T s, const _xmatxGTX<N, T>& m)
    {
		_xmatxGTX<N, T> result;
		for(int i = 0; i < N; ++i)
			result[i] = s / m[i];
		return result;
    }

    template <int N, typename T>
    inline _xvecxGTX<N, T> operator/ (const _xmatxGTX<N, T>& m, const _xvecxGTX<N, T>& v)
    {
        return m._inverse() * v;
    }

    template <int N, typename T>
    inline _xvecxGTX<N, T> operator/ (const _xvecxGTX<N, T>& v, const _xmatxGTX<N, T>& m)
    {
        return v * m._inverse();
    }
 
    template <int N, typename T>
    inline _xmatxGTX<N, T> operator/ (const _xmatxGTX<N, T>& m1, const _xmatxGTX<N, T>& m2)
    {
        return m1 * m2._inverse();
    }

	// Unary constant operators
    template <int N, typename T>
    inline const _xmatxGTX<N, T> operator- (const _xmatxGTX<N, T>& m)
    {
		_xmatxGTX<N, T> result;
		for(int i = 0; i < N; ++i)
			result[i] = -m[i];
		return result;
    }

    template <int N, typename T>
    inline const _xmatxGTX<N, T> operator++ (const _xmatxGTX<N, T>& m, int) 
    {
		_xmatxGTX<N, T> result;
		for(int i = 0; i < N; ++i)
			result[i] = m[i] + T(1);
		return result;
    }

    template <int N, typename T>
    inline const _xmatxGTX<N, T> operator-- (const _xmatxGTX<N, T>& m, int) 
    {
		_xmatxGTX<N, T> result;
		for(int i = 0; i < N; ++i)
			result[i] = m[i] - T(1);
		return result;
    }
}//namespace detail

	// Matrix Functions
	template <int N, typename T> 
	inline detail::_xmatxGTX<N, T> matrixCompMultGTX(const detail::_xmatxGTX<N, T>& x, const detail::_xmatxGTX<N, T>& y)
	{
        detail::_xmatxGTX<N, T> result;
        for(int j = 0; j < N; ++j)
            for(int i = 0; i < N; ++i)
                result[j][i] = x[j][i] * y[j][i];
        return result;
	}

	template <int N, typename T> 
	inline detail::_xmatxGTX<N, T> outerProductGTX(const detail::_xvecxGTX<N, T>& c, const detail::_xvecxGTX<N, T>& r)
	{
		detail::_xmatxGTX<N, T> result;
		for(int j = 0; j < N; ++j)
		for(int i = 0; i < N; ++i)
			result[j][i] = c[i] * r[j];
        return result;
	}

	template <int N, typename T> 
	inline detail::_xmatxGTX<N, T> transposeGTX(const detail::_xmatxGTX<N, T>& m)
	{
        detail::_xmatxGTX<N, T> result;
		for(int j = 0; j < N; ++j)
		for(int i = 0; i < N; ++i)
			result[j][i] = m[i][j];
		return result;
	}

    template <int N, typename T>
    inline T determinantGTX(const detail::_xmatxGTX<N, T>& m)
    {

    }

	template <int N, typename T> 
	inline detail::_xmatxGTX<N, T> inverseTransposeGTX(const detail::_xmatxGTX<N, T>& m)
	{
		
	}

	template <int N, typename T> 
	inline void columnGTX(detail::_xmatxGTX<N, T>& m, int ColIndex, const detail::_xvecxGTX<N, T>& v)
	{
		m[ColIndex] = v;
	}

	template <int N, typename T> 
	inline void rowGTX(detail::_xmatxGTX<N, T>& m, int RowIndex, const detail::_xvecxGTX<N, T>& v)
	{
		for(int i = 0; i < N; ++i)
			m[i][RowIndex] = v[i];
	}

	template <int N, typename T> 
	inline detail::_xvecxGTX<N, T> columnGTX(const detail::_xmatxGTX<N, T>& m, int ColIndex)
	{
		return m[ColIndex];
	}

	template <int N, typename T> 
	inline detail::_xvecxGTX<N, T> rowGTX(const detail::_xmatxGTX<N, T>& m, int RowIndex)
	{
		detail::_xvecxGTX<N, T> v;
		for(int i = 0; i < N; ++i)
			v[i] = m[i][RowIndex];
		return v;
	}
} //namespace glm
