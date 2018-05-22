///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2009-05-07
// Updated : 2009-05-07
// Licence : This source is under MIT License
// File    : glm/gtx/simd_vec4.inl
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm
{
	namespace detail
	{
		__m128 fvec4SIMD::one = _mm_set_ps1(1.f);

		//////////////////////////////////////
		// Implicit basic constructors

		inline fvec4SIMD::fvec4SIMD()
		{}

		inline fvec4SIMD::fvec4SIMD(__m128 const & Data) :
			Data(Data)
		{}

		inline fvec4SIMD::fvec4SIMD(fvec4SIMD const & v) :
			Data(v.Data)
		{}

		inline fvec4SIMD::fvec4SIMD(tvec4<float> const & v) :
			Data(_mm_set_ps(v.w, v.z, v.y, v.x))
		{}

		//////////////////////////////////////
		// Explicit basic constructors

		inline fvec4SIMD::fvec4SIMD(float const & s) :
			Data(_mm_set1_ps(s))
		{}

		inline fvec4SIMD::fvec4SIMD(float const & x, float const & y, float const & z, float const & w) :
	//		Data(_mm_setr_ps(x, y, z, w))
			Data(_mm_set_ps(w, z, y, x))
		{}

		inline fvec4SIMD::fvec4SIMD(float const v[4]) :
			Data(_mm_load_ps(v))
		{}

		//////////////////////////////////////
		// Swizzle constructors

		//fvec4SIMD(ref4<float> const & r);

		//////////////////////////////////////
		// Convertion vector constructors

		inline fvec4SIMD::fvec4SIMD(vec2 const & v, float const & s1, float const & s2) :
			Data(_mm_set_ps(s2, s1, v.y, v.x))
		{}

		inline fvec4SIMD::fvec4SIMD(float const & s1, vec2 const & v, float const & s2) :
			Data(_mm_set_ps(s2, v.y, v.x, s1))
		{}

		inline fvec4SIMD::fvec4SIMD(float const & s1, float const & s2, vec2 const & v) :
			Data(_mm_set_ps(v.y, v.x, s2, s1))
		{}

		inline fvec4SIMD::fvec4SIMD(vec3 const & v, float const & s) :
			Data(_mm_set_ps(s, v.z, v.y, v.x))
		{}

		inline fvec4SIMD::fvec4SIMD(float const & s, vec3 const & v) :
			Data(_mm_set_ps(v.z, v.y, v.x, s))
		{}

		inline fvec4SIMD::fvec4SIMD(vec2 const & v1, vec2 const & v2) :
			Data(_mm_set_ps(v2.y, v2.x, v1.y, v1.x))
		{}

		//inline fvec4SIMD::fvec4SIMD(ivec4SIMD const & v) :
		//	Data(_mm_cvtepi32_ps(v.Data))
		//{}

		//////////////////////////////////////
		// Unary arithmetic operators

		inline fvec4SIMD& fvec4SIMD::operator=(fvec4SIMD const & v)
		{
			this->Data = v.Data;
			return *this;
		}

		inline fvec4SIMD& fvec4SIMD::operator+=(float const & s)
		{
			this->Data = _mm_add_ps(Data, _mm_set_ps1(s));
			return *this;
		}

		inline fvec4SIMD& fvec4SIMD::operator+=(fvec4SIMD const & v)
		{
			this->Data = _mm_add_ps(this->Data , v.Data);
			return *this;
		}

		inline fvec4SIMD& fvec4SIMD::operator-=(float const & s)
		{
			this->Data = _mm_sub_ps(Data, _mm_set_ps1(s));
			return *this;
		}

		inline fvec4SIMD& fvec4SIMD::operator-=(fvec4SIMD const & v)
		{
			this->Data = _mm_sub_ps(this->Data , v.Data);
			return *this;
		}

		inline fvec4SIMD& fvec4SIMD::operator*=(float const & s)
		{
			this->Data = _mm_mul_ps(this->Data, _mm_set_ps1(s));
			return *this;
		}

		inline fvec4SIMD& fvec4SIMD::operator*=(fvec4SIMD const & v)
		{
			this->Data = _mm_mul_ps(this->Data , v.Data);
			return *this;
		}

		inline fvec4SIMD& fvec4SIMD::operator/=(float const & s)
		{
			this->Data = _mm_div_ps(Data, _mm_set1_ps(s));
			return *this;
		}

		inline fvec4SIMD& fvec4SIMD::operator/=(fvec4SIMD const & v)
		{
			this->Data = _mm_div_ps(this->Data , v.Data);
			return *this;
		}

		inline fvec4SIMD& fvec4SIMD::operator++()
		{
			this->Data = _mm_add_ps(this->Data , glm::detail::one);
			return *this;
		}

		inline fvec4SIMD& fvec4SIMD::operator--()
		{
			this->Data = _mm_sub_ps(this->Data , glm::detail::one);
			return *this;
		}

		//////////////////////////////////////
		// Swizzle operators

		//inline fvec4SIMD const fvec4SIMD::swizzle(int d, int c, int b, int a) const
		//{
		//	int const Mask = ((d << 6) | (c << 4) | (b << 2) | (a << 0));

		//	__m128 Data = _mm_shuffle_ps(this->Data, this->Data, Mask);
		//	return fvec4SIMD(Data);
		//}

		// operator+
		inline fvec4SIMD operator+ (fvec4SIMD const & v, float s)
		{
			return fvec4SIMD(_mm_add_ps(v.Data, _mm_set1_ps(s)));
		}

		inline fvec4SIMD operator+ (float s, fvec4SIMD const & v)
		{
			return fvec4SIMD(_mm_add_ps(_mm_set1_ps(s), v.Data));
		}

		inline fvec4SIMD operator+ (fvec4SIMD const & v1, fvec4SIMD const & v2)
		{
			return fvec4SIMD(_mm_add_ps(v1.Data, v2.Data));
		}

		//operator-
		inline fvec4SIMD operator- (fvec4SIMD const & v, float s)
		{
			return fvec4SIMD(_mm_sub_ps(v.Data, _mm_set1_ps(s)));
		}

		inline fvec4SIMD operator- (float s, fvec4SIMD const & v)
		{
			return fvec4SIMD(_mm_sub_ps(_mm_set1_ps(s), v.Data));
		}

		inline fvec4SIMD operator- (fvec4SIMD const & v1, fvec4SIMD const & v2)
		{
			return fvec4SIMD(_mm_sub_ps(v1.Data, v2.Data));
		}

		//operator*
		inline fvec4SIMD operator* (fvec4SIMD const & v, float s)
		{
			__m128 par0 = v.Data;
			__m128 par1 = _mm_set1_ps(s);
			return fvec4SIMD(_mm_mul_ps(par0, par1));
		}

		inline fvec4SIMD operator* (float s, fvec4SIMD const & v)
		{
			__m128 par0 = _mm_set1_ps(s);
			__m128 par1 = v.Data;
			return fvec4SIMD(_mm_mul_ps(par0, par1));
		}

		inline fvec4SIMD operator* (fvec4SIMD const & v1, fvec4SIMD const & v2)
		{
			return fvec4SIMD(_mm_mul_ps(v1.Data, v2.Data));
		}

		//operator/
		inline fvec4SIMD operator/ (fvec4SIMD const & v, float s)
		{
			__m128 par0 = v.Data;
			__m128 par1 = _mm_set1_ps(s);
			return fvec4SIMD(_mm_div_ps(par0, par1));
		}

		inline fvec4SIMD operator/ (float s, fvec4SIMD const & v)
		{
			__m128 par0 = _mm_set1_ps(s);
			__m128 par1 = v.Data;
			return fvec4SIMD(_mm_div_ps(par0, par1));
		}

		inline fvec4SIMD operator/ (fvec4SIMD const & v1, fvec4SIMD const & v2)
		{
			return fvec4SIMD(_mm_div_ps(v1.Data, v2.Data));
		}

		// Unary constant operators
		inline fvec4SIMD operator- (fvec4SIMD const & v)
		{
			return fvec4SIMD(_mm_sub_ps(_mm_setzero_ps(), v.Data));
		}

		inline fvec4SIMD operator++ (fvec4SIMD const & v, int)
		{
			return fvec4SIMD(_mm_add_ps(v.Data, glm::detail::one));
		}

		inline fvec4SIMD operator-- (fvec4SIMD const & v, int)
		{
			return fvec4SIMD(_mm_sub_ps(v.Data, glm::detail::one));
		}

	}//namespace detail

	namespace gtx{
	namespace simd_vec4
	{
		

	}//namespace simd_vec4
	}//namespace gtx
}//namespace glm
