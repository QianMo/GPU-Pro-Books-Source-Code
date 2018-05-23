
#ifndef __MATH_HPP__
#define __MATH_HPP__

#include <cmath>
#include <Common\Assert.hpp>
#include <Common\Types.hpp>

template<bool> struct CTAssert;
template<> struct CTAssert<true> {}; 

namespace M
{
	const float32 Pi = 3.14159265358979323846f;
	const float64 Pi64 = 3.14159265358979323846f;

	template<class T>
	void ThreeLaw(const T _a1, const T _d1, T& _a2, const T _d2)
	{
		_a2 = _a1*_d2/_d1; 
	}

	#ifndef EPSILON
	#define EPSILON 0.000001f
	#endif

	template<class T>
	inline T Min(const T& _a, const T& _b){return _a<_b ? _a : _b;}

	template<class T>
	inline T Max(const T& _a, const T& _b){return _a>_b ? _a : _b; }

	///<
	template<class T>
	T Abs(T _x){ return _x<0 ?  static_cast<T>(-1)*_x : _x; }

	template<class T>
	inline T Squared(const T _v){return _v*_v;}


/**************************************************************************************************/

	template<class T>
	T Divide(const T& _a, const T& _b)
	{
		if (M::Abs(_b)>static_cast<T>(EPSILON))
		{
			return _a/_b;
		}
		return 0;
	}


	/**************************************************************************************************/

	template< class T >
	inline T Clamp(const T& _x, const T& _min, const T& _max)
	{
		return M::Max(M::Min(_x,_max),_min);
	}

	/**************************************************************************************************/

	template<class T>
	inline T ClampVector(const T& _v, const T& _min, const T& _max)
	{
		T rV;
		for (int32 i=0; i<static_cast<int32>(_v.Size()); ++i)
			rV[i]=M::Clamp(_v[i],_min[i],_max[i]);

		return  rV;
	}

	/**************************************************************************************************/

	template<class T>
	inline const T Sign(const T val) 
	{
		return val<static_cast<T>(0) ? static_cast<T>(-1):static_cast<T>(1);
	}

	template<class T>
    T Cos(T _x){ return cos(_x); }

    template<class T>
    T Sin(T _x){ return sin(_x); }

	template<class F>
	F Expo(F _f, int32 _Num)
	{
		F fr=_f;
		for(int32 i=0;i<_Num-1;++i)
			fr*=_f;

		return fr;
	}

	template<int32 M, class T=float32>
	struct SExpo
	{
		static T Eval(const T& _x){return _x*SExpo<M-1,T>::Eval(_x);}
	};

	template<class T>
	struct SExpo<1,T>
	{
		static T Eval(const T& _x){return _x;}
	};

	inline float32 Pow(const float32 _x, const int32 M){ return ((M!=1) ? _x*Pow(_x,M-1) : 1.0f);}

	template<class T>
	inline const T Sqrt(const T _x)
	{
		return static_cast<T>(std::sqrt(static_cast<float64>(_x)));
	}

	/**************************************************************************************************/

	template<class T>
	inline T Exponent(T _v, int32 _n)
	{ 
		ASSERT(_n>-1, "Can't have negative values !");
		if(_n==0)
		{return static_cast<T>(1.0f);}
		else
		{
			T _v0=_v;
			for(int32 i=0; i<_n-1; ++i)
			{
				_v*=_v0;
			}
			return _v;
		} 
	}


	template<class T>
	inline T Cubed(const T _v){return _v*_v*_v;}

	

	/**************************************************************************************************/
	
	template<class T>
	float32 Rest(float32 _val)
	{
		int32 index=(int32)_val;
		return _val - Sign(_val)*(float32)index;
	}
	
	/**************************************************************************************************/


	struct IndexRest
	{
		int32		m_index;
		float32		m_rest;

		IndexRest(float32 _val) 
		{ 
			m_index = (int32)_val;
			m_rest	= _val-((float32)m_index);
		}	

		IndexRest(const float32 _t, const int32 _size) 
		{ 
			float32 sV=_t*(float32)_size;
			m_index = M::Min((int32)sV, _size);			
			m_rest	= sV-((float32)m_index);
		}
	};

	/**************************************************************************************************/

	struct FloorRest
	{
		int32		m_floor;
		float32		m_rest;

		explicit FloorRest(float32 _val)
		{
			m_floor = (int32)floor(_val);
			m_rest	= M::Abs(Rest<float32>(_val));
		}
	};

    /**************************************************************************************************/

    template<class T>
    inline T FloorVector(const T& _v)
    {
        T rV;
        for (uint32 i=0; i<static_cast<uint32>(_v.Size()); ++i)
            rV[i]=static_cast<T::value_type>( static_cast< FloatTypeTraits<T::value_type >::IntType >(_v[i]));

        return  rV;
    }

}


namespace Color
{
	///< Pixel to intensity
	template<class T>
	float Intensity(T _r, T _g, T _b)
	{
		return (T)0.3f*_r+(T)0.59f*_g+(T)0.11f*_b;
	}
	///<
	template<class T>
	T ToReal(T _byte)
	{
		return ((_byte/(T)255.0f)-(T)0.5f)*(T)2.0f;
	}
	///<
	template<class T>
	T ToByte(T _real)
	{
		return (_real/(T)2.0f + (T)0.5f)*(T)255.0f;
	}
}


#endif
