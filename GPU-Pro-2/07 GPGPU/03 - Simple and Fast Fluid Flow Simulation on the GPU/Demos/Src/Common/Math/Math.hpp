

#ifndef __MATH_HPP__
#define __MATH_HPP__


#include <cmath>
#include <Common\System\Assert.hpp>
#include <Common\System\Types.hpp>

#ifndef PI
#define PI       3.14159265358979323846
#endif

///<
template<class Type>
inline Type Min(const Type& _a, const Type& _b){return _a<_b ? _a : _b;}

///<
template<class Type>
Type Max(const Type& _a, const Type& _b){return _a>_b ? _a : _b; }

///<
template<int32 M>
struct Expo
{
	static float32 Eval(const float32 _x){return _x*Expo<M-1>::Eval(_x);}
};

template<>
struct Expo<1>
{
	static float32 Eval(const float32 _x){return _x;}
};

///<
template<class T>
inline const T Sqrt(const T _x)
{
	return static_cast<T>(std::sqrt(static_cast<float64>(_x)));
}

///<
template<class T>
inline T Squared( const T val ){ return val*val; }

///<
template<class T>
T Divide(const T& _a, const T& _b)
{
	if (_b!=static_cast<T>(0))
	{
		return _a/_b;
	}
	return 0;
}

///<
template<class T>
inline const T Sign(const T val) 
{
	return val<static_cast<T>(0) ? static_cast<T>(-1):static_cast<T>(1);
}

///<
template<class Type>
float32 Rest(float32 _val)
{
	int32 index=(int32)_val;
	return _val - Sign(_val)*(float32)index;
}

///<
struct IndexRest
{
	int32		m_index;
	float32		m_rest;

	explicit IndexRest(float32 _val) 
	{ 
		m_index = (int32)_val;
		m_rest	= Rest<float32>(_val);
	}	
};

///<
struct FloorRest
{
	int32		m_floor;
	float32		m_rest;

	explicit FloorRest(float32 _val)
	{
		m_floor = (int32)floor(_val);
		m_rest	= fabs(Rest<float32>(_val));
	}
};

///<
template< class Type >
inline Type Clamp(const Type& val, const Type& _min, const Type& _max)
{
	return Max(Min(val,_max),_min);
}

///<
template<class Type>
void CycleBase (Type& val, const Type& _min, const Type& _max)
{
	while (val >= _max)
	{
		val -= _max - _min;
	}
	while (val < _min)
	{
		val += _max - _min;
	}
}

///<
template<class Type>
class Cycle
{
	Type m_min,m_max;


public:

	explicit Cycle( const Type& _min, const Type& _max ) : m_min(_min),m_max(_max){}

	void operator()(Type& _x) const
	{
		CycleBase<Type>(_x, m_min,m_max);
	}
};

///<
const Cycle<float32> ZeroOneCycle(0.0f, 1.0f);


#endif
