#ifndef		_H_COMMON_ASSERT_
#define		_H_COMMON_ASSERT_

#include	<assert.h>

#ifndef Unconstify
#define Unconstify g_Unconstify
template<typename T> inline const T& g_Unconstify(const T& x)
{
    return x;
}
#endif

#ifndef	ASSERT
	#define ASSERT(e, msg)  (void)(e);assert(e && msg)//do { (void)(e); } while(Unconstify(false))
#endif


template<int> struct CompileTimeError;
template<> struct CompileTimeError<true> {};
#ifndef STATIC_CHECK
	#define STATIC_CHECK(expr) \
	{ CompileTimeError<((expr) != 0)> inst; (void)inst; } 
#endif

#endif	// #ifndef		_H_COMMON_ASSERT_
