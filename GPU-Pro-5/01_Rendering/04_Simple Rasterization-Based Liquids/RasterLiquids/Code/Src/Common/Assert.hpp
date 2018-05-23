#ifndef		__COMMON_ASSERT_HPP__
#define		__COMMON_ASSERT_HPP__

#ifndef	ASSERT

#ifdef _DEBUG
	#include	<assert.h>
	#define ASSERT(e, msg)  (void)(e);assert(e && msg)
#else
	#define ASSERT(e, msg) (void)(e); (void)(msg);
#endif
#endif

#endif	
