#pragma once


#ifdef MAXEST_FRAMEWORK_DEBUG
	#define ASSERT(condition)			NEssentials::Assert(condition)
	#define ASSERT_FUNCTION(function)	ASSERT((function) == true)
#else
	#define ASSERT(condition)			do {} while (0);
	#define ASSERT_FUNCTION(condition)	(condition)
#endif

#define ASSERT_RELEASE(condition)			NEssentials::Assert(condition)
#define ASSERT_FUNCTION_RELEASE(function)	ASSERT_RELEASE((function) == true)


namespace NEssentials
{
	inline void Assert(bool condition)
	{
		#ifdef MAXEST_FRAMEWORK_WINDOWS
			do
			{
				if (!(condition))
				{
					__debugbreak();
				}
			}
			while (0);
		#endif
	}
}
