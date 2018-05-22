#pragma once

#if defined(_MSC_VER) && defined(_DEBUG)
	#define USE_MEMORY_TRACKING
#endif

#ifdef USE_MEMORY_TRACKING
	#include <crtdbg.h>
	#include <windows.h>

	#define DEBUG_CLIENTBLOCK   new( _CLIENT_BLOCK, __FILE__, __LINE__)
	#define new DEBUG_CLIENTBLOCK

	extern _CrtMemState mem_state;

	#define  SET_CRT_DEBUG_FIELD(a) \
		_CrtSetDbgFlag((a) | _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG))

	inline void DebugDumpMemory()
	{
		_CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_DEBUG);
		_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
		_CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_DEBUG);
	}

	inline void InitMemoryTracker()
	{
		SET_CRT_DEBUG_FIELD(_CRTDBG_LEAK_CHECK_DF);

		_CrtMemCheckpoint(&mem_state);

	#ifdef _DEBUG
		//atexit(DebugDumpMemory);
	#endif
	}
#else // USE_MEMORY_TRACKING
	#define DEBUG_CLIENTBLOCK
		inline void gxDbgDumpMem() {}
		inline void gxInitDebug() {}
#endif // USE_MEMORY_TRACKING
