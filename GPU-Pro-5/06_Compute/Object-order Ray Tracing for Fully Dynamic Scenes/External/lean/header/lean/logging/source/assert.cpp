#ifdef LEAN_BUILD_LIB
#include "../../depconfig.h"
#endif

#include "../../lean.h"
#include <cstdlib>
#include <Windows.h>

namespace lean
{

// Returns true if execution should be continued normally.
LEAN_ALWAYS_LINK bool maybeIgnoreAssertion(const char *message, const char *file, unsigned int line)
{
	char fullText[32768];
	
	sprintf_s(fullText, "%s(%u): assertion violated:\n%s\n\n", file, line, message);
	::OutputDebugStringA(fullText);

	sprintf_s(fullText, "Assertion Violated!\n\nIn: %s(%u):\n\n%s\n\n\n(Press Retry to debug the application)", file, line, message);
	int promptResult = ::MessageBoxA(NULL, fullText, "Debug Assertion Violated", MB_ABORTRETRYIGNORE | MB_ICONERROR | MB_SETFOREGROUND | MB_TASKMODAL);
	if (promptResult == IDABORT)
		abort();
	return (promptResult == IDIGNORE);
}

} // namespace
