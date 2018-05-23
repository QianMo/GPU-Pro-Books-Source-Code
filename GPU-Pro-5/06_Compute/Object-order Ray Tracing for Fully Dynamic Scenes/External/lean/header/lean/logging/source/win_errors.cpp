#ifdef LEAN_BUILD_LIB
#include "../../depconfig.h"
#endif

#include "../win_errors.h"
#include "../../strings/utility.h"
#include <windows.h>

// Gets an error message describing the last WinAPI error that occurred. Returns the number of characters used.
LEAN_ALWAYS_LINK size_t lean::logging::get_last_win_error_msg(utf16_t *buffer, size_t maxCount)
{
	size_t count = ::FormatMessageW(
		FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
		nullptr, ::GetLastError(),
		0,
		buffer, static_cast<DWORD>(maxCount),
		nullptr );

	if (count == 0)
		count = wcsmcpy(buffer, L"Error unknown.", maxCount);
	else if (count >= maxCount)
		buffer[count = maxCount - 1] = 0;

	while (buffer[count - 1] == '\n' || buffer[count - 1] == '\r')
		buffer[--count] = 0;

	return count;
}
