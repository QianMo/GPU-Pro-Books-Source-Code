#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <stdarg.h>
#include <memory>


/* Prints to output console
Arg 1: LogLevel
Arg 2: fmt string
*/
#define PRINT(...) Log::LogText(__FUNCTION__, __LINE__, ##__VA_ARGS__)

namespace Log
{
	enum class LogLevel : unsigned
	{
		FATAL_ERROR,
		NON_FATAL_ERROR,
		WARNING,
		SUCCESS,
		DEBUG_PRINT,
		INIT_PRINT,
		START_PRINT,
		PINK_PRINT,
		NOLEVEL,
		HELP_PRINT,
		ASSERT,

		NUM_LOGLEVELS
	};

	void InitLogColors();
	void LogText(std::string p_func, int p_line, LogLevel p_vLevel, const char* p_format, ...);
}