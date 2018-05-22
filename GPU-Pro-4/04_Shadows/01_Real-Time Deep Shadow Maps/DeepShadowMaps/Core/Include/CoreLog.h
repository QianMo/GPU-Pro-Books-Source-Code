#pragma once

#include <stdio.h>
#include <stdarg.h>
#include <string>

#include "CoreError.h"
#include "CoreVector3.h"

//using namespace std;

class CoreLog
{
public:
	// Initiates the Log
	static CoreResult Init(std::wstring fileName);
	static void InitStdErr();
	// Initiates the Log
	static void InitWinDebug();
	// Stop the logging
	static void Stop();
	// Prints an information
	static void Information(std::wstring text, ...);
	static void Information(std::string text, ...);
	// Prints an Error and calls exit(1)
	static void Error(std::wstring text, ...);
	static void Error(std::string text, ...);
	// Writes a vector
	static void WriteVector3(CoreVector3 &v);
	// writes a matrix
	//static void WriteMatrix( enMatrix &m );
	// writes a plane
	//static void WritePlane( enPlane &p );
	// was th Log Initiated?
	inline static bool IsInitiated()		{ return initiated != NOT_INITIATED; }
private:
	enum INITTYPE
	{
		NOT_INITIATED = 0,
		INITIATED_FILE_HTML,
		INITIATED_STDERR,
		INITIATED_WINDEBUG
	};
	static FILE* out;
	static INITTYPE initiated;
};