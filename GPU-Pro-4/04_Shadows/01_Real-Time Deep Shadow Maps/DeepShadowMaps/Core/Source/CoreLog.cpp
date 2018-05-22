#include "Core.h"

FILE* CoreLog::out = NULL;
CoreLog::INITTYPE CoreLog::initiated = NOT_INITIATED;

#ifdef _DEBUG
#define CORELOG_ALWAYS_OUTPUT
#endif

// Initiates the Log
CoreResult CoreLog::Init(std::wstring fileName)
{
	#ifdef CORELOG_ALWAYS_OUTPUT
	if(!CoreLog::initiated)
	{	
		std::wstring newFileName = fileName + L".html";
		
		if(!_wfopen_s(&out, newFileName.c_str(), L"w+"))
		{
			CoreLog::initiated = INITIATED_FILE_HTML;
			fwprintf(out, L"<head><meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\"><title>Log</title><meta name=\"Microsoft Theme\" content=\"expeditn 011\"></head>");
			fwprintf(out, L"<center><h1>CoreLog</h1><br></center>");
			fflush(out);
			return CORE_OK;
		}
		else
			return CORE_MISC_ERROR;
	}
	else
		return CORE_ALREADY_INITIALIZED;
	
	#endif
	return CORE_OK;
}

// Initiates the Log
void CoreLog::InitStdErr()
{
	#ifdef CORELOG_ALWAYS_OUTPUT
	out = stderr;
	CoreLog::initiated = INITIATED_STDERR;
	fwprintf(out, L"CoreLog initiated\n");
	fflush(out);
	#endif
}

// Initiates the Log
void CoreLog::InitWinDebug()
{
	#ifdef CORELOG_ALWAYS_OUTPUT
	out = NULL;
	CoreLog::initiated = INITIATED_WINDEBUG;
	OutputDebugString(L"CoreLog initiated\n");
	#endif
}

// Stop the logging
void CoreLog::Stop()
{
	#ifdef CORELOG_ALWAYS_OUTPUT
	if(CoreLog::initiated)
	{
		CoreLog::Information(L"CoreLog stopped\n");
		if(initiated != INITIATED_WINDEBUG)
		{
			if(initiated == INITIATED_FILE_HTML)
			{
				fwprintf(out, L"</html>");
				fclose(out);
			}
			out = NULL;
		}
		CoreLog::initiated = NOT_INITIATED;
	}
	#endif
}

// Prints an information
void CoreLog::Information(std::wstring text, ...)
{
	#ifdef CORELOG_ALWAYS_OUTPUT
	va_list list;

	if(CoreLog::initiated != INITIATED_WINDEBUG)
	{
		if(initiated == INITIATED_FILE_HTML)
			fwprintf(out, L"<br><font color=\"#008000\">Information:  </font>");
		else
			fwprintf(out, L"Core Information: ");
		va_start(list, text);
		vfwprintf(out, text.c_str(), list);
		va_end(list);
		if(initiated != INITIATED_FILE_HTML)
			fwprintf(out, L"\n");
		fflush(out);
	}
	else
	{
		WCHAR buf[4096];

		int num = swprintf(buf, 4096, L"Core Information: ");
		va_start(list, text);
		int num2 = vswprintf(&buf[num], 4096 - num, text.c_str(), list);
		va_end(list);
		swprintf(&buf[num + num2], 4096 - num - num2, L"\n");
		OutputDebugString(buf);
	}
	
	#endif
}

void CoreLog::Information(std::string text, ...)
{
	#ifdef CORELOG_ALWAYS_OUTPUT
	va_list list;

	if(CoreLog::initiated != INITIATED_WINDEBUG)
	{
		if(initiated == INITIATED_FILE_HTML)
			fwprintf(out, L"<br><font color=\"#008000\">Information:  </font>");
		else
			fwprintf(out, L"Core Information: ");
		va_start(list, text);
		vfprintf(out, text.c_str(), list);
		va_end(list);
		if(initiated != INITIATED_FILE_HTML)
			fwprintf(out, L"\n");
		fflush(out);
	}
	else
	{
		char buf[4096];

		int num = sprintf_s(buf, 4096, "Core Information: ");
		va_start(list, text);
		int num2 = vsprintf_s(&buf[num], 4096 - num, text.c_str(), list);
		va_end(list);
		sprintf_s(&buf[num + num2], 4096 - num - num2, "\n");
		OutputDebugStringA(buf);
	}
	
	#endif
}

// Prints an error and calls exit(1)
void CoreLog::Error(std::wstring text, ... )
{
	va_list list;

	#ifdef CORELOG_ALWAYS_OUTPUT
	if(CoreLog::initiated)
	{
		if(CoreLog::initiated != INITIATED_WINDEBUG)
		{
			if(initiated == INITIATED_FILE_HTML)
				fwprintf(out, L"<br><font color=\"#FF0000\">Error: </font>");
			else
				fwprintf(out, L"Core Error: ");
			va_start(list, text);
			vfwprintf(out, text.c_str(), list);
			va_end(list);
			if(initiated != INITIATED_FILE_HTML)
				fwprintf(out, L"\n");
			fflush(out);
		}
		else
		{
			WCHAR buf[4096];

			int num = swprintf(buf, 4096, L"Core Error: ");
			va_start(list, text);
			int num2 = vswprintf(&buf[num], 4096 - num, text.c_str(), list);
			va_end(list);
			swprintf(&buf[num + num2], 4096 - num - num2, L"\n");
			
			OutputDebugString(buf);
		}
		exit(1);
	}
	else
	{
		MessageBox(NULL, text.c_str(), L"Critical Error, please call CoreLog::Init!", MB_ICONERROR + MB_OK);
		exit(1);
	}

	#else
	
	WCHAR boxtext[1000];
	va_start(list, boxtext);
	vswprintf_s(boxtext, 1000, text.c_str(), list);
	va_end(list);

	MessageBox(NULL, boxtext, L"Critical Error. Application is shutting down!", MB_ICONERROR + MB_OK);
	exit(1);
	#endif
}

void CoreLog::Error(std::string text, ... )
{
	va_list list;

	#ifdef CORELOG_ALWAYS_OUTPUT
	if(CoreLog::initiated)
	{
		if(CoreLog::initiated != INITIATED_WINDEBUG)
		{
			if(initiated == INITIATED_FILE_HTML)
				fwprintf(out, L"<br><font color=\"#FF0000\">Error: </font>");
			else
				fwprintf(out, L"Core Error: ");
			va_start(list, text);
			vfprintf(out, text.c_str(), list);
			va_end(list);
			if(initiated != INITIATED_FILE_HTML)
				fwprintf(out, L"\n");
			fflush(out);
		}
		else
		{
			char buf[4096];

			int num = sprintf_s(buf, 4096, "Core Error: ");
			va_start(list, text);
			int num2 = vsprintf_s(&buf[num], 4096 - num, text.c_str(), list);
			va_end(list);
			sprintf_s(&buf[num + num2], 4096 - num - num2, "\n");
			
			OutputDebugStringA(buf);
		}
		exit(1);
	}
	else
	{
		MessageBoxA(NULL, text.c_str(), "Critical Error, please call CoreLog::Init!", MB_ICONERROR + MB_OK);
		exit(1);
	}

	#else
	
	char boxtext[1000];
	va_start(list, boxtext);
	vsprintf_s(boxtext, 1000, text.c_str(), list);
	va_end(list);

	MessageBoxA(NULL, boxtext, "Critical Error. Application is shutting down!", MB_ICONERROR + MB_OK);
	exit(1);
	#endif
}

// Writes a vector
void CoreLog::WriteVector3(CoreVector3 &v)
{
	#ifdef CORELOG_ALWAYS_OUTPUT
	if(CoreLog::initiated)
	{
		if(CoreLog::initiated != INITIATED_WINDEBUG)
		{
			fwprintf(out, L"Vector: x=%f y=%f z=%f", v.x, v.y, v.z);
			if(initiated == INITIATED_FILE_HTML)
				fwprintf(out, L"<br />");
			else
				fwprintf(out, L"\n");
		}
		else
		{
			WCHAR buf[4096];

			swprintf(buf, 4096, L"Vector: x=%f y=%f z=%f\n", v.x, v.y, v.z);
			OutputDebugString(buf);
		}
	}

	#endif
}

// writes a matrix
/*void CoreLog::WriteMatrix( enMatrix &m )
{
	#ifdef CORELOG_ALWAYS_OUTPUT
	if( CoreLog::Initiated )
	{
		char s[1000];
		sprintf_s( s,"MatrixOut:<br>_11=%f _12=%f _13=%f _14=%f<br>_21=%f _22=%f _23=%f _24=%f<br>_31=%f _32=%f _33=%f _34=%f<br>_41=%f _42=%f _43=%f _44=%f", m._11, m._12, m._13, m._14, m._21, m._22, m._23, m._24, m._31, m._32, m._33, m._34, m._41, m._42, m._43, m._44 );
		CoreLog::Information( s );
	}

	#endif
}*/

// writes a plane
/*void CoreLog::WritePlane( enPlane &p )
{
	#ifdef CORELOG_ALWAYS_OUTPUT

	if( CoreLog::Initiated )
	{
		char s[500];
		sprintf_s(s,"PlaneOut: <br>n.x = %f <br> n.y = %f <br> n.z = %f <br> d = %f", p.a, p.b, p.c, p.d );
		CoreLog::Information( s );
	}

	#endif
}*/

