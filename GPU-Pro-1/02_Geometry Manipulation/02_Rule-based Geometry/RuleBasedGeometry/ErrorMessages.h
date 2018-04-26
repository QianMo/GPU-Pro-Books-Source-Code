/*
**********************************************************************
 * Demo program for
 * Rule-based Geometry Synthesis in Real-time
 * ShaderX 8 article.
 *
 * @author: Milan Magdics
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted for any non-commercial programs.
 * 
 * Use it for your own risk. The author(s) do(es) not take
 * responsibility or liability for the damages or harms caused by
 * this software.
**********************************************************************
*/

#pragma once

/////////////////////////////////////////////////////////
// some very basic error message macros to support 
//   code generation in both console and gui application
/////////////////////////////////////////////////////////

#define WINDOWS_APP

#ifdef WINDOWS_APP
#define ERROR_MSG(SHORT,LONG)				MessageBoxA( NULL, LONG, SHORT, MB_OK )
#else
	#ifdef CONSOLE_APP
	#define ERROR_MSG(SHORT,LONG)			std::cerr << LONG << std::endl;
	#endif
#endif

#define TO_C_STRING(STRING)					String(STRING).c_str()
