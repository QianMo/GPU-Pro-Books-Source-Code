//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (C) Tobias Zirr.  All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////////

#pragma once

#ifndef D3DEFFECTSLITE_BEGIN_LOG_LINE
	#define D3DEFFECTSLITE_BEGIN_LOG_LINE "D3DEffectsLite: "
#endif

#ifndef D3DEFFECTSLITE_END_LOG_LINE
	#define D3DEFFECTSLITE_END_LOG_LINE "\n"
#endif

#ifndef D3DEFFECTSLITE_MAKE_LINE
	#define D3DEFFECTSLITE_MAKE_LINE(m) D3DEFFECTSLITE_BEGIN_LOG_LINE m D3DEFFECTSLITE_END_LOG_LINE
#endif

#ifndef D3DEFFECTSLITE_LOG_LINE
	#define D3DEFFECTSLITE_LOG_LINE(m) ::D3DEffectsLite::Log( D3DEFFECTSLITE_MAKE_LINE(m) )
#endif

namespace D3DEffectsLite
{

void Log(const char *message);

inline void LogLine(const char *message, const char *and1 = nullptr, const char *and2 = nullptr)
{
	Log(D3DEFFECTSLITE_BEGIN_LOG_LINE);
	Log(message);
	if (and1)
		Log(and1);
	if (and2)
		Log(and2);
	Log(D3DEFFECTSLITE_END_LOG_LINE);
}

struct LoggedError { };

inline void LogLineAndThrow(const char *message, const char *and1 = nullptr, const char *and2 = nullptr)
{
	LogLine(message, and1, and2);
	throw LoggedError();
}

} // namespace