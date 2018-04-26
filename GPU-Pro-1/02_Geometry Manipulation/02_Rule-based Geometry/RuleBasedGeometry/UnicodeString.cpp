/*
**********************************************************************
 * Demo program for
 * Rule-based Geometry Synthesis in Real-time
 * ShaderX 8 article.
 *
 * @author: Laszlo Szecsi
 * Used with permission.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted for any non-commercial programs.
 * 
 * Use it for your own risk. The author(s) do(es) not take
 * responsibility or liability for the damages or harms caused by
 * this software.
**********************************************************************
*/

#include "DXUT.h"
#include "UnicodeString.h"

UnicodeString::UnicodeString(const wchar_t* a)
{
	length = wcslen(a);
	s = new wchar_t[length+1];
	StringCchCopyW(s, length+1, a);
}

UnicodeString::UnicodeString(const char* m)
{
	length = MultiByteToWideChar(CP_ACP, 0, m, -1, NULL, 0)-1;
	s = new wchar_t[length+1];
	MultiByteToWideChar(CP_ACP, 0, m, -1, s, length+1);
}

UnicodeString::~UnicodeString(void)
{
	delete s;
}

UnicodeString::UnicodeString(const UnicodeString& o)
{
	length = o.length;
	s = new wchar_t[length+1];
	StringCchCopyW(s, length+1, o.s);
}

bool UnicodeString::operator<(const UnicodeString& o) const
{
	return wcscmp(s, o.s) < 0;
}

UnicodeString::operator const wchar_t*() const
{
	return s;
}