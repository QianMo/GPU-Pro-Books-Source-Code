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

#pragma once

class UnicodeString
{
	unsigned int length;
	wchar_t* s;
public:
	UnicodeString(const wchar_t* a);
	UnicodeString(const char* m);
	UnicodeString(const UnicodeString& o);
	~UnicodeString(void);

	bool operator<(const UnicodeString& o) const;
	
	operator const wchar_t*() const;
};
