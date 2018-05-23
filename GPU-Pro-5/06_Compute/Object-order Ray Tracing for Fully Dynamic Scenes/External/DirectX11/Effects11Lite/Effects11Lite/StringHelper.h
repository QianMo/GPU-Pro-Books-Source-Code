//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (C) Tobias Zirr.  All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////////

#pragma once

#include <string>

inline size_t strcnt(const char *str, char c)
{
	size_t count = 0;
	while (*str)
		count += (*str++ == c);
	return count;
}

inline const char* strnns(const char *str)
{
	while (*str && isspace(*str))
		++str;
	return str;
}

inline const char* strnc(const char *str, char c)
{
	while (*str && *str != c)
		++str;
	return str;
}

inline const char* strnd(const char *str)
{
	while (*str && !isdigit(*str))
		++str;
	return str;
}

inline const char* strna(const char *str)
{
	while (*str && !isalpha(*str))
		++str;
	return str;
}

inline const char* strnna(const char *str)
{
	while (*str && isalpha(*str))
		++str;
	return str;
}
