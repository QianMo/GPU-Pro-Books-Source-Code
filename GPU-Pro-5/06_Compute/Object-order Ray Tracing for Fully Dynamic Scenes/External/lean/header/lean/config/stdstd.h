/*****************************************************/
/* lean Dependency Config       (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_CONFIG_STDSTD
#define LEAN_CONFIG_STDSTD

// Disable warnings
#define _CRT_SECURE_NO_WARNINGS 1 // Using _standard_ library
#define _SCL_SECURE_NO_WARNINGS 1

#ifdef _MSC_VER

#pragma warning(push)
// Some warnings won't go, even with _CRT_SECURE_NO_WARNINGS defined
#pragma warning(disable : 4996)

#endif

#endif