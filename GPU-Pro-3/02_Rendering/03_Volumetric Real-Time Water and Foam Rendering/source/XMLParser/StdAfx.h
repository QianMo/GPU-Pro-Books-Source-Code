// stdafx.h : include file for standard system include files,
//  or project specific include files that are used frequently, but
//      are changed infrequently
//

#if !defined(AFX_STDAFX_H__630E5A6D_77AE_4E4D_B006_DDFEB9100C68__INCLUDED_)
#define AFX_STDAFX_H__630E5A6D_77AE_4E4D_B006_DDFEB9100C68__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#define WIN32_LEAN_AND_MEAN		// Exclude rarely-used stuff from Windows headers

#include <atlbase.h>
#include <tchar.h>
#include <winsock2.h>


#ifdef _DEBUG

#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>

#define DEBUG_NEW   new( _NORMAL_BLOCK, __FILE__, __LINE__)
#endif

// TODO: reference additional headers your program requires here

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_STDAFX_H__630E5A6D_77AE_4E4D_B006_DDFEB9100C68__INCLUDED_)
