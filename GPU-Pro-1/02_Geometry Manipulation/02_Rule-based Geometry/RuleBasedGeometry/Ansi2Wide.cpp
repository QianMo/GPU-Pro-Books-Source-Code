/*
****************************************************************************
 * class CAnsi2Wide - implements ANSI to Unicode string conversion
 * ( found at: http://answers.google.com/answers/threadview/id/739419.html )
****************************************************************************
*/

#include "DXUT.h"
#include "Ansi2Wide.h"

// Default constructor
CAnsi2Wide::CAnsi2Wide():
	m_lpwszString(NULL)
{
}

// Constructs object and convert lpaszString to Unicode
CAnsi2Wide::CAnsi2Wide(LPCSTR lpaszString):
	m_lpwszString(NULL)
{
	int nLen = ::lstrlenA(lpaszString) + 1;
	m_lpwszString = new WCHAR[nLen];
	if (m_lpwszString == NULL)
	{
		return;
	}

	memset(m_lpwszString, 0, nLen * sizeof(WCHAR));

	if (::MultiByteToWideChar(CP_ACP, 0, lpaszString, nLen, m_lpwszString, nLen) == 0)
	{
		// Conversation failed
		return;
	}
}

// Destructor
CAnsi2Wide::~CAnsi2Wide()
{
	if (m_lpwszString != NULL)
	{
		delete [] m_lpwszString;
	}
}

// Returns converted string
CAnsi2Wide::operator LPCWSTR() const
{
	return m_lpwszString;
}
