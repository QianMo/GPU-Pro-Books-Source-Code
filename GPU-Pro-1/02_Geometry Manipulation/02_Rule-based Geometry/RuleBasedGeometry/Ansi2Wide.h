/*
****************************************************************************
 * class CAnsi2Wide - implements ANSI to Unicode string conversion
 * ( found at: http://answers.google.com/answers/threadview/id/739419.html )
****************************************************************************
*/

#pragma once

class CAnsi2Wide
{
public:
	CAnsi2Wide(LPCSTR lpaszString);
	~CAnsi2Wide();
	operator LPCWSTR() const;

protected:

private:
	CAnsi2Wide();
	LPWSTR m_lpwszString;
};
