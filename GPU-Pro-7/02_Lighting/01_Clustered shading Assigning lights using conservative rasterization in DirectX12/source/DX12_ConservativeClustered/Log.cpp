#include "Log.h"
#include <ctime>
#include <chrono>
#include <iomanip>
#include <windows.h>
#include <iosfwd>

#pragma warning( disable : 4996)

RECT secondaryRect = { 0,0,0,0 };

BOOL CALLBACK MonitorEnumProc(HMONITOR hMonitor, HDC hdc1, LPRECT lprcMonitor, LPARAM data)
{
	RECT rc = *lprcMonitor;

	MONITORINFOEX mInfo;
	mInfo.cbSize = sizeof(mInfo);
	::GetMonitorInfo(hMonitor, &mInfo);

	if (mInfo.dwFlags != MONITORINFOF_PRIMARY)
	{
		secondaryRect = mInfo.rcWork;
		return 0;
	}
	return 1;
}

enum class ConsoleColor : unsigned
{
	BLACK = 0,
	DARK_BLUE = 1,
	DARK_GREEN = 2,
	DARK_AQUA = 3,
	DARK_CYAN = 3,
	DARK_RED = 4,
	DARK_PURPLE = 5,
	DARK_PINK = 5,
	DARK_MANGENTA = 5,
	DARK_YELLOW = 6,
	DARK_WHITE = 7,
	GRAY = 8,
	BLUE = 9,
	GREEN = 10,
	AQUA = 11,
	CYAN = 11,
	RED = 12,
	PURPLE = 13,
	PINK = 13,
	MANGENTA = 13,
	YELLOW = 14,
	WHITE = 15
};

namespace ColorCMD
{
	HANDLE std_con_out;

	//Standard Output Handle
	bool colorprotect = false;

	ConsoleColor textcol, backcol, deftextcol, defbackcol;

	RECT m_secondaryRect;

	inline void UpdateColors()
	{
		CONSOLE_SCREEN_BUFFER_INFO csbi;
		GetConsoleScreenBufferInfo(std_con_out, &csbi);
		textcol = ConsoleColor(csbi.wAttributes & 15);
		backcol = ConsoleColor((csbi.wAttributes & 0xf0) >> 4);
	}

	inline void SetColor(ConsoleColor p_textcolor, ConsoleColor p_backcolor)
	{
		if (colorprotect && p_textcolor == p_backcolor)return;
		textcol = p_textcolor;backcol = p_backcolor;
		unsigned short wAttributes = ((unsigned int)backcol << 4) | (unsigned int)textcol;
		SetConsoleTextAttribute(std_con_out, wAttributes);
	}

	inline void SetTextColor(ConsoleColor p_textColor)
	{
		if (colorprotect && p_textColor == backcol)return;
		textcol = p_textColor;
		unsigned short wAttributes = ((unsigned int)backcol << 4) | (unsigned int)textcol;
		SetConsoleTextAttribute(std_con_out, wAttributes);
	}

	inline void SetBackColor(ConsoleColor p_backColor)
	{
		if (colorprotect && textcol == p_backColor)return;
		backcol = p_backColor;
		unsigned short wAttributes = ((unsigned int)backcol << 4) | (unsigned int)textcol;
		SetConsoleTextAttribute(std_con_out, wAttributes);
	}
	inline void UseSecondaryMonitor()
	{
		// init the memebr rectangle
		m_secondaryRect.top = 0;
		m_secondaryRect.left = 0;
		m_secondaryRect.right = 0;
		m_secondaryRect.bottom = 0;
		// enumerate
		::EnumDisplayMonitors(NULL, NULL, ::MonitorEnumProc, 0);
		// store the secondary monitor's rectangle in our member rectangle
		m_secondaryRect = secondaryRect;
	}

	inline void ConsoleColorInit()
	{
		std_con_out = GetStdHandle(STD_OUTPUT_HANDLE);

		COORD t;
		t.X = 900;
		t.Y = 10000;
		SetConsoleScreenBufferSize(std_con_out, t);
		UseSecondaryMonitor();
		if (((m_secondaryRect.right - m_secondaryRect.left) + (m_secondaryRect.bottom - m_secondaryRect.top)) == 0)
		{
			SetWindowPos(GetConsoleWindow(), HWND_NOTOPMOST, 0, 0, 900, 800, SWP_SHOWWINDOW);
		}
		else
		{
			SetWindowPos(GetConsoleWindow(), HWND_NOTOPMOST, m_secondaryRect.left, m_secondaryRect.top, (m_secondaryRect.right - m_secondaryRect.left), (m_secondaryRect.bottom - m_secondaryRect.top), SWP_SHOWWINDOW);
		}
		UpdateColors();
		deftextcol = textcol;
		defbackcol = backcol;
	}

	template<class elem, class traits>
	inline std::basic_ostream<elem, traits>& operator<<(std::basic_ostream<elem, traits>& os, ConsoleColor p_col)
	{
		os.flush();
		SetTextColor(p_col);
		return os;
	}

	template<class elem, class traits>
	inline std::basic_istream<elem, traits>& operator>>(std::basic_istream<elem, traits>& is, ConsoleColor p_col)
	{
		std::basic_ostream<elem, traits>* p = is.tie();
		if (p != NULL)p->flush();
		SetTextColor(p_col);
		return is;
	}

}

namespace Log
{
	static const std::pair<std::string, ConsoleColor> g_printInfo[] =
	{
		{ "FATAL_ERR  ", ConsoleColor::RED },
		{ "NON_FAT_ERR", ConsoleColor::RED },
		{ "WARNING    ", ConsoleColor::YELLOW },
		{ "SUCCESS    ", ConsoleColor::GREEN },
		{ "DEBUG_PRINT", ConsoleColor::WHITE },
		{ "INIT_PRINT ", ConsoleColor::BLUE },
		{ "START_PRINT", ConsoleColor::AQUA },
		{ "PINK_PRINT ", ConsoleColor::PINK },
		{ "NOLEVEL    ", ConsoleColor::GRAY },
		{ "HELP_PRINT ", ConsoleColor::DARK_GREEN },
		{ "ASSERT     ", ConsoleColor::RED },
	};

	void InitLogColors()
	{
		ColorCMD::ConsoleColorInit();
	}

	//////////////////////////////////////////////////////////////////////////
	//CONSOLE LOGGING
	//////////////////////////////////////////////////////////////////////////

	void WriteToConsole(std::string p_func, int p_line, LogLevel p_vLevel, std::string p_format, va_list p_args)
	{
		//Get local time for time stamp
		std::time_t t = std::time(nullptr);
		std::cout << std::put_time(std::localtime(&t), "%T");

		std::string output = "    " + g_printInfo[(unsigned)p_vLevel].first + "    " + p_format + "\n";

		if (p_vLevel == LogLevel::FATAL_ERROR)
			MessageBoxA(nullptr, output.c_str(), "Error", MB_ICONEXCLAMATION);

		//Set color 
		ColorCMD::SetColor(g_printInfo[(unsigned)p_vLevel].second, ColorCMD::defbackcol);
		std::cout << "";

		//Print text
		vprintf(output.c_str(), p_args);

		//Reset color to default
		ColorCMD::SetColor(ConsoleColor::WHITE, ColorCMD::defbackcol);
		std::cout << "";

	}

	void LogText(std::string p_func, int p_line, LogLevel p_vLevel, const char* p_format, ...)
	{
		va_list args;
		va_start(args, p_format);
		WriteToConsole(p_func, p_line, p_vLevel, p_format, args);
		va_end(args);
	}
}