#pragma once


namespace NSystem
{
	void ScreenSize(int& width, int& height);

	//

	void ScreenSize(int& width, int& height)
	{
		width = GetSystemMetrics(SM_CXSCREEN);
		height = GetSystemMetrics(SM_CYSCREEN);
	}
}
