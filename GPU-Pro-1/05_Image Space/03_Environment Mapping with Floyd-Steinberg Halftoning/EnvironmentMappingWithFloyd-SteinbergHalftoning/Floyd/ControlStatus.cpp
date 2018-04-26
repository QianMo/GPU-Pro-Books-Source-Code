#include "DXUT.h"
#include "ControlStatus.h"

ControlStatus::ControlStatus()
{
	for(unsigned int i=0; i<0xff; i++)
		keyPressed[i] = false;
	mouseButtonPressed[0] = false;
	mouseButtonPressed[1] = false;
	mouseButtonPressed[2] = false;
	mousePosition = D3DXVECTOR3(0, 0, 0);
}

void ControlStatus::handleInput(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	if(uMsg == WM_KEYDOWN)
		keyPressed[wParam] = true;
	else if(uMsg == WM_KEYUP)
		keyPressed[wParam] = false;
	else if(uMsg == WM_KILLFOCUS)
	{
		for(unsigned int i=0; i<0xff; i++)
			keyPressed[i] = false;
	}
	else if(uMsg == WM_MOUSEMOVE)
	{
/*		POINT pixPos;
		if(GetCursorPos(&pixPos))
			mousePosition = D3DXVECTOR3((double)pixPos.x / screenWidth * 2.0 - 1.0, (double)pixPos.y / screenHeight * 2.0 - 1.0, 0);*/

		mousePosition = D3DXVECTOR3(
			(double)(lParam & 0xffff) / viewportWidth * 2.0 - 1.0,
			-(double)(lParam >> 16) / viewportHeight * 2.0 + 1.0, 0);

	}
}