#include <stdafx.h>
#include <INPUT_MANAGER.h>

bool INPUT_MANAGER::GetInputMessages(UINT uMsg,WPARAM wParam)
{
	switch(uMsg)
	{
		// keyboard key is pressed down  
	case WM_KEYDOWN:
		{
			SetTriggerState(wParam,true);
			return true;
		}

		// keyboard key is released
	case WM_KEYUP:
		{
			SetTriggerState(wParam,false);
			return true;
		}

		// left mouse-button is pressed down
	case WM_LBUTTONDOWN:
		{
			SetTriggerState(VK_LBUTTON,true);
			return true;
		}

		// right mouse-button is pressed down
	case WM_RBUTTONDOWN:
		{
			SetTriggerState(VK_RBUTTON,true);
			return true;
		}

		// left mouse-button is released
	case WM_LBUTTONUP:
		{
			SetTriggerState(VK_LBUTTON,false);
			return true;
		}

		// right mouse-button is released
	case WM_RBUTTONUP:
		{
			SetTriggerState(VK_RBUTTON,false);
			return true;
		}
	}

	return false;
}

void INPUT_MANAGER::Update()
{
	for(int i=0;i<NUM_KEY_INFOS;i++)
	{
		if(keyInfos[i] & KEY_QUERIED)
			keyInfos[i] |= KEY_MULTI_PRESSED;
	}
}

void INPUT_MANAGER::SetTriggerState(unsigned char keyCode,bool pressed)
{
	if(pressed)
		keyInfos[keyCode] |= KEY_PRESSED;
	else
	{
		keyInfos[keyCode] &= ~KEY_PRESSED;
		keyInfos[keyCode] &= ~KEY_QUERIED;
		keyInfos[keyCode] &= ~KEY_MULTI_PRESSED;
	}
}

bool INPUT_MANAGER::GetTriggerState(unsigned char keyCode)
{
	if(keyInfos[keyCode] & KEY_PRESSED)
		return true;
	else
		return false;
}

bool INPUT_MANAGER::GetSingleTriggerState(unsigned char keyCode)
{
	if((keyInfos[keyCode] & KEY_PRESSED)&&(!(keyInfos[keyCode] & KEY_MULTI_PRESSED)))
	{
		keyInfos[keyCode] |= KEY_QUERIED;
		return true;
	}
	return false; 
}

POINT INPUT_MANAGER::GetMousePos() const
{ 
	POINT pos;
	GetCursorPos(&pos);
	return pos;
}

void INPUT_MANAGER::CenterMousePos() const
{
	SetCursorPos(SCREEN_WIDTH>>1,SCREEN_HEIGHT>>1);
}

