/*****************************************************/
/* breeze Framework Launch Lib  (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beLauncherInternal/stdafx.h"
#include "beLauncher/beInput.h"

#include <lean/logging/win_errors.h>

namespace beLauncher
{

// Constructor, acquires devices.
InputDevice::InputDevice(HWND hTargetWnd)
{
	RAWINPUTDEVICE devices[2];

	devices[0].usUsagePage = 1; // Generic desktop controls
	devices[1].usUsagePage = 1; // Generic desktop controls
	devices[0].usUsage = 6; // Keyboard
	devices[1].usUsage = 2; // Mouse
	devices[0].hwndTarget = hTargetWnd;
	devices[1].hwndTarget = hTargetWnd;
	devices[0].dwFlags = 0; // TODO: Fullscreen -> Disable system keys?
	devices[1].dwFlags = 0;

	if (!::RegisterRawInputDevices(devices, lean::arraylen(devices), sizeof(devices[0])))
		LEAN_THROW_WIN_ERROR_MSG("RegisterRawInputDevices");
}

// Destructor.
InputDevice::~InputDevice()
{
}

// Processes the given windows message, returning a pointer to the raw input data on success.
const RAWINPUT* InputDevice::Process(UINT msg, WPARAM wParam, LPARAM lParam)
{
	if (msg == WM_INPUT)
	{
		UINT bufferSize = 0;
		if (GetRawInputData((HRAWINPUT) lParam, RID_INPUT, NULL, &bufferSize, sizeof(RAWINPUTHEADER)) != 0)
		{
			LEAN_LOG_WIN_ERROR_MSG("GetRawInputData");
			return nullptr;
		}

		if (m_data.size() < bufferSize)
			m_data.resize(bufferSize);

		if (GetRawInputData((HRAWINPUT) lParam, RID_INPUT, (LPVOID) &m_data[0], &bufferSize, sizeof(RAWINPUTHEADER)) != bufferSize)
		{
			LEAN_LOG_WIN_ERROR_MSG("GetRawInputData");
			return nullptr;
		}

		return reinterpret_cast<const RAWINPUT*>(&m_data[0]);
	}

	return nullptr;
}

// Constructor.
Keyboard::Keyboard(InputDevice *device)
{
}

// Destructor.
Keyboard::~Keyboard()
{
}

// Processes keyboard input.
void Keyboard::Process(KeyboardState &state, const RAWINPUT &input) const
{
	if (input.header.dwType == RIM_TYPEKEYBOARD) 
	{
		state.State[input.data.keyboard.VKey] = !(input.data.keyboard.Flags & RI_KEY_BREAK);
		state.Changed[input.data.keyboard.VKey] = true;
	}
}

// Constructor.
Mouse::Mouse(InputDevice *device)
	: sensitivity(1.0f)
{
	int mouseSpeed = 10;
	if (::SystemParametersInfoW(SPI_GETMOUSESPEED, 0, &mouseSpeed, 0))
		sensitivity = (float) mouseSpeed / 10.0f;
	else
		LEAN_LOG_WIN_ERROR_MSG("SystemParametersInfo");
}

// Destructor.
Mouse::~Mouse()
{
}

namespace
{

inline void UpdateMouseButtonState(MouseState &state, const RAWINPUT &input, MouseButton::T button, USHORT downFlag, USHORT upFlag)
{
	if (input.data.mouse.usButtonFlags & downFlag)
	{
		state.State[button] = true;
		state.Changed[button] = true;
	}
	else if (input.data.mouse.usButtonFlags & upFlag)
	{
		state.State[button] = false;
		state.Changed[button] = true;
	}
}

} // namespace

// Processes keyboard input.
void Mouse::Process(MouseState &state, const RAWINPUT &input) const
{
	if (input.header.dwType == RIM_TYPEMOUSE) 
	{
		UpdateMouseButtonState(state, input, MouseButton::Left, RI_MOUSE_LEFT_BUTTON_DOWN, RI_MOUSE_LEFT_BUTTON_UP);
		UpdateMouseButtonState(state, input, MouseButton::Right, RI_MOUSE_RIGHT_BUTTON_DOWN, RI_MOUSE_RIGHT_BUTTON_UP);
		UpdateMouseButtonState(state, input, MouseButton::Middle, RI_MOUSE_MIDDLE_BUTTON_DOWN, RI_MOUSE_MIDDLE_BUTTON_UP);
		UpdateMouseButtonState(state, input, MouseButton::Button1, RI_MOUSE_BUTTON_4_DOWN, RI_MOUSE_BUTTON_4_UP);
		UpdateMouseButtonState(state, input, MouseButton::Button2, RI_MOUSE_BUTTON_5_DOWN, RI_MOUSE_BUTTON_5_UP);

		if (!(input.data.mouse.usFlags & MOUSE_MOVE_ABSOLUTE))
		{
			state.PtPosX += input.data.mouse.lLastX;
			state.PtPosY += input.data.mouse.lLastY;
			state.PtPosDeltaX += input.data.mouse.lLastX;
			state.PtPosDeltaY += input.data.mouse.lLastY;

			state.PosX = (float) state.PtPosX * sensitivity;
			state.PosY = (float) state.PtPosY * sensitivity;
			state.PosDeltaX = (float) state.PtPosDeltaX * sensitivity;
			state.PosDeltaY = (float) state.PtPosDeltaY * sensitivity;
		}

		if (input.data.mouse.usButtonFlags & RI_MOUSE_WHEEL)
		{
			state.PtWheel += input.data.mouse.usButtonData;
			state.PtWheelDelta += input.data.mouse.usButtonData;

			state.Wheel = (float) state.PtWheel;
			state.WheelDelta = (float) state.PtWheelDelta;
		}
	}
}

} // namespace
