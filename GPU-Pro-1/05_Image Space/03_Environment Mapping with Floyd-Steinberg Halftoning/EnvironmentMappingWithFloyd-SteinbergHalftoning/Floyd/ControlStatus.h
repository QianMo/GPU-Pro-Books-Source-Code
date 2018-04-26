#pragma once

/// Structure used both as an input state manager and passed as parameter to control calls.
class ControlStatus
{
public:
	/// Array of key pressed state variables. Addressed by virtual key codes, true if key is pressed.
	bool keyPressed[0xff];
	/// Mouse pointer position in normalized screen space.
	D3DXVECTOR3 mousePosition;
	/// Pressed state variable of left, center and right mouse buttons.
	bool mouseButtonPressed[3];

	/// Viewport width. Not set by constructor.
	unsigned int viewportWidth;
	/// Viewport height. Not set by constructor.
	unsigned int viewportHeight;

	/// Updates input state by processing message.
	virtual void handleInput(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

	/// Constructor. Intializes input state.
	ControlStatus();
};
