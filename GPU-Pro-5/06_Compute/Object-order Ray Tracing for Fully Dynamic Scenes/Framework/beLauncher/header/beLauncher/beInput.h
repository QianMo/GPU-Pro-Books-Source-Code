/*****************************************************/
/* breeze Framework Launch Lib  (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_LAUNCHER_INPUT
#define BE_LAUNCHER_INPUT

#include "beLauncher.h"
#include <Windows.h>
#include <beCore/beShared.h>

#include <lean/tags/transitive_ptr.h>
#include <lean/smart/resource_ptr.h>
#include <vector>

namespace beLauncher
{

/// Input device class.
class InputDevice : public beCore::Resource
{
private:
	std::vector<char> m_data;

public:
	/// Constructor, acquires devices.
	BE_LAUNCHER_API InputDevice(HWND hTargetWnd);
	/// Destructor.
	BE_LAUNCHER_API ~InputDevice();

	/// Processes the given windows message, returning a pointer to the raw input data on success.
	const RAWINPUT* Process(UINT msg, WPARAM wParam, LPARAM lParam);
};

/// Keyboard state.
struct KeyboardState
{
	static const lean::uint4 KeyCount = 256;
	
	bool State[KeyCount];		///< Key is down.
	bool Changed[KeyCount];		///< Key has just been pressed or released, depending on State.

	friend void Reset(KeyboardState &state);
	/// Initializes state.
	KeyboardState() { Reset(*this); }
};

inline bool Pressed(const KeyboardState &state, lean::uint4 key) { return state.State[key]; }
inline bool JustPressed(const KeyboardState &state, lean::uint4 key) { return state.State[key] && state.Changed[key]; }
inline bool JustReleased(const KeyboardState &state, lean::uint4 key) { return !state.State[key] && state.Changed[key]; }

/// Resets one-time key state.
inline void SetHandled(KeyboardState &state)
{
	ZeroMemory(state.Changed, sizeof(state.Changed));
}

/// Resets all key state.
inline void Reset(KeyboardState &state)
{
	ZeroMemory(state.State, sizeof(state.State));
	SetHandled(state);
}

/// Keyboard.
class Keyboard : public beCore::Resource
{
public:
	/// Constructor.
	BE_LAUNCHER_API Keyboard(InputDevice *device);
	/// Destructor.
	BE_LAUNCHER_API ~Keyboard();

	/// Processes keyboard input.
	BE_LAUNCHER_API void Process(KeyboardState &state, const RAWINPUT &input) const;
};

/// Tracked mouse buttons.
struct MouseButton
{
	/// Enum.
	enum T
	{
		Left,
		Right,
		Middle,
		Button1,
		Button2,

		Count
	};
};

/// Mouse state.
struct MouseState
{
	bool State[MouseButton::Count];		///< Button is down.
	bool Changed[MouseButton::Count];	///< Button has been pressed or released, depending on State.
	
	float PosX, PosY, Wheel;
	float PosDeltaX, PosDeltaY, WheelDelta;

	int PtPosX, PtPosY, PtWheel;
	int PtPosDeltaX, PtPosDeltaY, PtWheelDelta;

	friend void Reset(MouseState &state);
	/// Initializes state.
	MouseState() { Reset(*this); }
};

inline bool Pressed(const MouseState &state, MouseButton::T button) { return state.State[button]; }
inline bool JustPressed(const MouseState &state, MouseButton::T button) { return state.State[button] && state.Changed[button]; }
inline bool JustReleased(const MouseState &state, MouseButton::T button) { return !state.State[button] && state.Changed[button]; }

/// Resets one-time key state.
inline void SetHandled(MouseState &state)
{
	ZeroMemory(state.Changed, sizeof(state.Changed));
	state.PosDeltaX = 0.0f;
	state.PosDeltaY = 0.0f;
	state.WheelDelta = 0.0f;
	state.PtPosDeltaX = 0;
	state.PtPosDeltaY = 0;
	state.PtWheelDelta = 0;
}

/// Resets all mouse state.
inline void Reset(MouseState &state)
{
	ZeroMemory(state.State, sizeof(state.State));
	state.PosX = 0;
	state.PosY = 0;
	state.Wheel = 0;
	state.PtPosX = 0;
	state.PtPosY = 0;
	state.PtWheel = 0;
	SetHandled(state);
}

/// Mouse.
class Mouse : public beCore::Resource
{
public:
	float sensitivity;

	/// Constructor.
	BE_LAUNCHER_API Mouse(InputDevice *device);
	/// Destructor.
	BE_LAUNCHER_API ~Mouse();

	/// Processes keyboard input.
	BE_LAUNCHER_API void Process(MouseState &state, const RAWINPUT &input) const;
};

struct InputState
{
	lean::transitive_ptr<KeyboardState> KeyState;
	lean::transitive_ptr<MouseState> MouseState;

	InputState(struct KeyboardState *pKeyState, struct MouseState *pMouseState)
		: KeyState(pKeyState),
		MouseState(pMouseState) { }
};

} // namespace

#endif