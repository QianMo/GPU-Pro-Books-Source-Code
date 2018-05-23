

#ifndef __KEYBOARD_HPP__
#define __KEYBOARD_HPP__

#include <Common/Incopiable.hpp>
#include <Common/Common.hpp>

#include <Math/Matrix/Matrix.hpp>

#define DIRECTINPUT_VERSION 0x0800
#include <dinput.h>

///<
class Keyboard : public Incopiable
{
	HWND		m_hWnd;
	Vector2f	m_screenDims;
	

	static Keyboard m_inst;
	

public:

	static Keyboard& Get() { return m_inst; }

	IDirectInput8*			m_pDirectInput;
	IDirectInputDevice8*	m_pDirectInputKeyboard;    
	IDirectInputDevice8*	m_pDirectInputMouse;

	int8					KeyState[256];
	DIMOUSESTATE			MouseState;

	Keyboard(){ memset(this,0,sizeof(Keyboard)); }

	///<
	void Create(HINSTANCE _hInstance, HWND _hWnd, const Vector2f _dims);

	///<
	void Update();

	///<
	inline bool Key(const int8 _k) const{ return KeyState[_k] != 0; }

	inline bool		MouseClick1		() const	{return MouseState.rgbButtons[0]>0; }
	inline bool		MouseClick2		() const	{return MouseState.rgbButtons[1]>0; }
	inline bool		MouseClickWheel	() const	{return MouseState.rgbButtons[2]>0; }

	Vector3f		MouseDXYZ		() const;

	Vector2f		MouseX() const;
	Vector2f		ScreenSpaceMouseX() const;
	///<
	void Release();

};


#endif