
#include <Input/Keyboard.hpp>


#pragma comment( lib, "..\\..\\External\\DirectX\\Lib\\x86\\dinput8.lib")
#pragma comment( lib, "..\\..\\External\\DirectX\\Lib\\x86\\dxguid.lib")

Keyboard Keyboard::m_inst;

Vector3f Keyboard::MouseDXYZ()const {return Vector3f((float32)MouseState.lX,(float32)MouseState.lY,(float32)MouseState.lZ);}

///<
Vector2f Keyboard::ScreenSpaceMouseX() const
{
	Vector2f x = Keyboard::Get().MouseX();		
	if (m_screenDims.x() > 0 && m_screenDims.y() > 0)
	{
		return Vector2f( (-1.0f + 2.0f*x.x()/M::SCast<float32>(m_screenDims.x())),(-1.0f + 2.0f*(1.0f-x.y()/M::SCast<float32>(m_screenDims.y()))));
	}

	return x;
}

///<
Vector2f Keyboard::MouseX() const
{
	POINT cursorPos;
	GetCursorPos(&cursorPos);
	ScreenToClient(m_hWnd, &cursorPos);
	return Vector2f((float32)cursorPos.x,(float32)cursorPos.y);
}

///<
void Keyboard::Release()
{
	if (m_pDirectInputKeyboard)
		m_pDirectInputKeyboard->Unacquire();

	if (m_pDirectInputMouse)
		m_pDirectInputMouse->Unacquire();

	M::Release(&m_pDirectInputMouse);
	M::Release(&m_pDirectInputKeyboard);
	M::Release(&m_pDirectInput);		
}

///<
void Keyboard::Create(HINSTANCE _hInstance, HWND _hWnd, const Vector2f _dims)
{
	m_screenDims=_dims;
	m_hWnd = _hWnd;

	DirectInput8Create(_hInstance,		
		DIRECTINPUT_VERSION,			
		IID_IDirectInput8,    
		(void**)&m_pDirectInput,    
		NULL);  

	if (m_pDirectInput)
	{
		m_pDirectInput->CreateDevice(GUID_SysKeyboard, &m_pDirectInputKeyboard, NULL);
		m_pDirectInput->CreateDevice(GUID_SysMouse, &m_pDirectInputMouse, NULL);

		m_pDirectInputKeyboard->SetDataFormat(&c_dfDIKeyboard);
		m_pDirectInputMouse->SetDataFormat(&c_dfDIMouse);

		m_pDirectInputKeyboard->SetCooperativeLevel(m_hWnd, DISCL_NONEXCLUSIVE | DISCL_FOREGROUND);

		///< Set to exclusive mouse to make disappear.
		m_pDirectInputMouse->SetCooperativeLevel(m_hWnd, DISCL_NONEXCLUSIVE | DISCL_FOREGROUND);
	}
}

///<
void Keyboard::Update()
{
	if (m_pDirectInputKeyboard && m_pDirectInputMouse)
	{		
		///< Acquire keyboard Data:	
		m_pDirectInputKeyboard->Acquire();
		m_pDirectInputKeyboard->GetDeviceState(256, (LPVOID)KeyState);

		m_pDirectInputMouse->Acquire();
		m_pDirectInputMouse->GetDeviceState(sizeof(DIMOUSESTATE), (LPVOID)&MouseState);
	}
}