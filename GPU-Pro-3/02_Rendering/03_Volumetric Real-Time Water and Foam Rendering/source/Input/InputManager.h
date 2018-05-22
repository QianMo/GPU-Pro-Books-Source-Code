#ifndef __INPUTMANAGER__H__
#define __INPUTMANAGER__H__

#include "../Util/Singleton.h"

#define KEY_F1			(GLUT_KEY_F1+256)
#define KEY_F2			(GLUT_KEY_F2+256)
#define KEY_F3			(GLUT_KEY_F3+256)
#define KEY_F4			(GLUT_KEY_F4+256)
#define KEY_F5			(GLUT_KEY_F5+256)
#define KEY_F6			(GLUT_KEY_F6+256)
#define KEY_F7			(GLUT_KEY_F7+256)
#define KEY_F8			(GLUT_KEY_F8+256)
#define KEY_F9			(GLUT_KEY_F9+256)
#define KEY_F10			(GLUT_KEY_F10+256)
#define KEY_F11			(GLUT_KEY_F11+256)
#define KEY_F12			(GLUT_KEY_F12+256)

#define KEY_LEFT		(GLUT_KEY_LEFT+256)
#define KEY_UP			(GLUT_KEY_UP+256)
#define KEY_RIGHT		(GLUT_KEY_RIGHT+256)
#define KEY_DOWN		(GLUT_KEY_DOWN+256)
#define KEY_PAGE_UP		(GLUT_KEY_PAGE_UP+256)
#define KEY_PAGE_DOWN	(GLUT_KEY_PAGE_DOWN+256)
#define KEY_HOME		(GLUT_KEY_HOME+256)
#define KEY_END			(GLUT_KEY_END+256)
#define KEY_INSERT		(GLUT_KEY_INSERT+256)

#define KEY_CONSOLE		'`'
#define KEY_TAB			9
#define KEY_BACKSPACE	8
#define KEY_DELETE		127
#define KEY_SPACE		' '
#define KEY_ENTER		13
#define KEY_ESCAPE		27

#define MOUSE_LEFT		GLUT_LEFT_BUTTON
#define MOUSE_MIDDLE	GLUT_MIDDLE_BUTTON
#define MOUSE_RIGHT		GLUT_RIGHT_BUTTON


class InputManager : public Singleton<InputManager>
{
	friend class Singleton<InputManager>;

public:
	InputManager(void);
	~InputManager(void);

	/// Init the input manager
	void Init(void);

	/// Exit the input manager
	void Exit(void);

	/// Returns if a key is pressed or not
	const bool IsKeyPressed(const int& key) const;

	/// Returns if a key is pressed or not and sets the key state to false
	const bool IsKeyPressedAndReset(const int& key);

	/// Returns if a mouse key is pressed or not
	const bool IsMouseKeyPressed(const int& key) const;

	/// Returns the mouse position
	const POINT GetMousePosition(void);

	/// Callback functions for glut
	void KeyUpFunction(const unsigned char& key);
	void KeyDownFunction(const unsigned char& key);
	void SpecialKeyUpFunction(const int& key);
	void SpecialKeyDownFunction(const int& key);
	void MouseFunction(const int& button, const int& state);

private:

	/// Key states
	bool keyStates[512];
	
	/// Mouse state
	bool mouseStates[3];

	/// Last x mouse position
	int lastX;

	/// Last y mouse position
	int lastY;
};

#endif