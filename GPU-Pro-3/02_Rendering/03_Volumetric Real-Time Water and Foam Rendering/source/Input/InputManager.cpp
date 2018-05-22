#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <stdio.h>

#include "../Input/InputManager.h"

#include <GL/glut.h>


// -----------------------------------------------------------------------------
// ----------------------- InputManager::InputManager --------------------------
// -----------------------------------------------------------------------------
InputManager::InputManager(void)
{
	int i;
	for (i=0; i<512; i++)
		keyStates[i] = false;
	for (i=0; i<3; i++)
		mouseStates[i] = false;
	
	lastX = 0;
	lastY = 0;
}


// -----------------------------------------------------------------------------
// ----------------------- InputManager::~InputManager -------------------------
// -----------------------------------------------------------------------------
InputManager::~InputManager(void)
{

}


// -----------------------------------------------------------------------------
// ----------------------- InputManager::Init ----------------------------------
// -----------------------------------------------------------------------------
void InputManager::Init(void)
{

}

// -----------------------------------------------------------------------------
// ----------------------- InputManager::Exit ----------------------------------
// -----------------------------------------------------------------------------
void InputManager::Exit(void)
{

}

// -----------------------------------------------------------------------------
// ----------------------- InputManager::IsKeyPressed --------------------------
// -----------------------------------------------------------------------------
const bool InputManager::IsKeyPressed(const int& key) const
{
	if ((key >= 0) && (key < 512))
	{
		return keyStates[key];
	}
	return false;
}

// -----------------------------------------------------------------------------
// ----------------------- InputManager::IsKeyPressed --------------------------
// -----------------------------------------------------------------------------
const bool InputManager::IsKeyPressedAndReset(const int& key)
{
	if ((key>=0) && (key<512))
	{
		bool state = keyStates[key];
		keyStates[key] = false;
		return state;
	}
	return false;
}

// -----------------------------------------------------------------------------
// ----------------------- InputManager::IsMouseKeyPressed ---------------------
// -----------------------------------------------------------------------------
const bool InputManager::IsMouseKeyPressed(const int& key) const
{
	if ((key >= 0) && (key < 3))
	{
		return mouseStates[key];
	}
	return false;
}


// -----------------------------------------------------------------------------
// ----------------------- InputManager::GetMousePosition ----------------------
// -----------------------------------------------------------------------------
const POINT InputManager::GetMousePosition(void)
{
	POINT mPos, finalPos;
	GetCursorPos(&mPos);

	finalPos.x = mPos.x - lastX;
	finalPos.y = mPos.y - lastY;

	bool resetCursorPos = false;
	if (mPos.x == 0)
	{
		mPos.x = glutGet(GLUT_SCREEN_WIDTH) - 2;
		resetCursorPos = true;
	}
	if (mPos.x == (glutGet(GLUT_SCREEN_WIDTH) - 1))
	{
		mPos.x = 1;
		resetCursorPos = true;
	}
	if (mPos.y == 0)
	{
		mPos.y = glutGet(GLUT_SCREEN_HEIGHT) - 2;
		resetCursorPos = true;
	}
	if (mPos.y == (glutGet(GLUT_SCREEN_HEIGHT) - 1))
	{
		mPos.y = 1;
		resetCursorPos = true;
	}

	if (resetCursorPos)
		SetCursorPos(mPos.x, mPos.y);

	lastX = mPos.x;
	lastY = mPos.y;

	return finalPos;
}

// -----------------------------------------------------------------------------
// ----------------------- InputManager::KeyUpFunction -------------------------
// -----------------------------------------------------------------------------
void InputManager::KeyUpFunction(const unsigned char& key)
{
	InputManager::Instance()->keyStates[key]=false;
}


// -----------------------------------------------------------------------------
// ----------------------- InputManager::KeyDownFunction -----------------------
// -----------------------------------------------------------------------------
void InputManager::KeyDownFunction(const unsigned char& key)
{
	InputManager::Instance()->keyStates[key]=true;
}


// -----------------------------------------------------------------------------
// ----------------------- InputManager::SpecialKeyUpFunction ------------------
// -----------------------------------------------------------------------------
void InputManager::SpecialKeyUpFunction(const int& key)
{
	InputManager::Instance()->keyStates[key+256]=false;
}


// -----------------------------------------------------------------------------
// ----------------------- InputManager::SpecialKeyDownFunction ----------------
// -----------------------------------------------------------------------------
void InputManager::SpecialKeyDownFunction(const int& key)
{
	InputManager::Instance()->keyStates[key+256]=true;
}


// -----------------------------------------------------------------------------
// ----------------------- InputManager::MouseFunction -------------------------
// -----------------------------------------------------------------------------
void InputManager::MouseFunction(const int& button, const int& state)
{
	if (button == GLUT_LEFT_BUTTON)
		 InputManager::Instance()->mouseStates[0] = ((state == GLUT_DOWN) ? true:false);
	else if (button == GLUT_MIDDLE_BUTTON)
		InputManager::Instance()->mouseStates[1] = ((state == GLUT_DOWN) ? true:false);
	else if (button == GLUT_RIGHT_BUTTON)
		InputManager::Instance()->mouseStates[2] = ((state == GLUT_DOWN) ? true:false);
}
