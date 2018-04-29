/******************************************************************************

 @File         MDKInput.h

 @Title        MDKTools

 @Copyright    Copyright (C) 2010 by Imagination Technologies Limited.

 @Platform     Independent

 @Description  Wrapper around PVRShell keypad input functions
 
******************************************************************************/

#ifndef _MDK_INPUT_H_
#define _MDK_INPUT_H_

#include "PVRShell.h"



//! Class for checking input from keyboard or touchpad.
/*!
	
*/
class KeypadInput
{
public:
	//! Enumeration
	enum keysEnum 
	{ 
		UP, 		/*!< Corresponds to the up arrow key on a keyboard or the up direction on a joystick. */
		DOWN, 		/*!< Corresponds to the down arrow key on a keyboard or the down direction on a joystick. */
		LEFT, 		/*!< Corresponds to the left arrow key on a keyboard or the left direction on a joystick. */
		RIGHT, 		/*!< Corresponds to the right arrow key on a keyboard or the right direction on a joystick. */
		SELECT, 	/*!< Corresponds to the 'return' or 'enter' key a keyboard or the 'select' button on a keypad. */
		ACTION1, 	/*!< Corresponds to the '1' key on a keyboard or the 'action 1' button on a keypad. */
		ACTION2, 	/*!< Corresponds to the '2' key on a keyboard or the 'action 2' button on a keypad. */
		numKEYS
	};

	
	//! Constructor
	KeypadInput();

	//! Initializes the class.
	/*!
		\param bRotate
	*/
	void Init(PVRShell* pShell, bool bRotate=false);

	//! Updates the class to reflect the input device's state.
	/*!
		
	*/
	void CheckInput();

	//! Checks whether a particular input state is set.
	/*!
		\param key The member of Input::keysEnum corresponding to the state you want to know about.
		\return True if the key is down, false otherwise.
	*/
	bool KeyPress(keysEnum key) { return m_keys[key]; }
	bool KeyPressAny()
	{ 
		return m_keys[UP] || m_keys[DOWN] ||
			m_keys[LEFT] || m_keys[RIGHT] ||
			m_keys[SELECT] ||
			m_keys[ACTION1] || m_keys[ACTION2];
	}

	const bool (&GetState() const)[numKEYS] { return m_keys; };

private:
	bool m_bKeyRepeatFix;
	bool m_bRotate;
	bool m_keys[numKEYS];
	
	bool m_keysRepeat[numKEYS];
	void DoKeyPolicy(keysEnum key, bool shellPressed);

	PVRShell* m_pShell;
};

typedef bool KeypadState[KeypadInput::numKEYS];

#endif
