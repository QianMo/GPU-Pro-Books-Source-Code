/******************************************************************************

 @File         MDKInput.cpp

 @Title        MDKTools

 @Copyright    Copyright (C) 2010 by Imagination Technologies Limited.

 @Platform     Independent

 @Description  Wrapper around PVRShell keypad input functions
 
******************************************************************************/

#include "MDKInput.h"
#include "MDKMisc.h"
#include "assert.h"


KeypadInput::KeypadInput()
	: m_bKeyRepeatFix( false )
{
	for (unsigned int i = 0; i < numKEYS; i++)
	{
		m_keys[i] = false;
		m_keysRepeat[i] = false;
	}
}

void KeypadInput::Init(PVRShell* pShell, bool bRotate) {
	this->m_bRotate=bRotate;
	m_pShell=pShell;
}

void KeypadInput::DoKeyPolicy(keysEnum key, bool shellPressed)
{
	if (!m_keysRepeat[key] && shellPressed)
	{
		m_keys[key] = m_keysRepeat[key] = true;
	}
	else if (m_keysRepeat[key] && shellPressed)
	{
		m_keys[key] = false; 
	}
	else if (m_keysRepeat[key] && !shellPressed)
	{
		m_keys[key] = m_keysRepeat[key] = false;
	}		
}

void KeypadInput::CheckInput()
{
	if( m_bKeyRepeatFix )
	{
		DoKeyPolicy(RIGHT, m_pShell->PVRShellIsKeyPressed(PVRShellKeyNameRIGHT));
		DoKeyPolicy(LEFT, m_pShell->PVRShellIsKeyPressed(PVRShellKeyNameLEFT));
		DoKeyPolicy(UP, m_pShell->PVRShellIsKeyPressed(PVRShellKeyNameUP));
		DoKeyPolicy(DOWN, m_pShell->PVRShellIsKeyPressed(PVRShellKeyNameDOWN));
		DoKeyPolicy(SELECT, m_pShell->PVRShellIsKeyPressed(PVRShellKeyNameSELECT));
		DoKeyPolicy(ACTION1, m_pShell->PVRShellIsKeyPressed(PVRShellKeyNameACTION1));
		DoKeyPolicy(ACTION2, m_pShell->PVRShellIsKeyPressed(PVRShellKeyNameACTION2));
	}
	else
	{
		if(m_bRotate) {
			m_keys[RIGHT]	= m_pShell->PVRShellIsKeyPressed(PVRShellKeyNameUP);
			m_keys[LEFT]	= m_pShell->PVRShellIsKeyPressed(PVRShellKeyNameDOWN);
			m_keys[UP]		= m_pShell->PVRShellIsKeyPressed(PVRShellKeyNameLEFT);
			m_keys[DOWN]	= m_pShell->PVRShellIsKeyPressed(PVRShellKeyNameRIGHT);
		} else {
			m_keys[RIGHT]	= m_pShell->PVRShellIsKeyPressed(PVRShellKeyNameRIGHT);
			m_keys[LEFT]	= m_pShell->PVRShellIsKeyPressed(PVRShellKeyNameLEFT);
			m_keys[UP]		= m_pShell->PVRShellIsKeyPressed(PVRShellKeyNameUP);
			m_keys[DOWN]	= m_pShell->PVRShellIsKeyPressed(PVRShellKeyNameDOWN);
		}
		m_keys[SELECT]	= m_pShell->PVRShellIsKeyPressed(PVRShellKeyNameSELECT);
		m_keys[ACTION1]	= m_pShell->PVRShellIsKeyPressed(PVRShellKeyNameACTION1);
		m_keys[ACTION2]	= m_pShell->PVRShellIsKeyPressed(PVRShellKeyNameACTION2);
	}
}

