// ================================================================================ //
// Copyright (c) 2011 Arturo Garcia, Francisco Avila, Sergio Murguia and Leo Reyes	//
//																					//
// Permission is hereby granted, free of charge, to any person obtaining a copy of	//
// this software and associated documentation files (the "Software"), to deal in	//
// the Software without restriction, including without limitation the rights to		//
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies	//
// of the Software, and to permit persons to whom the Software is furnished to do	//
// so, subject to the following conditions:											//
//																					//
// The above copyright notice and this permission notice shall be included in all	//
// copies or substantial portions of the Software.									//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR		//
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,			//
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE		//
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER			//
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,	//
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE	//
// SOFTWARE.																		//
// ================================================================================ //

#ifdef WINDOWS

#include "ArgumentsParser.h"
#include "Input.h"

// -----------------------------------------------------------
// Constructor
// -----------------------------------------------------------
Input::Input()
{
	m_bMouseDown = false;
	m_bShadows = true;								// Cast shadows?
	m_bPhongShading = true;							// Use Phong or flat shading
	m_bMoveLights = false;							// indicates to move the lights instead of cameras
	m_bMuliplicativeReflex = false;
	m_bReflections = true;
	m_iEnvMappingFlag = -1;
	m_bNormalMapping = true;
	m_bGlossMapping = true;
	m_iAccelerationStructureFlag = 0;
	m_iNumBounces = 0;
}

// -----------------------------------------------------------
// Destructor
// -----------------------------------------------------------
Input::~Input(void)
{
}

// -----------------------------------------------------------
// Mouse-Move, if left click is pressed, then rotate
// camera
// -----------------------------------------------------------
BOOL Input::OnMouseMove(UINT& x, UINT& y, LPARAM& lParam)
{
	x = m_MouseX;
	y = m_MouseY;
	m_MouseX = LOWORD(lParam);
	m_MouseY = HIWORD(lParam);
	x -= m_MouseX;
	y -= m_MouseY;
	return true;
}



#endif //WINDOWS