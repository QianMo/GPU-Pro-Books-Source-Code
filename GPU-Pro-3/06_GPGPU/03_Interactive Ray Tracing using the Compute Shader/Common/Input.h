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

// ------------------------------------------------
// Input.h
// ------------------------------------------------
// Manipulates input from keyboard and mouse.

#ifndef INPUT_H
#define INPUT_H

#include "windows.h"

class Input
{
public:
	Input();
	~Input(void);

	int				IsShadow() { return m_bShadows; }
	int				IsFlatShading() { return m_bPhongShading; }
	int				IsMoveLights() { return m_bMoveLights; }
	int				IsMultiplicativeReflex() { return m_bMuliplicativeReflex; }
	int				IsReflections() { return m_bReflections; }
	int				IsNormalMappingOn()	{ return m_bNormalMapping; }
	int				IsGlossMappingOn()	{ return m_bGlossMapping; }
	int				IsMouseDown() { return m_bMouseDown; }
	unsigned int	GetAccelerationStructureFlag() { return m_iAccelerationStructureFlag; }
	unsigned int	GetNumBounces() { return m_iNumBounces; }
	int				GetEnvMappingFlag() { return m_iEnvMappingFlag; }

	void			SetMouseDown(BOOL bValue) { m_bMouseDown = bValue; }
	void			SetMouseCoordinates(UINT x, UINT y) { m_MouseX=x; m_MouseY = y; }
	void			SetAccelerationStructureFlag(UINT iOpt) { m_iAccelerationStructureFlag = iOpt; }
	void			SetNumBounces(UINT iBounces) { m_iNumBounces = iBounces; }
	
	void			OperateOnBounces(int a_Oper) { m_iNumBounces += a_Oper; printf("Number of bounces: %d\n", m_iNumBounces); }
	
	void			ChangeShadowingState() { m_bShadows = static_cast<bool>(!m_bShadows); printf("Shadows are %s\n", m_bShadows?"ON":"OFF"); }
	void			ChangePhongShadingState() { m_bPhongShading = static_cast<bool>(!m_bPhongShading); printf("%s shading is ON\n", m_bPhongShading?"Phong":"Flat"); }
	void			ChangeMoveLightsState() { m_bMoveLights = static_cast<bool>(!m_bMoveLights); printf("Move lights is %s\n", m_bMoveLights?"ON":"OFF");}
	void			ChangeMultiplicativeReflexState() { m_bMuliplicativeReflex = static_cast<bool>(!m_bMuliplicativeReflex); printf("Multiplicative is %s\n", m_bMuliplicativeReflex?"ON":"OFF");}
	void			ChangeReflectionsState() { m_bReflections = static_cast<bool>(!m_bReflections); printf("Reflections are %s\n", m_bReflections?"ON":"OFF");}
	void			ChangeNormalMappingState() { m_bNormalMapping = static_cast<bool>(!m_bNormalMapping); printf("Normal mapping is %s\n", m_bNormalMapping?"ON":"OFF");}
	void			ChangeGlossMappingState() { m_bGlossMapping = static_cast<bool>(!m_bGlossMapping); printf("Gloss mapping is %s\n", m_bGlossMapping?"ON":"OFF");}
	void			ChangeEnvMappingFlag() { m_iEnvMappingFlag = static_cast<int>((m_iEnvMappingFlag == 0))*-1; printf("Environment mapping is %s\n", m_iEnvMappingFlag!=0?"ON":"OFF");}

	int				OnMouseMove(UINT& x, UINT& y, LPARAM& lParam);	// On mouse move do something
private:
	int				m_bShadows;							// Cast shadows?
	int				m_bPhongShading;					// Use Phong or flat shading
	int				m_bMoveLights;						// Indicates to move the lights instead of cameras
	int				m_bNormalMapping;
	int				m_bGlossMapping;	
	int				m_iEnvMappingFlag;
	unsigned int	m_iAccelerationStructureFlag;
	unsigned int	m_iNumBounces;

	int				m_bMouseDown;						// Is left click pressed?
	UINT			m_MouseX;							// Position of the mouse on x-axis
	UINT			m_MouseY;							// Position of the mouse on y-axis

	int				m_bMuliplicativeReflex;				// Type of reflection
	int				m_bReflections;						// Cast reflections
};

#endif

#endif //WINDOWS-LINUX