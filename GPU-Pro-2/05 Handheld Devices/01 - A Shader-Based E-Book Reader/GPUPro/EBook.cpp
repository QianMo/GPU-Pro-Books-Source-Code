/*!****************************************************************************
 @File          EBook.cpp

 @Title         EBook Demo

 @Copyright     Copyright 2010 by Imagination Technologies Limited.

 @Platform      Independent

 @Description   
******************************************************************************/

#include "PVRShell.h"
#include "version.h"

// Touchscreen interface
#include "MDKTouch.h"
// Input keypad interface
#include "MDKInput.h"
// Utility functions
#include "MDKMisc.h"


#include "Page.h"
#include "Resource.h"

using namespace MDK;

/*!****************************************************************************
 Class implementing the PVRShell functions.
******************************************************************************/



/*!
	The EBook class can be used to render a book or a single page. It is the main
	application class.
*/
class EBook : public PVRShell
{
#ifdef DEBUG_MODE
	//! Print3D class used to display text
	CPVRTPrint3D	m_Print3D;
	//! Shader for drawing lines
	GLuint uiColor;
#endif

	//! Used in single page mode
	Page page;
	//! Textures for single page mode
	GLuint uiFrontTex, uiBackTex;

	//! Book instance
	Book book;

	//! Whether to render in single or double page mode
	bool bDoublePage;
	//! Whether to show fps info and debug information
	bool bShowInfo;

	//! Application timer
	PrecisionTimer timer;
	//! Wrapper around PVRShell keypad input functionality
	KeypadInput input;

public:
	EBook() : book(this) { }

	virtual bool InitApplication();
	virtual bool InitView();
	virtual bool RenderScene();
	virtual bool ReleaseView();
	//virtual bool QuitApplication();
};


/*!****************************************************************************
 @Function		InitApplication
 @Return		bool		true if no error occured
 @Description	Code in InitApplication() will be called by PVRShell once per
				run, before the rendering context is created.
				Used to initialize variables that are not dependant on it
				(e.g. external modules, loading meshes, etc.)
				If the rendering context is lost, InitApplication() will
				not be called again.
******************************************************************************/
bool EBook::InitApplication(){
	CPVRTResourceFile::SetReadPath((char*)PVRShellGet(prefReadPath));

	bShowInfo = false;

	return true;
}



/*!****************************************************************************
 @Function		InitView
 @Return		bool		true if no error occured
 @Description	Code in InitView() will be called by PVRShell upon
				initialization or after a change in the rendering context.
				Used to initialize variables that are dependant on the rendering
				context (e.g. textures, vertex buffers, etc.)
******************************************************************************/
bool EBook::InitView()
{
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

#ifdef DEBUG_MODE
	// Initialize Print3D
	bool bRotate = PVRShellGet(prefIsRotated) && PVRShellGet(prefFullScreen);
	if(m_Print3D.SetTextures(0,PVRShellGet(prefWidth),PVRShellGet(prefHeight), bRotate) != PVR_SUCCESS)
	{
		return false;
	}
#endif

	// Initialize single page
	if (!LoadTexture("data/EBook00.pvr", &uiFrontTex))
		return false;

	if (!LoadTexture("data/EBook01.pvr", &uiBackTex))
		return false;

	page.Init(&book, 0.0f);
	page.SetTextures(uiFrontTex, uiBackTex);

	// Initialize book
	if (!book.Init(14, "data/EBook%02d.pvr",
		(float)PVRShellGet(prefWidth) / (float)PVRShellGet(prefHeight)))
	{
		return false;
	}

#ifdef DEBUG_MODE
	// Shader used for drawing lines (debug only)
	GLuint vs, fs;
	const char *aszAttribs[] = { "inVertex" }; 
	if (!LoadShader("data/Color.vsh", "data/Color.fsh", aszAttribs, 1, 0, 0, vs, fs, uiColor))
		return false;
#endif

	bDoublePage = true;

	// Touch initialization
	if (!TouchDevice::Instance().Init(this, PVRShellGet(prefWidth), PVRShellGet(prefHeight), PVRShellGet(prefIsRotated)))
		return false;

	// Keypad initialization
	input.Init(this, PVRShellGet(prefIsRotated));

#ifdef DEBUG_MODE
	m_Print3D.Print3D(0.0f, 0.0f, 1.0f, 0xff0000ff, "Init");
	m_Print3D.Flush();
#endif
	// Start timing
	timer.Start();
	return true;
}

/*!****************************************************************************
 @Function		ReleaseView
 @Return		bool		true if no error occured
 @Description	Code in ReleaseView() will be called by PVRShell when the
				application quits or before a change in the rendering context.
******************************************************************************/
bool EBook::ReleaseView()
{	
#ifdef DEBUG_MODE
	m_Print3D.ReleaseTextures();
#endif
	return true;
}


bool EBook::RenderScene()
{
	// Input
	timer.Update();
	float time = timer.GetTimef();
	float deltaTime = timer.GetDeltaTimef();

	input.CheckInput();
	TouchDevice::Instance().Input();

#ifndef __APPLE__
	// Single / double page mode
	if (input.KeyPress(KeypadInput::LEFT) || input.KeyPress(KeypadInput::RIGHT))
	{
		if ((bDoublePage = !bDoublePage))
			book.ForceRender();
	}

	// Render on demand on/off
	if (input.KeyPress(KeypadInput::ACTION2))
		book.ToggleRenderOnDemand();

#ifdef DEBUG_MODE
	// Show debug info on/off (fps, direction)
	if (input.KeyPress(KeypadInput::ACTION1))
	{
		if (!(bShowInfo = !bShowInfo))
			book.ForceRender();
	}
#endif

	// Toggle wireframe
	if (input.KeyPress(KeypadInput::SELECT))
		book.ToggleWireframe();
#endif

	// Input
	const TouchState *pTouchState = &TouchDevice::Instance().GetState();
	if (bDoublePage)
		book.Input(time, deltaTime, pTouchState);
	else
		page.Input(time, deltaTime, pTouchState);


	// Render
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glDisable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	if (bDoublePage)
	{
		book.Render();
	}
	else
	{
		// Clears the color and depth buffer
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		page.Render(Page::PageInside);

		if (page.GetFoldValue() != 0.0f)
		{
			glEnable(GL_BLEND);
			glDisable(GL_CULL_FACE);
			page.RenderSide(Page::PageOutside);
			glEnable(GL_CULL_FACE);
			glDisable(GL_BLEND);
		}
	}
#ifdef DEBUG_MODE
	if (bShowInfo)
	{
		/* Draw input lines */
		glDisable(GL_DEPTH_TEST);
		glUseProgram(uiColor);

		PVRTVec4 red(1.0f, 0.0f, 0.0f, 1.0f);
		glUniform4fv(glGetUniformLocation(uiColor, "Color"), 1, red.ptr());
		glUniform1i(glGetUniformLocation(uiColor, "Rotate"), book.Rotated());

		if (bDoublePage)
		{
			book.RenderDebug();

			/* Draw quad */
			// If the framebuffer color is not cleared, print3D will draw on top of previous frames
			// making the text unreadable. For this reason a white fill quad is rendered
			// underneath
			if (!book.RenderThisFrame())
			{
				PVRTVec4 white(1.0f, 1.0f, 1.0f, 1.0f);
				glUniform4fv(glGetUniformLocation(uiColor, "Color"), 1, white.ptr());
				float afQuadStrip[] = {
						-1.0f,  0.85f, 
						-0.4f,  0.85, 
						-1.0f,  1.0f, 
						-0.4f,  1.0f,
				};
				glEnableVertexAttribArray(0);
				glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, afQuadStrip);
				glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
				glDisableVertexAttribArray(0);
			}
			m_Print3D.Print3D(0.0f, 0.0f, 1.0f, 0xff0000ff, "FPS=%.1f", 1.0f / deltaTime);
			m_Print3D.Print3D(0.0f, 5.0f, 0.5f, 0xff0000ff, "%s", book.IsRenderOnDemand() ? "OnDemand" : "Redraw");
			
			m_Print3D.Flush();
		}
		else
			page.RenderDebug();

		glEnable(GL_DEPTH_TEST);
	}
#endif
	PVRShellOutputDebug("FPS=%.2f\n", 1.0f / deltaTime);

	return true;
}
/*!****************************************************************************
 @Function		NewDemo
 @Return		PVRShell*		The demo supplied by the user
 @Description	This function must be implemented by the user of the shell.
				The user should return its PVRShell object defining the
				behaviour of the application.
******************************************************************************/
PVRShell* NewDemo()
{
	return new EBook();
}

/******************************************************************************
 End of file (EBook.cpp)
******************************************************************************/
