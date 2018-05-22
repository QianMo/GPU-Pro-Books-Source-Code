
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include "GL/glew.h"

#include <stdio.h>
#include <conio.h>
#include <assert.h>

#include "../Main/DemoManager.h"
#include "../Util/ConfigLoader.h"
#include "../Input/InputManager.h"

#include "../Util/AntTweakBar.h"

#include <GL/glut.h>

int windowHandle;

// -----------------------------------------------------------------------------
// ---------------------------------- Reshape ----------------------------------
// -----------------------------------------------------------------------------
void Reshape(int w, int h)
{
	TwWindowSize(w, h);
	DemoManager::Instance()->Reshape(w, h);
}

// -----------------------------------------------------------------------------
// ----------------------------------- Render ----------------------------------
// -----------------------------------------------------------------------------
void Render(void)
{
	DemoManager::Instance()->Render();
	DemoManager::Instance()->PostRenderUpdate();
}

// -----------------------------------------------------------------------------
// ----------------------------------- Update ----------------------------------
// -----------------------------------------------------------------------------
void Update(void)
{
	DemoManager::Instance()->Update();
}

// -----------------------------------------------------------------------------
// ------------------------------- KeyUpFunction -------------------------------
// -----------------------------------------------------------------------------
void KeyUpFunction(unsigned char key,int x,int y)
{
	InputManager::Instance()->KeyUpFunction(key);
}

// -----------------------------------------------------------------------------
// ------------------------------ KeyDownFunction ------------------------------
// -----------------------------------------------------------------------------
void KeyDownFunction(unsigned char key,int x,int y)
{
	InputManager::Instance()->KeyDownFunction(key);
}

// -----------------------------------------------------------------------------
// ---------------------------- SpecialKeyUpFunction ---------------------------
// -----------------------------------------------------------------------------
void SpecialKeyUpFunction(int key,int x,int y)
{
	InputManager::Instance()->SpecialKeyUpFunction(key);
}

// -----------------------------------------------------------------------------
// --------------------------- SpecialKeyDownFunction --------------------------
// -----------------------------------------------------------------------------
void SpecialKeyDownFunction(int key,int x,int y)
{
	InputManager::Instance()->SpecialKeyDownFunction(key);
}

// -----------------------------------------------------------------------------
// ------------------------------- MouseFunction -------------------------------
// -----------------------------------------------------------------------------
void MouseFunction(int button, int state, int x, int y)
{
	if (!TwEventMouseButtonGLUT(button, state, x, y))
		InputManager::Instance()->MouseFunction(button, state);
}

// -----------------------------------------------------------------------------
// --------------------------------- MouseMove ---------------------------------
// -----------------------------------------------------------------------------
void MouseMove(int x, int y)
{
	TwEventMouseMotionGLUT(x, y);
}

// -----------------------------------------------------------------------------
// ------------------------------------ Exit -----------------------------------
// -----------------------------------------------------------------------------
void Exit(void)
{
	TwTerminate();

	if(ConfigLoader::Instance()->IsFullScreen())
	{
		glutLeaveGameMode();
	}

	glutDestroyWindow(windowHandle);
}

// -----------------------------------------------------------------------------
// ------------------------------ CheckExtensions ------------------------------
// -----------------------------------------------------------------------------
void CheckExtensions(void)
{
	if (!glewIsExtensionSupported("GL_EXT_framebuffer_object"))
	{
		printf("GL_EXT_framebuffer_object not supported!\n");
		assert(false);
	}

	if (!glewIsExtensionSupported("GL_ARB_point_parameters"))
	{
		printf("GL_ARB_point_parameters not supported!\n");
		assert(false);
	}

	if (!glewIsExtensionSupported("GL_ARB_texture_rectangle"))
	{
		printf("GL_ARB_texture_rectangle not supported!\n");
		assert(false);
	}

	if (!glewIsExtensionSupported("GL_ARB_vertex_program"))
	{
		printf("GL_ARB_vertex_program not supported!\n");
		assert(false);
	}

	if (!glewIsExtensionSupported("GL_ARB_fragment_program"))
	{
		printf("GL_ARB_fragment_program not supported!\n");
		assert(false);
	}

	if (!glewIsExtensionSupported("GL_EXT_texture_array"))
	{
		printf("GL_EXT_texture_array not supported!\n");
		assert(false);
	}

	if (!glewIsExtensionSupported("GL_ARB_occlusion_query"))
	{
		printf("GL_ARB_occlusion_query not supported!\n");
		assert(false);
	}
}

// -----------------------------------------------------------------------------
// ------------------------------------ main -----------------------------------
// -----------------------------------------------------------------------------
int main(int argc, char **argv)
{
	// initialise glut
	glutInit(&argc,argv);

	ConfigLoader::Instance()->LoadConfigFile("config.txt");

	if (ConfigLoader::Instance()->GetBitsPerPixel() == 32)
	{
		glutInitDisplayMode(GLUT_DEPTH|GLUT_RGBA|GLUT_DOUBLE|GLUT_ALPHA);
	}
	else
	{
		glutInitDisplayMode(GLUT_DEPTH|GLUT_RGB|GLUT_DOUBLE|GLUT_ALPHA);
	}

	if (ConfigLoader::Instance()->IsFullScreen())
	{
		char gameModeString[128];
		sprintf_s(gameModeString, "%dx%d:%d@%d", ConfigLoader::Instance()->GetScreenWidth(),
											   ConfigLoader::Instance()->GetScreenHeight(),
											   ConfigLoader::Instance()->GetBitsPerPixel(),
											   ConfigLoader::Instance()->GetRefreshRate());
		glutGameModeString(gameModeString);
		glutEnterGameMode();
	}
	else
	{
		glutInitWindowSize(ConfigLoader::Instance()->GetScreenWidth(), ConfigLoader::Instance()->GetScreenHeight());
		windowHandle = glutCreateWindow(ConfigLoader::Instance()->GetWindowName());
		glutPositionWindow(350, 240);
	}

	// glut callback functions
	glutDisplayFunc(Render);
	glutReshapeFunc(Reshape);
	glutIdleFunc(Update);

	glutKeyboardFunc(KeyDownFunction);
	glutKeyboardUpFunc(KeyUpFunction);
	glutSpecialFunc(SpecialKeyDownFunction);
	glutSpecialUpFunc(SpecialKeyUpFunction);
	glutMouseFunc(MouseFunction);
	glutMotionFunc(MouseMove);
	glutPassiveMotionFunc((GLUTmousemotionfun)TwEventMouseMotionGLUT);

	// call exit on program end
	atexit(Exit);

	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		assert(false);
	}

	CheckExtensions();

	int maxBuffers;
	glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS_EXT, &maxBuffers);
	int maxDrawBuffers;
	glGetIntegerv(GL_MAX_DRAW_BUFFERS, &maxDrawBuffers);

	if ((maxBuffers < 4) || (maxDrawBuffers < 4))
	{
		printf("Not enough draw buffers!");
		assert(false);
	}

	// disable vsync
	typedef bool (APIENTRY *PFNWGLSWAPINTERVALFARPROC)(int);

	PFNWGLSWAPINTERVALFARPROC wglSwapIntervalEXT = 0;

	wglSwapIntervalEXT = (PFNWGLSWAPINTERVALFARPROC)wglGetProcAddress("wglSwapIntervalEXT");

	if( wglSwapIntervalEXT )
		wglSwapIntervalEXT(false);

	// init gui
	if(!TwInit(TW_OPENGL, NULL))
	{
		char buffer [256];
		sprintf_s(buffer, "AntTweakBar initialization failed: %s\n", TwGetLastError());
		MessageBox(NULL, "TwInit", buffer, MB_OK);
		return 1;
	}
	TwGLUTModifiersFunc(glutGetModifiers);

	//DemoManager::Instance()->Init();

	// glut main loop
	glutMainLoop();

	return 0;
}