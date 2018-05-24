#include <system/application.h>
#include <system/time.h>
#include <essentials/macros.h>

#ifdef MAXEST_FRAMEWORK_SYSTEM_APPLICATION_OPENGL
	#include <glew.h>
#endif


NSystem::Application::Application()
{
	for (int i = 0; i < ARRAY_SIZE(keys); i++)
		keys[i] = false;
	mouseLeftButtonDown = false;
	mouseMiddleButtonDown = false;
	mouseRightButtonDown = false;
	mouseWindowX = 0;
	mouseWindowY = 0;
	mouseDesktopX = 0;
	mouseDesktopY = 0;
	mouseRelX = 0;
	mouseRelY = 0;
	cursorVisible = true;

	runStartTime = runStopTime = 0;
	lastFrameTime = 0.0f;

	keyDownFunction = nullptr;
	mouseMotionFunction = nullptr;
	mouseLeftButtonDownFunction = nullptr;
	mouseRightButtonDownFunction = nullptr;
}


bool NSystem::Application::Create(int width, int height, bool fullScreen)
{
	this->fullScreen = fullScreen;

	if (SDL_Init(SDL_INIT_VIDEO) < 0)
		return false;

	if (width == 0 && height == 0)
	{
		SDL_DisplayMode displayMode;
		SDL_GetCurrentDisplayMode(0, &displayMode);

		width = displayMode.w;
		height = displayMode.h;
	}

	Uint32 windowCreationFlags = 0;
	if (fullScreen)
		windowCreationFlags |= SDL_WINDOW_FULLSCREEN;
	#ifdef MAXEST_FRAMEWORK_SYSTEM_APPLICATION_OPENGL
		windowCreationFlags |= SDL_WINDOW_OPENGL;
	#endif

	window = SDL_CreateWindow(
		"MaxestFrameworkWindow", 
		SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED,
		width,
		height,
		windowCreationFlags
	);

	#ifdef MAXEST_FRAMEWORK_SYSTEM_APPLICATION_OPENGL
		SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, true); 
		SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
		SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
		SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
		SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);
		SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
		SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
		SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, true);

		glContext = SDL_GL_CreateContext(window);

		SDL_GL_SetSwapInterval(0);

		#ifdef MAXEST_FRAMEWORK_SYSTEM_APPLICATION_OPENGL
			glewExperimental = GL_TRUE;
			if (glewInit() != GLEW_OK)
				return false;
		#endif
	#endif

	return true;
}


void NSystem::Application::Destroy()
{
	#ifdef MAXEST_FRAMEWORK_SYSTEM_APPLICATION_OPENGL
		SDL_GL_DeleteContext(glContext);
	#endif

	SDL_DestroyWindow(window);
}


void NSystem::Application::Run(bool(*runFunction)())
{
	int prevMouseX;
	int prevMouseY;

	SDL_DisplayMode displayMode;
	SDL_GetCurrentDisplayMode(0, &displayMode);
	int screenCenterX = displayMode.w / 2;
	int screenCenterY = displayMode.h / 2;

	while (true)
	{
		runStartTime = TickCount();

		SDL_Event event;

		while (SDL_PollEvent(&event))
		{
			if (event.type == SDL_QUIT)
			{
				return;
			}
			else if (event.type == SDL_KEYDOWN)
			{
				if (keyDownFunction != NULL)
					keyDownFunction((Keys)event.key.keysym.scancode);

				keys[event.key.keysym.scancode] = true;
			}
			else if (event.type == SDL_KEYUP)
			{
				keys[event.key.keysym.scancode] = false;
			}
			else if (event.type == SDL_MOUSEMOTION)
			{
				mouseWindowX = event.motion.x;
				mouseWindowY = event.motion.y;

				if (mouseMotionFunction)
					mouseMotionFunction(event.button.x, event.button.y);
			}
			else if (event.type == SDL_MOUSEBUTTONDOWN)
			{
				if (event.button.button == SDL_BUTTON_LEFT)
				{
					mouseLeftButtonDown = true;
					if (mouseLeftButtonDownFunction)
						mouseLeftButtonDownFunction(event.button.x, event.button.y);
				}
				else if (event.button.button == SDL_BUTTON_MIDDLE)
				{
					mouseMiddleButtonDown = true;
				}
				else if (event.button.button == SDL_BUTTON_RIGHT)
				{
					mouseRightButtonDown = true;
					if (mouseRightButtonDownFunction)
						mouseRightButtonDownFunction(event.button.x, event.button.y);
				}
			}
			else if (event.type == SDL_MOUSEBUTTONUP)
			{
				if (event.button.button == SDL_BUTTON_LEFT)
				{
					mouseLeftButtonDown = false;
				}
				else if (event.button.button == SDL_BUTTON_MIDDLE)
				{
					mouseMiddleButtonDown = false;
				}
				else if (event.button.button == SDL_BUTTON_RIGHT)
				{
					mouseRightButtonDown = false;
				}
			}
		}

		if (SDL_GetWindowFlags(window) & SDL_WINDOW_INPUT_FOCUS)
		{
			if (cursorVisible)
			{
				prevMouseX = mouseDesktopX;
				prevMouseY = mouseDesktopY;

				SDL_GetGlobalMouseState(&mouseDesktopX, &mouseDesktopY);

				mouseRelX = mouseDesktopX - prevMouseX;
				mouseRelY = mouseDesktopY - prevMouseY;
			}
			else
			{
				SDL_GetGlobalMouseState(&mouseDesktopX, &mouseDesktopY);

				mouseRelX = mouseDesktopX - screenCenterX;
				mouseRelY = mouseDesktopY - screenCenterY;

				SDL_WarpMouseGlobal(screenCenterX, screenCenterY);
			}
		}

		if (runFunction)
		{
			if (!runFunction())
				return;
		}

		#ifdef MAXEST_FRAMEWORK_SYSTEM_APPLICATION_OPENGL
			SDL_GL_SwapWindow(window);
		#endif

		runStopTime = TickCount();
		lastFrameTime = (float)(runStopTime - runStartTime) / 1000.0f;
	}
}


bool NSystem::Application::IsKeyDown(Keys key)
{
	return keys[(int)key];
}


bool NSystem::Application::MouseLeftButtonDown()
{
	return mouseLeftButtonDown;
}


bool NSystem::Application::MouseMiddleButtonDown()
{
	return mouseMiddleButtonDown;
}


bool NSystem::Application::MouseRightButtonDown()
{
	return mouseRightButtonDown;
}


int NSystem::Application::MouseWindowX()
{
	return mouseWindowX;
}


int NSystem::Application::MouseWindowY()
{
	return mouseWindowY;
}


int NSystem::Application::MouseDesktopX()
{
	return mouseDesktopX;
}


int NSystem::Application::MouseDesktopY()
{
	return mouseDesktopY;
}


int NSystem::Application::MouseRelX()
{
	return mouseRelX;
}


int NSystem::Application::MouseRelY()
{
	return mouseRelY;
}


void NSystem::Application::ShowCursor(bool show)
{
	cursorVisible = show;
	SDL_ShowCursor(show);
}


float NSystem::Application::LastFrameTime()
{
	return lastFrameTime;
}


void NSystem::Application::SetKeyDownFunction(void (*keyDownFunction)(Keys key))
{
	this->keyDownFunction = keyDownFunction;
}


void NSystem::Application::SetMouseMotionFunction(void (*mouseMotionFunction)(int x, int y))
{
	this->mouseMotionFunction = mouseMotionFunction;
}


void NSystem::Application::SetMouseLeftButtonDownFunction(void (*mouseLeftButtonDownFunction)(int x, int y))
{
	this->mouseLeftButtonDownFunction = mouseLeftButtonDownFunction;
}


void NSystem::Application::SetMouseRightButtonDownFunction(void (*mouseRightButtonDownFunction)(int x, int y))
{
	this->mouseRightButtonDownFunction = mouseRightButtonDownFunction;
}
