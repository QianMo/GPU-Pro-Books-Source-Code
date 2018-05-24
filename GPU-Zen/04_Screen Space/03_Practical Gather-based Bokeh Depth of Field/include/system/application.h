#pragma once


#include "keys.h"
#include <essentials/types.h>

#include <SDL.h>
#undef main // because of SDL


using namespace NEssentials;


namespace NSystem
{
	class Application
	{
	public:
		Application();

		bool Create(int width, int height, bool fullScreen);
		void Destroy();

		void Run(bool(*runFunction)());

		bool IsKeyDown(Keys key);
		bool MouseLeftButtonDown();
		bool MouseMiddleButtonDown();
		bool MouseRightButtonDown();
		int MouseWindowX();
		int MouseWindowY();
		int MouseDesktopX();
		int MouseDesktopY();
		int MouseRelX();
		int MouseRelY();
		void ShowCursor(bool show);

		float LastFrameTime();

		void SetKeyDownFunction(void (*keyDownFunction)(Keys key));
		void SetMouseMotionFunction(void (*mouseLeftButtonDownFunction)(int x, int y));
		void SetMouseLeftButtonDownFunction(void (*mouseLeftButtonDownFunction)(int x, int y));
		void SetMouseRightButtonDownFunction(void (*mouseRightButtonDownFunction)(int x, int y));

	private:
		bool fullScreen;

		bool keys[512];
		bool mouseLeftButtonDown;
		bool mouseMiddleButtonDown;
		bool mouseRightButtonDown;
		int mouseWindowX;
		int mouseWindowY;
		int mouseDesktopX;
		int mouseDesktopY;
		int mouseRelX;
		int mouseRelY;
		bool cursorVisible;

		uint64 runStartTime, runStopTime;
		float lastFrameTime;

		SDL_Window* window;
		SDL_GLContext glContext;

		void (*keyDownFunction)(Keys key);
		void (*mouseMotionFunction)(int x, int y);
		void (*mouseLeftButtonDownFunction)(int x, int y);
		void (*mouseRightButtonDownFunction)(int x, int y);
	};
}
