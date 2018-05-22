
/* * * * * * * * * * * * * Author's note * * * * * * * * * * * *\
*   _       _   _       _   _       _   _       _     _ _ _ _   *
*  |_|     |_| |_|     |_| |_|_   _|_| |_|     |_|  _|_|_|_|_|  *
*  |_|_ _ _|_| |_|     |_| |_|_|_|_|_| |_|     |_| |_|_ _ _     *
*  |_|_|_|_|_| |_|     |_| |_| |_| |_| |_|     |_|   |_|_|_|_   *
*  |_|     |_| |_|_ _ _|_| |_|     |_| |_|_ _ _|_|  _ _ _ _|_|  *
*  |_|     |_|   |_|_|_|   |_|     |_|   |_|_|_|   |_|_|_|_|    *
*                                                               *
*                     http://www.humus.name                     *
*                                                                *
* This file is a part of the work done by Humus. You are free to   *
* use the code in any way you like, modified, unmodified or copied   *
* into your own work. However, I expect you to respect these points:  *
*  - If you use this file and its contents unmodified, or use a major *
*    part of this file, please credit the author and leave this note. *
*  - For use in anything commercial, please request my approval.     *
*  - Share your work and ideas too as much as you can.             *
*                                                                *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "../CPU.h"
#include "../BaseApp.h"

#include <sys/resource.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include <gtk/gtk.h>

#include <linux/joystick.h>

extern BaseApp *app;

int main(int argc, char *argv[]){
	// Be nice to other processes, helps reduce mouse lag
	setpriority(PRIO_PROCESS, 0, 20);

	gtk_init(NULL, NULL);

	initCPU();

	// Make sure we're running in the exe's directory
	char path[PATH_MAX];
	if (realpath("/proc/self/exe", path)){
		char *slash = strrchr(path, '/');
		if (slash) *slash = '\0';
		chdir(path);
	}

	// Initialize joystick
	int joy = open("/dev/input/js0", O_RDONLY);
	if (joy != -1){
		// Found joystick, set non-blocking mode
		fcntl(joy, F_SETFL, O_NONBLOCK);
	}


	// Initialize timer
	app->initTime();

	app->loadConfig();
	app->initGUI();

	if (app->init()){
		app->resetCamera();

		Display *display = XOpenDisplay(0);
		app->setDisplay(display);

		do {
			app->loadConfig();

			//if (!app->initCaps()) break;
			if (!app->initAPI()) break;

			if (!app->load()){
				app->closeWindow(true, false);
				break;
			}


			XEvent event;
			unsigned int key;
			bool done = false;

			while (true){

				while (XPending(display) > 0){
					XNextEvent(display, &event);

					//printf("%d\n", event.type);

					switch (event.type){
						case Expose:
							if (event.xexpose.count != 0) break;
							break;
						case MotionNotify:
							static int lastX, lastY;
							app->onMouseMove(event.xmotion.x, event.xmotion.y, event.xmotion.x - lastX, event.xmotion.y - lastY);
							lastX = event.xmotion.x;
							lastY = event.xmotion.y;
							break;
						case ConfigureNotify:
							app->onSize(event.xconfigure.width, event.xconfigure.height);
							break;
						case ButtonPress:
							app->onMouseButton(event.xbutton.x, event.xbutton.y, (MouseButton) (event.xbutton.button - 1), true);
							break;
						case ButtonRelease:
							app->onMouseButton(event.xbutton.x, event.xbutton.y, (MouseButton) (event.xbutton.button - 1), false);
							break;
						case KeyPress:
							key = XLookupKeysym(&event.xkey, 0);
							if (key == XK_Return && (event.xkey.state & Mod1Mask)){
								app->toggleFullscreen();
							} else {
								app->onKey(key, true);

								//char str[8];
								//int nChar = XLookupString(&event.xkey, str, sizeof(str), NULL, NULL);
								//for (int i = 0; i < nChar; i++) app->processChar(str[i]);
							}
							break;
						case KeyRelease:
							key = XLookupKeysym(&event.xkey, 0);
							app->onKey(key, false);
							break;
						case ClientMessage:
							if (*XGetAtomName(display, event.xclient.message_type) == *"WM_PROTOCOLS"){
								app->closeWindow(true, true);
							}
							break;
						case DestroyNotify:
							done = true;
							break;
						default:
							break;
					}
				}

				if (done) break;

				/*
					Joystick support
				*/
				if (joy != -1){
			        js_event js;
					while (read(joy, &js, sizeof(js)) > 0){
						switch (js.type & ~JS_EVENT_INIT){
							case JS_EVENT_AXIS:
								//printf("Axis %d: %d\n", js.number, js.value);
								app->onJoystickAxis(js.number, float(js.value) / 32767.0f);
								break;
							case JS_EVENT_BUTTON:
								//printf("Button %d: %d\n", js.number, js.value);
								app->onJoystickButton(js.number, js.value != 0);
								break;
						}
					}
				}

				app->updateTime();
				app->makeFrame();
			}


		} while (!app->isDone());

		XCloseDisplay(display);

		app->exit();
	}

	delete app;

	close(joy);

	return 0;
}
