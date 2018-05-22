
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

#include <unistd.h>

extern BaseApp *app;

OSStatus key(EventHandlerCallRef nextHandler, EventRef event, void *userData){
	uint32_t kind = GetEventKind(event);

	static uint32_t lastModifiers;
	uint32_t modifiers;
	GetEventParameter(event, kEventParamKeyModifiers, typeUInt32, NULL, sizeof(modifiers), NULL, &modifiers);
	
	if (kind == kEventRawKeyModifiersChanged){
		uint32_t changed = modifiers ^ lastModifiers;
		if (changed & (1 << shiftKeyBit)){
			app->onKey(KEY_SHIFT, (modifiers >> shiftKeyBit) & 0x1);
		}
		if (changed & (1 << controlKeyBit)){
			app->onKey(KEY_CTRL, (modifiers >> controlKeyBit) & 0x1);
		}
		if (changed & (1 << rightShiftKeyBit)){
			app->onKey(KEY_SHIFT, (modifiers >> rightShiftKeyBit) & 0x1);
		}
		if (changed & (1 << rightControlKeyBit)){
			app->onKey(KEY_CTRL, (modifiers >> rightControlKeyBit) & 0x1);
		}

		lastModifiers = modifiers;
	} else {
		uint32_t key;
		GetEventParameter(event, kEventParamKeyCode, typeUInt32, NULL, sizeof(key), NULL, &key);
		if (kind == kEventRawKeyDown){
			if (key == KEY_ENTER && (modifiers & (1 << optionKeyBit))){
				app->toggleFullscreen();
			} else {
				app->onKey(key, true);
			}
		} else if (kind == kEventRawKeyUp){
			app->onKey(key, false);
		}
	}

	return noErr;
}

bool toLocal(Point &dest, Point &src){
	Rect structureBounds;
	GetWindowBounds(app->getWindow(), kWindowStructureRgn, &structureBounds);

	int w = app->getWidth();
	int h = app->getHeight();

	dest.h = src.h - structureBounds.left;
	dest.v = src.v - structureBounds.bottom + h;

	return (dest.h >= 0 && dest.v >= 0 && dest.h < w && dest.v < h);
}

OSStatus mouseButton(EventHandlerCallRef nextHandler, EventRef event, void *userData){
	uint32_t kind = GetEventKind(event);

	uint16_t button;
	GetEventParameter(event, kEventParamMouseButton, typeMouseButton, NULL, sizeof(button), NULL, &button);

	MouseButton b;
	if (button == kEventMouseButtonPrimary)
		b = MOUSE_LEFT;
	else if (button == kEventMouseButtonSecondary)
		b = MOUSE_RIGHT;
	else if (button == kEventMouseButtonTertiary)
		b = MOUSE_MIDDLE;
	else
		return noErr;

	Point p, point;
	GetEventParameter(event, kEventParamMouseLocation, typeQDPoint, NULL, sizeof(p), NULL, &p);
	if (toLocal(point, p)){
		if (kind == kEventMouseDown){
			app->onMouseButton(point.h, point.v, b, true);
		} else if (kind = kEventMouseUp){
			app->onMouseButton(point.h, point.v, b, false);
		}
	}

	return noErr;
}

OSStatus mouseMove(EventHandlerCallRef nextHandler, EventRef event, void *userData){
	Point p, point;
	GetEventParameter(event, kEventParamMouseLocation, typeQDPoint, NULL, sizeof(p), NULL, &p);
	if (toLocal(point, p)){
		GetEventParameter(event, kEventParamMouseDelta, typeQDPoint, NULL, sizeof(p), NULL, &p);

		app->onMouseMove(point.h, point.v, p.h, p.v);
	}

	return noErr;
}

OSStatus mouseWheel(EventHandlerCallRef nextHandler, EventRef event, void *userData){
	Point p;
	GetEventParameter(event, kEventParamMouseLocation, typeQDPoint, NULL, sizeof(p), NULL, &p);

	EventMouseWheelAxis axis;
	GetEventParameter(event, kEventParamMouseWheelAxis, typeMouseWheelAxis, NULL, sizeof(axis), NULL, &axis);
	if (axis == kEventMouseWheelAxisY){
		SInt32 delta;
		GetEventParameter(event, kEventParamMouseWheelDelta, typeLongInteger, NULL, sizeof(delta), NULL, &delta);
		app->onMouseWheel(p.h, p.v, delta);
	}

	return noErr;
}

OSStatus resize(EventHandlerCallRef nextHandler, EventRef event, void *userData){
	Rect rect;
	GetWindowBounds(app->getWindow(), kWindowContentRgn, &rect);

	app->onSize(rect.right - rect.left, rect.bottom - rect.top);

	return noErr;
}

OSStatus close(EventHandlerCallRef nextHandler, EventRef event, void *userData){
	app->closeWindow(true, true);
	return noErr;
}

OSErr quit(const AppleEvent *appleEvent, AppleEvent *reply, long refcon){
	app->closeWindow(true, true);
	return noErr;
}

void idle(EventLoopTimerRef inTimer, void *userData){
	app->updateTime();
	app->makeFrame();
}

int main(int argc, char *argv[]){
	// Be nice to other processes, helps reduce mouse lag
	setpriority(PRIO_PROCESS, 0, 20);

	// Make sure we're running in the exe's directory
/*
	char path[PATH_MAX];
	if (realpath("/proc/self/exe", path)){
		char *slash = strrchr(path, '/');
		if (slash) *slash = '\0';
		chdir(path);
	}
*/
	initCPU();


	// Initialize timer
	app->initTime();

	app->loadConfig();
	app->initGUI();

	if (app->init()){
		app->resetCamera();

		do {
			app->loadConfig();

			//if (!app->initCaps()) break;
			if (!app->initAPI()) break;

			if (!app->load()){
				app->closeWindow(true, false);
				break;
			}

			EventTypeSpec events[3];
			events[0].eventClass = kEventClassKeyboard;
			events[0].eventKind  = kEventRawKeyDown;
			events[1].eventClass = kEventClassKeyboard;
			events[1].eventKind  = kEventRawKeyUp;
			events[2].eventClass = kEventClassKeyboard;
			events[2].eventKind  = kEventRawKeyModifiersChanged;
 			EventHandlerUPP keyUPP = NewEventHandlerUPP(key);
			InstallApplicationEventHandler(keyUPP, 3, events, NULL, NULL);

			events[0].eventClass = kEventClassMouse;
			events[0].eventKind  = kEventMouseDown;
			events[1].eventClass = kEventClassMouse;
			events[1].eventKind  = kEventMouseUp;
 			EventHandlerUPP mouseButtonUPP = NewEventHandlerUPP(mouseButton);
			InstallApplicationEventHandler(mouseButtonUPP, 2, events, NULL, NULL);

			events[0].eventClass = kEventClassMouse;
			events[0].eventKind  = kEventMouseMoved;
			events[1].eventClass = kEventClassMouse;
			events[1].eventKind  = kEventMouseDragged;
 			EventHandlerUPP mouseMoveUPP = NewEventHandlerUPP(mouseMove);
			InstallApplicationEventHandler(mouseMoveUPP, 2, events, NULL, NULL);

			events[0].eventClass = kEventClassMouse;
			events[0].eventKind  = kEventMouseWheelMoved;
 			EventHandlerUPP mouseWheelUPP = NewEventHandlerUPP(mouseWheel);
			InstallApplicationEventHandler(mouseWheelUPP, 1, events, NULL, NULL);

			AEEventHandlerUPP quitUPP = NewAEEventHandlerUPP(quit);
			AEInstallEventHandler(kCoreEventClass, kAEQuitApplication, quitUPP, 0, false);
					
			events[0].eventClass = kEventClassWindow;
			events[0].eventKind  = kEventWindowClose;
			EventHandlerUPP closeUPP = NewEventHandlerUPP(close);
			InstallWindowEventHandler(app->getWindow(), closeUPP, 1, events, NULL, NULL);

			events[0].eventClass = kEventClassWindow;
			events[0].eventKind  = kEventWindowResizeCompleted;
			EventHandlerUPP resizeUPP = NewEventHandlerUPP(resize);
			InstallWindowEventHandler(app->getWindow(), resizeUPP, 1, events, NULL, NULL);

			EventLoopTimerRef timer;
			EventLoopTimerUPP idleUPP = NewEventLoopTimerUPP(idle);
			InstallEventLoopTimer(GetMainEventLoop(), 0.0, 0.0001, idleUPP, NULL, &timer);


			RunApplicationEventLoop();


			RemoveEventLoopTimer(timer);
			DisposeEventLoopTimerUPP(idleUPP);
			DisposeAEEventHandlerUPP(quitUPP);

			DisposeEventHandlerUPP(keyUPP);
			DisposeEventHandlerUPP(mouseButtonUPP);
			DisposeEventHandlerUPP(mouseMoveUPP);
			DisposeEventHandlerUPP(mouseWheelUPP);
			DisposeEventHandlerUPP(resizeUPP);
			DisposeEventHandlerUPP(closeUPP);
			
		} while (!app->isDone());

		app->exit();
	}

	delete app;

	return 0;
}
