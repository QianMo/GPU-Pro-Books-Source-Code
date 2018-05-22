
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

#ifndef _DIALOG_H_
#define _DIALOG_H_

#include "Button.h"
#include "../Util/Queue.h"

struct WInfo {
	Widget *widget;
	float x, y;
};

struct DialogTab {
	Queue <WInfo> widgets;
	char *caption;
	float rightX;
};

class Dialog : public Widget, public PushButtonListener {
public:
	Dialog(const float x, const float y, const float w, const float h, const bool modal, const bool hideOnClose);
	virtual ~Dialog();

	int addTab(const char *caption);
	void addWidget(const int tab, Widget *widget, const uint flags = 0);

	void setCurrentTab(const int tab){ currTab = tab; }

	void updateWidgets();

	bool onMouseMove(const int x, const int y);
	bool onMouseButton(const int x, const int y, const MouseButton button, const bool pressed);
	bool onMouseWheel(const int x, const int y, const int scroll);
	bool onKey(const unsigned int key, const bool pressed);
	bool onJoystickAxis(const int axis, const float value);
	bool onJoystickButton(const int button, const bool pressed);
	void onButtonClicked(PushButton *button);

	void draw(Renderer *renderer, const FontID defaultFont, const SamplerStateID linearClamp, const BlendStateID blendSrcAlpha, const DepthStateID depthState);

protected:
	void close();

	float tabHeight;
	float borderWidth;

	Array <DialogTab *> tabs;
	uint currTab;

	PushButton *closeButton;

	int sx, sy;
	bool draging;
	bool closeModeHide;
	bool showSelection;
	bool isModal;
};

#endif // _DIALOG_H_
