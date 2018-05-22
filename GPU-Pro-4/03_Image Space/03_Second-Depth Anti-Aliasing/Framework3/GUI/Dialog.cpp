
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

#include "Dialog.h"

Dialog::Dialog(const float x, const float y, const float w, const float h, const bool modal, const bool hideOnClose){
	setPosition(x, y);
	setSize(w, h);

	borderWidth = 8;
	tabHeight = 32;

	color = vec4(0.75f, 1, 0.5f, 0.65f);

	draging = false;
	closeModeHide = hideOnClose;
	showSelection = false;
	isModal = modal;
	currTab = 0;

	closeButton = new PushButton(x + w - 2 * borderWidth - 24, y + 2 * borderWidth, 24, 24, "x");
	closeButton->setListener(this);
}

Dialog::~Dialog(){
	for (uint i = 0; i < tabs.getCount(); i++){
		DialogTab *tab = tabs[i];
		if (tab->widgets.goToFirst()){
			do {
				delete tab->widgets.getCurrent().widget;
			} while (tab->widgets.goToNext());
		}
		delete tab->caption;
		delete tab;
	}

	delete closeButton;
}

int Dialog::addTab(const char *caption){
	DialogTab *tab = new DialogTab;
	tab->caption = new char[strlen(caption) + 1];
	strcpy(tab->caption, caption);
	return tabs.add(tab);
}

void Dialog::addWidget(const int tab, Widget *widget, const uint flags){
	WInfo wInfo;
	wInfo.widget = widget;
	wInfo.x = widget->getX();
	wInfo.y = widget->getY();

	widget->setPosition(xPos + wInfo.x + 2 * borderWidth, yPos + wInfo.y + 2 * borderWidth + tabHeight);
	tabs[tab]->widgets.addFirst(wInfo);
}

void Dialog::updateWidgets()
{
	for (uint i = 0; i < tabs.getCount(); i++){
		DialogTab *tab = tabs[i];

		if (tab->widgets.goToFirst()){
			do {
				WInfo wi = tab->widgets.getCurrent();
				wi.widget->setPosition(xPos + wi.x + 2 * borderWidth, yPos + wi.y + 2 * borderWidth + tabHeight);
			} while (tab->widgets.goToNext());
		}
	}
	closeButton->setPosition(xPos + width - 2 * borderWidth - 24, yPos + 2 * borderWidth);
}

bool Dialog::onMouseMove(const int x, const int y){
	if (currTab < tabs.getCount()){
		if (draging){
			float dx = float(x - sx);
			float dy = float(y - sy);

			xPos += dx;
			yPos += dy;

			updateWidgets();

			sx = x;
			sy = y;
		} else {
			DialogTab *tab = tabs[currTab];

			if (tab->widgets.goToFirst()){
				do {
					Widget *widget = tab->widgets.getCurrent().widget;
					if (widget->isEnabled() && widget->isVisible() && (widget->isInWidget(x, y) || widget->isCapturing())){
						if (widget->onMouseMove(x, y)){
							tab->widgets.moveCurrentToTop();
							capture = isModal || widget->isCapturing();
							return true;
						}
					}
				} while (tab->widgets.goToNext());
			}

		}
	}

	return true;
}

bool Dialog::onMouseButton(const int x, const int y, const MouseButton button, const bool pressed){
	showSelection = false;

	if (closeButton->isCapturing() || closeButton->isInWidget(x, y)){
		closeButton->onMouseButton(x, y, button, pressed);
		capture = true;
		return true;
	}

	if (currTab < tabs.getCount()){
		DialogTab *tab = tabs[currTab];

		if (tab->widgets.goToFirst()){
			do {
				Widget *widget = tab->widgets.getCurrent().widget;
				if (widget->isEnabled() && widget->isVisible() && (widget->isInWidget(x, y) || widget->isCapturing())){
					if (widget->onMouseButton(x, y, button, pressed)){
						tab->widgets.moveCurrentToTop();
						capture = isModal || widget->isCapturing();
						return true;
					}
				}
			} while (tab->widgets.goToNext());
		}
	}

	if (button == MOUSE_LEFT){
		capture = isModal || pressed;
		if (isInWidget(x, y)){
			draging = pressed;
			sx = x;
			sy = y;

			if (pressed){
				if (x > xPos + 2 * borderWidth && y > yPos + 2 * borderWidth && y < yPos + 2 * borderWidth + tabHeight){
					for (uint i = 0; i < tabs.getCount(); i++){
						if (x < tabs[i]->rightX){
							currTab = i;
							draging = false;
							break;
						}
					}
				}
			}
		}
	}

	return true;
}

bool Dialog::onMouseWheel(const int x, const int y, const int scroll){
	if (currTab < tabs.getCount()){
		DialogTab *tab = tabs[currTab];

		if (tab->widgets.goToFirst()){
			do {
				Widget *widget = tab->widgets.getCurrent().widget;
				if (widget->isEnabled() && widget->isVisible() && (widget->isInWidget(x, y) || widget->isCapturing())){
					if (widget->onMouseWheel(x, y, scroll)){
						tab->widgets.moveCurrentToTop();
						capture = isModal || widget->isCapturing();
						return true;
					}
				}
			} while (tab->widgets.goToNext());
		}
	}

	return true;
}

bool Dialog::onKey(const unsigned int key, const bool pressed){
	if (currTab < tabs.getCount()){
		DialogTab *tab = tabs[currTab];

		if (tab->widgets.goToFirst()){
			if (tab->widgets.getCurrent().widget->onKey(key, pressed)) return true;
		}
		if (pressed){
			if (key == KEY_ESCAPE){
				close();
				return true;
			} else if (key == KEY_TAB){
				if (tab->widgets.goToFirst()){
					Widget *currTop = tab->widgets.getCurrent().widget;

					tab->widgets.goToLast();
					do {
						Widget *widget = tab->widgets.getCurrent().widget;
						if (widget->isEnabled()){
							tab->widgets.moveCurrentToTop();

							currTop->onFocus(false);
							widget->onFocus(true);

							showSelection = true;
							break;
						}
					} while (tab->widgets.goToPrev());
				}
				return true;
			}
			showSelection = false;
		}
	}

	return false;
}

bool Dialog::onJoystickAxis(const int axis, const float value){
	if (currTab < tabs.getCount()){
		DialogTab *tab = tabs[currTab];

		if (tab->widgets.goToFirst()){
			if (tab->widgets.getCurrent().widget->onJoystickAxis(axis, value)) return true;
		}
	}

	return false;
}

bool Dialog::onJoystickButton(const int button, const bool pressed){
	if (currTab < tabs.getCount()){
		DialogTab *tab = tabs[currTab];

		if (tab->widgets.goToFirst()){
			if (tab->widgets.getCurrent().widget->onJoystickButton(button, pressed)) return true;
		}
	}

	return false;
}

void Dialog::onButtonClicked(PushButton *button){
	close();
}

void Dialog::draw(Renderer *renderer, const FontID defaultFont, const SamplerStateID linearClamp, const BlendStateID blendSrcAlpha, const DepthStateID depthState){
	drawSoftBorderQuad(renderer, linearClamp, blendSrcAlpha, depthState, xPos, yPos, xPos + width, yPos + height, borderWidth, 1, 1);

	vec4 black(0, 0, 0, 1);
	vec4 blue(0.3f, 0.4f, 1.0f, 0.65f);

	float x = xPos + 2 * borderWidth;
	float y = yPos + 2 * borderWidth;
	for (uint i = 0; i < tabs.getCount(); i++){
		float tabWidth = 0.75f * tabHeight;
		float cw = renderer->getTextWidth(defaultFont, tabs[i]->caption);
		float newX = x + tabWidth * cw + 6;

		if (i == currTab){
			vec2 quad[] = { MAKEQUAD(x, y, newX, y + tabHeight, 2) };
			renderer->drawPlain(PRIM_TRIANGLE_STRIP, quad, elementsOf(quad), blendSrcAlpha, depthState, &blue);
		}

		vec2 rect[] = { MAKERECT(x, y, newX, y + tabHeight, 2) };
		renderer->drawPlain(PRIM_TRIANGLE_STRIP, rect, elementsOf(rect), BS_NONE, depthState, &black);

		renderer->drawText(tabs[i]->caption, x + 3, y, tabWidth, tabHeight, defaultFont, linearClamp, blendSrcAlpha, depthState);

		tabs[i]->rightX = x = newX;
	}

	vec2 line[] = { MAKEQUAD(xPos + 2 * borderWidth, y + tabHeight - 1, xPos + width - 2 * borderWidth, y + tabHeight + 1, 0) };
	renderer->drawPlain(PRIM_TRIANGLE_STRIP, line, elementsOf(line), BS_NONE, depthState, &black);

	closeButton->draw(renderer, defaultFont, linearClamp, blendSrcAlpha, depthState);

	if (currTab < tabs.getCount()){
		DialogTab *tab = tabs[currTab];

		if (tab->widgets.goToLast()){
			do {
				Widget *widget = tab->widgets.getCurrent().widget;
				if (widget->isVisible()) widget->draw(renderer, defaultFont, linearClamp, blendSrcAlpha, depthState);
			} while (tab->widgets.goToPrev());
		}
		if (showSelection){
			if (tab->widgets.goToFirst()){
				Widget *w = tab->widgets.getCurrent().widget;

				float x = w->getX();
				float y = w->getY();
				vec2 rect[] = { MAKERECT(x - 5, y - 5, x + w->getWidth() + 5, y + w->getHeight() + 5, 1) };
				renderer->drawPlain(PRIM_TRIANGLE_STRIP, rect, elementsOf(rect), BS_NONE, depthState, &black);
			}
		}
	}
}

void Dialog::close(){
	if (closeModeHide){
		visible = false;
		showSelection = false;
	} else {
		dead = true;
	}
	capture = false;
}
