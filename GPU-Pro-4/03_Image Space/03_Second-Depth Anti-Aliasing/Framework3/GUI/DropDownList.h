
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

#ifndef _DROPDOWNLIST_H_
#define _DROPDOWNLIST_H_

#include "Widget.h"

class DropDownList;
class DropDownListener {
public:
	virtual ~DropDownListener(){}

	virtual void onDropDownChanged(DropDownList *dropDownList) = 0;
};

class DropDownList : public Widget {
public:
	DropDownList(const float x, const float y, const float w, const float h);
	virtual ~DropDownList();

	int addItem(const char *str);
	int addItemUnique(const char *str);
	void selectItem(const int item);
	int getItem(const char *str) const;
	const char *getText(const int index) const { return items[index]; }
	const char *getSelectedText() const { return items[selectedItem]; }
	int getSelectedItem() const { return selectedItem; }
	void sort();
	void clear();


	void setListener(DropDownListener *listener){ dropDownListener = listener; }

	bool onMouseButton(const int x, const int y, const MouseButton button, const bool pressed);
	bool onMouseWheel(const int x, const int y, const int scroll);
	bool onKey(const unsigned int key, const bool pressed);
	void onFocus(const bool focus);

	void draw(Renderer *renderer, const FontID defaultFont, const SamplerStateID linearClamp, const BlendStateID blendSrcAlpha, const DepthStateID depthState);

protected:
	Array <char *> items;
	int selectedItem;

	DropDownListener *dropDownListener;

	bool isDroppedDown;
};

#endif // _DROPDOWNLIST_H_
