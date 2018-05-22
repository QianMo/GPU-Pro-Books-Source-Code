
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

#ifndef _CHECKBOX_H_
#define _CHECKBOX_H_

#include "Widget.h"

class CheckBox;
class CheckBoxListener {
public:
	virtual ~CheckBoxListener(){}

	virtual void onCheckBoxClicked(CheckBox *checkBox) = 0;
};


class CheckBox : public Widget {
public:
	CheckBox(const float x, const float y, const float w, const float h, const char *txt, const bool check = false);
	virtual ~CheckBox();

	void setListener(CheckBoxListener *listener){ checkBoxListener = listener; }
	
	void setChecked(const bool ch){ checked = ch; }
	bool isChecked() const { return checked; }

	bool onMouseButton(const int x, const int y, const MouseButton button, const bool pressed);
	bool onKey(const unsigned int key, const bool pressed);

	void draw(Renderer *renderer, const FontID defaultFont, const SamplerStateID linearClamp, const BlendStateID blendSrcAlpha, const DepthStateID depthState);

protected:
	char *text;

	CheckBoxListener *checkBoxListener;

	bool checked;
};

#endif // _CHECKBOX_H_
