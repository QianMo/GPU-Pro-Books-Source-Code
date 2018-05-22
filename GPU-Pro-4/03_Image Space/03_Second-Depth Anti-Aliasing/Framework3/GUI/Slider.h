
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

#ifndef _SLIDER_H_
#define _SLIDER_H_

#include "Widget.h"

class Slider;
class SliderListener {
public:
	virtual ~SliderListener(){}

	virtual void onSliderChanged(Slider *Slider) = 0;
};


class Slider : public Widget {
public:
	Slider(const float x, const float y, const float w, const float h, const float minVal = 0, const float maxVal = 1, const float val = 0);
	virtual ~Slider();

	float getValue() const { return value; }
	void setValue(const float val);
	void setRange(const float minVal, const float maxVal);

	void setListener(SliderListener *listener){ sliderListener = listener; }

	bool onMouseMove(const int x, const int y);
	bool onMouseButton(const int x, const int y, const MouseButton button, const bool pressed);

	void draw(Renderer *renderer, const FontID defaultFont, const SamplerStateID linearClamp, const BlendStateID blendSrcAlpha, const DepthStateID depthState);

protected:
	void updateValue(const int x);

	char *text;

	SliderListener *sliderListener;

	float minValue, maxValue, value;
};

#endif // _SLIDER_H_
