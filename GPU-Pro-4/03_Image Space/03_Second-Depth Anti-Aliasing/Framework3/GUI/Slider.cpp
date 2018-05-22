
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

#include "Slider.h"

Slider::Slider(const float x, const float y, const float w, const float h, const float minVal, const float maxVal, const float val){
	setPosition(x, y);
	setSize(w, h);

	sliderListener = NULL;

	color = vec4(1, 0.2f, 0.2f, 0.65f);

	minValue = minVal;
	maxValue = maxVal;
	setValue(val);
}

Slider::~Slider(){
}

void Slider::setValue(const float val){
	value = clamp(val, minValue, maxValue);
}

void Slider::setRange(const float minVal, const float maxVal){
	minValue = minVal;
	maxValue = maxVal;
	value = clamp(value, minVal, maxVal);
}

bool Slider::onMouseMove(const int x, const int y){
	if (capture){
		updateValue(x);
		return true;
	}
	return false;
}

bool Slider::onMouseButton(const int x, const int y, const MouseButton button, const bool pressed){
	if (button == MOUSE_LEFT){
		if (pressed){
			updateValue(x);
		}
		capture = pressed;
	}

	return true;
}

void Slider::draw(Renderer *renderer, const FontID defaultFont, const SamplerStateID linearClamp, const BlendStateID blendSrcAlpha, const DepthStateID depthState){
	vec4 black(0, 0, 0, 1);

	vec2 quad[] = { MAKEQUAD(xPos, yPos, xPos + width, yPos + height, 2) };
	renderer->drawPlain(PRIM_TRIANGLE_STRIP, quad, elementsOf(quad), blendSrcAlpha, depthState, &color);

	vec2 rect[] = { MAKERECT(xPos, yPos, xPos + width, yPos + height, 2) };
	renderer->drawPlain(PRIM_TRIANGLE_STRIP, rect, elementsOf(rect), BS_NONE, depthState, &black);

	vec2 line[] = { MAKEQUAD(xPos + 0.5f * height, yPos + 0.5f * height - 1, xPos + width - 0.5f * height, yPos + 0.5f * height + 1, 0) };
	renderer->drawPlain(PRIM_TRIANGLE_STRIP, line, elementsOf(line), BS_NONE, depthState, &black);

	float x = lerp(xPos + 0.5f * height, xPos + width - 0.5f * height, (value - minValue) / (maxValue - minValue));
	vec2 marker[] = { MAKEQUAD(x - 0.2f * height, yPos + 0.2f * height, x + 0.2f * height, yPos + 0.8f * height, 0) };
	renderer->drawPlain(PRIM_TRIANGLE_STRIP, marker, elementsOf(marker), BS_NONE, depthState, &black);
}

void Slider::updateValue(const int x){
	float t = saturate((x - (xPos + 0.5f * height)) / (width - height));
	value = lerp(minValue, maxValue, t);

	if (sliderListener) sliderListener->onSliderChanged(this);
}
