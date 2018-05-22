
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

#include "Button.h"

PushButton::PushButton(const float x, const float y, const float w, const float h, const char *txt){
	setPosition(x, y);
	setSize(w, h);

	buttonListener = NULL;

	text = new char[strlen(txt) + 1];
	strcpy(text, txt);

	color = vec4(0.5f, 0.75f, 1, 1);

	pushed = false;
}

PushButton::~PushButton(){
	delete text;
}

bool PushButton::onMouseButton(const int x, const int y, const MouseButton button, const bool pressed){
	if (pressed == pushed) return false;

	if (button == MOUSE_LEFT){
		if (pressed){
			pushed  = true;
			capture = true;
		} else {
			if (buttonListener && isInWidget(x, y)) buttonListener->onButtonClicked(this);
			pushed  = false;
			capture = false;
		}
	}

	return true;
}

bool PushButton::onKey(const unsigned int key, const bool pressed){
	if (key == KEY_ENTER || key == KEY_SPACE){
		if (buttonListener && pushed && !pressed) buttonListener->onButtonClicked(this);
		pushed = pressed;
		return true;
	}

	return false;
}

void PushButton::draw(Renderer *renderer, const FontID defaultFont, const SamplerStateID linearClamp, const BlendStateID blendSrcAlpha, const DepthStateID depthState){
	drawButton(renderer, text, defaultFont, linearClamp, blendSrcAlpha, depthState);
}

void PushButton::drawButton(Renderer *renderer, const char *text, const FontID defaultFont, const SamplerStateID linearClamp, const BlendStateID blendSrcAlpha, const DepthStateID depthState){
	vec4 black(0, 0, 0, 1);
	vec4 col = color;
	if (pushed) col *= vec4(0.5f, 0.5f, 0.5f, 1);

	vec2 quad[] = { MAKEQUAD(xPos, yPos, xPos + width, yPos + height, 2) };
	renderer->drawPlain(PRIM_TRIANGLE_STRIP, quad, elementsOf(quad), blendSrcAlpha, depthState, &col);

	vec2 rect[] = { MAKERECT(xPos, yPos, xPos + width, yPos + height, 2) };
	renderer->drawPlain(PRIM_TRIANGLE_STRIP, rect, elementsOf(rect), BS_NONE, depthState, &black);


	float textWidth = 0.75f * height;

	float tw = renderer->getTextWidth(defaultFont, text);
	float maxW = width / tw;
	if (textWidth > maxW) textWidth = maxW;

	float x = 0.5f * (width - textWidth * tw);

	renderer->drawText(text, xPos + x, yPos, textWidth, height, defaultFont, linearClamp, blendSrcAlpha, depthState);
}

/***********************************************************************************************************/

KeyWaiterButton::KeyWaiterButton(const float x, const float y, const float w, const float h, const char *txt, uint *key) : PushButton(x, y, w, h, txt){
	targetKey = key;
	waitingForKey = false;
	if (*key == 0){
		// Disabled
		color = vec4(0.1f, 0.1f, 0.1f, 1);
	}
}

KeyWaiterButton::~KeyWaiterButton(){

}

bool KeyWaiterButton::onMouseButton(const int x, const int y, const MouseButton button, const bool pressed){
	if (pressed == pushed) return false;

	if (button == MOUSE_LEFT){
		if (pressed){
			pushed  = true;
			capture = true;
		} else {
			if (isInWidget(x, y)){
				if (buttonListener) buttonListener->onButtonClicked(this);

				waitingForKey = true;
			}
			pushed = false;
		}
	}

	return true;
}

bool KeyWaiterButton::onKey(const unsigned int key, const bool pressed){
	if (waitingForKey){
		if (key != KEY_ESCAPE){
			if (key == KEY_DELETE && *targetKey != 0){
				*targetKey = 0;
				color = vec4(0.1f, 0.1f, 0.1f, 1);
			} else {
				*targetKey = key;
				color = vec4(0.5f, 0.1f, 1, 1);
			}
		}
		waitingForKey = false;
		capture = false;
		return true;
	} else if (key == KEY_SPACE || key == KEY_ENTER){
		if (pushed && !pressed){
			if (buttonListener) buttonListener->onButtonClicked(this);
			waitingForKey = true;
		}
		pushed = pressed;
		return true;
	}

	return false;
}

void KeyWaiterButton::draw(Renderer *renderer, const FontID defaultFont, const SamplerStateID linearClamp, const BlendStateID blendSrcAlpha, const DepthStateID depthState){
	if (waitingForKey){
		drawButton(renderer, "<Press key>", defaultFont, linearClamp, blendSrcAlpha, depthState);
	} else {
		drawButton(renderer, text, defaultFont, linearClamp, blendSrcAlpha, depthState);
	}
}

/***********************************************************************************************************/

AxisWaiterButton::AxisWaiterButton(const float x, const float y, const float w, const float h, const char *txt, int *axis, bool *axisInvert) : PushButton(x, y, w, h, txt){
	targetAxis = axis;
	targetAxisInvert = axisInvert;
	waitingForAxis = false;
	if (*axis < 0){
		// Disabled
		color = vec4(0.1f, 0.1f, 0.1f, 1);
	}
}

AxisWaiterButton::~AxisWaiterButton(){

}

bool AxisWaiterButton::onMouseButton(const int x, const int y, const MouseButton button, const bool pressed){
	if (pressed == pushed) return false;

	if (button == MOUSE_LEFT){
		if (pressed){
			pushed  = true;
			capture = true;
		} else {
			if (isInWidget(x, y)){
				if (buttonListener) buttonListener->onButtonClicked(this);

				waitingForAxis = true;
			}
			pushed = false;
		}
	}

	return true;
}

bool AxisWaiterButton::onKey(const unsigned int key, const bool pressed){
	if (waitingForAxis){
		if (pressed && key == KEY_DELETE){
			*targetAxis = -1;
			*targetAxisInvert = false;
			color = vec4(0.1f, 0.1f, 0.1f, 1);

			waitingForAxis = false;
			capture = false;
			return true;
		}
	} else if (key == KEY_SPACE || key == KEY_ENTER){
		if (pushed && !pressed){
			if (buttonListener) buttonListener->onButtonClicked(this);
			waitingForAxis = true;
		}
		pushed = pressed;
		return true;
	}

	return false;
}

bool AxisWaiterButton::onJoystickAxis(const int axis, const float value){
	if (waitingForAxis && fabsf(value) > 0.5f){
		*targetAxis = axis;
		*targetAxisInvert = (value < 0.0f);
		color = vec4(0.5f, 0.1f, 1, 1);

		waitingForAxis = false;
		capture = false;
		return true;
	}

	return false;
}

void AxisWaiterButton::draw(Renderer *renderer, const FontID defaultFont, const SamplerStateID linearClamp, const BlendStateID blendSrcAlpha, const DepthStateID depthState){
	if (waitingForAxis){
		drawButton(renderer, "<Push stick>", defaultFont, linearClamp, blendSrcAlpha, depthState);
	} else {
		drawButton(renderer, text, defaultFont, linearClamp, blendSrcAlpha, depthState);
	}
}

/***********************************************************************************************************/

ButtonWaiterButton::ButtonWaiterButton(const float x, const float y, const float w, const float h, const char *txt, int *button) : PushButton(x, y, w, h, txt){
	targetButton = button;
	waitingForButton = false;
	if (button < 0){
		// Disabled
		color = vec4(0.1f, 0.1f, 0.1f, 1);
	}
}

ButtonWaiterButton::~ButtonWaiterButton(){

}

bool ButtonWaiterButton::onMouseButton(const int x, const int y, const MouseButton button, const bool pressed){
	if (pressed == pushed) return false;

	if (button == MOUSE_LEFT){
		if (pressed){
			pushed  = true;
			capture = true;
		} else {
			if (isInWidget(x, y)){
				if (buttonListener) buttonListener->onButtonClicked(this);

				waitingForButton = true;
			}
			pushed = false;
		}
	}

	return true;
}

bool ButtonWaiterButton::onKey(const unsigned int key, const bool pressed){
	if (waitingForButton){
		if (pressed && key == KEY_DELETE){
			*targetButton = -1;
			color = vec4(0.1f, 0.1f, 0.1f, 1);

			waitingForButton = false;
			capture = false;
			return true;
		}
	} else if (key == KEY_SPACE || key == KEY_ENTER){
		if (pushed && !pressed){
			if (buttonListener) buttonListener->onButtonClicked(this);
			waitingForButton = true;
		}
		pushed = pressed;
		return true;
	}

	return false;
}

bool ButtonWaiterButton::onJoystickButton(const int button, const bool pressed){
	if (waitingForButton){
		*targetButton = button;
		color = vec4(0.5f, 0.1f, 1, 1);

		waitingForButton = false;
		capture = false;
		return true;
	}

	return false;
}

void ButtonWaiterButton::draw(Renderer *renderer, const FontID defaultFont, const SamplerStateID linearClamp, const BlendStateID blendSrcAlpha, const DepthStateID depthState){
	if (waitingForButton){
		drawButton(renderer, "<Press button>", defaultFont, linearClamp, blendSrcAlpha, depthState);
	} else {
		drawButton(renderer, text, defaultFont, linearClamp, blendSrcAlpha, depthState);
	}
}
