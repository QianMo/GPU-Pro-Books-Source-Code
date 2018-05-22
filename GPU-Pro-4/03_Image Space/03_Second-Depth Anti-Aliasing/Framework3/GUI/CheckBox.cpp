
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

#include "CheckBox.h"

CheckBox::CheckBox(const float x, const float y, const float w, const float h, const char *txt, const bool check){
	setPosition(x, y);
	setSize(w, h);

	checkBoxListener = NULL;

	text = new char[strlen(txt) + 1];
	strcpy(text, txt);

	color = vec4(0.5f, 0.75f, 1, 1);

	checked = check;
}

CheckBox::~CheckBox(){
	delete text;
}

bool CheckBox::onMouseButton(const int x, const int y, const MouseButton button, const bool pressed){
	if (button == MOUSE_LEFT && pressed){
		checked = !checked;
		if (checkBoxListener) checkBoxListener->onCheckBoxClicked(this);
	}

	return true;
}

bool CheckBox::onKey(const unsigned int key, const bool pressed){
	if (key == KEY_ENTER || key == KEY_SPACE){
		if (pressed){
			checked = !checked;
			if (checkBoxListener) checkBoxListener->onCheckBoxClicked(this);
		}
		return true;
	}

	return false;
}

void CheckBox::draw(Renderer *renderer, const FontID defaultFont, const SamplerStateID linearClamp, const BlendStateID blendSrcAlpha, const DepthStateID depthState){
	if (check == TEXTURE_NONE){
		uint32 checkPic[] = {
			0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xf5ffffff,0x9744619d,0xffffffff,0xffffffff,
			0x25d8ffff,0xce1a0000,0xffffffff,0xffffffff,0x001adeff,0xffd81a00,0xffffffff,0xffffffff,0x00002aef,0xffffd81a,
			0xffffffff,0xfcffffff,0x1000004a,0xffffffd5,0xffffffff,0x7cffffff,0xc9090000,0xffffffff,0xffffffff,0x01b3ffff,
			0xffb30400,0xffffffff,0xf6d6f5ff,0x0011deff,0xffff9300,0xffffffff,0x6b000fa8,0x000035f6,0xffffff67,0xffffffff,
			0x0c00004b,0x39000055,0xfffffff9,0xffffffff,0x00000075,0xe0130000,0xffffffff,0xffffffff,0x000000c9,0xffaf0000,
			0xffffffff,0xffffffff,0x000034ff,0xffff5c00,0xffffffff,0xffffffff,0x4b5bdaff,0xfffff575,0xffffffff,0xffffffff,
			0xffffffff,0xffffffff,0xffffffff,0xffffffff,
		};
		Image img;
		img.loadFromMemory(checkPic, FORMAT_I8, 16, 16, 1, 1, false);
		img.convert(FORMAT_RGBA8); // For DX10
		check = renderer->addTexture(img, false, linearClamp);
	}

	if (checked){
		TexVertex quad[] = { MAKETEXQUAD(xPos, yPos + 0.2f * height, xPos + 0.6f * height, yPos + 0.8f * height, 3) };
		renderer->drawTextured(PRIM_TRIANGLE_STRIP, quad, elementsOf(quad), check, linearClamp, BS_NONE, depthState);
	} else {
		vec2 quad[] = { MAKEQUAD(xPos, yPos + 0.2f * height, xPos + 0.6f * height, yPos + 0.8f * height, 3) };
		renderer->drawPlain(PRIM_TRIANGLE_STRIP, quad, elementsOf(quad), BS_NONE, depthState);
	}

	vec2 rect[] = { MAKERECT(xPos, yPos + 0.2f * height, xPos + 0.6f * height, yPos + 0.8f * height, 3) };
	vec4 black(0, 0, 0, 1);
	renderer->drawPlain(PRIM_TRIANGLE_STRIP, rect, elementsOf(rect), BS_NONE, depthState, &black);


	float textWidth = 0.75f * height;

	float w = width - 0.7f * height;
	float tw = renderer->getTextWidth(defaultFont, text);
	float maxW = w / tw;
	if (textWidth > maxW) textWidth = maxW;

	float x = 0.7f * height;

	renderer->drawText(text, xPos + x, yPos, textWidth, height, defaultFont, linearClamp, blendSrcAlpha, depthState);
}
