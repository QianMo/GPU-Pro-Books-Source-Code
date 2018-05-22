
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

#include "DropDownList.h"

DropDownList::DropDownList(const float x, const float y, const float w, const float h){
	setPosition(x, y);
	setSize(w, h);

	color = vec4(1, 1, 0, 0.65f);

	dropDownListener = NULL;

	selectedItem = -1;
	isDroppedDown = false;
}

DropDownList::~DropDownList(){
	clear();
}

int DropDownList::addItem(const char *str){
	char *string = new char[strlen(str) + 1];
	strcpy(string, str);

	return items.add(string);
}

int DropDownList::addItemUnique(const char *str){
	for (uint i = 0; i < items.getCount(); i++){
		if (strcmp(str, items[i]) == 0){
			return i;
		}
	}
	return addItem(str);
}

void DropDownList::selectItem(const int item){
	selectedItem = item;
	if (dropDownListener) dropDownListener->onDropDownChanged(this);
}

int DropDownList::getItem(const char *str) const {
	for (uint i = 0; i < items.getCount(); i++){
		if (strcmp(str, items[i]) == 0){
			return i;
		}
	}

	return -1;
}

void DropDownList::clear(){
	for (uint i = 0; i < items.getCount(); i++){
		delete items[i];
	}
	items.clear();
}

int comp(char * const &elem0, char * const &elem1){
	return strcmp(elem0, elem1);
}

void DropDownList::sort(){
	char *curr = NULL;
	if (selectedItem >= 0) curr = items[selectedItem];

	items.sort(comp);

	if (selectedItem >= 0){
		for (uint i = 0; i < items.getCount(); i++){
			if (strcmp(curr, items[i]) == 0){
				selectedItem = i;
				break;
			}
		}
	}
}

bool DropDownList::onMouseButton(const int x, const int y, const MouseButton button, const bool pressed){
	if (button == MOUSE_LEFT){
		if (pressed){
			if (x > xPos + width - height && x < xPos + width && y > yPos && y < yPos + height){
				if (y < yPos + 0.5f * height){
					if (selectedItem > 0) selectItem(selectedItem - 1);
				} else {
					if (selectedItem + 1 < (int) items.getCount()) selectItem(selectedItem + 1);
				}
			} else {
				if (isDroppedDown){
					int item = int((y - yPos) / height + selectedItem);
					if (item >= 0 && item < (int) items.getCount()) selectItem(item);
				}
				isDroppedDown = !isDroppedDown;
			}
		}
		capture = isDroppedDown;

		return true;
	}

	return false;
}

bool DropDownList::onMouseWheel(const int x, const int y, const int scroll){
	selectedItem -= scroll;

	int count = items.getCount();

	if (selectedItem >= count){
		selectedItem = count - 1;
	} else if (selectedItem < 0){
		selectedItem = 0;
	}

	return true;
}

bool DropDownList::onKey(const unsigned int key, const bool pressed){
	if (pressed){
		switch (key){
		case KEY_UP:
			if (selectedItem > 0) selectItem(selectedItem - 1);
			return true;
		case KEY_DOWN:
			if (selectedItem + 1 < (int) items.getCount()) selectItem(selectedItem + 1);
			return true;
		case KEY_ENTER:
		case KEY_SPACE:
			isDroppedDown = !isDroppedDown;
			return true;
		case KEY_ESCAPE:
			if (!isDroppedDown) return false;
			capture = isDroppedDown = false;
			return true;
		}
	}

	return false;
}

void DropDownList::onFocus(const bool focus){
	capture = isDroppedDown = false;
}

void DropDownList::draw(Renderer *renderer, const FontID defaultFont, const SamplerStateID linearClamp, const BlendStateID blendSrcAlpha, const DepthStateID depthState){
	vec4 col = enabled? color : vec4(color.xyz() * 0.5f, 1);
	vec4 black(0, 0, 0, 1);

	vec2 quad[] = { MAKEQUAD(xPos, yPos, xPos + width, yPos + height, 2) };
	renderer->drawPlain(PRIM_TRIANGLE_STRIP, quad, elementsOf(quad), blendSrcAlpha, depthState, &col);

	vec2 rect[] = { MAKERECT(xPos, yPos, xPos + width, yPos + height, 2) };
	renderer->drawPlain(PRIM_TRIANGLE_STRIP, rect, elementsOf(rect), BS_NONE, depthState, &black);

	vec2 line0[] = { MAKEQUAD(xPos + width - height, yPos + 2, xPos + width - height + 2, yPos + height - 2, 0) };
	renderer->drawPlain(PRIM_TRIANGLE_STRIP, line0, elementsOf(line0), BS_NONE, depthState, &black);
	vec2 line1[] = { MAKEQUAD(xPos + width - height + 1, yPos + 0.5f * height - 1, xPos + width - 2, yPos + 0.5f * height + 1, 0) };
	renderer->drawPlain(PRIM_TRIANGLE_STRIP, line1, elementsOf(line1), BS_NONE, depthState, &black);

	vec2 triangles[] = {
		vec2(xPos + width - 0.5f * height, yPos + 0.1f * height),
		vec2(xPos + width - 0.2f * height, yPos + 0.4f * height),
		vec2(xPos + width - 0.8f * height, yPos + 0.4f * height),
		vec2(xPos + width - 0.5f * height, yPos + 0.9f * height),
		vec2(xPos + width - 0.8f * height, yPos + 0.6f * height),
		vec2(xPos + width - 0.2f * height, yPos + 0.6f * height),
	};
	renderer->drawPlain(PRIM_TRIANGLES, triangles, elementsOf(triangles), BS_NONE, depthState, &black);

	float textWidth = 0.75f * height;
	float w = width - 1.3f * height;
	if (selectedItem >= 0){
		float tw = renderer->getTextWidth(defaultFont, items[selectedItem]);
		float maxW = w / tw;
		if (textWidth > maxW) textWidth = maxW;

		renderer->drawText(items[selectedItem], xPos + 0.15f * height, yPos, textWidth, height, defaultFont, linearClamp, blendSrcAlpha, depthState);
	}

	if (isDroppedDown){
		vec2 quad[] = { MAKEQUAD(xPos, yPos - selectedItem * height, xPos + width - height + 2, yPos + (items.getCount() - selectedItem) * height, 2) };
		renderer->drawPlain(PRIM_TRIANGLE_STRIP, quad, elementsOf(quad), blendSrcAlpha, depthState, &col);

		vec2 rect[] = { MAKERECT(xPos, yPos - selectedItem * height, xPos + width - height + 2, yPos + (items.getCount() - selectedItem) * height, 2) };
		renderer->drawPlain(PRIM_TRIANGLE_STRIP, rect, elementsOf(rect), BS_NONE, depthState, &black);

		for (uint i = 0; i < items.getCount(); i++){
			float tw = renderer->getTextWidth(defaultFont, items[i]);
			float maxW = w / tw;
			if (textWidth > maxW) textWidth = maxW;

			renderer->drawText(items[i], xPos + 0.15f * height, yPos + (int(i) - selectedItem) * height, textWidth, height, defaultFont, linearClamp, blendSrcAlpha, depthState);
		}		
	}

}
