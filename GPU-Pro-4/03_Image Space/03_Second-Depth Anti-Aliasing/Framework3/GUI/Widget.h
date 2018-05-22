
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

#ifndef _WIDGET_H_
#define _WIDGET_H_

#include "../Renderer.h"

enum MouseButton {
	MOUSE_LEFT   = 0,
	MOUSE_MIDDLE = 1,
	MOUSE_RIGHT  = 2,
};


class Widget {
public:
	Widget(){
		visible = true;
		capture = false;
		dead = false;
		enabled = true;
	}
	virtual ~Widget(){}

	virtual bool isInWidget(const int x, const int y) const;

	virtual bool onMouseMove(const int x, const int y){ return false; }
	virtual bool onMouseButton(const int x, const int y, const MouseButton button, const bool pressed){ return false; }
	virtual bool onMouseWheel(const int x, const int y, const int scroll){ return false; }
	virtual bool onKey(const unsigned int key, const bool pressed){ return false; }
	virtual bool onJoystickAxis(const int axis, const float value){ return false; }
	virtual bool onJoystickButton(const int button, const bool pressed){ return false; }
	virtual void onSize(const int w, const int h){}
	virtual void onFocus(const bool focus){}

	virtual void draw(Renderer *renderer, const FontID defaultFont, const SamplerStateID linearClamp, const BlendStateID blendSrcAlpha, const DepthStateID depthState) = 0;

	void setPosition(const float x, const float y);
	void setSize(const float w, const float h);
	float getX() const { return xPos; }
	float getY() const { return yPos; }
	float getWidth()  const { return width;  }
	float getHeight() const { return height; }
	void setColor(const vec4 &col){ color = col; }
//	void setCapturing(const bool capturing){ capture = capturing; }
	void setVisible(const bool isVisible){ visible = isVisible; }
	void setEnabled(const bool isEnabled){ enabled = isEnabled; }

	bool isVisible() const { return visible; }
	bool isCapturing() const { return capture; }
	bool isDead() const { return dead; }
	bool isEnabled() const { return enabled; }

	static void clean(){ corner = check = TEXTURE_NONE; }
protected:

	static TextureID corner, check;

	void drawSoftBorderQuad(Renderer *renderer, const SamplerStateID linearClamp, const BlendStateID blendSrcAlpha, const DepthStateID depthState, const float x0, const float y0, const float x1, const float y1, const float borderWidth, const float colScale = 1, const float transScale = 1);

	vec4 color;
	float xPos, yPos, width, height;

	bool visible;
	bool capture;
	bool dead;
	bool enabled;
};

#endif // _WIDGET_H_
