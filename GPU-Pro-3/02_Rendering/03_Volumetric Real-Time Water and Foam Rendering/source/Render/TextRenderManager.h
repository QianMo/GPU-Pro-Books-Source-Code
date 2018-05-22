#ifndef __TEXTRENDERMANAGER__H__
#define __TEXTRENDERMANAGER__H__

#include <string>

#include "../Util/Singleton.h"

class TextRenderManager : public Singleton<TextRenderManager>
{
	friend class Singleton<TextRenderManager>;

public:
	enum FontType
	{
		FONT_8_BY_13 = 0,
		FONT_9_BY_15,
		FONT_TIMES_ROMAN_10,
		FONT_TIMES_ROMAN_24,
		FONT_HELVETICA_10,
		FONT_HELVETICA_12,
		FONT_HELVETICA_18
	};

	TextRenderManager(void);

	// begin text rendering
	void Begin(void);

	// end text rendering
	void End(void);

	// sets spec. projection matrix
	void ResetProjection(const int& width, const int& height);

	// renders a text
	void RenderText(const char* text, const FontType& font, const int& x, const int& y, const float& r, const float& g, const float& b);

private:
	int currentWidth;
	int currentHeight;
};

#endif