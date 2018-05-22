#include <iostream>
#include <fstream>
#include <string>

#include "../Render/TextRenderManager.h"

#include <GL/glut.h>


// -----------------------------------------------------------------------------
// ----------------------- TextRenderManager::TextRenderManager ----------------
// -----------------------------------------------------------------------------
TextRenderManager::TextRenderManager(void)
{
}


// -----------------------------------------------------------------------------
// ----------------------- TextRenderManager::Begin ----------------------------
// -----------------------------------------------------------------------------
void TextRenderManager::Begin()
{
	glDisable(GL_TEXTURE_2D);
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glOrtho(0, currentWidth, currentHeight, 0, -100, 100);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}


// -----------------------------------------------------------------------------
// ----------------------- TextRenderManager::End ------------------------------
// -----------------------------------------------------------------------------
void TextRenderManager::End()
{
	glEnable(GL_TEXTURE_2D);
}


// -----------------------------------------------------------------------------
// ----------------------- TextRenderManager::ResetProjection ------------------
// -----------------------------------------------------------------------------
void TextRenderManager::ResetProjection(const int& width, const int& height)
{
	currentWidth = width;
	currentHeight = height;
}

// -----------------------------------------------------------------------------
// ----------------------- ConfigLoader::TextRenderManager ---------------------
// -----------------------------------------------------------------------------
void TextRenderManager::RenderText(const char* text, const FontType& font, const int& x, const int& y, const float& r, const float& g, const float& b)
{
	int length;
	length = (int) strlen(text);

	glColor3f(r, g, b);
	glRasterPos2f(x, y);

	int i;
	for (i = 0; i < length; i++)
	{
		switch(font)
		{
		case FONT_8_BY_13:
			glutBitmapCharacter(GLUT_BITMAP_8_BY_13, text[i]);
			break;
		case FONT_9_BY_15:
			glutBitmapCharacter(GLUT_BITMAP_9_BY_15, text[i]);
			break;
		case FONT_TIMES_ROMAN_10:
			glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_10, text[i]);
			break;
		case FONT_TIMES_ROMAN_24:
			glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, text[i]);
			break;
		case FONT_HELVETICA_10:
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, text[i]);
			break;
		case FONT_HELVETICA_12:
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, text[i]);
			break;
		case FONT_HELVETICA_18:
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, text[i]);
			break;
		}
	}
}