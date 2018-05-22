
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

#ifndef _TEXTUREPACKER_H_
#define _TEXTUREPACKER_H_

#include "../Platform.h"
#include "Array.h"

struct TextureRectangle {
	uint x, y;
	uint width, height;
};


typedef int (*compareRectFunc)(TextureRectangle *const &elem0, TextureRectangle *const &elem1);

int originalAreaComp(TextureRectangle *const &elem0, TextureRectangle *const &elem1);
int areaComp(TextureRectangle *const &elem0, TextureRectangle *const &elem1);
int widthComp(TextureRectangle *const &elem0, TextureRectangle *const &elem1);
int heightComp(TextureRectangle *const &elem0, TextureRectangle *const &elem1);

class TexturePacker {
public:
	~TexturePacker();

	void addRectangle(uint width, uint height);
	bool assignCoords(uint *width, uint *height, compareRectFunc compRectFunc = originalAreaComp);

	TextureRectangle *getRectangle(uint index) const { return rects[index]; }

protected:
	Array <TextureRectangle *> rects;
};

#endif // _TEXTUREPACKER_H_
