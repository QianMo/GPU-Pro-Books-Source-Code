#ifndef _BITMAP_H_
#define _BITMAP_H_

struct bmpColor
{
	unsigned char B;
	unsigned char G;
	unsigned char R;
};

class Bitmap
{
public:
	Bitmap();
	Bitmap(const char *path);
	Bitmap(int width, int height);

	~Bitmap();

	bool create(const char *path);
	bool create(int width, int height);

	bool save(const char *path);

	inline bmpColor * getData();

	inline int getWidth();
	inline int getHeight();

	inline void getColor(int x, int y, unsigned char &red, unsigned char &green, unsigned char &blue);
	inline void setColor(int x, int y, unsigned char red, unsigned char green, unsigned char blue);

#pragma pack(push, 1)
	struct BmpHeaderInfo
	{
		// BMP File Header
		unsigned short bfType;
		unsigned int bfSize;
		unsigned short bfReserved1;
		unsigned short bfReserved2;
		unsigned int bfOffBits;

		// Bitmap Information
		unsigned int biSize;
		int biWidth;
		int biHeight;
		unsigned short biPlanes;
		unsigned short biBitCount;
		unsigned int biCompression;
		unsigned int biSizeImage;
		int biXpelsPerMeter;
		int biYpelsPerMeter;
		unsigned int biClrUsed;
		unsigned int biClrImportant;
	};
#pragma pack(pop)

	int _width;
	int _height;
	bmpColor *_data;

};

bmpColor * Bitmap::getData()
{
	return _data;
}

int Bitmap::getWidth()
{
	return _width;
}

int Bitmap::getHeight()
{
	return _height;
}

void Bitmap::getColor(int x, int y, unsigned char &r, unsigned char &g, unsigned char &b)
{
	int pos = x + (_height - 1 - y) * _width;
	bmpColor rgb = _data[pos];
	r = rgb.R;
	g = rgb.G;
	b = rgb.B;	
}

void Bitmap::setColor(int x, int y, unsigned char r, unsigned char g, unsigned char b)
{

	int pos = x + (_height - 1 - y) * _width;
	_data[pos].R = r;
	_data[pos].G = g;
	_data[pos].B = b;
	
}

#endif