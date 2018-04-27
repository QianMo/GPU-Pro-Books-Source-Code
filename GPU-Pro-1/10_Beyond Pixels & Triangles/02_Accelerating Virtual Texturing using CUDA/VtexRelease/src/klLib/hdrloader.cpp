
/***********************************************************************************
	Created:	17:9:2002
	FileName: 	hdrloader.cpp
	Author:		Igor Kravtchenko
	
	Info:		Load HDR image and convert to a set of float32 RGB triplet.
************************************************************************************/
#include "shared.h"
#include "hdrloader.h"
/*
#include <math.h>
#include <memory.h>
#include <stdio.h>*/
#include "FileSystem.h"

typedef unsigned char RGBE[4];
#define R			0
#define G			1
#define B			2
#define E			3

#define  MINELEN	8				// minimum scanline length for encoding
#define  MAXELEN	0x7fff			// maximum scanline length for encoding

static void workOnRGBE(RGBE *scan, int len, klVec4 *cols);

static bool retroDecrunch(RGBE *scanline, int len, std::istream &stream)
{
	int i;
	int rshift = 0;
	
	while (len > 0) {
        scanline[0][R] = stream.get();
		scanline[0][G] = stream.get();
		scanline[0][B] = stream.get();
		scanline[0][E] = stream.get();
        if (stream.eof())
			return false;

		if (scanline[0][R] == 1 &&
			scanline[0][G] == 1 &&
			scanline[0][B] == 1) {
			for (i = scanline[0][E] << rshift; i > 0; i--) {
				memcpy(&scanline[0][0], &scanline[-1][0], 4);
				scanline++;
				len--;
			}
			rshift += 8;
		}
		else {
			scanline++;
			len--;
			rshift = 0;
		}
	}
	return true;
}

static bool decrunch(RGBE *scanline, int len, std::istream &stream)
{
	int  i, j;
					
    if (len < MINELEN || len > MAXELEN) {
		return retroDecrunch(scanline, len, stream);
    }

    i = stream.get();
	if (i != 2) {
        stream.seekg(-1, std::ios_base::cur );
		return retroDecrunch(scanline, len, stream);
	}

    scanline[0][G] = stream.get();
    scanline[0][B] = stream.get();
    i = stream.get();

	if (scanline[0][G] != 2 || scanline[0][B] & 128) {
		scanline[0][R] = 2;
		scanline[0][E] = i;
		return retroDecrunch(scanline + 1, len - 1, stream);
	}

	// read each component
	for (i = 0; i < 4; i++) {
	    for (j = 0; j < len; ) {
            unsigned char code = stream.get();
			if (code > 128) { // run
			    code &= 127;
			    unsigned char val = stream.get();
			    while (code--)
					scanline[j++][i] = val;
			}
			else  {	// non-run
			    while(code--)
					scanline[j++][i] = stream.get();
			}
		}
    }

	//return feof(file) ? false : true;
    return stream.eof() ? false : true;
}

bool HDRLoader::load(std::istream &stream, HDRLoaderResult &res)
{
	int i;
	char str[200];

	stream.read(str, 10);
	if (memcmp(str, "#?RADIANCE", 10)) {
		return false;
	}

    stream.seekg(1, std::ios_base::cur );

    // Parse the header
    // (end is indicated by a blank line)
	char cmd[200];
	i = 0;
	char c = 0, oldc;
	while(true) {
		oldc = c;
        c = stream.get();
		if (c == 0xa && oldc == 0xa)
			break;
		cmd[i++] = c;
	}

    // Parse the resoltuion string
	char reso[200];
	i = 0;
	while(true) {
		c = stream.get();
		reso[i++] = c;
		if (c == 0xa)
			break;
	}

	int w, h;
	if (!sscanf(reso, "-Y %ld +X %ld", &h, &w)) {
        // Unsupported coordinate system
		return false;
	}

	res.width = w;
	res.height = h;

	klVec4 *cols = new klVec4[w * h];
	res.pixels = cols;

	RGBE *scanline = new RGBE[w];
	if (!scanline) {
		return false;
	}

	// convert image 
	for (int y = h - 1; y >= 0; y--) {
		if (decrunch(scanline, w, stream) == false)
			break;
		workOnRGBE(scanline, w, cols);
		cols += w;
	}

	delete [] scanline;
	return true;
}

float convertComponent(int expo, int val)
{
	float v = val / 256.0f;
	float d = (float) pow(2.0, expo);
	return v * d;
}

void workOnRGBE(RGBE *scan, int len, klVec4 *cols)
{
	while (len-- > 0) {
		int expo = scan[0][E] - 128;
		(*cols)[0] = convertComponent(expo, scan[0][R]);
		(*cols)[1] = convertComponent(expo, scan[0][G]);
		(*cols)[2] = convertComponent(expo, scan[0][B]);
        (*cols)[3] = 1.0f;
		cols++;
		scan++;
	}
}
