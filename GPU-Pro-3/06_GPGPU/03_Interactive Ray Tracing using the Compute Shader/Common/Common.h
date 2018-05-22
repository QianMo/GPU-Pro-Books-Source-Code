// ================================================================================ //
// Copyright (c) 2011 Arturo Garcia, Francisco Avila, Sergio Murguia and Leo Reyes	//
//																					//
// Permission is hereby granted, free of charge, to any person obtaining a copy of	//
// this software and associated documentation files (the "Software"), to deal in	//
// the Software without restriction, including without limitation the rights to		//
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies	//
// of the Software, and to permit persons to whom the Software is furnished to do	//
// so, subject to the following conditions:											//
//																					//
// The above copyright notice and this permission notice shall be included in all	//
// copies or substantial portions of the Software.									//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR		//
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,			//
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE		//
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER			//
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,	//
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE	//
// SOFTWARE.																		//
// ================================================================================ //

// ------------------------------------------------
// Common.h
// ------------------------------------------------
// Defines common functions and macros 
// for all projects.

#ifndef COMMON_H
#define COMMON_H

// Keeps track of wich Ray trace mode is being used
// 1 = CPU Version
// 2 = GPU Version
// 3 = GPU/CS Version
#ifndef APP_MODE
#define APP_MODE 3
#endif

// Include Common Libraries
#ifdef WINDOWS
#include "windows.h"
#include <tchar.h>
#endif

#include <sys/stat.h> 
#include <sys/types.h>
#include <stdlib.h>
#include <malloc.h>
#include <memory.h>
#include <stdio.h>
#include <utility>
#include <vector>
#include <map>
#include <string>
#include <stdio.h>
#include <cassert>

#ifdef LINUX
#include <DataTypes.h>
#endif

// Constant Variables
const int TEXTURE_WIDTH = 1024;									// Client Area Width
const int TEXTURE_HEIGHT = 1024;								// Client Area Height
const int SCREEN_MULTIPLIER = 1;								// Size of every texel on screen
const int NUM_PIXELS = TEXTURE_WIDTH * TEXTURE_HEIGHT;			// Total Number of Pixels in the Texture
const unsigned int WIDTH = TEXTURE_WIDTH*SCREEN_MULTIPLIER;				// Width of the Screen
const unsigned int HEIGHT = TEXTURE_HEIGHT*SCREEN_MULTIPLIER;			// Height of the Screen
const int SIZE_OF_RESULT = NUM_PIXELS;							// Size of the Result Array
const int NUM_LIGHTS = 1;										// Total number of lights

const int NUM_MATERIALS = 7;									// Number of materials in GPU version
const unsigned int NUM_RAYS = HEIGHT*WIDTH;								// Total number of primary rays in GPU version

const double PI = 3.14159265;
const double T_PI = 6.28318531;
const float	CLEAR_BLACK[]={0.0f,0.0f,0.0f,0.0f};

#ifndef clamp
#define clamp(m,M,v) (v<m ? m: v>M ? M:v)
#endif

// Safe Release Macro
#ifndef SAFE_RELEASE
#define SAFE_RELEASE(P) if (P!=NULL){P->Release();P=NULL;}
#endif

// Safe Delete Macro
#ifndef SAFE_DELETE
#define SAFE_DELETE(P) if (P!=NULL){delete[] P;P=NULL;}
#endif

#if defined(DEBUG) | defined(_DEBUG)
	#ifndef HR
	#define HR(x){HRESULT hr = x; if(FAILED(hr)) return hr; }
	#endif
#else
	#ifndef HR
	#define HR(x){ HRESULT hr = x; if(FAILED(hr)) return hr; }
	#endif
#endif

#ifdef WINDOWS

inline void listCurDir(std::string dir, std::vector<string>& fnVec)
{
	WIN32_FIND_DATA fd;
	//HANDLE h = FindFirstFile((LPCWSTR)dir.c_str(), &fd);
	HANDLE h = FindFirstFile((LPCWSTR)"*.*", &fd);
	fnVec.clear();

	string str;
	int size = sizeof(fd.cFileName)/sizeof(fd.cFileName[0]);
	for(int i = 0; i < size; ++i)
	{
		str += fd.cFileName[i];
	}
	fnVec.push_back(str);
	while (FindNextFile(h, &fd))
	{
		//fnVec.push_back(fd.cFileName);
	}
}
#endif

inline bool fileExists(string strFilename) 
{ 
	struct stat stFileInfo; 

	// Attempt to get the file attributes
	return !stat(strFilename.c_str(),&stFileInfo);
}

inline void startTimer(LARGE_INTEGER &LocalTimer, LARGE_INTEGER &Freq)
{
	#ifdef WINDOWS
		QueryPerformanceFrequency(&Freq);
		QueryPerformanceCounter(&LocalTimer);
	#elif defined(LINUX)
		timeval t1;
		gettimeofday(&t1,NULL);
		LocalTimer = t1.tv_sec + (t1.tv_usec/1000000.0);
	#endif
}

inline void calculateTime(LARGE_INTEGER &LocalTimer, LARGE_INTEGER &Freq, float &totalTime)
{
	#ifdef WINDOWS
		LARGE_INTEGER FinalTimer;
		QueryPerformanceCounter(&FinalTimer);
		totalTime = float(FinalTimer.QuadPart-LocalTimer.QuadPart)/float(Freq.QuadPart);
	#elif defined(LINUX)
		timeval t2;
		gettimeofday(&t2,NULL);
		FinalTimer = t2.tv_sec + (t2.tv_usec/1000000.0);
		totalTime = FinalTimer-LocalTimer;
	#endif
}

#ifdef LINUX
inline void getTime(double* time)
{
	timeval t1;
	gettimeofday(&t1,NULL);
    *time = t1.tv_sec + (t1.tv_usec/1000000.0);
}


#include <stdarg.h>
#include <unistd.h>
#include <jpeglib.h>
#include <png.h>

void abort_(const char * s, ...)
{
	va_list args;
	va_start(args, s);
	vfprintf(stderr, s, args);
	fprintf(stderr, "\n");
	va_end(args);
	abort();
}

int write_jpeg_file( char *filename )
{
	/* The g_Result buffer uses 4 bytes per pixel, RGBA.
	 * We need 3 bytes per pixel. The copy_buffer function,
	 * copies the 4 bytes pixel array into a 3 bytes pixel array
	 */
	unsigned char* rgb_buffer = new unsigned char[TEXTURE_WIDTH * TEXTURE_HEIGHT * 3];
	copy_buffer(g_Result,rgb_buffer,TEXTURE_WIDTH,TEXTURE_HEIGHT);

	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;
	
	/* this is a pointer to one row of image data */
	JSAMPROW row_pointer[1];
	FILE *outfile = fopen( filename, "wb" );
	
	if ( !outfile )
	{
		printf("Error opening output jpeg file %s\n!", filename );
		return -1;
	}
	cinfo.err = jpeg_std_error( &jerr );
	jpeg_create_compress(&cinfo);
	jpeg_stdio_dest(&cinfo, outfile);

	/* Setting the parameters of the output file here */
	cinfo.image_width = TEXTURE_WIDTH;	
	cinfo.image_height = TEXTURE_HEIGHT;
	cinfo.input_components = 3;
	cinfo.in_color_space = JCS_RGB;
    /* default compression parameters, we shouldn't be worried about these */
	jpeg_set_defaults( &cinfo );

	jpeg_set_quality(&cinfo,255,TRUE);
	/* Now do the compression .. */
	jpeg_start_compress( &cinfo, TRUE );
	/* like reading a file, this time write one row at a time */
	while( cinfo.next_scanline < cinfo.image_height )
	{
		row_pointer[0] = &rgb_buffer[ cinfo.next_scanline * cinfo.image_width *  cinfo.input_components];
		jpeg_write_scanlines( &cinfo, row_pointer, 1 );
	}
	/* similar to read file, clean up after we're done compressing */
	jpeg_finish_compress( &cinfo );
	jpeg_destroy_compress( &cinfo );
	fclose( outfile );
	delete [] rgb_buffer;
	/* success code is 1! */
	return 1;
}

void copy_buffer(unsigned char* old_buffer, unsigned char* new_buffer,  int Width, int Height)
{
	int x, y, Ofs = 0, Ofs2 = 0;
	unsigned char R, G, B;
	for (y = 0; y < Height; y++)
	{
		for (x = 0; x < Width; x++)
		{
			R = old_buffer[Ofs2++];
			G = old_buffer[Ofs2++];
			B = old_buffer[Ofs2++];
			Ofs2++;
			// Attention: Blue is stored first! Red comes last!
			new_buffer[Ofs++] = R;
			new_buffer[Ofs++] = G;
			new_buffer[Ofs++] = B;
		}
	}
}
#endif
#endif