/**
*	EWA filtering on the GPU
*	Copyright 2010-2011 Pavlos Mavridis, All rights reserved
*/

#include "tga.h"

bool Texture::LoadTGA(char *filename,
			 GLenum MINF,
			 GLenum MAGF,
			 int mipmap)
{

	GLubyte		TGAheader[12]={0,0,2,0,0,0,0,0,0,0,0,0};
	GLubyte		TGAcompare[12];								
	GLubyte		header[6];									
	GLuint		bytesPerPixel;								
	GLuint		imageSize;									
	GLuint		temp;										
	GLuint		type=GL_RGBA;

	FILE *file = fopen(filename, "rb");
	
	if(	file==NULL)
		return 0;

	if(	fread(TGAcompare,1,sizeof(TGAcompare),file)!=sizeof(TGAcompare) ||	
		memcmp(TGAheader,TGAcompare,sizeof(TGAheader))!=0				||	
		fread(header,1,sizeof(header),file)!=sizeof(header))			
	{
		fclose(file);
		return 0;
	}

	width  = header[1] * 256 + header[0];
	height = header[3] * 256 + header[2];
    
 	if(	width	<=0	||								
		height	<=0	||								
		(header[4]!=24 && header[4]!=32))					// Is The TGA 24 or 32 Bit?
	{
		fclose(file);										
		return 0;									
	}

	bpp	= header[4];							
	bytesPerPixel	= bpp/8;					
	imageSize		= width*height*bytesPerPixel;	

	imageData=(GLubyte *)malloc(imageSize);		// Reserve Memory To Hold The TGA Data

	if(	imageData==NULL ||							
		fread(imageData, 1, imageSize, file)!=imageSize)	
	{
		if(imageData!=NULL)					
			free(imageData);						

		fclose(file);									
		return 0;									
	}

	for(GLuint i=0; i<int(imageSize); i+=bytesPerPixel)		// Loop Through The Image Data
	{														// Swaps The 1st And 3rd Bytes ('R'ed and 'B'lue)
		temp=imageData[i];							// Temporarily Store The Value At Image Data 'i'
		imageData[i] = imageData[i + 2];	// Set The 1st Byte To The Value Of The 3rd Byte
		imageData[i + 2] = temp;					// Set The 3rd Byte To The Value In 'temp' (1st Byte Value)
	}

	fclose (file);											// Close The File

	glGenTextures(1, &texID);				

	if (bpp==24)								
		type=GL_RGB;										

	glBindTexture(GL_TEXTURE_2D,texID);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, MINF);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, MAGF);

		glTexImage2D(GL_TEXTURE_2D, 0, type, width, height, 0, type, GL_UNSIGNED_BYTE,imageData);


	return true;											
}
