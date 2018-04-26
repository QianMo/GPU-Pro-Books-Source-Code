/**********************************
** bmp.cpp                       **
** -------                       **
**                               **
** Read/write 24bit uncompressed **
**   BMP images.  Code borrowed  **
**   from U-Virginia intro GFX   **
**   class pages and modified.   **
**                               **
** Chris Wyman (2/08/2005)       **
**********************************/

#include <stdio.h>
#include <stdlib.h>
#include "bmp.h"

int ReadInt(FILE *fp);
void WriteInt(int x, FILE *fp);
unsigned short int ReadUnsignedShort(FILE *fp);
void WriteUnsignedShort(unsigned short int x, FILE *fp);
unsigned int ReadUnsignedInt(FILE *fp);
void WriteUnsignedInt(unsigned int x, FILE *fp);


unsigned char *ReadBMP( char *f, int *width, int *height, bool invertY )
{
    unsigned char *img;
	unsigned char *tmp;
    int x, y;
    int lineLength;
    
	FILE *fp;
	char buf[1024];
	unsigned short int bmpType, bmpReserved1, bmpReserved2;
	unsigned int bmpSize, bmpOffBits;

	unsigned int imgSize, imgSizeImage, imgCompression, imgClrUsed, imgClrImportant;
	int imgWidth, imgHeight, imgXPelsPerMeter, imgYPelsPerMeter;
	unsigned short int imgPlanes, imgBitCount;

	fp = fopen( f, "rb" );
	if (!fp)
	{
		sprintf( buf, "ReadBMP() unable to open file '%s'!", f );
		FatalError( buf );
	}

    /* Read file header */
    bmpType = ReadUnsignedShort(fp);
    bmpSize = ReadUnsignedInt(fp);
    bmpReserved1 = ReadUnsignedShort(fp);
    bmpReserved2 = ReadUnsignedShort(fp);
    bmpOffBits = ReadUnsignedInt(fp);

    /* Check file header */
    if (bmpType != MYBMP_BF_TYPE || bmpOffBits != MYBMP_BF_OFF_BITS)
	{
		sprintf( buf, "ReadBMP() encountered bad header in file '%s'!", f);
		FatalError( buf );
	}

    /* Read info header */
    imgSize = ReadUnsignedInt(fp);
    imgWidth = ReadInt(fp);
    imgHeight = ReadInt(fp);
    imgPlanes = ReadUnsignedShort(fp);
    imgBitCount = ReadUnsignedShort(fp);
    imgCompression = ReadUnsignedInt(fp);
    imgSizeImage = ReadUnsignedInt(fp);
    imgXPelsPerMeter = ReadInt(fp);
    imgYPelsPerMeter = ReadInt(fp);
    imgClrUsed = ReadUnsignedInt(fp);
    imgClrImportant = ReadUnsignedInt(fp);

    /* Check info header */
    if( imgSize != MYBMP_BI_SIZE || imgWidth <= 0 || 
		imgHeight <= 0 || imgPlanes != 1 || 
		imgBitCount != 24 || imgCompression != MYBMP_BI_RGB ||
		imgSizeImage == 0 )
	{
		sprintf( buf, "ReadBMP() encountered unsupported bitmap type in '%s'!", f);
		FatalError( buf );
	}
    
	/* compute the line length */
    lineLength = imgWidth * 3; 
    if ((lineLength % 4) != 0) 
		lineLength = (lineLength / 4 + 1) * 4;

    /* Creates the image */
    img = (unsigned char *) malloc( 3 * imgWidth * imgHeight * sizeof( unsigned char ) );
	tmp = (unsigned char *) malloc( 3 * lineLength * sizeof( unsigned char ) );
	if (!tmp || !img)
		FatalError( "Unable to allocate memory in ReadBMP()!");
   
	/* Position the file after header.  Header should be 54 bytes long -- checked above */
    fseek(fp, (long) bmpOffBits, SEEK_SET);  

	/* Read the image */
    for (y = 0; y < imgHeight; y++) {
		int yy = invertY ? (imgHeight-1-y) : y;
        fread(tmp, 1, lineLength, fp);

        /* Copy into permanent structure */
        for (x = 0; x < imgWidth; x++)
		{
			*(img+(yy*3*imgWidth)+3*x+2) = tmp[3*x+0]; 
			*(img+(yy*3*imgWidth)+3*x+1) = tmp[3*x+1]; 
			*(img+(yy*3*imgWidth)+3*x+0) = tmp[3*x+2]; 
		}
    }

	/* cleanup */
    free( tmp );
	fclose( fp );
 
	/* return our image */
	*width = imgWidth;
	*height = imgHeight;
    return img;
}




int WriteBMP( char *f, int width, int height, unsigned char *ptr )
{
	FILE *fp;
	char buf[1024];
    int x, y;
    int lineLength;

	fp = fopen( f, "wb" );
	if (!fp)
	{
		sprintf( buf, "WriteBMP() unable to open file '%s' for writing!\n", f );
		Error( buf );
		return GFXIO_OPENERROR;
	}

    lineLength = width * 3;  
    if ((lineLength % 4) != 0)
		lineLength = (lineLength / 4 + 1) * 4;
    
    /* Write file header */
    WriteUnsignedShort( (unsigned short int) MYBMP_BF_TYPE,								fp);
    WriteUnsignedInt  ( (unsigned int)		(MYBMP_BF_OFF_BITS + lineLength * height),	fp);
    WriteUnsignedShort( (unsigned short int) 0,											fp);
    WriteUnsignedShort( (unsigned short int) 0,											fp);
    WriteUnsignedInt  ( (unsigned short)     MYBMP_BF_OFF_BITS,							fp);

    /* Write info header */
    WriteUnsignedInt  ( (unsigned short int) MYBMP_BI_SIZE,								fp);
    WriteInt          ( (int)				width,										fp);
    WriteInt          ( (int)				height,										fp);
    WriteUnsignedShort( (unsigned short int) 1,											fp);
    WriteUnsignedShort( (unsigned short int) 24,											fp);
    WriteUnsignedInt  ( (unsigned int)		MYBMP_BI_RGB,								fp);
    WriteUnsignedInt  ( (unsigned int)		(lineLength * (unsigned int) height),		fp);
    WriteInt          ( (int)				2925,										fp);
    WriteInt          ( (int)				2925,										fp);
    WriteUnsignedInt  ( (int)				0,											fp);
    WriteUnsignedInt  ( (int)				0,											fp);

    /* Write pixels */
    for (y = 0; y < height; y++) 
	{
		int nbytes = 0;
		for (x = 0; x < width; x++) 
		{
			putc( *(ptr+(y*3*width)+3*x+2), fp), nbytes++;
			putc( *(ptr+(y*3*width)+3*x+1), fp), nbytes++;
			putc( *(ptr+(y*3*width)+3*x+0), fp), nbytes++;
		}
		/* Padding for 32-bit boundary */
		while ((nbytes % 4) != 0) 
		{
			putc(0, fp);
			nbytes++;
		}
    }

	fclose( fp );

	return GFXIO_OK;
}





/* Reads an unsigned short from a file in little endian format */
static unsigned short int ReadUnsignedShort(FILE *fp)
{
    unsigned short int lsb, msb;

    lsb = getc(fp);
    msb = getc(fp);
    return (msb << 8) | lsb;
}



/* Writes as unsigned short to a file in little endian format */
static void WriteUnsignedShort(unsigned short int x, FILE *fp)
{
    unsigned char lsb, msb;

    lsb = (unsigned char) (x & 0x00FF);
    msb = (unsigned char) (x >> 8);
    putc(lsb, fp);
    putc(msb, fp);
}

/* Reads as unsigned int word from a file in little endian format */
static unsigned int ReadUnsignedInt(FILE *fp)
{
    unsigned int b1, b2, b3, b4;

    b1 = getc(fp);
    b2 = getc(fp);
    b3 = getc(fp);
    b4 = getc(fp);
    return (b4 << 24) | (b3 << 16) | (b2 << 8) | b1;
}



/* Writes an unsigned int to a file in little endian format */
static void WriteUnsignedInt(unsigned int x, FILE *fp)
{
    unsigned char b1, b2, b3, b4;

    b1 = (unsigned char) (x & 0x000000FF);
    b2 = (unsigned char) ((x >> 8) & 0x000000FF);
    b3 = (unsigned char) ((x >> 16) & 0x000000FF);
    b4 = (unsigned char) ((x >> 24) & 0x000000FF);
    putc(b1, fp);
    putc(b2, fp);
    putc(b3, fp);
    putc(b4, fp);
}


/* Reads an int word from a file in little endian format */
static int ReadInt(FILE *fp)
{
    int b1, b2, b3, b4;

    b1 = getc(fp);
    b2 = getc(fp);
    b3 = getc(fp);
    b4 = getc(fp);
    return (b4 << 24) | (b3 << 16) | (b2 << 8) | b1;
}


/* Writes an int to a file in little endian format */
static void WriteInt(int x, FILE *fp)
{
    char b1, b2, b3, b4;

    b1 = (x & 0x000000FF);
    b2 = ((x >> 8) & 0x000000FF);
    b3 = ((x >> 16) & 0x000000FF);
    b4 = ((x >> 24) & 0x000000FF);
    putc(b1, fp);
    putc(b2, fp);
    putc(b3, fp);
    putc(b4, fp);
}