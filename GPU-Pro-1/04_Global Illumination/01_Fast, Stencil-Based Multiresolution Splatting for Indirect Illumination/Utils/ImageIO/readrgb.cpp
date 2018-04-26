/* This was grabbed from here:
**   http://www.cs.utah.edu/classes/cs5610/assignments/lab2/
**  
** Though I modified the final function name to be a little clearer.
** See the header file for usage information.
** 
**   -Chris (1/23/2006)
*/


#include <stdio.h>
#include <stdlib.h> 
#include <string.h>

void
bwtorgba(unsigned char *b,unsigned char *l,int n) {
    while(n--) {
	l[0] = *b;
	l[1] = *b;
	l[2] = *b;
	l[3] = 0xff;
	l += 4; b++;
    }
}

void
latorgba(unsigned char *b, unsigned char *a,unsigned char *l,int n) {
    while(n--) {
	l[0] = *b;
	l[1] = *b;
	l[2] = *b;
	l[3] = *a;
	l += 4; b++; a++;
    }
}

void
rgbtorgba(unsigned char *r,unsigned char *g,unsigned char *b,unsigned char *l,int n) {
    while(n--) {
	l[0] = r[0];
	l[1] = g[0];
	l[2] = b[0];
	l[3] = 0xff;
	l += 4; r++; g++; b++;
    }
}

void
rgbatorgba(unsigned char *r,unsigned char *g,unsigned char *b,unsigned char *a,unsigned char *l,int n) {
    while(n--) {
	l[0] = r[0];
	l[1] = g[0];
	l[2] = b[0];
	l[3] = a[0];
        l += 4; r++; g++; b++; a++;
    }
}

typedef struct _ImageRec {
    unsigned short imagic;
    unsigned short type;
    unsigned short dim;
    unsigned short xsize, ysize, zsize;
    unsigned int min, max;
    unsigned int wasteBytes;
    char name[80];
    unsigned long colorMap;
    FILE *file;
    unsigned char *tmp, *tmpR, *tmpG, *tmpB;
    unsigned long rleEnd;
    unsigned int *rowStart;
    int *rowSize;
} ImageRec;

static void
ConvertShort(unsigned short *array, long length) {
    unsigned b1, b2;
    unsigned char *ptr;

    ptr = (unsigned char *)array;
    while (length--) {
	b1 = *ptr++;
	b2 = *ptr++;
	*array++ = (b1 << 8) | (b2);
    }
}

static void
ConvertLong(unsigned *array, long length) {
    unsigned b1, b2, b3, b4;
    unsigned char *ptr;

    ptr = (unsigned char *)array;
    while (length--) {
	b1 = *ptr++;
	b2 = *ptr++;
	b3 = *ptr++;
	b4 = *ptr++;
	*array++ = (b1 << 24) | (b2 << 16) | (b3 << 8) | (b4);
    }
}

static ImageRec *ImageOpen(const char *fileName)
{
    union {
	int testWord;
	char testByte[4];
    } endianTest;
    ImageRec *image;
    int swapFlag;
    int x;

    endianTest.testWord = 1;
    if (endianTest.testByte[0] == 1) {
	swapFlag = 1;
    } else {
	swapFlag = 0;
    }

    image = (ImageRec *)malloc(sizeof(ImageRec));
    if (image == NULL) {
	fprintf(stderr, "Out of memory!\n");
	exit(1);
    }
    if ((image->file = fopen(fileName, "rb")) == NULL) {
	perror(fileName);
	exit(1);
    }

    fread(image, 1, 12, image->file);

    if (swapFlag) {
	ConvertShort(&image->imagic, 6);
    }

    image->tmp = (unsigned char *)malloc(image->xsize*256);
    image->tmpR = (unsigned char *)malloc(image->xsize*256);
    image->tmpG = (unsigned char *)malloc(image->xsize*256);
    image->tmpB = (unsigned char *)malloc(image->xsize*256);
    if (image->tmp == NULL || image->tmpR == NULL || image->tmpG == NULL ||
	image->tmpB == NULL) {
	fprintf(stderr, "Out of memory!\n");
	exit(1);
    }

    if ((image->type & 0xFF00) == 0x0100) {
	x = image->ysize * image->zsize * sizeof(unsigned);
	image->rowStart = (unsigned *)malloc(x);
	image->rowSize = (int *)malloc(x);
	if (image->rowStart == NULL || image->rowSize == NULL) {
	    fprintf(stderr, "Out of memory!\n");
	    exit(1);
	}
	image->rleEnd = 512 + (2 * x);
	fseek(image->file, 512, SEEK_SET);
	fread(image->rowStart, 1, x, image->file);
	fread(image->rowSize, 1, x, image->file);
	if (swapFlag) {
	    ConvertLong(image->rowStart, x/sizeof(unsigned));
	    ConvertLong((unsigned *)image->rowSize, x/sizeof(int));
	}
    }
    return image;
}

static void
ImageClose(ImageRec *image) {
    fclose(image->file);
    free(image->tmp);
    free(image->tmpR);
    free(image->tmpG);
    free(image->tmpB);
    free(image);
}

static void
ImageGetRow(ImageRec *image, unsigned char *buf, int y, int z) {
    unsigned char *iPtr, *oPtr, pixel;
    int count;

    if ((image->type & 0xFF00) == 0x0100) {
	fseek(image->file, image->rowStart[y+z*image->ysize], SEEK_SET);
	fread(image->tmp, 1, (unsigned int)image->rowSize[y+z*image->ysize],
	      image->file);

	iPtr = image->tmp;
	oPtr = buf;
	while (1) {
	    pixel = *iPtr++;
	    count = (int)(pixel & 0x7F);
	    if (!count) {
		return;
	    }
	    if (pixel & 0x80) {
		while (count--) {
		    *oPtr++ = *iPtr++;
		}
	    } else {
		pixel = *iPtr++;
		while (count--) {
		    *oPtr++ = pixel;
		}
	    }
	}
    } else {
	fseek(image->file, 512+(y*image->xsize)+(z*image->xsize*image->ysize),
	      SEEK_SET);
	fread(buf, 1, image->xsize, image->file);
    }
}

unsigned char *
ReadRGB(const char *name, int *width, int *height, int *components, bool invertY=false ) {
    unsigned *base, *lptr;
    unsigned char *rbuf, *gbuf, *bbuf, *abuf;
    ImageRec *image;
    int y;

    image = ImageOpen(name);
    
    if(!image)
	return NULL;
    (*width)=image->xsize;
    (*height)=image->ysize;
    (*components)=image->zsize;
    base = (unsigned *)malloc(image->xsize*image->ysize*sizeof(unsigned));
    rbuf = (unsigned char *)malloc(image->xsize*sizeof(unsigned char));
    gbuf = (unsigned char *)malloc(image->xsize*sizeof(unsigned char));
    bbuf = (unsigned char *)malloc(image->xsize*sizeof(unsigned char));
    abuf = (unsigned char *)malloc(image->xsize*sizeof(unsigned char));
    if(!base || !rbuf || !gbuf || !bbuf)
      return NULL;
    lptr = base;
    for(y=0; y<image->ysize; y++) {
	int yy = invertY ? (image->ysize-1-y) : y;
	if(image->zsize>=4) {
	    ImageGetRow(image,rbuf,yy,0);
	    ImageGetRow(image,gbuf,yy,1);
	    ImageGetRow(image,bbuf,yy,2);
	    ImageGetRow(image,abuf,yy,3);
	    rgbatorgba(rbuf,gbuf,bbuf,abuf,(unsigned char *)lptr,image->xsize);
	    lptr += image->xsize;
	} else if(image->zsize==3) {
	    ImageGetRow(image,rbuf,yy,0);
	    ImageGetRow(image,gbuf,yy,1);
	    ImageGetRow(image,bbuf,yy,2);
	    rgbtorgba(rbuf,gbuf,bbuf,(unsigned char *)lptr,image->xsize);
	    lptr += image->xsize;
	} else if(image->zsize==2) {
	    ImageGetRow(image,rbuf,yy,0);
	    ImageGetRow(image,abuf,yy,1);
	    latorgba(rbuf,abuf,(unsigned char *)lptr,image->xsize);
	    lptr += image->xsize;
	} else {
	    ImageGetRow(image,rbuf,yy,0);
	    bwtorgba(rbuf,(unsigned char *)lptr,image->xsize);
	    lptr += image->xsize;
	}
    }
    ImageClose(image);
    free(rbuf);
    free(gbuf);
    free(bbuf);
    free(abuf);

    return (unsigned char*) base;
}

