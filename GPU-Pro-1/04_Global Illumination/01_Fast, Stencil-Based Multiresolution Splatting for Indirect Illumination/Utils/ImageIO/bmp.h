/**********************************
** bmp.h                         **
** -----                         **
**                               **
** Read/write 24bit uncompressed **
**   BMP images.  Code borrowed  **
**   from U-Virginia intro GFX   **
**   class pages and modified.   **
**                               **
** Chris Wyman (2/08/2005)       **
**********************************/
#ifndef _MYBMP_H_
#define _MYBMP_H_

#pragma warning( disable: 4996 )


/* define return codes for WriteBMP() */
#ifndef GFXIO_ERRORS
#define GFXIO_ERRORS
    #define GFXIO_OK            0
    #define GFXIO_OPENERROR     1
    #define GFXIO_BADFILE       2
    #define GFXIO_UNSUPPORTED   3
#endif

/* some useful bitmap constants, prefixed with nonsense to not overlap with */
/*    potential MS Windows definitions...                                   */
#define MYBMP_BF_TYPE           0x4D42
#define MYBMP_BF_OFF_BITS       54
#define MYBMP_BI_SIZE           40
#define MYBMP_BI_RGB            0L
#define MYBMP_BI_RLE8           1L
#define MYBMP_BI_RLE4           2L
#define MYBMP_BI_BITFIELDS      3L


/* Reads a 24-bit uncompressed BMP from the file 'f'                        */
/*    Returns:  A pointer to a character array storing the image.  Data is  */
/*              laid out how it is arranged in a PPM file.  1st byte is     */
/*              the red component of the upper left pixel, 2nd byte is the  */
/*              green component of the ul-pixel, then the blue component,   */
/*              the next 3 bytes contain the rgb values of the pixel to the */
/*              right, and pixels are stored in scan-line order.            */
/*              NOTE: Not the layout from the file (where its BGR not RGB)  */
/*    The values stored in *w and *h are the image width & height           */
unsigned char *ReadBMP( char *f, int *width, int *height, bool invertY=false );


/* Writes an uncompressed 24-bit BMP to the file 'f'                        */
/*    Returns:  One of the error codes from above or GFXIO_OK               */
/*    Input:  'f', the filename to write to                                 */
/*            width, the image width                                        */
/*            height, the image height                                      */
/*            ptr, a pointer to an unsigned character / unsigned byte array */
/*                 NOTE: ptr should have length 3*w*h laied out as R,G,B    */
/*                       starting from the upper left pixel, scanline order */
int WriteBMP( char *f, int width, int height, unsigned char *ptr );


/* Utility functions */
void FatalError( char *msg );    /* Prints "Fatal Error: " + msg, then exits */
void Error( char *msg );         /* Prints "Error: " + msg, then continues   */
void Warning( char *msg );       /* Prints "Warning: " + msg, then continues */


#endif
