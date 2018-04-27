/**
 *
 *  This software module was originally developed for research purposes,
 *  by Multimedia Lab at Ghent University (Belgium).
 *  Its performance may not be optimized for specific applications.
 *
 *  Those intending to use this software module in hardware or software products
 *  are advized that its use may infringe existing patents. The developers of 
 *  this software module, their companies, Ghent Universtity, nor Multimedia Lab 
 *  have any liability for use of this software module or modifications thereof.
 *
 *  Ghent University and Multimedia Lab (Belgium) retain full right to modify and
 *  use the code for their own purpose, assign or donate the code to a third
 *  party, and to inhibit third parties from using the code for their products. 
 *
 *  This copyright notice must be included in all copies or derivative works.
 *
 *  For information on its use, applications and associated permission for use,
 *  please contact Prof. Rik Van de Walle (rik.vandewalle@ugent.be). 
 *
 *  Detailed information on the activities of
 *  Ghent University Multimedia Lab can be found at
 *  http://multimedialab.elis.ugent.be/.
 *
 *  Copyright (c) Ghent University 2004-2009.
 *
 **/

/* Based on code by Tony Lin, which is based on code by the IJG 
    http://www.codeproject.com/KB/graphics/tonyjpeglib.aspx
*/
#include <string.h> //for memset
#include "SimpleDCTDec.h"

SimpleDCTDec::SimpleDCTDec( int _quality ) : quality(_quality) {
	initHuffmanTables();
    setQuality(_quality);
    setColorSpace(CS_RGBA);
}

void SimpleDCTDec::setColorSpace(ColorSpace cs) {
    if ( cs == CS_RGBA ) {
        colorConverter = YCoCgAToRGBA;
    } else {
        colorConverter = YCoCgAToYCoCgA;
    }
}

SimpleDCTDec::~SimpleDCTDec(void) {}

void SimpleDCTDec::setQuality( int quality ) {
    this->quality = quality;

	// These are the sample quantization tables given in JPEG spec section K.1.
	// The spec says that the values given produce "good" quality, and
	// when divided by 2, "very good" quality.	

	static unsigned short std_luminance_quant_tbl[64] = 
	{
			16,  11,  10,  16,  24,  40,  51,  61,
			12,  12,  14,  19,  26,  58,  60,  55,
			14,  13,  16,  24,  40,  57,  69,  56,
			14,  17,  22,  29,  51,  87,  80,  62,
			18,  22,  37,  56,  68, 109, 103,  77,
			24,  35,  55,  64,  81, 104, 113,  92,
			49,  64,  78,  87, 103, 121, 120, 101,
			72,  92,  95,  98, 112, 100, 103,  99
	};
	static unsigned short std_chrominance_quant_tbl[64] = 
	{
			17,  18,  24,  47,  99,  99,  99,  99,
			18,  21,  26,  66,  99,  99,  99,  99,
			24,  26,  56,  99,  99,  99,  99,  99,
			47,  66,  99,  99,  99,  99,  99,  99,
			99,  99,  99,  99,  99,  99,  99,  99,
			99,  99,  99,  99,  99,  99,  99,  99,
			99,  99,  99,  99,  99,  99,  99,  99,
			99,  99,  99,  99,  99,  99,  99,  99
	};


	/*  For AA&N IDCT method, divisors are equal to quantization
	*	coefficients scaled by scalefactor[row]*scalefactor[col], where
	*		scalefactor[0] = 1
	*		scalefactor[k] = cos(k*PI/16) * sqrt(2)    for k=1..7
	*	We apply a further scale factor of 8.
	*/	
	static unsigned short aanscales[64] = {
			/* precomputed values scaled up by 14 bits */
			16384, 22725, 21407, 19266, 16384, 12873,  8867,  4520,
			22725, 31521, 29692, 26722, 22725, 17855, 12299,  6270,
			21407, 29692, 27969, 25172, 21407, 16819, 11585,  5906,
			19266, 26722, 25172, 22654, 19266, 15137, 10426,  5315,
			16384, 22725, 21407, 19266, 16384, 12873,  8867,  4520,
			12873, 17855, 16819, 15137, 12873, 10114,  6967,  3552,
			 8867, 12299, 11585, 10426,  8867,  6967,  4799,  2446,
			 4520,  6270,  5906,  5315,  4520,  3552,  2446,  1247
	};
	
	// Safety checking. Convert 0 to 1 to avoid zero divide. 
	scale = quality;
	if (scale <= 0) 
		scale = 1;
	if (scale > 100) 
		scale = 100;
	
	//	Non-linear map: 1->5000, 10->500, 25->200, 50->100, 75->50, 100->0
	if (scale < 50)
		scale = 5000 / scale;
	else
		scale = 200 - scale*2;

	//	Scale the Y and CbCr quant table, respectively
	scaleQuantTable( qtblY,	 std_luminance_quant_tbl, aanscales );
	scaleQuantTable( qtblCoCg, std_chrominance_quant_tbl, aanscales );
}

void SimpleDCTDec::scaleQuantTable(
			unsigned short* tblRst,		//result quant table
			unsigned short* tblStd,		//standard quant table
			unsigned short* tblAan		//scale factor for AAN dct
			)
{
	int i, temp, half = 1<<11;

	for (i = 0; i < 64; i++) 
	{
		// (1) user scale up
		temp = (int)(( scale * tblStd[i] + 50 ) / 100 );

		// limit to baseline range 
		if (temp <= 0) 
			temp = 1;
		if (temp > 255)
			temp = 255;		

		// (2) scaling needed for AA&N algorithm
		tblRst[i] = (unsigned short)(( temp * tblAan[i] + half ) >> 12 );
	}
}

/*	Prepare four Huffman tables:
*		HUFFMAN_TABLE htblYDC, htblYAC, htblCoCgDC, htblCoCgAC;
*/
void SimpleDCTDec::initHuffmanTables( void )
{
	//	Y dc component
	static unsigned char bitsYDC[17] =
    { 0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 };
	static unsigned char valYDC[] =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };	

	//	CbCr dc
	static unsigned char bitsCbCrDC[17] =
    { 0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 };
	static unsigned char valCbCrDC[] =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };	

	//	Y ac component
	static unsigned char bitsYAC[17] =
    { 0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d };
	static unsigned char valYAC[] =
    { 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
	0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
	0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
	0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
	0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
	0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
	0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
	0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
	0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
	0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
	0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
	0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
	0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
	0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
	0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
	0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
	0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
	0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
	0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
	0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
	0xf9, 0xfa };	

	//	CbCr ac
	static unsigned char bitsCbCrAC[17] =
    { 0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77 };
	static unsigned char valCbCrAC[] =
    { 0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
	0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
	0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
	0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
	0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
	0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
	0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
	0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
	0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
	0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
	0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
	0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
	0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
	0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
	0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
	0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
	0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
	0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
	0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
	0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
	0xf9, 0xfa };

	//	Compute four derived Huffman tables
	computeHuffmanTable( bitsYDC, valYDC, &htblYDC );
	computeHuffmanTable( bitsYAC, valYAC, &htblYAC );
	computeHuffmanTable( bitsCbCrDC, valCbCrDC, &htblCoCgDC );
	computeHuffmanTable( bitsCbCrAC, valCbCrAC, &htblCoCgAC );
}

////////////////////////////////////////////////////////////////////////////////

//	Compute the derived values for a Huffman table.	

void SimpleDCTDec::computeHuffmanTable(
		unsigned char *	pBits, 
		unsigned char * pVal,
		HUFFTABLE * dtbl	
		)
{
	int p, i, l, si;
	int lookbits, ctr;
	char huffsize[257];
	unsigned int huffcode[257];
	unsigned int code;

	memcpy( dtbl->bits, pBits, 17 );
	memcpy( dtbl->huffval, pVal, 256 );
	
	/* Figure C.1: make table of Huffman code length for each symbol */
	/* Note that this is in code-length order. */
	p = 0;
	for (l = 1; l <= 16; l++) {
		for (i = 1; i <= (int) pBits[l]; i++)
			huffsize[p++] = (char) l;
	}
	huffsize[p] = 0;
	
	/* Figure C.2: generate the codes themselves */
	/* Note that this is in code-length order. */
	
	code = 0;
	si = huffsize[0];
	p = 0;
	while (huffsize[p]) {
		while (((int) huffsize[p]) == si) {
			huffcode[p++] = code;
			code++;
		}
		code <<= 1;
		si++;
	}
	
	/* Figure F.15: generate decoding tables for bit-sequential decoding */
	
	p = 0;
	for (l = 1; l <= 16; l++) {
		if (pBits[l]) {
			dtbl->valptr[l] = p; /* huffval[] index of 1st symbol of code length l */
			dtbl->mincode[l] = huffcode[p]; /* minimum code of length l */
			p += pBits[l];
			dtbl->maxcode[l] = huffcode[p-1]; /* maximum code of length l */
		} else {
			dtbl->maxcode[l] = -1;	/* -1 if no codes of this length */
		}
	}
	dtbl->maxcode[17] = 0xFFFFFL; /* ensures jpeg_huff_decode terminates */
	
	/* Compute lookahead tables to speed up decoding.
	 * First we set all the table entries to 0, indicating "too long";
	 * then we iterate through the Huffman codes that are short enough and
	 * fill in all the entries that correspond to bit sequences starting
	 * with that code.	 */
	
	memset( dtbl->look_nbits, 0, sizeof(int)*256 );
	
	int HUFF_LOOKAHEAD = 8;
	p = 0;
	for (l = 1; l <= HUFF_LOOKAHEAD; l++) 
	{
		for (i = 1; i <= (int) pBits[l]; i++, p++) 
		{
			/* l = current code's length, 
			p = its index in huffcode[] & huffval[]. Generate left-justified
			code followed by all possible bit sequences */
			lookbits = huffcode[p] << (HUFF_LOOKAHEAD-l);
			for (ctr = 1 << (HUFF_LOOKAHEAD-l); ctr > 0; ctr--) 
			{
				dtbl->look_nbits[lookbits] = l;
				dtbl->look_sym[lookbits] = pVal[p];
				lookbits++;
			}
		}
	}
}


bool SimpleDCTDec::decompress(unsigned char *dest, unsigned char *source, int width, int height, int size) {
	
    nGetBits = 0;
	nGetBuff = 0;
	dcA = dcY = dcCo = dcCg = 0;

    this->width = width;
    this->height = height;

	//	Error handling
	if(( dest == 0 )||( source == 0 ))
		return false;

	//	declares
	int xPixel, yPixel, xTile, yTile, cxTile, cyTile, cxBlock, cyBlock;
	int y, nTrueRows, nTrueCols;
	unsigned char byTile[1024], *pTileRow;
		
	//	horizontal and vertical count of tile, macroblocks, 
	//	or MCU(Minimum Coded Unit), in 16*16 pixels
	cxTile = (width + 15) >> 4;	
	cyTile = (height + 15) >> 4;

	//	horizontal and vertical count of block, in 8*8 pixels
	cxBlock = cxTile << 1;
	cyBlock = cyTile << 1;

	int pitch = width * 4;

	pData = source;
	nDataBytes = size;

	//	Decompress all the tiles, or macroblocks, or MCUs
	for( yTile = 0; yTile < cyTile; yTile++ )
	{
		for( xTile = 0; xTile < cxTile; xTile++ )
		{
			//	Decompress one macroblock started from m_pData;
			//	This function will push m_pData ahead
			//	Result is storing in byTile
			if( ! decompressOneTile( byTile ))
				return false;

			//	Get tile starting pixel position
			xPixel = xTile << 4;
			yPixel = yTile << 4;

			//	Get the true number of tile columns and rows
			nTrueRows = 16;
			nTrueCols = 16;			
			if( yPixel + nTrueRows > height )
				nTrueRows = height - yPixel;
			if( xPixel + nTrueCols > width )
				nTrueCols = width - xPixel;

			//	Write the output bgr data
			pTileRow = dest + yPixel * pitch + xPixel * 4;
			for( y = 0; y < nTrueRows; y ++ )
			{
				memcpy( pTileRow, byTile + y * 64, nTrueCols * 4 );
				pTileRow += pitch;			
			}		
		}
	}	
	return true;
}

bool SimpleDCTDec::decompressOneTile(unsigned char *rgba)
{
	unsigned char pYCoCgA[640];//Four color components, 256 + 64 + 64 + 256 bytes 
	short coef[64];	
	
	//	Do Y/Co/Cg/A components, Y: 4 blocks; Cb: 1 block; Cr: 1 block; A: 4 blocks
	for( int i=0; i<10; i++ )
	{
		huffmanDecode(coef, i);	//source is m_pData; coef is result
		inverseDct(coef, pYCoCgA + i*64, i);	//De-scale and inverse dct		
	}

	//	Color conversion and up-sampling
	colorConverter( pYCoCgA, rgba);
	return true;
}

static int ma(int a, int b) {
    return (a > b) ? a : b;
}

static int mi(int a, int b) {
    return (a < b) ? a : b;
}

static void convertToRGB(int Y, int Co, int Cg, int *dst) {
    dst[2] = ma(mi(255,Y + Co - Cg),0);
    dst[1] = ma(mi(255,Y + Cg     ),0);
    dst[0] = ma(mi(255,Y - Co - Cg),0);
}


void SimpleDCTDec::YCoCgAToRGBA(unsigned char * pYCoCgA,//in, Y: 256 bytes; Cb: 64 bytes; Cr: 64 bytes; A: 256 bytes
		unsigned char * rgba )
{
	unsigned char *pByte, *py[4], *pa[4], *pco, *pcg;

	pByte = rgba;
    for( int i=0; i<4; i++ ) {
		py[i] = pYCoCgA + i * 64;
        pa[i] = pYCoCgA + i * 64 + 384;
    }
	pco	  = pYCoCgA + 256;
	pcg   = pYCoCgA + 320;

	for(int j=0; j<16; j++ ) {
		for(int  i=0; i<16; i++ ) {			

			//	block number is ((j/8) * 2 + i/8): {0, 1, 2, 3}
			int y = *( py[(j>>3) * 2 + (i>>3)] ++ );
			int a = *( pa[(j>>3) * 2 + (i>>3)] ++ );

			int co = pco[j/2 * 8 + i/2]-128;
            int cg = pcg[j/2 * 8 + i/2]-128; 

            int rgb[3];
            convertToRGB(y,co,cg,rgb);

            *(pByte++) = rgb[0];
            *(pByte++) = rgb[1];
            *(pByte++) = rgb[2];
            *(pByte++) = a;
		}
	}
}

/**
    This only cancels the 4:2:0 color compression but leaves data in the YCoCg color space ( with [0..255] fields )
*/
void SimpleDCTDec::YCoCgAToYCoCgA(unsigned char * pYCoCgA,//in, Y: 256 bytes; Cb: 64 bytes; Cr: 64 bytes; A: 256 bytes
		unsigned char * out )
{
    unsigned int *pOut;
	unsigned char *py[4], *pa[4], *pco, *pcg;

	pOut = (unsigned int *)out;
    for( int i=0; i<4; i++ ) {
		py[i] = pYCoCgA + i * 64;
        pa[i] = pYCoCgA + i * 64 + 384;
    }
	pco	  = pYCoCgA + 256;
	pcg   = pYCoCgA + 320;

	for(int j=0; j<16; j++ ) {
		for(int  i=0; i<16; i++ ) {			

			//	block number is ((j/8) * 2 + i/8): {0, 1, 2, 3}
			int y = *( py[((j>>3) << 1) + (i>>3)]++ );
			int a = *( pa[((j>>3) << 1) + (i>>3)]++ );

			int co = pco[((j>>1) << 3) + (i>>1)];
            int cg = pcg[((j>>1) << 3) + (i>>1)]; 

            int packed = y | (co << 8) | (cg << 16) | (a << 24);
            *(pOut++) = packed;
		}
	}
}

#define FIXED_POINT
#ifdef FIXED_POINT
    //Uses 24.8 fixed point

#define FIX_1_082392200  ((int)277)		/* FIX(1.082392200) */
#define FIX_1_414213562  ((int)362)		/* FIX(1.414213562) */
#define FIX_1_847759065  ((int)473)		/* FIX(1.847759065) */
#define FIX_2_613125930  ((int)669)		/* FIX(2.613125930) */

#define MULTIPLY(var,cons)  (int)(((cons) * (var)) >> 8 )
#define SCALAR_TYPE int
#define IDESCALE(x,n)  ((int) ((x)>>(n)))

#else

#define FIX_1_082392200  1.082392200f
#define FIX_1_414213562  1.414213562f
#define FIX_1_847759065  1.847759065f
#define FIX_2_613125930  2.613125930f

#define MULTIPLY(var,cons)  ((cons) * (var))
#define SCALAR_TYPE float
#define IDESCALE(x,n)  ((int) ((x)/(float)(1<<n)) )

#endif


//	AA&N DCT algorithm implemention

void SimpleDCTDec::inverseDct( 
		short* coef, 			//in, dct coefficients, length = 64
		unsigned char* data, 	//out, 64 bytes		
		int nBlock				//block index: 0~3:Y; 4:Cb; 5:Cr
		)
{
	SCALAR_TYPE tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
	SCALAR_TYPE tmp10, tmp11, tmp12, tmp13;
	SCALAR_TYPE z5, z10, z11, z12, z13;
	SCALAR_TYPE workspace[64];		/* buffers data between passes */

	short* inptr = coef;
	unsigned short* quantptr;	
	SCALAR_TYPE* wsptr = workspace;
	unsigned char* outptr;
	int ctr, dcval, DCTSIZE = 8;

	if( nBlock < 4 || nBlock > 5 )
		quantptr = qtblY;
	else
		quantptr = qtblCoCg;
	
	//Pass 1: process columns from input (inptr), store into work array(wsptr)
	
	for (ctr = 8; ctr > 0; ctr--) {
    /* Due to quantization, we will usually find that many of the input
	* coefficients are zero, especially the AC terms.  We can exploit this
	* by short-circuiting the IDCT calculation for any column in which all
	* the AC terms are zero.  In that case each output is equal to the
	* DC coefficient (with scale factor as needed).
	* With typical images and quantization tables, half or more of the
	* column DCT calculations can be simplified this way.
	*/
		
		if ((inptr[DCTSIZE*1] | inptr[DCTSIZE*2] | inptr[DCTSIZE*3] |
			inptr[DCTSIZE*4] | inptr[DCTSIZE*5] | inptr[DCTSIZE*6] |
			inptr[DCTSIZE*7]) == 0) 
		{
			/* AC terms all zero */
			dcval = (int)( inptr[DCTSIZE*0] * quantptr[DCTSIZE*0] );
			
			wsptr[DCTSIZE*0] = dcval;
			wsptr[DCTSIZE*1] = dcval;
			wsptr[DCTSIZE*2] = dcval;
			wsptr[DCTSIZE*3] = dcval;
			wsptr[DCTSIZE*4] = dcval;
			wsptr[DCTSIZE*5] = dcval;
			wsptr[DCTSIZE*6] = dcval;
			wsptr[DCTSIZE*7] = dcval;
			
			inptr++;			/* advance pointers to next column */
			quantptr++;
			wsptr++;
			continue;
		}
		
		/* Even part */
		
		tmp0 = inptr[DCTSIZE*0] * quantptr[DCTSIZE*0];
		tmp1 = inptr[DCTSIZE*2] * quantptr[DCTSIZE*2];
		tmp2 = inptr[DCTSIZE*4] * quantptr[DCTSIZE*4];
		tmp3 = inptr[DCTSIZE*6] * quantptr[DCTSIZE*6];
		
		tmp10 = tmp0 + tmp2;	/* phase 3 */
		tmp11 = tmp0 - tmp2;
		
		tmp13 = tmp1 + tmp3;	/* phases 5-3 */
		tmp12 = MULTIPLY(tmp1 - tmp3, FIX_1_414213562) - tmp13; /* 2*c4 */
		
		tmp0 = tmp10 + tmp13;	/* phase 2 */
		tmp3 = tmp10 - tmp13;
		tmp1 = tmp11 + tmp12;
		tmp2 = tmp11 - tmp12;
		
		/* Odd part */
		
		tmp4 = inptr[DCTSIZE*1] * quantptr[DCTSIZE*1];
		tmp5 = inptr[DCTSIZE*3] * quantptr[DCTSIZE*3];
		tmp6 = inptr[DCTSIZE*5] * quantptr[DCTSIZE*5];
		tmp7 = inptr[DCTSIZE*7] * quantptr[DCTSIZE*7];
		
		z13 = tmp6 + tmp5;		/* phase 6 */
		z10 = tmp6 - tmp5;
		z11 = tmp4 + tmp7;
		z12 = tmp4 - tmp7;
		
		tmp7  = z11 + z13;		/* phase 5 */
		tmp11 = MULTIPLY(z11 - z13, FIX_1_414213562); /* 2*c4 */
		
		z5	  = MULTIPLY(z10 + z12, FIX_1_847759065); /* 2*c2 */
		tmp10 = MULTIPLY(z12, FIX_1_082392200) - z5; /* 2*(c2-c6) */
		tmp12 = MULTIPLY(z10, - FIX_2_613125930) + z5; /* -2*(c2+c6) */
		
		tmp6 = tmp12 - tmp7;	/* phase 2 */
		tmp5 = tmp11 - tmp6;
		tmp4 = tmp10 + tmp5;
		
		wsptr[DCTSIZE*0] = (int) (tmp0 + tmp7);
		wsptr[DCTSIZE*7] = (int) (tmp0 - tmp7);
		wsptr[DCTSIZE*1] = (int) (tmp1 + tmp6);
		wsptr[DCTSIZE*6] = (int) (tmp1 - tmp6);
		wsptr[DCTSIZE*2] = (int) (tmp2 + tmp5);
		wsptr[DCTSIZE*5] = (int) (tmp2 - tmp5);
		wsptr[DCTSIZE*4] = (int) (tmp3 + tmp4);
		wsptr[DCTSIZE*3] = (int) (tmp3 - tmp4);
		
		inptr++;			/* advance pointers to next column */
		quantptr++;
		wsptr++;
	}
	
	/* Pass 2: process rows from work array, store into output array. */
	/* Note that we must descale the results by a factor of 8 == 2**3, */
	/* and also undo the PASS1_BITS scaling. */

int RANGE_MASK = 1023; //2 bits wider than legal samples
#define PASS1_BITS  2
	
	wsptr = workspace;
	for (ctr = 0; ctr < DCTSIZE; ctr++) {
		outptr = data + ctr * 8;
		
		/* Even part */
		
		tmp10 = ((int) wsptr[0] + (int) wsptr[4]);
		tmp11 = ((int) wsptr[0] - (int) wsptr[4]);
		
		tmp13 = ((int) wsptr[2] + (int) wsptr[6]);
		tmp12 = MULTIPLY((int) wsptr[2] - (int) wsptr[6], FIX_1_414213562)
			- tmp13;
		
		tmp0 = tmp10 + tmp13;
		tmp3 = tmp10 - tmp13;
		tmp1 = tmp11 + tmp12;
		tmp2 = tmp11 - tmp12;
		
		/* Odd part */
		
		z13 = (int) wsptr[5] + (int) wsptr[3];
		z10 = (int) wsptr[5] - (int) wsptr[3];
		z11 = (int) wsptr[1] + (int) wsptr[7];
		z12 = (int) wsptr[1] - (int) wsptr[7];
		
		tmp7 = z11 + z13;		/* phase 5 */
		tmp11 = MULTIPLY(z11 - z13, FIX_1_414213562); /* 2*c4 */
		
		z5    = MULTIPLY(z10 + z12, FIX_1_847759065); /* 2*c2 */
		tmp10 = MULTIPLY(z12, FIX_1_082392200) - z5; /* 2*(c2-c6) */
		tmp12 = MULTIPLY(z10, - FIX_2_613125930) + z5; /* -2*(c2+c6) */
		
		tmp6 = tmp12 - tmp7;	/* phase 2 */
		tmp5 = tmp11 - tmp6;
		tmp4 = tmp10 + tmp5;
		
		/* Final output stage: scale down by a factor of 8 and range-limit */
        outptr[0] = mi(ma(IDESCALE(tmp0 + tmp7, PASS1_BITS+3)+128,0),255);
		outptr[7] = mi(ma(IDESCALE(tmp0 - tmp7, PASS1_BITS+3)+128,0),255);
		outptr[1] = mi(ma(IDESCALE(tmp1 + tmp6, PASS1_BITS+3)+128,0),255);
		outptr[6] = mi(ma(IDESCALE(tmp1 - tmp6, PASS1_BITS+3)+128,0),255);
		outptr[2] = mi(ma(IDESCALE(tmp2 + tmp5, PASS1_BITS+3)+128,0),255);
		outptr[5] = mi(ma(IDESCALE(tmp2 - tmp5, PASS1_BITS+3)+128,0),255);
		outptr[4] = mi(ma(IDESCALE(tmp3 + tmp4, PASS1_BITS+3)+128,0),255);
		outptr[3] = mi(ma(IDESCALE(tmp3 - tmp4, PASS1_BITS+3)+128,0),255);
		
		wsptr += DCTSIZE;		/* advance pointer to next row */
	}
}

void SimpleDCTDec::huffmanDecode( 
		short* coef,			//	out, DCT coefficients
		int iBlock				//	0,1,2,3:Y; 4:Cb; 5:Cr
		)
{	
/*
* jpeg_natural_order[i] is the natural-order position of the i'th 
* element of zigzag order.
*
* When reading corrupted data, the Huffman decoders could attempt
* to reference an entry beyond the end of this array (if the decoded
* zero run length reaches past the end of the block).  To prevent
* wild stores without adding an inner-loop test, we put some extra
* "63"s after the real entries.  This will cause the extra coefficient
* to be stored in location 63 of the block, not somewhere random.
* The worst case would be a run-length of 15, which means we need 16
* fake entries.
*/	
	static const int jpeg_natural_order[64+16] = {
			0,  1,  8, 16,  9,  2,  3, 10,
			17, 24, 32, 25, 18, 11,  4,  5,
			12, 19, 26, 33, 40, 48, 41, 34,
			27, 20, 13,  6,  7, 14, 21, 28,
			35, 42, 49, 56, 57, 50, 43, 36,
			29, 22, 15, 23, 30, 37, 44, 51,
			58, 59, 52, 45, 38, 31, 39, 46,
			53, 60, 61, 54, 47, 55, 62, 63,
			63, 63, 63, 63, 63, 63, 63, 63,//extra entries for safety
			63, 63, 63, 63, 63, 63, 63, 63
	};

	int* pLastDC;
	int s, k, r;

	HUFFTABLE *dctbl, *actbl;

	if( iBlock < 4 || iBlock > 5 )
	{
		dctbl = &htblYDC;
		actbl = &htblYAC;
        if ( iBlock < 4 ) {
		    pLastDC = &dcY;
        } else {
		    pLastDC = &dcA;
        }
	}
	else
	{
		dctbl = &htblCoCgDC;
		actbl = &htblCoCgAC;
		if( iBlock == 4 )
			pLastDC = &dcCo;
		else
			pLastDC = &dcCg;
	}

	memset( coef, 0, sizeof(short) * 64 );
	
    /* Section F.2.2.1: decode the DC coefficient difference */
	s = getCategory( dctbl );		//get dc category number, s

	if (s) {
		r = getBits(s);					//get offset in this dc category
		s = valueFromCategory(s, r);	//get dc difference value
    }
	
    /* Convert DC difference to actual value, update last_dc_val */
    s += *pLastDC;
    *pLastDC = s;

    /* Output the DC coefficient (assumes jpeg_natural_order[0] = 0) */
    coef[0] = (short) s;	
    
	/* Section F.2.2.2: decode the AC coefficients */
	/* Since zeroes are skipped, output area must be cleared beforehand */
	for (k = 1; k < 64; k++) 
	{
		s = getCategory( actbl );	//s: (run, category)
		r = s >> 4;			//	r: run length for ac zero, 0 <= r < 16
		s &= 15;			//	s: category for this non-zero ac
		
		if( s ) 
		{
			k += r;					//	k: position for next non-zero ac
			r = getBits(s);			//	r: offset in this ac category
			s = valueFromCategory(s, r);	//	s: ac value

			coef[ jpeg_natural_order[ k ] ] = (short) s;
		} 
		else // s = 0, means ac value is 0 ? Only if r = 15.  
		{
			if (r != 15)	//means all the left ac are zero
				break;
			k += 15;
		}
	}		
}



////////////////////////////////////////////////////////////////////////////////
//get category number for dc, or (0 run length, ac category) for ac

//	The max length for Huffman codes is 15 bits; so we use 32 bits buffer	
//	m_nGetBuff, with the validated length is m_nGetBits.
//	Usually, more than 95% of the Huffman codes will be 8 or fewer bits long
//	To speed up, we should pay more attention on the codes whose length <= 8

inline int SimpleDCTDec::getCategory( HUFFTABLE* htbl )
{
	//	If left bits < 8, we should get more data
	if(nGetBits < 8)
		fillBitBuffer();

	//	Call special process if data finished; min bits is 1
	if(nGetBits < 8)
		return specialDecode( htbl, 1 );

	//	Peek the first valid byte	
	int look = ((nGetBuff>>(nGetBits - 8))& 0xFF);
	int nb = htbl->look_nbits[look];

	if( nb ) 
	{ 
		nGetBits -= nb;
		return htbl->look_sym[look]; 
	} 
	else	//Decode long codes with length >= 9
		return specialDecode( htbl, 9 );
}

////////////////////////////////////////////////////////////////////////////////

void SimpleDCTDec::fillBitBuffer( void )
{
	unsigned char uc;
	while( nGetBits < 25 )	//#define MIN_GET_BITS  (32-7)
	{
		if( nDataBytes > 0 )//We have more data
		{
			uc = *pData++;
			nGetBuff = (nGetBuff << 8) | ((int) uc);
			nGetBits += 8;
			nDataBytes --;
		}
		else
			break;
	}
}

////////////////////////////////////////////////////////////////////////////////

inline int SimpleDCTDec::getBits(int nbits) 
{
	if( nGetBits < nbits )//we should read nbits bits to get next data
		fillBitBuffer();
	nGetBits -= nbits;
	return (int) (nGetBuff >> nGetBits) & ((1<<nbits)-1);
}

////////////////////////////////////////////////////////////////////////////////
//	Special Huffman decode:
//	(1) For codes with length > 8
//	(2) For codes with length < 8 while data is finished

int SimpleDCTDec::specialDecode( HUFFTABLE* htbl, int nMinBits )
{
	
	int l = nMinBits;
	int code;
	
	/* HUFF_DECODE has determined that the code is at least min_bits */
	/* bits long, so fetch that many bits in one swoop. */

	code = getBits(l);
	
	/* Collect the rest of the Huffman code one bit at a time. */
	/* This is per Figure F.16 in the JPEG spec. */
	while (code > htbl->maxcode[l]) {
		code <<= 1;
		code |= getBits(1);
		l++;
	}
	
	/* With garbage input we may reach the sentinel value l = 17. */
	if (l > 16) {
		return 0;			/* fake a zero as the safest result */
	}
	
	return htbl->huffval[ htbl->valptr[l] +	(int)(code - htbl->mincode[l]) ];
}

/*	To find dc or ac value according to category and category offset*/

inline int SimpleDCTDec::valueFromCategory(int nCate, int nOffset)
{
/*	//Method 1: 
	//On some machines, a shift and add will be faster than a table lookup.
	#define HUFF_EXTEND(x,s) \
	((x)< (1<<((s)-1)) ? (x) + (((-1)<<(s)) + 1) : (x)) 
*/
	//Method 2: Table lookup
	
	//If (nOffset < half[nCate]), then value is below zero
	//Otherwise, value is above zero, and just the nOffset
	static const int half[16] =		/* entry n is 2**(n-1) */
	{ 0, 0x0001, 0x0002, 0x0004, 0x0008, 0x0010, 0x0020, 0x0040, 0x0080,
    0x0100, 0x0200, 0x0400, 0x0800, 0x1000, 0x2000, 0x4000 };
	
	//start[i] is the starting value in this category; surely it is below zero
	static const int start[16] =	/* entry n is (-1 << n) + 1 */
	{ 0, ((-1)<<1) + 1, ((-1)<<2) + 1, ((-1)<<3) + 1, ((-1)<<4) + 1,
    ((-1)<<5) + 1, ((-1)<<6) + 1, ((-1)<<7) + 1, ((-1)<<8) + 1,
    ((-1)<<9) + 1, ((-1)<<10) + 1, ((-1)<<11) + 1, ((-1)<<12) + 1,
    ((-1)<<13) + 1, ((-1)<<14) + 1, ((-1)<<15) + 1 };	

	return ( nOffset < half[nCate] ? nOffset + start[nCate] : nOffset);	
}
