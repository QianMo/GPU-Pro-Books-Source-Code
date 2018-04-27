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
class SimpleDCTDec {

	unsigned short quality, scale;
	
	//	To speed up, we precompute two DCT quant tables
	unsigned short qtblY[64], qtblCoCg[64];

	// Derived data constructed for each Huffman table 
	typedef struct {
		int				mincode[17];	// smallest code of length k 
		int				maxcode[18];	// largest code of length k (-1 if none) 
		int				valptr[17];		// huffval[] index of 1st symbol of length k
		unsigned char	bits[17];		// bits[k] = # of symbols with codes of 
		unsigned char	huffval[256];	// The symbols, in order of incr code length 
		int				look_nbits[256];// # bits, or 0 if too long
		unsigned char	look_sym[256];	// symbol, or unused
	} HUFFTABLE;

	HUFFTABLE htblYDC, htblYAC, htblCoCgDC, htblCoCgAC;
	
	unsigned short width, height;
	int dcY, dcCo, dcCg, dcA;

	int nGetBits, nGetBuff, nDataBytes;
	unsigned char * pData;

	void scaleQuantTable(unsigned short* tblRst, unsigned short* tblStd, unsigned short* tblAan);
	void initHuffmanTables(void);
	void computeHuffmanTable(unsigned char *	pBits, unsigned char * pVal, HUFFTABLE * dtbl);

	bool decompressOneTile(unsigned char *rgba);
	void inverseDct(short* coef, unsigned char* data, int nBlock);	
	void huffmanDecode(short* coef, int iBlock);
	int getCategory( HUFFTABLE* htbl );
	void fillBitBuffer( void );
	int getBits(int nbits);	
	int specialDecode( HUFFTABLE* htbl, int nMinBits );
	int valueFromCategory(int nCate, int nOffset);

	static void YCoCgAToRGBA(unsigned char * yCoCgA, unsigned char * rgba);
    static void YCoCgAToYCoCgA(unsigned char * pYCoCgA, unsigned char * out );

    void (*colorConverter)(unsigned char * pYCoCgA, unsigned char * out );

public:	
	SimpleDCTDec(int quality = 50);	
	~SimpleDCTDec();
	
    //Set the quality the next decompress call will use...
    //you can call this once for a number of decomrpess calls to
    //gain some speed.
    void setQuality(int quality);

    enum ColorSpace {
        CS_RGBA,
        CS_YCOCGA
    };

    void setColorSpace(ColorSpace cs);

	bool decompress(unsigned char *dest, unsigned char *source, int width, int height, int compressedSize);
};

