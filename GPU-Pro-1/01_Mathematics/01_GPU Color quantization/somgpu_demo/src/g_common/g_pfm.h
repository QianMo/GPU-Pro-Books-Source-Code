#ifndef G_PFM_H
#define G_PFM_H

#include <stdio.h>
#include "g_common.h"
#include "g_vector.h"

class GPf1;

class GPfm
{
  public:
    int w;
    int h;
    FLOAT3 *fm;
    FLOAT3 **pm;

    GPfm();
    ~GPfm();

    // clear memory
    void clear();

    void load( int width, int height, const BYTE3 *bm );
    void load( int width, int height, const float *pa );
    void load( int width, int height, float a );
    void load( int width, int height, const float *pr, const float *pg, const float *pb );
    void load( int width, int height, const FLOAT3 *prgb );
    void load( int width, int height, float r, float g, float b );
    void load( int width, int height, const FLOAT3 &col );
    void load( int width, int height );

    // supported format "pfm", "ppm", "bmp", "png", "jpg", "tga", "gif", "dds", "raw", "pim"
    void load( const char *spath );

    // default format "pfm", supported "pfm", "ppm", "bmp", "png", "raw", "jpg", "tga", "dds", "pim"
    void save( const char *spath, const char *type="pfm" ) const;


    void draw( const FLOAT3 *prgb, int x, int y, int blkw, int blkh);
    void draw( const GPfm &blk, int x, int y);
    void getchannel( float *pr, float *pg, float *pb ) const;
    void getchannel( GPf1 &pr, GPf1 &pg, GPf1 &pb ) const;
    void getblk( GPfm &blk, int x, int y, int blkw, int blkh ) const;

    void flip_vertical();
    void flip_horizontal();
    void transpose();
    void rotate( int angle );

    // scale this pfm to dimension (width, height)
    void scale( GPfm &pfm, int width, int height ) const;
    void scale( int width, int height );
    void resample( int width, int height );
    void resize( int width, int height );

    void add( const GPfm &A );
    void add( FLOAT3 a );
    void add( float a );
  
    void sub( const GPfm &A );
    void sub( FLOAT3 a );
    void sub( float a );
  
    void mul( const GPfm &A );
    void mul( FLOAT3 a );
    void mul( float a );

    void sqrt();


    void yuv();
    void yuv_inverse();

    FLOAT3 lookup_linear( float x, float y ) const;
    FLOAT3 lookup_nearest( float x, float y ) const;
    FLOAT3 lookup_bicubic( float x, float y ) const;
    int match( FLOAT3 v0 );

    FLOAT3 amax() const;
    FLOAT3 amin() const;
    FLOAT3 vmax() const;
    FLOAT3 vmin() const;
    FLOAT3 vmean() const;
    FLOAT3 variance() const;

    void vclamp( const FLOAT3 &lb, const FLOAT3 &ub );

    static float mse( const GPfm &pfm0, const GPfm &pfm1 );

    static void fill_partitions
    (
      float t0, float t1,  // boundaries values in destination scale
      float s0, float s1,  // source and destination scale factors
      float *p, // weighting of each partitions
      int   *q, // quantized index of source
      int  &np  // number of partition
    );
  private:
    GPfm( const GPfm &pfm );
    GPfm& operator=( const GPfm &pfm );

};

class GPf4
{
  public:
    int w,h;
    FLOAT4 *fm;
    FLOAT4 **pm;
    GPf4();
    ~GPf4();
    void flip_vertical();
    void flip_horizontal();
    void load( int width, int height, const float *r, const float *g, const float *b, const float *a );
    void load( int width, int height, const FLOAT4 *rgba );
    void load( int width, int height, const FLOAT3 *rgb );
    void load( int width, int height );

    void load( const char *spath );
    void save( const char *spath ) const;


    void getchannel( float *r, float *g, float *b, float *a ) const;
    void getchannel( GPfm &vec_xyz, GPf1 &vec_w ) const;

    FLOAT4 vmax() const;
    FLOAT4 vmin() const;
    FLOAT4 vmean() const;
    FLOAT4 variance() const;
    void mul( FLOAT4 a );
    void add( FLOAT4 a );

    void draw( const GPf4 &blk, int x, int y);
    //void load( GPfm &pfm );
  private:
    GPf4( const GPf4 &pfm );
    GPf4& operator=( const GPf4 &pfm );
};

class GPf1
{
  public:
    int w,h;
    float *fm;
    float **pm;
    GPf1();
    ~GPf1();
    void flip_vertical();
    void flip_horizontal();
    void load_no_flip( const char *spath );
    void save( const char *spath ) const;
    void load( int width, int height, const FLOAT3 *fm );
    void load( int width, int height, const float *pa );
    void load( int width, int height, float a );
    void load( int width, int height );
    void load( const char *spath );
    void getblk( GPf1 &blk, int x, int y, int blkw, int blkh ) const;
    void draw( const GPf1 &blk, int x, int y);
    void draw( const float *src, int x, int y, int blkw, int blkh );
    void scale( int width, int height );
  private:
    GPf1( const GPf1 &pfm ){}
    GPf1& operator=( const GPf1 &pfm ){}
};

class GBmp
{
  public:
    int w,h;
    BYTE3 *bm;
    BYTE3 **pm;

    GBmp();
    ~GBmp();

    void load( int width, int height, const FLOAT3 *fm );
    void load( int width, int height, const BYTE3 *bm );
    void load( int width, int height );
    void load( const char *spath );

    void save( const char *spath ) const;
    void flip_vertical();
    void getblk( GBmp &blk, int x, int y, int blkw, int blkh ) const;
    void rb_swap();

#pragma pack( push, 1 )
    struct bmp_header_info
    {
      unsigned short bfType;
      unsigned int bfSize;
      unsigned short bfReserved1;
      unsigned short bfReserved2;
      unsigned int bfOffBits;

      // bitmap header
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
};


class GPPm
{
  public:
    int w;
    int h;
    BYTE3 *bm;
    BYTE3 **pm;

    GPPm();
    ~GPPm();

    void load( const char *spath, int x, int y, int blkw, int blkh );
    void load( int width, int height, const BYTE3 *bm );
    void load( int width, int height, const FLOAT3 *fm );
    void load( int width, int height );
    void load( const char *spath );

    void save( const char *spath ) const;
    void flip_vertical();
    void flip_horizontal();
    void getblk( GPPm &blk, int x, int y, int blkw, int blkh ) const;
};





/////////////////////////////////////////////////////////////////////
//
//  g_png.h
//

class GPng
{
  public:
    int w;
    int h;
    BYTE3* bm;
    BYTE3** pm;

    GPng();
    ~GPng();
    void load( int width, int height, const FLOAT3 *fm );
    void load( int width, int height, const BYTE3 *bm );
    void load( int width, int height );
    void load( const char *spath );
    void save( const char *spath ) const;
    void getblk( GPng &blk, int x, int y, int blkw, int blkh ) const;
    void flip_vertical();

    static void decompress( unsigned char *src, unsigned int n_src, unsigned char *des, unsigned int n_des );
    static void   compress( unsigned char *src, unsigned int n_src, unsigned char *des, unsigned int &n_des );
  private:

    static int crc_table_computed;
    static unsigned int crc_table[256];
    static void make_crc_table();
    static unsigned int update_crc(unsigned int crc, void *src_buf, int len);
    static unsigned int crc( void *buf, int len );


    void filter( BYTE3 *curline, BYTE3 *prvline, int type );
    static inline unsigned char Paeth( unsigned int ua, unsigned int ub, unsigned int uc );
    void write_chunk( FILE *f0, unsigned int chk_id, const void *chk_dat, unsigned int chk_len ) const;

#pragma pack( push, 1 )
typedef struct _IHDR
{
  unsigned int Width;
  unsigned int Height;
  unsigned char Bit; 
  unsigned char Colour;
  unsigned char Compression;
  unsigned char Filter;
  unsigned char Interlace;
}IHDR;
#pragma pack(pop)

};
//
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
//
//  g_jpg.h
//

class GJpg
{
  public:
    int w;
    int h;
    BYTE3* bm;
    BYTE3** pm;

    GJpg();
    ~GJpg();
    void load( int width, int height, const FLOAT3 *fm );
    void load( int width, int height, const BYTE3 *bm );
    void load( int width, int height );
    void load( const char *spath );
    void save( const char *spath ) const;
    void getblk( GJpg &blk, int x, int y, int blkw, int blkh ) const;

};
//
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
//
//  g_tga.h
//

class GTga
{
	public:
    int w;
    int h;
    BYTE3* bm;
    BYTE3** pm;

	  GTga();
	  ~GTga();
    void load( int width, int height, const FLOAT3 *fm );
    void load( int width, int height, const BYTE3 *bm );
    void load( int width, int height );

    void load( const char *spath );
    void load( const char *spath, const FLOAT4 &bgcolor );
    void save( const char *spath ) const;

	  void flip_vertical();

#pragma pack( push, 1 )
  typedef struct _HEADER {
     char      idlength;
     char      colourmaptype;
     char      datatypecode;
     short int colourmaporigin;
     short int colourmaplength;
     char      colourmapdepth;
     short int x_origin;
     short int y_origin;
     short     width;
     short     height;
     char      bitsperpixel;
     char      imagedescriptor;
  } HEADER;

  typedef struct _IDENTIFIELD {
	  char     desc[256];
  } IDENTIFIELD;
#pragma pack(pop)

};
//
/////////////////////////////////////////////////////////////////////

class GPim
{
  public:
    GPfm coltbl;
    int **pm;
    int *im;
    int w, h, n_code;

    GPim();
    ~GPim();

    void load( int width, int height, int number_of_code );
    void load( int width, int height, int number_of_code, const FLOAT3 *color_table, const GStack<int> *cluster );
    void load( const char *spath );
    void load( const GPfm &img, int number_of_code, int max_cycle );
    void load( const GPfm &img, int number_of_code, const GPim &pim );

    void save( const char *spath ) const;
    void save_indices( GPf1 &indices ) const;
    void decode( GPfm &decimg ) const;

    void getidx( void *index, int bytes_per_element ) const;

    void flip_vertical();
  private:
    static void lbg_codebook_initialization( const GPf1 &src, int n_code, GPf1 &codebook );
    static void lbg_training( const GPf1 &src, int max_cycle, GPf1 &codebook, GStack<int> *cluster );
    static void lbg_codeword_update( float *codeword, int n_member, const float *vec, int vsize );
    static bool lbg_codebook_converge( const GPf1 &old_codebook, const GPf1 &new_codebook );
};

class GGif
{
  public:
    int w, h, n;
    float *delay;
    BYTE3 *bm;
    BYTE3 **pm;
    GGif();
    ~GGif();
    void load( int width, int height );
    void load( int width, int height, int nframe );
    void load( const char *spath );
    void load( const char *spath, FLOAT3 col );

  private:
    void load( const char *spath, FLOAT3 *col );
    #pragma pack( push, 1 )
      struct gif_header_info
      {
        unsigned char  header[6];    
        unsigned short width;
        unsigned short height;
        unsigned char gct;
        unsigned char bgcolor;
        unsigned char aratio;
      };
      struct gif_control_ex
      {
        unsigned char instructor;
        unsigned char control_label;
        unsigned char block_size;
        unsigned char disposal_method; // interlace flag
        unsigned short delay_time;  // for animation;
        unsigned char color_index;
        unsigned char end_flag;
      };
      struct gif_image_descriptor
      {
        unsigned char instructor;
        unsigned short left;
        unsigned short top;
        unsigned short width;
        unsigned short height;
        unsigned char localflag;  // local color table information here;
        unsigned char min_bit;  //lzw minimal bit;
      };
      struct gif_app_ex
      {
        unsigned char instructor;
        unsigned char label;
        unsigned char size;
        unsigned char identifier[11];  //Identifier and Authentication Code
      };
      struct gif_comment_ex
      {
        unsigned char introducer;
        unsigned char label;
      };
    #pragma pack(pop)
    int DecodeImageData( const unsigned char *src, int dataSize, unsigned char *des, int width, int height );
    void InverseInterlace( const unsigned char *input, unsigned char **output, int width, int height );
    void frameresize( 
      BYTE3 **output, 
      int left, int top, int width, int height, 
      const BYTE3 *global_ct, const BYTE3 *local_ct, unsigned char **gif_index,
      bool transparent_bit, unsigned char transparent_index );

};

class GDds
{
  public:
    const static int GDDS_RGB_DXT1;
    const static int GDDS_RGBA_DXT3;
    const static int GDDS_RGBA_DXT5;

    int mipmaplevel, compressmethod, alphaimage, iscubemap;
    int w, h, d;
    int datasize;
    unsigned int rmask, gmask, bmask, amask;
    unsigned char *data;
    int *mipmapw, *mipmaph;

    //beginning each layers
    unsigned char **lm;

    GDds();
    ~GDds();
    void load( const char *spath );
    void load( int width, int height, int fourcc, int n_mip_layer );
    void decode( GPfm &pfm ) const;
    void save( const char *spath ) const;
    int getmemsize( int klevel ) const;

    // fourcc: 0    uncompressed
    // fourcc: DXT1 GDDS_RGB_DXT1
    void load( int width, int height, const FLOAT3 *fm, int fourcc, int genmipmap );

  private:
    static float comparecolor( FLOAT3 color1, FLOAT3 color2 );
    static unsigned int shiftcount(unsigned int mask);
    static int mipmaplevelcount(int width, int height);
    static void doblock( FLOAT3 *pix, int compressmethod, int aimage, const void *input );
    static unsigned __int64 compressblock( int width, const unsigned char *tmp, int x, int y, int wsize, int hsize, int fourcc);
    static void downsample( const unsigned char *src, int w0, int h0, unsigned char *des );

    #pragma pack( push, 1 )
    typedef struct
    {
      unsigned int dwSize;
      unsigned int dwFlags;
      unsigned int dwFourCC;
      unsigned int dwRGBBitCount;
      unsigned int dwRBitMask, dwGBitMask, dwBBitMask;
      unsigned int dwABitMask;
    } DDPIXELFORMAT;

    typedef struct
    {
      unsigned int dwCaps1;
      unsigned int dwCaps2;
      unsigned int dwCaps3;
      unsigned int dwCaps4;
    } DDSCAPS2;

    typedef struct
    {
      unsigned int dwSize;
      unsigned int dwFlags;
      unsigned int dwHeight;
      unsigned int dwWidth;
      unsigned int dwPitchOrLinearSize;
      unsigned int dwDepth;
      unsigned int dwMipMapCount;
      unsigned int dwRserved1[11];
      DDPIXELFORMAT ddpfPixelFormat;
      DDSCAPS2 ddsCaps;
      unsigned int dwReserved2;
    } DDSURFACEDESC2;

    typedef struct
    {
      unsigned int dxgiFormat;
      unsigned int resourceDimension;
      unsigned int miscFlag;
      unsigned int arraySize;
      unsigned int reserved;
    } DDS_HEADER_DXT10;
    #pragma pack(pop)
};
#endif
