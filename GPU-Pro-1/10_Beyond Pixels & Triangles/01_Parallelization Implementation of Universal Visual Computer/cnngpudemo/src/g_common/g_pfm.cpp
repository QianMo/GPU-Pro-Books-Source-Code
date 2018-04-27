#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <malloc.h>
#include <time.h>

#include "g_common.h"
#include "g_pfm.h"

GPfm::GPfm( const GPfm &pfm ){ exit(-1); }
GPfm& GPfm::operator=( const GPfm &pfm ){ exit(-1); return *this; }

GPfm::GPfm()
{
  w = 0;
  h = 0;
  fm = NULL;
  pm = NULL;
}

GPfm::~GPfm()
{
  SAFE_FREE( fm );
  SAFE_FREE( pm );
}

void GPfm::clear()
{
  w = 0;
  h = 0;
  SAFE_FREE( fm );
  SAFE_FREE( pm );
}

void GPfm::load( const char *spath )
{
  {
    GPath gp = parse_spath( spath );
    if( strcmp( _strlwr(gp.ename), "dds" )==0 )
    {
      GDds dds;
        dds.load( spath );
        dds.decode( *this );
      return;
    }
    if( strcmp( _strlwr(gp.ename), "gif" )==0 )
    {
      GGif gif;
        gif.load( spath );
        load( gif.w, gif.h, gif.bm );
      return;
    }
    if( strcmp( _strlwr(gp.ename), "tga" )==0 )
    {
      GTga tga;
        tga.load( spath );
        load( tga.w, tga.h, tga.bm );
      return;
    }
    if( strcmp( _strlwr(gp.ename), "raw" )==0 )
    {
      GPath gp = parse_spath( spath );
      char info_path[256];
        sprintf( info_path, "%s%s.raw.txt", gp.dname, gp.fname );

      if( !fexist(spath) )
      {
        printf( "[Error] : GPfm::load(), \"%s\" not found.\n", spath );
        exit(-1);
      }
      if( !fexist(info_path) )
      {
        printf( "[Error] : GPfm::load(), \"%s\" not found.\n", info_path );
        exit(-1);
      }

      unsigned short mode;
      int width, height;
      float byte_ordering;
      char str[256];
      FILE *f1 = fopen( info_path, "rb" );
        fread( &mode, sizeof(unsigned short), 1, f1 ); swapbyte( &mode, sizeof(unsigned short) );
        fgets( str, 256, f1 );
        while( fgets( str, 256, f1 ) && str[0] == '#' );
        sscanf(str, "%i %i", &width, &height);
        fgets( str, 256, f1 );
        sscanf(str, "%f", &byte_ordering);
      fclose(f1);


      int i, j;
      if( mode=='PF' )
      {
        FILE *f0 = fopen( spath, "rb" );
          load(width,height);
          fread( fm, sizeof(FLOAT3), w*h, f0);
          flip_vertical();
          if( byte_ordering==1 )
            for( j=0; j<h; j++ )
              for( i=0; i<w; i++ )
              {
                swapbyte( &pm[j][i].x, sizeof(float) );
                swapbyte( &pm[j][i].y, sizeof(float) );
                swapbyte( &pm[j][i].z, sizeof(float) );
              }
        fclose(f0);
      }else if( mode=='P6' )
      {
        GPPm ppm;
          ppm.load(width, height);
          FILE *f0 = fopen( spath, "rb" );
          fread( ppm.bm, sizeof(BYTE3), ppm.w*ppm.h, f0);
          fclose(f0);
        this->load( ppm.w, ppm.h, ppm.bm );
      }else
      {
        printf( "[Error] : GPfm::load(), unsupported file format.\n" );
        exit(-1);
      }

      return;
    }
  }

  char str[256], info[2];
  unsigned short mode;


  FILE *f0 = fopen( spath, "rb" );
    if( f0==NULL )
    {
      printf( "[Error] : GPfm::load(), \"%s\" not found.\n", spath );
      exit(-1);
    }
    fread( &mode, sizeof(unsigned short), 1, f0 ); swapbyte( &mode, sizeof(unsigned short) );
    fread( info, sizeof(char), 2, f0 );
  fclose(f0);


  int width, height;

  switch( mode )
  {
    case 'PF':
    {
      float byte_ordering;

      FILE *f0 = fopen( spath, "rb" );

        if( info[0]=='\n' || info[1]=='\n' )
        {
          fgets( str, 256, f0 );
          while( fgets( str, 256, f0 ) && str[0] == '#' );
          sscanf(str, "%i %i", &width, &height);
          fgets( str, 256, f0 );
          sscanf(str, "%f", &byte_ordering);
        }else if( info[0]==' ' )
        {
          fgets( str, 256, f0 );
          char tmp[16];
          sscanf( str, "%s %i %i %f", tmp, &width, &height, &byte_ordering );
        }else if( info[0]=='\r' && info[1]!='\n' )
        {
          fscanf( f0, "%[^\r]", str ); fgetc(f0);
          do{ fscanf( f0, "%[^\r]", str ); fgetc(f0); }while( str[0] == '#' );
          sscanf(str, "%i %i", &width, &height);
          fscanf( f0, "%[^\r]", str ); fgetc(f0);
          sscanf(str, "%f", &byte_ordering);
        }else
        {
          printf( "[Error] GPfm::load, incorrect file format\n" );
          exit(-1);
        }

        load(width,height);
        fread( fm, sizeof(FLOAT3), w*h, f0);
        flip_vertical();

        int i,j;
        if( byte_ordering==1 )
          for( j=0; j<h; j++ )
            for( i=0; i<w; i++ )
            {
              swapbyte( &pm[j][i].x, sizeof(float) );
              swapbyte( &pm[j][i].y, sizeof(float) );
              swapbyte( &pm[j][i].z, sizeof(float) );
            }

      fclose(f0);
    }
    break;

    case 'Pf':
    {
      GPf1 pf1;
        pf1.load( spath );
        load( pf1.w, pf1.h, pf1.fm );
    }
    break;

    case 'pf':
    {
      GPf4 pf4;
        pf4.load( spath );
      GPf1 a;
        pf4.getchannel( *this, a );
    }
    break;

    case 'P5':
    case 'P6':
    {
      GPPm ppm;
        ppm.load( spath );
        load( ppm.w, ppm.h, ppm.bm );
    }
    break;

    case 'BM':
    {
      GBmp bmp;
        bmp.load( spath );
        load( bmp.w, bmp.h, bmp.bm );
    }
    break;

    case 0x8950:
    {
      GPng png;
        png.load( spath );
        load( png.w, png.h, png.bm );
    }
    break;

    case 0xFFD8:
    {
      GJpg jpg;
        jpg.load( spath );
        load( jpg.w, jpg.h, jpg.bm );
    }
    break;

    case 'PI':
    {
      GPim pim;
        pim.load( spath );
        pim.decode( *this );
    }
    break;

    default:
      printf( "[Error] : GPfm::load(), unsupported file format.\n" );
      exit(-1);
  }

}

void GPfm::load( int width, int height )
{
  SAFE_FREE( fm );
  SAFE_FREE( pm );

  w = width;
  h = height;
  fm = (FLOAT3*) malloc( w * h * sizeof(FLOAT3) );
  memset( fm, 0, w*h*sizeof(FLOAT3) );

  pm = (FLOAT3**) malloc( h * sizeof(FLOAT3*) );
  int j;
  for( j=0; j<h; j++ )
    pm[j] = &fm[j*w];

}

void GPfm::load( int width, int height, const BYTE3 *bm )
{
  if( w!=width || h!=height )
    load( width, height );

  int i,j;
  for( j=0; j<h; j++ )
    for( i=0; i<w; i++ )
    {
      fm[j*w+i] = bm[j*w+i];
      fm[j*w+i] = fm[j*w+i] / 255;
    }

}

void GPfm::load( int width, int height, float a )
{
  load(width,height);

  FLOAT3 *tm = fm;
  FLOAT3 c = FLOAT3(a,a,a);

  int i,j;
  for( j=0; j<h; j++ )
    for( i=0; i<w; i++, tm++ )
    {
      *tm = c;
    }
}

void GPfm::load( int width, int height, const FLOAT3 &col )
{
  load( width, height, col.x, col.y, col.z );
}

void GPfm::load( int width, int height, float r, float g, float b )
{
  load(width,height);


  FLOAT3 *tm = fm;
  FLOAT3 c = FLOAT3(r,g,b);

  int i,j;
  for( j=0; j<h; j++ )
    for( i=0; i<w; i++, tm++ )
      *tm = c;
}

void GPfm::load( int width, int height, const float *pa )
{
  load(width,height);

  FLOAT3 *tm = fm;
  const float *ta = pa;

  int i,j;
  for( j=0; j<h; j++ )
    for( i=0; i<w; i++, tm++, ta++ )
    {
      tm->x = *ta;
      tm->y = *ta;
      tm->z = *ta;
    }
}

void GPfm::load( int width, int height, const FLOAT3 *prgb )
{
  load(width,height);
  memcpy( fm, prgb, w*h*sizeof(FLOAT3) );
}

void GPfm::load( int width, int height, const float *pr, const float *pg, const float *pb )
{
  load(width,height);

  FLOAT3 *tm = fm;
  const float *tr = pr;
  const float *tg = pg;
  const float *tb = pb;

  int i,j;
  for( j=0; j<h; j++ )
    for( i=0; i<w; i++, tm++, tr++, tg++, tb++ )
    {
      if(pr) tm->x = *tr;
      if(pg) tm->y = *tg;
      if(pb) tm->z = *tb;
    }
}

void GPfm::draw( const FLOAT3 *prgb, int x, int y, int blkw, int blkh )
{
  int src_w, src_h;
    src_w = blkw;
    src_h = blkh;

  int des_w, des_h;
    des_w = w;
    des_h = h;

  FLOAT3 *src, *des;
    src = (FLOAT3*)prgb;
    des = &fm[y*w + x];

  int j;
  for( j=0; j<src_h; j++, src+=src_w, des+=des_w )
    memcpy( des, src, src_w * sizeof(FLOAT3) );
}


void GPfm::draw( const GPfm &blk, int x, int y)
{
  int src_w, src_h;
    src_w = blk.w;
    src_h = blk.h;

  int des_w, des_h;
    des_w = w;
    des_h = h;

  FLOAT3 *src, *des;
    src = blk.fm;
    des = &fm[y*w + x];

  int j;
  for( j=0; j<src_h; j++, src+=src_w, des+=des_w )
    memcpy( des, src, src_w * sizeof(FLOAT3) );
}

void GPfm::flip_vertical()
{
  FLOAT3 *tline, *line0, *line1;
    tline = (FLOAT3*) malloc( w * sizeof(FLOAT3) );
    line0 = fm;
    line1 = fm + w*(h-1);

  int j, jlen = h/2;
  for( j=0; j<jlen; j++, line0+=w, line1-=w )
  {
    memcpy( tline, line0, w*sizeof(FLOAT3) );
    memcpy( line0, line1, w*sizeof(FLOAT3) );
    memcpy( line1, tline, w*sizeof(FLOAT3) );
  }

  free(tline);
}

void GPfm::getblk( GPfm &blk, int x, int y, int blkw, int blkh ) const
{
  if( blk.w != blkw || blk.h != blkh )
    blk.load( blkw,blkh );

  int j;

  for( j=0; j<blkh; j++ )
    memcpy( blk.pm[j], &pm[y+j][x], blkw * sizeof(FLOAT3) );
}


void GPfm::flip_horizontal()
{
  int i,j;
  int ilen = w/2;

  FLOAT3 *line0 = fm;
  FLOAT3 c;

  for( j=0; j<h; j++, line0+=w )
    for( i=0; i<ilen; i++ )
    {
      c = line0[i];
      line0[i] = line0[w-i-1];
      line0[w-i-1] = c;
    }
}

FLOAT3 GPfm::lookup_linear( float sx, float sy ) const
{
  float qx, qy;
  float rx, ry;
    qx = floorf(sx-.5f)+.5f;
    qy = floorf(sy-.5f)+.5f;
    rx = sx - qx;
    ry = sy - qy;

  int x0,y0, x1,y1;
    x0 = (int)G_CLAMP( qx  , 0, w-1  );
    x1 = (int)G_CLAMP( qx+1, 0, w-1  );
    y0 = (int)G_CLAMP( qy  , 0, h-1  );
    y1 = (int)G_CLAMP( qy+1, 0, h-1  );

  FLOAT3 val;
    val = 
    ( (1-rx)*pm[y0][x0] + rx*pm[y0][x1] ) * (1-ry) +
    ( (1-rx)*pm[y1][x0] + rx*pm[y1][x1] ) * ry;

  return val;
}

FLOAT3 GPfm::lookup_bicubic( float sx, float sy ) const
{
  float qx, qy;
  float rx, ry;
    qx = floorf(sx-.5f)+.5f;
    qy = floorf(sy-.5f)+.5f;
    rx = sx - qx;
    ry = sy - qy;


  FLOAT3 cval[4], rval[4];
  FLOAT3 p, q, r;

  int i, j;
  for( j=0; j<4; j++ )
  {
    for( i=0; i<4; i++ )
      rval[i] = lookup_nearest( qx-1+i, qy-1+j );

    q = rval[0] - rval[1];
    p = rval[3] - rval[2] - q;
    q -= p;
    r = rval[2] - rval[0];
    cval[j] = rx * (rx * (rx * p + q) + r) + rval[1];
  }

  FLOAT3 val;
    q = cval[0] - cval[1];
    p = cval[3] - cval[2] - q;
    q -= p;
    r = cval[2] - cval[0];
    val = ry * (ry * (ry * p + q) + r) + cval[1];

  return val;
}

FLOAT3 GPfm::lookup_nearest( float sx, float sy ) const
{
  int x0,y0;
    x0 = (int)G_CLAMP( sx  , 0, w-1  );
    y0 = (int)G_CLAMP( sy  , 0, h-1  );
  return pm[y0][x0];
}

int GPfm::match( FLOAT3 v0 )
{
  int lidx;
  float l, ln;
  int i;
  FLOAT3 v1;

  v1 = fm[0];
  ln = (v0-v1).norm();
  lidx = 0;
  for( i=0; i<w*h; i++ )
  {
    v1 = fm[i];
    l = (v0-v1).norm();
    if( ln>l )
    {
      ln = l;
      lidx = i;
    }
  }

  return lidx;
}

void GPfm::save( const char *spath, const char *mode ) const
{
  if( strcmp( mode, "pfm" )==0)
  {
    FILE *f0 = fopen( spath, "wb" );
      fprintf( f0, "PF\n");
	    fprintf( f0, "%i %i\n", w, h );
	    fprintf( f0, "%f\n", -1.f );
      for( int j=h-1; j>=0; j-- )
        fwrite( pm[j], sizeof(FLOAT3), w, f0 );
    fclose( f0 );
  }else if( strcmp( mode, "raw" )==0)
  {
    FILE *f0 = fopen( spath, "wb" );
      for( int j=h-1; j>=0; j-- )
        fwrite( pm[j], sizeof(FLOAT3), w, f0 );
    fclose( f0 );

    GPath gp = parse_spath( spath );
    char info_path[256];
      sprintf( info_path, "%s%s.raw.txt", gp.dname, gp.fname );
    FILE *f1 = fopen( info_path, "wt" );
      fprintf( f1, "PF\n");
	    fprintf( f1, "%i %i\n", w, h );
	    fprintf( f1, "%f\n", -1.f);
    fclose(f1);
  }else if( strcmp( mode, "ppm" )==0)
  {
    GPPm tmp;
      tmp.load( w,h,fm );
      tmp.save( spath );
  }else if( strcmp( mode, "bmp" )==0)
  {
    GBmp tmp;
      tmp.load( w,h,fm );
      tmp.save( spath );
  }else if( strcmp( mode, "png" )==0)
  {
    GPng tmp;
      tmp.load( w,h,fm );
      tmp.save( spath );
  }else if( strcmp( mode, "jpg" )==0)
  {
    GJpg tmp;
      tmp.load( w,h,fm );
      tmp.save( spath );
  }else if( strcmp( mode, "tga" )==0)
  {
    GTga tmp;
      tmp.load( w,h,fm );
      tmp.save( spath );
  }else if( strcmp( mode, "pim" )==0)
  {
    GPim tmp;
      tmp.load( *this, 256, 3000 );
      tmp.save( spath );
  }else if( strcmp( mode, "pf1" )==0)
  {
    GPf1 tmp;
      tmp.load( w,h,fm );
      tmp.save( spath );
  }else if( strcmp( mode, "dds" )==0)
  {
    GDds tmp;
      tmp.load( w,h,fm,0,0 );
      tmp.save( spath );
  }else if( strcmp( mode, "dds,mip" )==0)
  {
    GDds tmp;
      tmp.load( w,h,fm,0,1 );
      tmp.save( spath );
  }else if( strcmp( mode, "dxt1" )==0)
  {
    GDds tmp;
      tmp.load( w,h,fm,GDds::GDDS_RGB_DXT1,0 );
      tmp.save( spath );
  }else if( strcmp( mode, "dxt1,mip" )==0)
  {
    GDds tmp;
      tmp.load( w,h,fm,GDds::GDDS_RGB_DXT1,1 );
      tmp.save( spath );
  }else
  {
    printf( "[Error] : GPfm::Save(), unsupported mode %s.\n", mode );
    exit(-1);
  }
}

void GPfm::rotate( int angle )
{
  switch( angle )
  {
    case 0:
      break;
    case 90:
      {
        int i,j;
        GPfm tmp;
          tmp.load( h,w );

        for( j=0; j<h; j++ )
          for( i=0; i<w; i++ )
            tmp.fm[ i*h+h-j-1 ] = fm[ j*w+i ];

        this->load( tmp.w, tmp.h, tmp.fm );
      }
      break;
    case -180:
    case 180:
      this->flip_vertical();
      this->flip_horizontal();
      break;
    case -90:
    case 270:
      {
        int i,j;
        GPfm tmp;
          tmp.load( h,w );

        for( j=0; j<h; j++ )
          for( i=0; i<w; i++ )
            tmp.fm[ (w-i-1)*h+j ] = fm[ j*w+i ];

        this->load( tmp.w, tmp.h, tmp.fm );
      }
      break;

    default:
      printf( "[Error] GPfm::rotate, only 0, +/-90, 180, 270 degree are supported.\n" );
      exit(-1);
  }
}

void GPfm::transpose()
{
  GPfm pfm;
    pfm.load( w,h, fm );
    this->load( h,w );

  int i,j;
  for( j=0; j<h; j++ )
    for( i=0; i<w; i++ )
      pm[j][i] = pfm.pm[i][j];

}

float GPfm::mse( const GPfm &pfm0, const GPfm &pfm1 )
{

  if( pfm0.w != pfm1.w || pfm0.h != pfm1.h )
  {
    printf( "[Error] GPfm::mse(), images is not of the same dimension.\n" );
    exit(-1);
  }

  int w,h;
    w = pfm0.w;
    h = pfm0.h;

  FLOAT3 f, row_sum;
  f=0;
  FLOAT3 *f0, *f1;
    f0 = pfm0.fm;
    f1 = pfm1.fm;

  int i,j;
  for( j=0; j<h; j++ )
  {
    row_sum = 0;
    for( i=0; i<w; i++, f0++, f1++ )
    {
      //f = f + (*f0 - *f1)*(*f0 - *f1);
      row_sum = row_sum + (*f0 - *f1)*(*f0 - *f1);
    }
    f = f + row_sum/(float)w;
  }
  f = f/(float)h;
    
  //f = f/((float)w*h);
  float mse = (f.x + f.y + f.z)/3;
    
  return mse;
}






////////////////////////////////////////////////////////////////////////
//
//  add
//
void GPfm::add( const GPfm &A )
{
  if( w!=A.w || h!=A.h )
  {
    printf( "[Error] : GPfm::add(), the pfm are not of the same dimension.\n" );
    exit(-1);
  }

  int t, l;
  float *tn, *sn;
    tn = (float*)fm;
    sn = (float*)A.fm;

  l = w*h*3;

  for( t=0; t<l; t++, tn++, sn++ )
    *tn += *sn;
}

void GPfm::add( FLOAT3 a )
{
  int t, l;
  FLOAT3 *tm;

  tm = fm;
  l = w*h;
  for( t=0; t<l; t++, tm++ )
    *tm += a;
}

void GPfm::add( float a )
{
  int t, l;
  FLOAT3 *tm;

  tm = fm;
  l = w*h;
  for( t=0; t<l; t++, tm++ )
    *tm += a;
}
//
//
////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////
//
//  sub
//
void GPfm::sub( const GPfm &A )
{
  if( w!=A.w || h!=A.h )
  {
    printf( "[Error] : GPfm::add(), the pfm are not of the same dimension.\n" );
    exit(-1);
  }

  int t, l;
  float *tn, *sn;
    tn = (float*)fm;
    sn = (float*)A.fm;

  l = w*h*3;

  for( t=0; t<l; t++, tn++, sn++ )
    *tn -= *sn;
}

void GPfm::sub( FLOAT3 a )
{
  int t, l;
  FLOAT3 *tm;

  tm = fm;
  l = w*h;
  for( t=0; t<l; t++, tm++ )
    *tm -= a;
}

void GPfm::sub( float a )
{
  int t, l;
  FLOAT3 *tm;

  tm = fm;
  l = w*h;
  for( t=0; t<l; t++, tm++ )
    *tm -= a;
}
//
//
////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////
//
//  mul
//
void GPfm::mul( const GPfm &A )
{
  if( w!=A.w || h!=A.h )
  {
    printf( "[Error] : GPfm::mul(), the pfm are not of the same dimension.\n" );
    exit(-1);
  }

  int t, l;
  float *tn, *sn;
    tn = (float*)fm;
    sn = (float*)A.fm;

  l = w*h*3;

  for( t=0; t<l; t++, tn++, sn++ )
    *tn *= *sn;
}

void GPfm::mul( FLOAT3 a )
{
  int t, l;
  FLOAT3 *tm;

  tm = fm;
  l = w*h;
  for( t=0; t<l; t++, tm++ )
    *tm *= a;
}

void GPfm::mul( float a )
{
  int t, l;
  FLOAT3 *tm;

  tm = fm;
  l = w*h;
  for( t=0; t<l; t++, tm++ )
    *tm *= a;
}

void GPfm::sqrt()
{
  int t, l;
  FLOAT3 *tm;

  tm = fm;
  l = w*h;
  for( t=0; t<l; t++, tm++ )
    *tm = sqrtf3(*tm);
}

//
//
////////////////////////////////////////////////////////////////////////




FLOAT3 GPfm::amax() const
{
  int t, l;
  FLOAT3 *tm;
  float x,y,z;
  float mx,my,mz;

  l = w*h;
  tm = fm;

  mx = fabsf( tm->x );
  my = fabsf( tm->y );
  mz = fabsf( tm->z );

  for( t=0; t<l; t++, tm++ )
  {
    x = fabsf( tm->x );
    y = fabsf( tm->y );
    z = fabsf( tm->z );

    if( x > mx ) mx=x;
    if( y > my ) my=y;
    if( z > mz ) mz=z;
  }

  return FLOAT3( mx,my,mz );
}

FLOAT3 GPfm::amin() const
{
  int t, l;
  FLOAT3 *tm;
  float x,y,z;
  float mx,my,mz;

  l = w*h;
  tm = fm;

  mx = fabsf( tm->x );
  my = fabsf( tm->y );
  mz = fabsf( tm->z );

  for( t=0; t<l; t++, tm++ )
  {
    x = fabsf( tm->x );
    y = fabsf( tm->y );
    z = fabsf( tm->z );

    if( x < mx ) mx=x;
    if( y < my ) my=y;
    if( z < mz ) mz=z;
  }

  return FLOAT3( mx,my,mz );
}

FLOAT3 GPfm::vmax() const
{
  int t, l;
  FLOAT3 *tm;
  float x,y,z;
  float mx,my,mz;

  l = w*h;
  tm = fm;

  mx = tm->x;
  my = tm->y;
  mz = tm->z;

  for( t=0; t<l; t++, tm++ )
  {
    x = tm->x;
    y = tm->y;
    z = tm->z;

    if( x > mx ) mx=x;
    if( y > my ) my=y;
    if( z > mz ) mz=z;
  }

  return FLOAT3( mx,my,mz );
}

FLOAT3 GPfm::vmin() const
{
  int t, l;
  FLOAT3 *tm;
  float x,y,z;
  float mx,my,mz;

  l = w*h;
  tm = fm;

  mx = tm->x;
  my = tm->y;
  mz = tm->z;

  for( t=0; t<l; t++, tm++ )
  {
    x = tm->x;
    y = tm->y;
    z = tm->z;

    if( x < mx ) mx=x;
    if( y < my ) my=y;
    if( z < mz ) mz=z;
  }

  return FLOAT3( mx,my,mz );
}

FLOAT3 GPfm::vmean() const
{
  int t, l;
  FLOAT3 *tm;
  FLOAT3 res;

  l = w*h;
  tm = fm;

  res = 0;
  for( t=0; t<l; t++, tm++ )
    res = res + *tm;

  return res/float(l);
}

FLOAT3 GPfm::variance() const
{
  int t, l;
  FLOAT3 *tm;
  FLOAT3 mn;
  FLOAT3 var;
  FLOAT3 tmp;
    
  l = w*h;
  tm = fm;
  mn = vmean();
  var = 0;
    
  for( t=0; t<l; t++, tm++ )
  {
    tmp = *tm - mn;
    var = var + tmp*tmp;
  }

  return var/float(l);
}

void GPfm::vclamp( const FLOAT3 &lb, const FLOAT3 &ub )
{
  FLOAT3 *tm;
  int t, l;

  l = w*h;
  tm = fm;
  for( t=0; t<l; t++, tm++ )
  {
    FLOAT3 &c = *tm;
    c.x = G_CLAMP( c.x, lb.x, ub.x );
    c.y = G_CLAMP( c.y, lb.y, ub.y );
    c.z = G_CLAMP( c.z, lb.z, ub.z );
  }
}


void GPfm::yuv()
{
  FLOAT3 f0, f1, f2;
    f0 =  FLOAT3(  0.2990f,  0.5870f,  0.1140f );
    f1 =  FLOAT3(  0.5000f, -0.4187f, -0.0813f );
    f2 =  FLOAT3( -0.1687f, -0.3313f,  0.5000f );

  int t, l;
  FLOAT3 *tm;

  l = w*h;
  tm = fm;

  for( t=0; t<l; t++, tm++ )
    *tm = FLOAT3( vdot(*tm,f0), vdot(*tm,f1), vdot(*tm,f2) );
}

void GPfm::yuv_inverse()
{
  FLOAT3 f0, f1, f2;
    f0 =  FLOAT3(  1,  1.40200f,  0        );
    f1 =  FLOAT3(  1, -0.71414f, -0.34414f );
    f2 =  FLOAT3(  1,  0       ,  1.77200f );

  int t, l;
  FLOAT3 *tm;

  l = w*h;
  tm = fm;

  for( t=0; t<l; t++, tm++ )
    *tm = FLOAT3( vdot(*tm,f0), vdot(*tm,f1), vdot(*tm,f2) );
}

void GPfm::getchannel( float *pr, float *pg, float *pb ) const
  {
  int t, l;
  float *tc;
  FLOAT3 *tm;

  l=w*h;
  
  if(pr) 
    for( t=0, tm=fm, tc=pr; t<l; t++, tm++, tc++ )
      *tc = tm->x;
  
  if(pg) 
    for( t=0, tm=fm, tc=pg; t<l; t++, tm++, tc++ )
      *tc = tm->y;
  
  if(pb) 
    for( t=0, tm=fm, tc=pb; t<l; t++, tm++, tc++ )
      *tc = tm->z;
}

void GPfm::getchannel( GPf1 &pr, GPf1 &pg, GPf1 &pb ) const
{
  pr.load(w,h);
  pg.load(w,h);
  pb.load(w,h);
  getchannel( pr.fm, pg.fm, pb.fm );
}

void GPfm::fill_partitions
(
  float t0, float t1,  // boundaries values in destination scale
  float s0, float s1,  // source and destination scale factors
  float *p, // weighting of each partitions
  int   *q, // quantized index of source
  int  &np  // number of partition
){
  float s01, s10;
    s01 = s1/s0;
    s10 = s0/s1;

  t0 *= s10;
  t1 *= s10;

  int f0, c0;
    f0 = (int)floor(t0);
    c0 = (int)ceil(t0);

  int f1, c1;
    f1 = (int)floor(t1);
    c1 = (int)ceil(t1);

  np=0;


  if( c0-t0 > FLT_EPSILON )
  {
    p[np] = c0-t0;
    q[np] = f0;
    np++;
  }

  int i;
  for( i=c0; i<f1; i++, np++ )
  {
    p[np] = 1;
    q[np] = i;
  }

  if( t1-f1 > FLT_EPSILON )
  {
    p[np] = t1-f1;
    q[np] = f1;
    np++;
  }

  float sp=0;
  for( i=0; i<np; i++ )
    sp+=p[i];
  for( i=0; i<np; i++ )
    p[i]/=sp;


}

void GPfm::scale( GPfm &pfm, int width, int height ) const
{
  pfm.load( width, height );

  int w0,h0, w1,h1;
    w0 = w;
    h0 = h;
    w1 = pfm.w;
    h1 = pfm.h;

  int max_px, max_py;
    max_px = w0 / w1  + 2;
    max_py = h0 / h1  + 2;

  int i,j;
  int x,y;

  int    *pnx,  *pny;
  float **ppx, **ppy;
  int   **pqx, **pqy;
    pnx = (int*)    malloc  (         w1 * sizeof(int) );
    pny = (int*)    malloc  (         h1 * sizeof(int) );
    ppx = (float**) malloc2d( max_px, w1, sizeof(float) );
    ppy = (float**) malloc2d( max_py, h1, sizeof(float) );
    pqx = (int**)   malloc2d( max_px, w1, sizeof(int) );
    pqy = (int**)   malloc2d( max_py, h1, sizeof(int) );

  for( j=0; j<h1; j++ )
    fill_partitions( (float)j, (float)j+1, (float)h0, (float)h1, ppy[j], pqy[j], pny[j] );
  for( i=0; i<w1; i++ )
    fill_partitions( (float)i, (float)i+1, (float)w0, (float)w1, ppx[i], pqx[i], pnx[i] );

  int    nx,  ny;
  float *px, *py;
  int   *qx, *qy;

  for( j=0; j<h1; j++ )
  {
    py = ppy[j];  qy = pqy[j];  ny = pny[j];
    for( i=0; i<w1; i++ )
    {
      px = ppx[i];  qx = pqx[i];  nx = pnx[i];
      for( y=0; y<ny; y++ )
        for( x=0; x<nx; x++ )
          pfm.fm[j*w1+i] += fm[ qy[y]*w0 + qx[x] ] * py[y]*px[x];
    }
  }

  free( pnx );  free( pny );
  free( ppx );  free( ppy );
  free( pqx );  free( pqy );


//  qx = new int[max_px]; 
//  qy = new int[max_py];
//  px = new float[max_px];
//  py = new float[max_py];
//
//  for( j=0; j<h1; j++ )
//  {
//    fill_partitions( (float)j, (float)j+1, (float)h0, (float)h1, py, qy, ny );
//    for( i=0; i<w1; i++ )
//    {
//      fill_partitions( (float)i, (float)i+1, (float)w0, (float)w1, px, qx, nx );
//
//      int x,y;
//      for( y=0; y<ny; y++ )
//        for( x=0; x<nx; x++ )
//          pfm.fm[j*w1+i] = pfm.fm[j*w1+i] + fm[ qy[y]*w0 + qx[x] ] * py[y]*px[x];
//    }
//  }
//
//  delete[] px;
//  delete[] qx;
//  delete[] py;
//  delete[] qy;
}

void GPfm::scale( int width, int height )
{
  if( w==width && h==height )
    return;

  GPfm pfm;
    scale( pfm, width, height );

  load( pfm.w, pfm.h, pfm.fm );
}

void GPfm::resample( int width, int height )
{
  if( w==width && h==height )
    return;

  int i, j;
  GPfm des;
  GPfm &src = *this;

  des.load( width, height );
  for( j=0; j<des.h; j++ )
    for( i=0; i<des.w; i++ )
      des.pm[j][i] = src.lookup_linear( (i+.5f)/des.w*src.w, (j+.5f)/des.h*src.h );

  src.load( des.w, des.h, des.fm );
}

void GPfm::resize( int width, int height )
{
  if( w==width && h==height )
    return;

  GPfm &pfm = *this;

  if( width>pfm.w )
    pfm.resample( width, pfm.h );
  else
    pfm.scale( width, pfm.h );

  if( height>pfm.h )
    pfm.resample( width, height );
  else
    pfm.scale( width, height );
}



/////////////////////////
// g_bmp.cpp
//
// Created by Gary Ho, ma_hty@hotmail.com, 2005
//

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

GBmp::GBmp()
{
  memset( this, 0, sizeof(GBmp) );
}

GBmp::~GBmp()
{
  SAFE_FREE( bm );
  SAFE_FREE( pm );
}

void GBmp::load( const char *spath )
{

  bmp_header_info bmi;
  {
    FILE *f0 = fopen( spath, "rb" );
      if( f0==NULL )
      {
        printf( "[Error] GBmp::load, file %s not found.\n", spath );
        exit(-1);
      }
      memset( &bmi, 0, sizeof(bmp_header_info) );
      fread( &bmi, 18, 1, f0 );
      if( bmi.biSize==12 )
      {
        fread( &bmi.biWidth, sizeof(short), 1, f0 );
        fread( &bmi.biHeight, sizeof(short), 1, f0 );
        fread( &bmi.biPlanes, sizeof(short), 1, f0 );
        fread( &bmi.biBitCount, sizeof(short), 1, f0 );
      }else if( bmi.biSize==40 )
      {
        fread( &bmi.biWidth, 36, 1, f0 );
      }else
      {
        printf( "[Error] GBmp::load, unsupported header info format.\n" );
        exit(-1);
      }
    fclose(f0);
  }

  if( bmi.bfType != 'MB' )
  {
    printf( "[Error] GBmp::load, not bitmap file.\n" );
    exit(-1);
  }

  if( bmi.biCompression==0 && ( bmi.biBitCount==8 || bmi.biBitCount==4 || bmi.biBitCount==1 ) )
  {
    int i, j, l, pos, ncolor;

    if( bmi.biBitCount==8 )
      l = ( bmi.biWidth + (4-1) )/4*4;
    if( bmi.biBitCount==4 )
      l = ( (bmi.biWidth+1)/2 + (4-1) )/4*4;
    if( bmi.biBitCount==1 )
      l = ( (bmi.biWidth+7)/8 + (4-1) )/4*4;

    ncolor = bmi.biClrUsed ? bmi.biClrUsed : 1<<bmi.biBitCount;
    
    BYTE4 *palette;
    GBYTE *dat;
      palette = (BYTE4*) malloc( ncolor * sizeof(BYTE4) );
      dat = (GBYTE*) malloc( bmi.biHeight * l );
      FILE *f0 = fopen( spath, "rb" );
        fseek( f0, bmi.biSize+14, SEEK_SET );
        fread( palette, sizeof(BYTE4), ncolor, f0 );
        fseek( f0, bmi.bfOffBits, SEEK_SET );
        fread( dat, l, bmi.biHeight, f0 );
      fclose(f0);

    load( bmi.biWidth, bmi.biHeight );
    for( j=0; j<h; j++ )
    {
      GBYTE *src = (GBYTE*)&dat[ (bmi.biHeight-1-j)*l ];
      BYTE3 *des = pm[j];
      for( i=0; i<w; i++ )
      {
        if( bmi.biBitCount==8 )
          pos = src[i];
        if( bmi.biBitCount==4 )
          pos = i%2 ? (src[i/2]&0xF) : (src[i/2]>>4);
        if( bmi.biBitCount==1 )
          pos = ((src[i/8])>>(7-i%8))&1;
        des[i].x = palette[pos].z;
        des[i].y = palette[pos].y;
        des[i].z = palette[pos].x;
      }
    }
    free( palette );
    free( dat );
  }else
  if( bmi.biCompression==0 && bmi.biBitCount == 24 && bmi.biClrUsed==0 )
  {
    int i, j, l;

    l = ( bmi.biWidth*3 + (4-1) )/4*4;
    
    GBYTE *dat;
      dat = (GBYTE*) malloc( bmi.biHeight * l );
      FILE *f0 = fopen( spath, "rb" );
        fseek( f0, bmi.bfOffBits, SEEK_SET );
        fread( dat, l, bmi.biHeight, f0 );
      fclose(f0);
    
    load( bmi.biWidth, bmi.biHeight );
    for( j=0; j<h; j++ )
    {
      BYTE3 *src = (BYTE3*)&dat[ (bmi.biHeight-1-j)*l ];
      BYTE3 *des = pm[j];
      for( i=0; i<w; i++ )
      {
        des[i].x = src[i].z;
        des[i].y = src[i].y;
        des[i].z = src[i].x;
      }
    }

    free( dat );

  }else if( (bmi.biCompression==0 || bmi.biCompression==3) && bmi.biBitCount == 32 && bmi.biClrUsed==0 )
  {
    printf( "[Warning] GBmp::load, alpha channel is dropped.\n" );

    int i, j, l;
    int rmask, gmask, bmask;

    l = bmi.biWidth*4;

    GBYTE *dat;
      dat = (GBYTE*) malloc( bmi.biHeight * l );
      FILE *f0 = fopen( spath, "rb" );
        if( bmi.biCompression==3 )
        {
          fseek( f0, bmi.biSize+14, SEEK_SET );
          fread( &rmask, 1, sizeof(int), f0 );
          fread( &gmask, 1, sizeof(int), f0 );
          fread( &bmask, 1, sizeof(int), f0 );
        }else
        {
          rmask = 0xFF<<16;
          gmask = 0xFF<<8;
          bmask = 0xFF;
        }

        fseek( f0, bmi.bfOffBits, SEEK_SET );
        fread( dat, l, bmi.biHeight, f0 );
      fclose(f0);
    
    load( bmi.biWidth, bmi.biHeight );
    for( j=0; j<h; j++ )
    {
      BYTE4 *src = (BYTE4*)&dat[ (bmi.biHeight-1-j)*l ];
      BYTE3 *des = pm[j];
      for( i=0; i<w; i++ )
      {
        des[i].x = src[i].z;
        des[i].y = src[i].y;
        des[i].z = src[i].x;
      }
    }

    free( dat );

  }else if( (bmi.biCompression==0 || bmi.biCompression==3) && bmi.biBitCount == 16 && bmi.biClrUsed==0 )
  {
    int i, j, l;
    int rmask, gmask, bmask;
    int rshift, gshift, bshift;
    int rmax, gmax, bmax;
    GBYTE *dat;
    float r, g , b;

    l = ( bmi.biWidth*2 + (4-1) )/4*4;

    dat = (GBYTE*) malloc( bmi.biHeight * l );
    FILE *f0 = fopen( spath, "rb" );
      if( bmi.biCompression==3 )
      {
        fseek( f0, bmi.biSize+14, SEEK_SET );
        fread( &rmask, 1, sizeof(int), f0 );
        fread( &gmask, 1, sizeof(int), f0 );
        fread( &bmask, 1, sizeof(int), f0 );
      }else
      {
        rmask = 0x1F<<10;
        gmask = 0x1F<<5;
        bmask = 0x1F;
      }
      fseek( f0, bmi.bfOffBits, SEEK_SET );
      fread( dat, l, bmi.biHeight, f0 );
    fclose(f0);

    {
      int rgbmask[3] = { rmask, gmask, bmask };
      int rgbshift[3], rgbmax[3], nbit, idx[3];
      float val[] = { (float)rmask, (float)gmask, (float)bmask };
      bubble_sort( val, idx, 3 );
      for( i=0, nbit=0; i<3; i++ )
      {
        rgbshift[idx[i]] = nbit;
        rgbmax[idx[i]] = rgbmask[idx[i]]>>rgbshift[idx[i]];
        nbit += ROUND( log(rgbmax[idx[i]]+1.0)/log(2.0) );
      }
      rshift=rgbshift[0];  gshift=rgbshift[1];  bshift=rgbshift[2];
      rmax=rgbmax[0];  gmax=rgbmax[1];  bmax=rgbmax[2];
    }

    load( bmi.biWidth, bmi.biHeight );
    for( j=0; j<h; j++ )
    {
      unsigned short *src = (unsigned short*)&dat[ (bmi.biHeight-1-j)*l ];
      BYTE3 *des = pm[j];
      for( i=0; i<w; i++ )
      {
        r = float( (src[i]&rmask)>>rshift ) / rmax;
        g = float( (src[i]&gmask)>>gshift ) / gmax;
        b = float( (src[i]&bmask)>>bshift ) / bmax;
        des[i].x = (GBYTE)ftoi(r,255);
        des[i].y = (GBYTE)ftoi(g,255);
        des[i].z = (GBYTE)ftoi(b,255);
      }
    }

    free( dat );

  }else
  {
    printf( "[Error] GBmp::load, unsupport bitmap format.\n" );
    exit(-1);
  }
}

void GBmp::load( int width, int height )
{
  SAFE_FREE( bm );
  SAFE_FREE( pm );

  w = width;
  h = height;
  bm = (BYTE3*) malloc( w * h * sizeof(BYTE3) );
  memset( bm, 0, w*h*sizeof(BYTE3) );

  pm = (BYTE3**) malloc( h * sizeof(BYTE3*) );
  int j;
  for( j=0; j<h; j++ )
    pm[j] = &bm[j*w];
}

void GBmp::flip_vertical()
{
  GBmp tmp;
    tmp.load(w,h);

  int j;
  for( j=0; j<h; j++ )
    memcpy( tmp.pm[j], pm[h-j-1], w*sizeof(BYTE3) );
    memcpy( bm, tmp.bm, w*h*sizeof(BYTE3) );
}

void GBmp::rb_swap()
{
  GBYTE tmp;
  int i,j;
  for( j=0; j<h; j++ )
    for( i=0; i<w; i++ )
    {
     tmp = pm[j][i].x;
     pm[j][i].x = pm[j][i].z;
     pm[j][i].z = tmp;
    }
}
void GBmp::save( const char *spath ) const
{
  bmp_header_info bhi;
    bhi.bfType = 'MB';
    bhi.bfSize = w*h*3*sizeof(unsigned char) + sizeof(bhi);
    bhi.bfReserved1 = 0;
    bhi.bfReserved2 = 0;
    bhi.bfOffBits = sizeof(bhi);

    bhi.biSize = 40;
    bhi.biWidth = w;
    bhi.biHeight = h;
    bhi.biPlanes = 1;
    bhi.biBitCount = 24;
    bhi.biCompression = 0;
    bhi.biSizeImage = 0;
    bhi.biXpelsPerMeter = 0;
    bhi.biYpelsPerMeter = 0;
    bhi.biClrUsed = 0;
    bhi.biClrImportant = 0;

  int j;
  GBmp a;
    a.load( w,h );
    memcpy( a.bm, bm, w*h*sizeof(BYTE3) );
    a.rb_swap();
    a.flip_vertical();
  unsigned char pad[3] = {0,0,0};

  FILE *f0 = fopen( spath, "wb" );
    fwrite( &bhi, sizeof(bmp_header_info), 1, f0 );
    for( j=0; j<h; j++ )
    {
      fwrite( a.pm[j], sizeof(BYTE3), w, f0 );
      fwrite( pad, sizeof(unsigned char), (4-w*3%4)%4, f0 );
    }
  fclose(f0);
}

void GBmp::getblk( GBmp &blk, int x, int y, int blkw, int blkh ) const
{
  if( blk.w != blkw || blk.h != blkh )
    blk.load( blkw,blkh );

  int j;

  for( j=0; j<blkh; j++ )
    memcpy( blk.pm[j], &pm[y+j][x], blkw * sizeof(BYTE3) );
}

void GBmp::load( int width, int height, const BYTE3 *prgb )
{
  load(width,height);
  memcpy( bm, prgb, w*h*sizeof(BYTE3) );
}

void GBmp::load( int width, int height, const FLOAT3 *fm )
{
  load( width, height );
  int i,j;
  for( j=0; j<h; j++ )
    for( i=0; i<w; i++ )
    {
      bm[j*w+i].x = (GBYTE)ftoi(fm[j*w+i].x,255);
      bm[j*w+i].y = (GBYTE)ftoi(fm[j*w+i].y,255);
      bm[j*w+i].z = (GBYTE)ftoi(fm[j*w+i].z,255);
    }
}


GPPm::GPPm()
{
  memset( this, 0, sizeof(GPPm) );
}

GPPm::~GPPm()
{
  SAFE_FREE( bm );
  SAFE_FREE( pm );
}

void GPPm::load( int width, int height )
{
  SAFE_FREE( bm );
  SAFE_FREE( pm );

  w = width;
  h = height;
  bm = (BYTE3*) malloc( w * h * sizeof(BYTE3) );
  memset( bm, 0, w*h*sizeof(BYTE3) );

  pm = (BYTE3**) malloc( h * sizeof(BYTE3*) );
  int j;
  for( j=0; j<h; j++ )
    pm[j] = &bm[j*w];
}

void GPPm::save( const char *spath ) const
{
  FILE *f0 = fopen( spath, "wb" );
    fprintf( f0, "P6\n");
	fprintf( f0, "%i %i\n", w, h );
	fprintf( f0, "%i\n", 255 );
    fwrite( bm, sizeof(BYTE3), w*h, f0 );
  fclose( f0 );
}

void GPPm::flip_vertical()
{
  GPPm tmp;
    tmp.load(w,h);

  int j;
  for( j=0; j<h; j++ )
    memcpy( tmp.pm[j], pm[h-j-1], w*sizeof(BYTE3) );
    memcpy( bm, tmp.bm, w*h*sizeof(BYTE3) );
}

void GPPm::load( const char *spath )
{
  load( spath, 0,0,0,0 );
}

void GPPm::load( const char *spath, int x, int y, int blkw, int blkh )
{
  char str[256], info[2];
  int width, height;
  short mode;


  FILE *f0 = fopen( spath, "rb" );
    if( f0==NULL )
    {
      printf( "[Error] : GPfm::load(), \"%s\" not found.\n", spath );
      exit(-1);
    }
    fread( &mode, sizeof(short), 1, f0 ); swapbyte( &mode, sizeof(mode) );
    fread( info, sizeof(char), 2, f0 );

  fclose(f0);


  if( mode=='P5' || mode=='P6'  )
  {
    FILE *f0 = fopen( spath, "rb" );

      if( info[0]=='\n' || info[1]=='\n' )
      {
        fgets( str, 256, f0 );
        while( fgets( str, 256, f0 ) && str[0] == '#' );
        sscanf(str, "%i %i", &width, &height);
        fgets( str, 256, f0 );
      }else if( info[0]==' ' )
      {
        fgets( str, 256, f0 );
        char tmp[16];
        sscanf( str, "%s %i %i", tmp, &width, &height );
      }else if( info[0]=='\r' && info[1]!='\n' )
      {
        fscanf( f0, "%[^\r]", str ); fgetc(f0);
        do{ fscanf( f0, "%[^\r]", str ); fgetc(f0); }while( str[0] == '#' );
        sscanf(str, "%i %i", &width, &height);
        fscanf( f0, "%[^\r]", str ); fgetc(f0);
      }else
      {
        printf( "[Error] GPPm::load, incorrect file format\n" );
        exit(-1);
      }

      if( x==0 && y==0 && blkw==0 && blkh==0 )
      {
        load(width,height);
        fread( bm, sizeof(BYTE3), w*h, f0 );
      }else
      {
        GPPm tmp;
          tmp.load( width, blkh );
          fseek( f0, y*width*sizeof(BYTE3), SEEK_CUR );
          fread( tmp.bm, sizeof(BYTE3), tmp.w*tmp.h, f0 );
          tmp.getblk( *this, x,0, blkw,blkh );
      }

    fclose(f0);

  }else
  {
    printf( "[Error] : GPPm::load(), unsupported file format\n" );
    exit(-1);
  }

}

void GPPm::getblk( GPPm &blk, int x, int y, int blkw, int blkh ) const
{
  if( blk.w != blkw || blk.h != blkh )
    blk.load( blkw,blkh );

  int j;

  for( j=0; j<blkh; j++ )
    memcpy( blk.pm[j], &pm[y+j][x], blkw * sizeof(BYTE3) );
}

void GPPm::load( int width, int height, const BYTE3 *prgb )
{
  load(width,height);
  memcpy( bm, prgb, w*h*sizeof(BYTE3) );
}

void GPPm::load( int width, int height, const FLOAT3 *fm )
{
  load( width, height );

  int i,j;
  for( j=0; j<h; j++ )
    for( i=0; i<w; i++ )
    {
      bm[j*w+i].x = (GBYTE)ftoi(fm[j*w+i].x,255);
      bm[j*w+i].y = (GBYTE)ftoi(fm[j*w+i].y,255);
      bm[j*w+i].z = (GBYTE)ftoi(fm[j*w+i].z,255);
    }
}

void GPPm::flip_horizontal()
{
  int i,j;
  int ilen = w/2;

  BYTE3 *line0 = bm;
  BYTE3 c;

  for( j=0; j<h; j++, line0+=w )
    for( i=0; i<ilen; i++ )
    {
      c = line0[i];
      line0[i] = line0[w-i-1];
      line0[w-i-1] = c;
    }

}

GPf4::GPf4()
{
  w = 0;
  h = 0;
  fm = NULL;
  pm = NULL;
}

GPf4::~GPf4()
{
  SAFE_FREE( fm );
  SAFE_FREE( pm );
}

void GPf4::load( int width, int height )
{
  SAFE_FREE( fm );
  SAFE_FREE( pm );

  w = width;
  h = height;
  fm = (FLOAT4*) malloc( w * h * sizeof(FLOAT4) );

  memset( fm, 0, w*h*sizeof(FLOAT4) );

  pm = (FLOAT4**) malloc( h * sizeof(FLOAT4*) );
  int j;
  for( j=0; j<h; j++ )
    pm[j] = &fm[j*w];
}

void GPf4::flip_vertical()
{
  FLOAT4 *tline, *line0, *line1;
    tline = (FLOAT4*) malloc( w * sizeof(FLOAT4) );
    line0 = fm;
    line1 = fm + w*(h-1);

  int j, jlen = h/2;
  for( j=0; j<jlen; j++, line0+=w, line1-=w )
  {
    memcpy( tline, line0, w*sizeof(FLOAT4) );
    memcpy( line0, line1, w*sizeof(FLOAT4) );
    memcpy( line1, tline, w*sizeof(FLOAT4) );
  }

  free(tline);
}

void GPf4::flip_horizontal()
{
  int i,j;
  int ilen = w/2;

  FLOAT4 *line0 = fm;
  FLOAT4 c;

  for( j=0; j<h; j++, line0+=w )
    for( i=0; i<ilen; i++ )
    {
      c = line0[i];
      line0[i] = line0[w-i-1];
      line0[w-i-1] = c;
    }
}


void GPf4::load( const char *spath )
{
  char str[256], info[2];
  int width, height;
  short mode;


  FILE *f0 = fopen( spath, "rb" );
    if( f0==NULL )
    {
      printf( "[Error] : GPf4::load(), \"%s\" not found.\n", spath );
      exit(-1);
    }
    fread( &mode, sizeof(short), 1, f0 ); swapbyte( &mode, sizeof(mode) );
    fread( info, sizeof(char), 2, f0 );

  fclose(f0);


  if( mode=='pf'  )
  {
    FILE *f0 = fopen( spath, "rb" );

      float byte_ordering;
      if( info[0]=='\n' || info[1]=='\n' )
      {
        fgets( str, 256, f0 );
        while( fgets( str, 256, f0 ) && str[0] == '#' );
        sscanf(str, "%i %i", &width, &height);
        fgets( str, 256, f0 );
        sscanf(str, "%f", &byte_ordering);
      }else if( info[0]==' ' )
      {
        fgets( str, 256, f0 );
        char tmp[16];
        sscanf( str, "%s %i %i %f", tmp, &width, &height, &byte_ordering );
      }else if( info[0]=='\r' && info[1]!='\n' )
      {
        fscanf( f0, "%[^\r]", str ); fgetc(f0);
        do{ fscanf( f0, "%[^\r]", str ); fgetc(f0); }while( str[0] == '#' );
        sscanf(str, "%i %i", &width, &height);
        fscanf( f0, "%[^\r]", str ); fgetc(f0);
        sscanf(str, "%f", &byte_ordering);
      }else
      {
        printf( "[Error] GPf4::load, incorrect file format\n" );
        exit(-1);
      }

      load(width,height);
      fread( fm, sizeof(FLOAT4), w*h, f0 );
      flip_vertical();

      int i,j;
      if( byte_ordering==1 )
        for( j=0; j<h; j++ )
          for( i=0; i<w; i++ )
          {
            swapbyte( &pm[j][i].x, sizeof(float) );
            swapbyte( &pm[j][i].y, sizeof(float) );
            swapbyte( &pm[j][i].z, sizeof(float) );
            swapbyte( &pm[j][i].w, sizeof(float) );
          }

    fclose(f0);

  }else
  {
    printf( "[Error] : GPf4::load(), unsupported file format\n" );
    exit(-1);
  }
}

void GPf4::save( const char *spath ) const
{
  int j;

  FILE *f0 = fopen( spath, "wb" );
    fprintf( f0, "pf\n");
	  fprintf( f0, "%i %i\n", w, h );
	  fprintf( f0, "%f\n", -1.f );
    for( j=h-1; j>=0; j-- )
      fwrite( pm[j], sizeof(FLOAT4), w, f0 );
  fclose( f0 );
}

void GPf4::load( int width, int height, const float *r, const float *g, const float *b, const float *a )
{
  load( width, height );

  int i, l;
  FLOAT4 *tm;
  l = w*h;

  if(r)
    for( i=0, tm=fm; i<l; i++, tm++, r++ )
      tm->x = *r;
  if(g)
    for( i=0, tm=fm; i<l; i++, tm++, g++ )
      tm->y = *g;
  if(b)
    for( i=0, tm=fm; i<l; i++, tm++, b++ )
      tm->z = *b;
  if(a)
    for( i=0, tm=fm; i<l; i++, tm++, a++ )
      tm->w = *a;
}

void GPf4::load( int width, int height, const FLOAT3 *rgb )
{
  load( width, height );

  int i, l;
  l = w*h;
  for( i=0; i<l; i++, rgb++ )
    fm[i] = FLOAT4( *rgb, 1 );
}

void GPf4::load( int width, int height, FLOAT4 a )
{
  load( width, height );

  int i, l;
  l = w*h;
  for( i=0; i<l; i++ )
    fm[i] = a;
}

void GPf4::draw( const GPf4 &blk, int x, int y )
{
  int src_w, src_h;
    src_w = blk.w;
    src_h = blk.h;

  int des_w, des_h;
    des_w = w;
    des_h = h;

  FLOAT4 *src, *des;
    src = blk.fm;
    des = &fm[y*w + x];

  int j;
  for( j=0; j<src_h; j++, src+=src_w, des+=des_w )
    memcpy( des, src, src_w * sizeof(FLOAT4) );
}


void GPf4::getchannel( float *r, float *g, float *b, float *a ) const
{
  int i, l;
  FLOAT4 *tm;
  l=w*h;

  if(r)
  for( i=0, tm=fm; i<l; i++, tm++, r++ )
    *r = tm->x;
  if(g)
  for( i=0, tm=fm; i<l; i++, tm++, g++ )
    *g = tm->y;
  if(b)
  for( i=0, tm=fm; i<l; i++, tm++, b++ )
    *b = tm->z;
  if(a)
  for( i=0, tm=fm; i<l; i++, tm++, a++ )
    *a = tm->w;
}

void GPf4::getchannel( GPfm &vec_xyz, GPf1 &vec_w ) const
{
  vec_xyz.load( w, h );
  vec_w.load( w, h );

  int t, l;
  FLOAT4 *tm;

  l=w*h;
  tm=fm;

  FLOAT3 *xyz = vec_xyz.fm;
  float  *w   = vec_w.fm;

  for( t=0; t<l; t++, tm++, xyz++, w++ )
  {
    *xyz = FLOAT3( tm->x, tm->y, tm->z );
    *w = tm->w;
  }
}

void GPf4::load( int width, int height, const FLOAT4 *rgba )
{
  load( width, height );
  memcpy( fm, rgba, w * h * sizeof(FLOAT4) );
}

FLOAT4 GPf4::vmax() const
{
  int t, l;
  FLOAT4 *tm;
  float mx,my,mz,mw;

  l = w*h;
  tm = fm;

  mx = tm->x;
  my = tm->y;
  mz = tm->z;
  mw = tm->w;

  for( t=0; t<l; t++, tm++ )
  {
    if( tm->x > mx ) mx=tm->x;
    if( tm->y > my ) my=tm->y;
    if( tm->z > mz ) mz=tm->z;
    if( tm->w > mw ) mw=tm->w;
  }

  return FLOAT4( mx,my,mz,mw );
}

FLOAT4 GPf4::vmin() const
{
  int t, l;
  FLOAT4 *tm;
  float mx,my,mz,mw;

  l = w*h;
  tm = fm;

  mx = tm->x;
  my = tm->y;
  mz = tm->z;
  mw = tm->w;

  for( t=0; t<l; t++, tm++ )
  {
    if( tm->x < mx ) mx=tm->x;
    if( tm->y < my ) my=tm->y;
    if( tm->z < mz ) mz=tm->z;
    if( tm->w < mw ) mw=tm->w;
  }

  return FLOAT4( mx,my,mz,mw );
}

FLOAT4 GPf4::vmean() const
{
  int t, l;
  FLOAT4 *tm;
  FLOAT4 res;

  l = w*h;
  tm = fm;

  res = 0;
  for( t=0; t<l; t++, tm++ )
    res = res + *tm;

  return res/float(l);
}

FLOAT4 GPf4::variance() const
{
  int t, l;
  FLOAT4 *tm;
  FLOAT4 mn;
  FLOAT4 var;
  FLOAT4 tmp;
    
  l = w*h;
  tm = fm;
  mn = vmean();
  var = 0;
    
  for( t=0; t<l; t++, tm++ )
  {
    tmp = *tm - mn;
    var = var + tmp*tmp;
  }

  return var/float(l);
}

void GPf4::mul( FLOAT4 a )
{
  int t, l;
  FLOAT4 *tm;

  tm = fm;
  l = w*h;
  for( t=0; t<l; t++, tm++ )
    *tm = *tm * a;
}

void GPf4::add( FLOAT4 a )
{
  int t, l;
  FLOAT4 *tm;

  tm = fm;
  l = w*h;
  for( t=0; t<l; t++, tm++ )
    *tm = *tm + a;
}







GPf1::GPf1()
{
  w = 0;
  h = 0;
  fm = NULL;
  pm = NULL;
}

GPf1::~GPf1()
{
  SAFE_FREE( fm );
  SAFE_FREE( pm );
}

void GPf1::load( int width, int height, const float *pa )
{
  load( width, height );
  memcpy( fm, pa, w * h * sizeof(float) );
}

void GPf1::load( int width, int height, const FLOAT3 *gm )
{
  load( width, height );
  int i, n = w*h;
  float *tm = fm;

  for( i=0; i<n; i++, tm++ )
    *tm = (gm[i].x+gm[i].y+gm[i].z)/3;
}

void GPf1::load( int width, int height, float a )
{
  load( width, height );

  int i;
  int n = w*h;
  float *tm = fm;

  for( i=0; i<n; i++, tm++ )
    *tm = a;
}


void GPf1::load( int width, int height )
{
  SAFE_FREE( fm );
  SAFE_FREE( pm );

  w = width;
  h = height;
  fm = (float*) malloc( w * h * sizeof(float) );
  memset( fm, 0, w*h*sizeof(float) );

  pm = (float**) malloc( h * sizeof(float*) );
  int j;
  for( j=0; j<h; j++ )
    pm[j] = &fm[j*w];
}

void GPf1::load_no_flip( const char *spath )
{
  char str[256], info[2];
  int width, height;
  short mode;


  FILE *f0 = fopen( spath, "rb" );
    if( f0==NULL )
    {
      printf( "[Error] : GPf1::load(), \"%s\" not found.\n", spath );
      exit(-1);
    }
    fread( &mode, sizeof(short), 1, f0 ); swapbyte( &mode, sizeof(mode) );
    fread( info, sizeof(char), 2, f0 );

  fclose(f0);


  if( mode=='Pf'  )
  {
    FILE *f0 = fopen( spath, "rb" );

      float byte_ordering;
      if( info[0]=='\n' || info[1]=='\n' )
      {
        fgets( str, 256, f0 );
        while( fgets( str, 256, f0 ) && str[0] == '#' );
        sscanf(str, "%i %i", &width, &height);
        fgets( str, 256, f0 );
        sscanf(str, "%f", &byte_ordering);
      }else if( info[0]==' ' )
      {
        fgets( str, 256, f0 );
        char tmp[16];
        sscanf( str, "%s %i %i %f", tmp, &width, &height, &byte_ordering );
      }else if( info[0]=='\r' && info[1]!='\n' )
      {
        fscanf( f0, "%[^\r]", str ); fgetc(f0);
        do{ fscanf( f0, "%[^\r]", str ); fgetc(f0); }while( str[0] == '#' );
        sscanf(str, "%i %i", &width, &height);
        fscanf( f0, "%[^\r]", str ); fgetc(f0);
        sscanf(str, "%f", &byte_ordering);
      }else
      {
        printf( "[Error] GPf1::load, incorrect file format\n" );
        exit(-1);
      }

      load(width,height);
      fread( fm, sizeof(float), w*h, f0 );

      int i,j;
      if( byte_ordering==1 )
        for( j=0; j<h; j++ )
          for( i=0; i<w; i++ )
            swapbyte( &pm[j][i], sizeof(float) );

    fclose(f0);

  }else
  {
    printf( "[Error] : GPf1::load(), unsupported file format\n" );
    exit(-1);
  }
}

void GPf1::load( const char *spath )
{
  char str[256], info[2];
  int width, height;
  short mode;


  FILE *f0 = fopen( spath, "rb" );
    if( f0==NULL )
    {
      printf( "[Error] : GPf1::load(), \"%s\" not found.\n", spath );
      exit(-1);
    }
    fread( &mode, sizeof(short), 1, f0 ); swapbyte( &mode, sizeof(mode) );
    fread( info, sizeof(char), 2, f0 );

  fclose(f0);


  if( mode=='Pf'  )
  {
    FILE *f0 = fopen( spath, "rb" );

      float byte_ordering;
      if( info[0]=='\n' || info[1]=='\n' )
      {
        fgets( str, 256, f0 );
        while( fgets( str, 256, f0 ) && str[0] == '#' );
        sscanf(str, "%i %i", &width, &height);
        fgets( str, 256, f0 );
        sscanf(str, "%f", &byte_ordering);
      }else if( info[0]==' ' )
      {
        fgets( str, 256, f0 );
        char tmp[16];
        sscanf( str, "%s %i %i %f", tmp, &width, &height, &byte_ordering );
      }else if( info[0]=='\r' && info[1]!='\n' )
      {
        fscanf( f0, "%[^\r]", str ); fgetc(f0);
        do{ fscanf( f0, "%[^\r]", str ); fgetc(f0); }while( str[0] == '#' );
        sscanf(str, "%i %i", &width, &height);
        fscanf( f0, "%[^\r]", str ); fgetc(f0);
        sscanf(str, "%f", &byte_ordering);
      }else
      {
        printf( "[Error] GPf1::load, incorrect file format\n" );
        exit(-1);
      }

      load(width,height);
      fread( fm, sizeof(float), w*h, f0 );
      flip_vertical();

      int i,j;
      if( byte_ordering==1 )
        for( j=0; j<h; j++ )
          for( i=0; i<w; i++ )
            swapbyte( &pm[j][i], sizeof(float) );

    fclose(f0);

  }else
  {
    printf( "[Error] : GPf1::load(), unsupported file format\n" );
    exit(-1);
  }
}


void GPf1::getblk( GPf1 &blk, int x, int y, int blkw, int blkh ) const
{
  if( blk.w != blkw || blk.h != blkh )
    blk.load( blkw,blkh );

  int j;

  for( j=0; j<blkh; j++ )
    memcpy( blk.pm[j], &pm[y+j][x], blkw * sizeof(float) );
}

void GPf1::draw( const GPf1 &blk, int x, int y )
{
  float *src, *des;
  int j;

  src = blk.fm;
  des = &fm[y*w + x];
  for( j=0; j<blk.h; j++, src+=blk.w, des+=w )
    memcpy( des, src, blk.w * sizeof(float) );
}

void GPf1::draw( const float *src, int x, int y, int blkw, int blkh )
{
  float *des;
  int j;

  des = &fm[y*w + x];
  for( j=0; j<blkh; j++, src+=blkw, des+=w )
    memcpy( des, src, blkw * sizeof(float) );
}


void GPf1::save( const char *spath ) const
{
  int j;

  FILE *f0 = fopen( spath, "wb" );
    fprintf( f0, "Pf\n" );
	  fprintf( f0, "%i %i\n", w, h );
	  fprintf( f0, "%f\n", -1.f );
    for( j=h-1; j>=0; j-- )
      fwrite( pm[j], sizeof(float), w, f0 );
  fclose( f0 );
}

void GPf1::flip_vertical()
{
  float *tline, *line0, *line1;
    tline = (float*) malloc( w * sizeof(float) );
    line0 = fm;
    line1 = fm + w*(h-1);

  int j, jlen = h/2;
  for( j=0; j<jlen; j++, line0+=w, line1-=w )
  {
    memcpy( tline, line0, w*sizeof(float) );
    memcpy( line0, line1, w*sizeof(float) );
    memcpy( line1, tline, w*sizeof(float) );
  }

  free(tline);
}

void GPf1::flip_horizontal()
{
  int i,j;
  int ilen = w/2;

  float *line0 = fm;
  float c;

  for( j=0; j<h; j++, line0+=w )
    for( i=0; i<ilen; i++ )
    {
      c = line0[i];
      line0[i] = line0[w-i-1];
      line0[w-i-1] = c;
    }

}

void GPf1::scale( int width, int height )
{
  GPf1 tmp;
  tmp.load( width, height );

  int w0,h0, w1,h1;
    w0 = w;
    h0 = h;
    w1 = tmp.w;
    h1 = tmp.h;

  int max_px, max_py;
    max_px = w0 / w1  + 2;
    max_py = h0 / h1  + 2;

  int i,j;
  int x,y;

  int    *pnx,  *pny;
  float **ppx, **ppy;
  int   **pqx, **pqy;
    pnx = (int*)    malloc  (         w1 * sizeof(int) );
    pny = (int*)    malloc  (         h1 * sizeof(int) );
    ppx = (float**) malloc2d( max_px, w1, sizeof(float) );
    ppy = (float**) malloc2d( max_py, h1, sizeof(float) );
    pqx = (int**)   malloc2d( max_px, w1, sizeof(int) );
    pqy = (int**)   malloc2d( max_py, h1, sizeof(int) );

  for( j=0; j<h1; j++ )
    GPfm::fill_partitions( (float)j, (float)j+1, (float)h0, (float)h1, ppy[j], pqy[j], pny[j] );
  for( i=0; i<w1; i++ )
    GPfm::fill_partitions( (float)i, (float)i+1, (float)w0, (float)w1, ppx[i], pqx[i], pnx[i] );

  int    nx,  ny;
  float *px, *py;
  int   *qx, *qy;

  for( j=0; j<h1; j++ )
  {
    py = ppy[j];  qy = pqy[j];  ny = pny[j];
    for( i=0; i<w1; i++ )
    {
      px = ppx[i];  qx = pqx[i];  nx = pnx[i];
      for( y=0; y<ny; y++ )
        for( x=0; x<nx; x++ )
          tmp.fm[j*w1+i] = tmp.fm[j*w1+i] + fm[ qy[y]*w0 + qx[x] ] * py[y]*px[x];
    }
  }

  free( pnx );  free( pny );
  free( ppx );  free( ppy );
  free( pqx );  free( pqy );

  load( tmp.w, tmp.h, tmp.fm );
}



/////////////////////////////////////////////////////////////////////
//
//  g_png.cpp

GPng::GPng()
{
  memset(this,0,sizeof(GPng)); 
}

GPng::~GPng()
{
  SAFE_FREE(bm);
  SAFE_FREE(pm);
}

void GPng::load( const char *spath )
{
  IHDR ihdr;
  FILE *f0 = fopen( spath, "rb" );
  if( f0==NULL )
  {
    printf( "[Error] GPng::load, file %s not found.\n", spath );
    exit(-1);
  }

  unsigned char pngsignature[] = { 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n' };

  unsigned char signature[8];
    fread( signature, 1, 8, f0 );

  if( memcmp( signature, pngsignature, 8 ) )
  {
    printf( "[Error] GPng::load, not GPng file.\n" );
    exit(-1);
  }else
  {
    int n_src;
    int chk_len;
    int chk_id;
    int chk_crc;

    unsigned char *src;
    BYTE3 *palette;


    palette = NULL;
    n_src = 0;


    do
    {
      fread( &chk_len, sizeof(int), 1, f0 ); 
      swapbyte( &chk_len, 4 );
      if( fread( &chk_id , sizeof(int), 1, f0 )==0 )
      {
        printf("[Error] GPng::load(), End of file reached.  File may be corrupted.");
        exit(-1);
      }
      swapbyte( &chk_id, 4 );
  
      switch( chk_id )
      {
        case 'IHDR':
          fread( &ihdr, sizeof(IHDR), 1, f0 );
          swapbyte( &ihdr.Width, 4 );
          swapbyte( &ihdr.Height, 4 );

          if( ihdr.Interlace!=0 && ihdr.Interlace!=1 )
          {
            printf("[Error] GPng::load(), Interlace type not supported.  Interlace=%i\n", ihdr.Interlace);
            exit(-1);
          }
          if( ihdr.Bit!=8 )
          {
            printf("[Error] GPng::load(), Color Depth not supported.  Colour Depth=%i\n", ihdr.Bit);
            exit(-1);
          }
          if( ihdr.Colour!=2 && ihdr.Colour!=3 )
          {
            printf("[Error] GPng::load(), Color Type not supported.  Colour=%i\n", ihdr.Colour);
            exit(-1);
          }

          load( ihdr.Width, ihdr.Height );
          src = (unsigned char*)malloc( (w*3+1)*h*sizeof(unsigned char) );
        break;

        case 'PLTE':
        {
          palette = (BYTE3*)malloc( chk_len );
          fread( palette, sizeof(unsigned char), chk_len, f0 );
        }
        break;
    
        case 'IDAT':
        {
          if( n_src+chk_len > (w*3+1)*h )
            src = (unsigned char*)realloc( src, n_src+chk_len );
          fread( &src[n_src], sizeof(unsigned char), chk_len, f0 );
          //if( (src[2]&0x06)==0 )
          //{
          //  printf("[Error] GPng::load(), Uncompressed format not supported.\n" );
          //  exit(-1);
          //}
          n_src += chk_len;
        }
        break;
    
        case 'IEND':
        break;

        default:
        {
          char *str = (char*) &chk_id;
          // printf( "unknown chunk - %c%c%c%c\n", str[3], str[2], str[1], str[0] );
          fseek( f0, chk_len, SEEK_CUR );
        }
        break;
      }
      fread( &chk_crc, sizeof(int), 1, f0 ); swapbyte(&chk_crc,4);

    }while( chk_id!='IEND' );
    
    int i, j;
    int type, kb;
    BYTE3 *null_row;
    int n_des;
    unsigned char *des, *trow;


    if(ihdr.Colour==2)
      kb=3;
    if(ihdr.Colour==3)
      kb=1;
    null_row = (BYTE3*)malloc( w * sizeof(BYTE3) );
    memset( null_row, 0, w*sizeof(BYTE3) );

    if( ihdr.Interlace==0 )
    {
      n_des = (w*kb+1)*h;
      des = (unsigned char*)malloc( n_des*sizeof(unsigned char) );
      decompress( src, n_src, des, n_des );

      if( ihdr.Colour==2 )
      {
        type = des[0];
        memcpy( pm[0], &des[1], w * sizeof(BYTE3) );
        filter( pm[0], null_row, type );
        for( j=1; j<h; j++ )
        {
          trow = &des[(w*3+1)*j];
          type = trow[0];
          memcpy( pm[j], &trow[1], w * sizeof(BYTE3) );
          filter( pm[j], pm[j-1], type );
        }
      }
      if( ihdr.Colour==3 )
      {
        for( j=0; j<h; j++ )
        {
          trow = &des[(w+1)*j];
          for( i=0, trow++; i<w; i++ )
            pm[j][i] = palette[ trow[i] ];
        }
      }
    }

    if( ihdr.Interlace==1 )
    {
      int k, kw, kh;
      int ps[][4] =
      {
        {8,8,0,0}, {8,8,4,0}, {4,8,0,4}, 
        {4,4,2,0}, {2,4,0,2}, {2,2,1,0},
        {1,2,0,1},
      };

      for( k=0, n_des=0; k<8; k++ )
      {
        kw = (w+ps[k][0]-1-ps[k][2])/ps[k][0];
        kh = (h+ps[k][1]-1-ps[k][3])/ps[k][1];
        n_des += (1+kw*kb)*kh;
      }
      des = (unsigned char*)malloc(n_des*sizeof(unsigned char));
      decompress( src, n_src, des, n_des );
      if(ihdr.Colour==2)
      {
        for( k=0, trow=des; k<7; k++ )
          for( j=ps[k][3]; j<h; j+=ps[k][1] )
            for( i=ps[k][2], trow++; i<w; i+=ps[k][0], trow+=3 )
              pm[j][i] = *((BYTE3*)trow);
      }
      if(ihdr.Colour==3)
      {
        for( k=0, trow=des; k<7; k++ )
          for( j=ps[k][3]; j<h; j+=ps[k][1] )
            for( i=ps[k][2], trow++; i<w; i+=ps[k][0], trow++ )
              pm[j][i] = palette[ *trow ];
      }
    }

    free(src);
    free(des);
    free(null_row);
    SAFE_FREE(palette);
  }

  fclose(f0);
}

void GPng::load( int width, int height )
{
  SAFE_FREE(bm);
  SAFE_FREE(pm);

  w = width;
  h = height;
  bm = (BYTE3*)malloc( w * h * sizeof(BYTE3) );
  memset( bm, 0, w * h * sizeof(BYTE3) );
  pm = (BYTE3**) malloc( h * sizeof(BYTE3*) );
  int j;
  for( j = 0; j<h; j++ )
    pm[j] = &bm[j*w];
}

void GPng::load( int width, int height, const FLOAT3 *fm )
{
  load( width, height );

  int i,j;
  for( j=0; j<h; j++ )
    for( i=0; i<w; i++ )
    {
      bm[j*w+i].x = (GBYTE)ftoi(fm[j*w+i].x,255);
      bm[j*w+i].y = (GBYTE)ftoi(fm[j*w+i].y,255);
      bm[j*w+i].z = (GBYTE)ftoi(fm[j*w+i].z,255);
    }
}

void GPng::load( int width, int height, const BYTE3 *prgb )
{
  load(width,height);
  memcpy( bm, prgb, w*h*sizeof(BYTE3) );
}

inline void GPng::filter(BYTE3 *curline, BYTE3 *prvline, int type)
{
  
  int i;
  switch(type)
  {
    case 1:
      for( i=1; i<w; i++ )
      {
        curline[i].x += curline[i-1].x;
        curline[i].y += curline[i-1].y;
        curline[i].z += curline[i-1].z;
      }
      break;
    case 2:
      for( i=0; i<w; i++ )
      {
        curline[i].x += prvline[i].x;
        curline[i].y += prvline[i].y;
        curline[i].z += prvline[i].z;
      }
      break;
    case 3:
      curline[0].x += ( prvline[0].x )/2;
      curline[0].y += ( prvline[0].y )/2;
      curline[0].z += ( prvline[0].z )/2;
    
      for( i=1; i<w; i++ )
      {
        curline[i].x += ( prvline[i].x + curline[i-1].x )/2;
        curline[i].y += ( prvline[i].y + curline[i-1].y )/2;
        curline[i].z += ( prvline[i].z + curline[i-1].z )/2;
      }
      break;
    case 4:
      curline[0].x += Paeth( 0,prvline[0].x,0 );
      curline[0].y += Paeth( 0,prvline[0].y,0 );
      curline[0].z += Paeth( 0,prvline[0].z,0 );
    
      unsigned int a,b,c;
      for( i=1; i<w; i++ )
      {
        a = curline[i-1].x;
        b = prvline[i].x;
        c = prvline[i-1].x;
        curline[i].x += Paeth(a,b,c);
        a = curline[i-1].y;
        b = prvline[i].y;
        c = prvline[i-1].y;
        curline[i].y += Paeth(a,b,c);
        a = curline[i-1].z;
        b = prvline[i].z;
        c = prvline[i-1].z;
        curline[i].z += Paeth(a,b,c);
      }
      break;
  }
}

inline unsigned char GPng::Paeth( unsigned int ua, unsigned int ub, unsigned int uc )
{
	int a, b, c, p, pa, pb, pc;

	a = (int)ua;
	b = (int)ub;
	c = (int)uc;

	p = a + b - c;
	pa = ( p>a ? p-a : a-p );
	pb = ( p>b ? p-b : b-p );
	pc = ( p>c ? p-c : c-p );

	if( pa<=pb && pa<=pc )
		return a;
	else if( pb<=pc )
		return b;
	else
		return c;
}

void GPng::getblk( GPng &blk, int x, int y, int blkw, int blkh ) const
{
  if( blk.w != blkw || blk.h != blkh )
    blk.load( blkw,blkh );

  int j;

  for( j=0; j<blkh; j++ )
    memcpy( blk.pm[j], &pm[y+j][x], blkw * sizeof(BYTE3) );
}

void GPng::flip_vertical()
{
  GPng tmp;
    tmp.load(w,h);

  int j;
  for( j=0; j<h; j++ )
    memcpy( tmp.pm[j], pm[h-j-1], w*sizeof(BYTE3) );
    memcpy( bm, tmp.bm, w*h*sizeof(BYTE3) );
}



#include "../../lib/zlib/include/zlib.h"

#ifdef _DEBUG
#pragma comment (lib, "../../lib/zlib/lib/zlibd.lib")  // Link with zlib in Debug Mode
#endif
#ifdef NDEBUG
#pragma comment (lib, "../../lib/zlib/lib/zlib.lib")  // Link with zlib in Release Mode
#endif

void GPng::compress( unsigned char *src, unsigned int n_src, unsigned char *des, unsigned int &n_des )
{
  z_stream c_stream; /* compression stream */
  int err;
  
  c_stream.zalloc = (alloc_func)0;
  c_stream.zfree = (free_func)0;
  c_stream.opaque = (voidpf)0;
  
  err = deflateInit(&c_stream, Z_DEFAULT_COMPRESSION);
  
  c_stream.next_in  = (Bytef*)src;
  c_stream.next_out = des;
  
  while( c_stream.total_in != n_src && c_stream.total_out < n_src ) 
  {
    c_stream.avail_in = c_stream.avail_out = 1;
    err = deflate(&c_stream, Z_NO_FLUSH);
  }
  for (;;) 
  {
    c_stream.avail_out = 1;
    err = deflate(&c_stream, Z_FINISH);
    if (err == Z_STREAM_END) break;
  }
  err = deflateEnd(&c_stream);
  n_des = c_stream.total_out;
}

void GPng::decompress( unsigned char *src, unsigned int n_src, unsigned char *des, unsigned int n_des )
{
  int err;
  z_stream d_stream; // decompression stream
  
  d_stream.zalloc = (alloc_func)0;
  d_stream.zfree = (free_func)0;
  d_stream.opaque = (voidpf)0;
  
  d_stream.next_in  = src;
  d_stream.avail_in = 0;
  d_stream.next_out = des;
  
  err = inflateInit(&d_stream);
  
  while (d_stream.total_out < n_des && d_stream.total_in < n_src)
  {
    d_stream.avail_in = d_stream.avail_out = 1; // force small buffers 
    err = inflate(&d_stream, Z_NO_FLUSH);
    if (err == Z_STREAM_END) break;
  }
  
  err = inflateEnd(&d_stream);
  
  if (err != Z_OK) {
    printf("[Error] GPng::decompress(), inflate error found!, %i\n", err);
    exit(-1);
  }
}


// Table of CRCs of all 8-bit messages.
unsigned int GPng::crc_table[256];

// Flag: has the table been computed? Initially false.
int GPng::crc_table_computed = 0;

// Make the table for a fast CRC.
void GPng::make_crc_table(void)
{
  unsigned int c;
  int n, k;
  
  for (n = 0; n < 256; n++) {
    c = (unsigned int) n;
    for (k = 0; k < 8; k++) {
      if (c & 1)
        c = 0xedb88320L ^ (c >> 1);
      else
        c = c >> 1;
    }
    crc_table[n] = c;
  }
  crc_table_computed = 1;
}

unsigned int GPng::update_crc(unsigned int crc, void *src_buf, int len)
{
  unsigned char *buf = (unsigned char*)src_buf;
  unsigned int c = crc;
  int n;
  
  if (!crc_table_computed)
    make_crc_table();
  for (n = 0; n < len; n++) {
    c = crc_table[(c ^ buf[n]) & 0xff] ^ (c >> 8);
  }
  return c;
}

// Return the CRC of the bytes buf[0..len-1].
unsigned int GPng::crc( void *buf, int len )
{
  return update_crc(0xffffffffL, (unsigned char*)buf, len) ^ 0xffffffffL;
}

void GPng::save( const char *spath ) const
{
  // write:: (raw data) + Filter + deflate + split + IDAT(len, id, crc) + IHDR/IEND

  int i, j;

  int filtertype;
    filtertype = 0;

  unsigned char *cdata;
  unsigned int n_cdata;  
    cdata = (unsigned char*) malloc( 2*(w*3+1) * h * sizeof(unsigned char) );
    {
      unsigned char *fdata, *rdata;
      unsigned int n_fdata;
        fdata = (unsigned char*) malloc( (w*3+1) * h * sizeof(unsigned char) );
        n_fdata = (w*3+1) * h;
        n_cdata = n_fdata;
        for( j=0; j<h; j++ )
        {
          rdata = &fdata[j*(w*3+1)];
          rdata[0] = filtertype;

          // implementation of filter type 0
          memcpy( &rdata[1], pm[j], w * sizeof(BYTE3) );
        }    

      // Deflate
      compress( fdata, n_fdata, cdata, n_cdata );
    
      free(fdata);
    }
  

  #define MAX_DAT_LEN 65521

  FILE *f0 = fopen( spath, "wb" );

    // PNG Signature
      unsigned char png_signature_info[] = { 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n' };
      fwrite( png_signature_info, sizeof(png_signature_info), 1, f0 );

    // PNG IHDR
      IHDR ihdr;
        ihdr.Width       = w;    swapbyte( &ihdr.Width,  4 );
        ihdr.Height      = h;    swapbyte( &ihdr.Height, 4 );
        ihdr.Bit         = 8;
        ihdr.Colour      = 2;
        ihdr.Compression = 0;
        ihdr.Filter      = 0;
        ihdr.Interlace   = 0;
      write_chunk( f0, 'IHDR', &ihdr, sizeof(IHDR) );

    // PNG IDAT (multiple)
    if( n_cdata )
    {
      int n_pass = ( n_cdata - 1 ) / MAX_DAT_LEN;
      unsigned char *tdat = cdata;
      for( i=0; i<n_pass; i++, tdat+=MAX_DAT_LEN )
        write_chunk( f0, 'IDAT', tdat, MAX_DAT_LEN );
        write_chunk( f0, 'IDAT', tdat, n_cdata - n_pass*MAX_DAT_LEN );
    }

    // PNG IEND
      write_chunk( f0, 'IEND', NULL, 0 );

  fclose(f0);
  free(cdata);
}



void GPng::write_chunk( FILE *f0, unsigned int chk_id, const void *chk_dat, unsigned int chk_len ) const
{
  unsigned char buf[65535];
  unsigned char *tbuf = buf;
  unsigned int chk_crc;

  memcpy( tbuf, &chk_len, 4       );  swapbyte( tbuf, 4 );  tbuf+=4;
  memcpy( tbuf, &chk_id , 4       );  swapbyte( tbuf, 4 );  tbuf+=4;
  memcpy( tbuf,  chk_dat, chk_len );                        tbuf+=chk_len;

  chk_crc = crc( &buf[4], chk_len+4 );
  memcpy( tbuf, &chk_crc , 4      );  swapbyte( tbuf, 4 );  tbuf+=4;

  fwrite( &buf, sizeof(unsigned char), chk_len+12, f0 );
}

//
/////////////////////////////////////////////////////////////////////




GJpg::GJpg()
{
  memset(this,0,sizeof(GJpg)); 
}

GJpg::~GJpg()
{
  SAFE_FREE(bm);
  SAFE_FREE(pm);
}

void GJpg::load( int width, int height )
{
  SAFE_FREE(bm);
  SAFE_FREE(pm);

  w = width;
  h = height;
  bm = (BYTE3*)malloc( w * h * sizeof(BYTE3) );
  memset( bm, 0, w * h * sizeof(BYTE3) );
  pm = (BYTE3**) malloc( h * sizeof(BYTE3*) );
  int j;
  for( j = 0; j<h; j++ )
    pm[j] = &bm[j*w];
}

void GJpg::load( int width, int height, const FLOAT3 *fm )
{
  load( width, height );

  int i,j;
  for( j=0; j<h; j++ )
    for( i=0; i<w; i++ )
    {
      bm[j*w+i].x = (GBYTE)ftoi(fm[j*w+i].x,255);
      bm[j*w+i].y = (GBYTE)ftoi(fm[j*w+i].y,255);
      bm[j*w+i].z = (GBYTE)ftoi(fm[j*w+i].z,255);
    }
}

void GJpg::load( int width, int height, const BYTE3 *prgb )
{
  load(width,height);
  memcpy( bm, prgb, w*h*sizeof(BYTE3) );
}

void GJpg::getblk( GJpg &blk, int x, int y, int blkw, int blkh ) const
{
  if( blk.w != blkw || blk.h != blkh )
    blk.load( blkw,blkh );

  int j;

  for( j=0; j<blkh; j++ )
    memcpy( blk.pm[j], &pm[y+j][x], blkw * sizeof(BYTE3) );
}



//  http://graphics.im.ntu.edu.tw/~vivace/gil/io/jpg.cpp

extern "C" {  
#include "../../lib/JpegLib/include/jpeglib.h" 
}
#ifdef _DEBUG
#pragma comment (lib, "../../lib/JpegLib/lib/JpegLibd.lib")  // Link with JpegLib in Debug Mode
#endif
#ifdef NDEBUG
#pragma comment (lib, "../../lib/JpegLib/lib/JpegLib.lib")  // Link with JpegLib in Release Mode
#endif


void GJpg::load( const char *spath )
{
	FILE *f0;
	if ((f0 = fopen(spath, "rb")) == NULL) 
  {
    printf( "[Error] : GJpg::load(), \"%s\" not found.\n", spath );
    exit(-1);
	}
	
	struct jpeg_error_mgr jerr;
	struct jpeg_decompress_struct cinfo;
  int i, j;

	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&cinfo);
	jpeg_stdio_src(&cinfo, f0);
	jpeg_read_header(&cinfo, TRUE);
	jpeg_start_decompress(&cinfo);

  if( cinfo.data_precision!=8 )
  {
    printf( "[Error] : GJpg::load(), pixel precision must be 8 bits\n" );
    exit(-1);
  }

  load( cinfo.output_width, cinfo.output_height );

  if( cinfo.output_components==3 )
  {
    for( j=0; j<h; j++ )
		  jpeg_read_scanlines( &cinfo, (unsigned char**) &pm[j], 1 );
  }
  if( cinfo.output_components==1 )
  {
    GBYTE *row;
    row = (GBYTE*) malloc( w*sizeof(GBYTE) );
    for( j=0; j<h; j++ )
    {
		  jpeg_read_scanlines( &cinfo, (unsigned char**) &row, 1 );
      for( i=0; i<w; i++ )
        pm[j][i] = row[i];
    }
    free(row);
  }

	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);
	fclose(f0);
}


void GJpg::save( const char *spath ) const
{
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  int j;

  FILE *f0 = fopen( spath, "wb" );
    cinfo.err = jpeg_std_error(&jerr);
	  jpeg_create_compress(&cinfo);
	  jpeg_stdio_dest(&cinfo, f0);
	  cinfo.image_width      = w; 	
	  cinfo.image_height     = h;
	  cinfo.input_components = 3;
    cinfo.in_color_space   = JCS_RGB;
	  jpeg_set_defaults( &cinfo );
	  jpeg_set_quality( &cinfo, 75, TRUE );
	  jpeg_start_compress(&cinfo, TRUE);
    for( j=0; j<h; j++ )
		  jpeg_write_scanlines(&cinfo, (unsigned char**) &pm[j], 1);
    jpeg_finish_compress(&cinfo);
	fclose(f0);
	jpeg_destroy_compress(&cinfo);

}






GTga::GTga()
{
  memset(this,0,sizeof(GTga)); 
}

GTga::~GTga()
{
  SAFE_FREE(bm);
  SAFE_FREE(pm);
}

void GTga::load( int width, int height )
{
  SAFE_FREE(bm);
  SAFE_FREE(pm);

  w = width;
  h = height;
  bm = (BYTE3*)malloc( w * h * sizeof(BYTE3) );
  memset( bm, 0, w * h * sizeof(BYTE3) );
  pm = (BYTE3**) malloc( h * sizeof(BYTE3*) );
  int j;
  for( j = 0; j<h; j++ )
    pm[j] = &bm[j*w];
}

void GTga::load( int width, int height, const FLOAT3 *fm )
{
  load( width, height );

  int i,j;
  for( j=0; j<h; j++ )
    for( i=0; i<w; i++ )
    {
      bm[j*w+i].x = (GBYTE)ftoi(fm[j*w+i].x,255);
      bm[j*w+i].y = (GBYTE)ftoi(fm[j*w+i].y,255);
      bm[j*w+i].z = (GBYTE)ftoi(fm[j*w+i].z,255);
    }
}

void GTga::load( int width, int height, const BYTE3 *prgb )
{
  load(width,height);
  memcpy( bm, prgb, w*h*sizeof(BYTE3) );
}

void GTga::flip_vertical()
{
  GBmp tmp;
    tmp.load(w,h);

  int j;
  for( j=0; j<h; j++ )
    memcpy( tmp.pm[j], pm[h-j-1], w*sizeof(BYTE3) );
    memcpy( bm, tmp.bm, w*h*sizeof(BYTE3) );
}

void GTga::load( const char *spath, const FLOAT4 &bgcolor )
{
	HEADER header;
	IDENTIFIELD identifield;
  memset( &header, 0, sizeof(HEADER) );
  memset( &identifield, 0, sizeof(IDENTIFIELD) );

	FILE *f0;
	if( f0 = fopen( spath, "rb" ) )
	{
		fread( &header, sizeof(HEADER), 1, f0 );
		w = header.width;
		h = header.height;
		switch( header.datatypecode )
		{
		case 0: // No image data included.
			printf( "Image contains zero image data.\n" );
			exit(0);
			break;
		case 1: // Uncompressed, color-mapped images.
			printf( "Err: load(), uncompressed color-mapped image not supported.\n" );
			exit(-1);
			break;
		case 2: // Uncompressed, RGB images.
			break;
		case 3: // Uncompressed, black and white images.
			printf( "Err: load(), uncompressed B/W image not supported.\n" );
			exit(-1);
			break;
		case 9: // Runlength encoded color-mapped images.
			printf( "Err: load(), runlength image not supported.\n" );
			exit(-1);
			break;
		case 10: // Runlength encoded RGB images.
			printf( "Err: load(), runlength image not supported.\n" );
			exit(-1);
			break;
		case 11: // Compressed, black and white images.
			printf( "Err: load(), compressed image not supported.\n" );
			exit(-1);
			break;
		case 32: // Compressed color-mapped data, using Huffman, Delta, and runlength encoding.
			printf( "Err: load(), compressed image not supported.\n" );
			exit(-1);
			break;
		case 33: // Compressed color-mapped data, using Huffman, Delta, and runlength encoding.  4-pass quadtree-type process.
			printf( "Err: load(), compressed image not supported.\n" );
			exit(-1);
			break;
		default:
			printf( "Err: load(), image type not supported.\n" );
			exit(-1);
			break;
		}
		fread( identifield.desc, sizeof(char), header.idlength, f0 );
		
		switch( header.datatypecode )
		{
		case 2:
			{
				switch( header.bitsperpixel )
				{
				case 16:
          printf( "[Error] : GTga::load(), 16bit RGB not supported.\n" );
					exit(-1);
					break;
				case 24:
					{
						int width, height;
						width = header.width;
						height = header.height;
						
						bm = (BYTE3*)malloc( width * height * sizeof(BYTE3) );
						pm = (BYTE3**)malloc( height * sizeof(BYTE3*) );
						
						int i;
						for( i = 0; i<height; i++ )
							pm[i] = bm + i * width;
						
						fread( bm, sizeof(BYTE3), width * height, f0 );
						
						char temp;
						for( i = 0; i<height*width; i++ )
						{
							temp = bm[i].x;
							bm[i].x = bm[i].z;
							bm[i].z = temp;
						}

            if( !(header.imagedescriptor & 1<<5) )
  						flip_vertical();
					}
					break;
				case 32:
					{
            printf( "[Warning] : GTga::load(), 32bit RGB not supported; instead, alpha channel is pre-multiplied to color channels.\n" );
						int width, height;
						width = header.width;
						height = header.height;
						
						bm = (BYTE3*)malloc( width * height * sizeof(BYTE3) );
						pm = (BYTE3**)malloc( height * sizeof(BYTE3*) );
						
						GBYTE *tempbuf = (GBYTE*)malloc( width * height * sizeof(GBYTE) * 4 );
						fread( tempbuf, 4 * sizeof(GBYTE), width * height, f0 );
						
            float r, g, b, a;
						int i, j;

						for( i = 0; i<height; i++ )
							pm[i] = bm + i * width;
						
						for( j=0; j<height; j++ )
							for( i=0; i<width; i++ )
							{
                a = (1-float(tempbuf[(j*width+i)*4+3])/255) * bgcolor.w;
                r = float(tempbuf[(j*width+i)*4+2])/255*(1-a) + bgcolor.x*a;
                g = float(tempbuf[(j*width+i)*4+1])/255*(1-a) + bgcolor.y*a;
                b = float(tempbuf[(j*width+i)*4+0])/255*(1-a) + bgcolor.z*a;
                pm[j][i].x = (GBYTE)ftoi(r,255);
                pm[j][i].y = (GBYTE)ftoi(g,255);
                pm[j][i].z = (GBYTE)ftoi(b,255);
							}

            if( !(header.imagedescriptor & 1<<5) )
              flip_vertical();
					}
					break;
				default:
          printf( "[Warning] : GTga::load(), unknown colour depth RGB not supported.\n" );
					exit(-1);
					break;
				}
			}
		}
	}else
	{
		printf( "Err: load(), File not found.\n" );
		exit(-1);
	}
}

void GTga::load( const char *spath )
{
  load( spath, 0 );
}

void GTga::save( const char *spath ) const
{
	// 24 bit only
	FILE *f0;
	if( f0=fopen( spath, "wb" ) )
	{
		GTga temp;
		temp.load( w, h );

	  HEADER header;
	  IDENTIFIELD identifield;
    memset( &header, 0, sizeof(HEADER) );
    memset( &identifield, 0, sizeof(IDENTIFIELD) );

    header.idlength = 0;
    header.colourmaptype = 0;
    header.datatypecode = 2;
    header.colourmaporigin = 0;
    header.colourmaplength = 0;
    header.colourmapdepth = 0;
    header.x_origin = 0;
    header.y_origin = 0;
    header.width = w;
    header.height = h;
    header.bitsperpixel = 24;
    header.imagedescriptor = 1<<5;

		int i, j;
		for( j = 0; j<h; j++ )
			for( i = 0; i<w; i++ )
			{
				temp.pm[j][i].x = pm[j][i].z;
				temp.pm[j][i].y = pm[j][i].y;
				temp.pm[j][i].z = pm[j][i].x;
			}
		fwrite( &header, sizeof(HEADER), 1, f0 );
		fwrite( &identifield, sizeof(char), header.idlength, f0 );
		fwrite( temp.bm, sizeof(BYTE3), w * h, f0 );
		fclose(f0);

	}else
	{
		printf( "[Error]: GTga::save(), cannot open file %s for write.\n", spath );
	}
}


GPim::GPim()
{
  memset( this, 0, sizeof(GPim) );
}

GPim::~GPim()
{
  SAFE_FREE( pm );
}

void GPim::load( int width, int height, int number_of_code )
{
  SAFE_FREE( pm );
  w = width;
  h = height;
  n_code = number_of_code;

  coltbl.load( n_code, 1 );
  pm = (int**) malloc2d( w, h, sizeof(int) );
  im = pm[0];
  memset( im, 0, w*h*sizeof(int) );
}

void GPim::load( int width, int height, int number_of_code, const FLOAT3 *color_table, const GStack<int> *cluster )
{
  load( width, height, number_of_code );

  coltbl.load( n_code, 1, color_table );

  int i, j, pos;
  for( j=0; j<n_code; j++ )
    for( i=0; i<cluster[j].ns; i++ )
    {
      pos = cluster[j].buf[i];
      pm[pos/w][pos%w] = j;
    }
}

void GPim::load( const char *spath )
{
  if( !fexist(spath) )
  {
    printf( "[Error] GPim::load(), file %s not found.\n", spath );
    exit(-1);
  }

  char str[256];
  FILE *f0 = fopen( spath, "rb" );
    fgets( str, 256, f0 );
    int mode = str[0]*256 + str[1];
    if( mode!='PI' )
    {
      printf( "[Error] GPim::load(), incorrect file format\n" );
      exit(-1);
    }

    fgets( str, 256, f0 );
    sscanf( str, "%i %i %i", &w, &h, &n_code );
    load( w, h, n_code );
    fgets( str, 256, f0 );

    int n_palette;
    sscanf( str, "%i", &n_palette );
    if( n_palette==-1 )
      n_palette = n_code;
    else
      coltbl.load( n_palette, 1 );

    fread( coltbl.fm, sizeof(FLOAT3), n_palette, f0 );

    //fread( im, sizeof(int), w*h, f0 );
    //int n_byte = (int)ceil( log( n_code-.5 ) / log(256) );
    //int i, j;
    //for( j=0; j<h; j++ )
    //  for( i=0; i<w; i++ )
    //    fread( &pm[j][i], n_byte, 1, f0 );

    int n_byte = (int) ceil( log( n_code-.5 ) / log(256.0) );
    unsigned int n_src, n_des;
    unsigned char *src, *des, *tdes;
    int i, j;

      fread( &n_src, sizeof(unsigned int), 1, f0 );
      src = (unsigned char*) malloc( n_src );
      fread( src, sizeof(unsigned char), n_src, f0 );

      n_des = w*h*n_byte;
      des = (unsigned char*) malloc( n_des );
      GPng::decompress( src, n_src, des, n_des );
      
      for( j=0, tdes = des; j<h; j++ )
        for( i=0; i<w; i++, tdes+=n_byte )
          memcpy( &pm[j][i], tdes, n_byte );

      free( src );
      free( des );

  fclose(f0);
}

void GPim::save( const char *spath ) const
{
  FILE *f0 = fopen( spath, "wb" );
    fprintf( f0, "PI\n" );
    fprintf( f0, "%i %i %i\n", w, h, n_code );
    fprintf( f0, "%i\n", coltbl.w );

    fwrite( coltbl.fm, sizeof(FLOAT3), coltbl.w, f0 );

    //fwrite( im, sizeof(int), w*h, f0 );
    //int n_byte = (int) ceil( log( n_code-.5 ) / log(256) );
    //int i, j;
    //for( j=0; j<h; j++ )
    //  for( i=0; i<w; i++ )
    //    fwrite( &pm[j][i], sizeof(GBYTE), n_byte, f0 );

    int n_byte = (int) ceil( log( n_code-.5 ) / log(256.0) );
    unsigned int n_src, n_des;
    unsigned char *src, *des, *tsrc;
    int i, j;

      n_src = w*h*n_byte;
      src = (unsigned char*) malloc( n_src );
      des = (unsigned char*) malloc( n_src );
      for( j=0, tsrc = src; j<h; j++ )
        for( i=0; i<w; i++, tsrc+=n_byte )
          memcpy( tsrc, &pm[j][i], n_byte );

      GPng::compress( src, n_src, des, n_des );

      fwrite( &n_des, sizeof(unsigned int), 1, f0 );
      fwrite( des, sizeof(unsigned char), n_des, f0 );

      free( src );
      free( des );

  fclose(f0);
}

void GPim::save_indices( GPf1 &indices ) const
{
  int i, j;

  indices.load( w, h );
  for( j=0; j<indices.h; j++ )
    for( i=0; i<indices.w; i++ )
      indices.pm[j][i] = im[j*w+i] / float(n_code-1);
}

void GPim::decode( GPfm &decimg ) const
{
  int i, j;
  decimg.load( w, h );
  for( j=0; j<h; j++ )
    for( i=0; i<w; i++ )
      //decimg.pm[j][i] = coltbl.fm[ pm[j][i] ];
      decimg.pm[j][i] = coltbl.lookup_linear( float(pm[j][i])/n_code*coltbl.w+.5f, .5f );
}

void GPim::load( const GPfm &img, int number_of_code, int max_cycle )
{
  GStack<int> *cluster;
  GPf1 codebook;
  GPf1 src;

    cluster = new GStack<int>[number_of_code]; 
    src.load( 3, img.w * img.h, (float*)img.fm );

    lbg_codebook_initialization( src, number_of_code, codebook );
    lbg_training( src, max_cycle, codebook, cluster );

    load( img.w, img.h, number_of_code, (FLOAT3*)codebook.fm, cluster );

  delete[] cluster;
}

void GPim::load( const GPfm &img, int number_of_code, const GPim &pim )
{

  int lidx;
  float ln;
  int i;
  ln = pim.coltbl.fm[0].norm();
  lidx = 0;
  for( i=0; i<pim.coltbl.w; i++ )
  {
    if( ln<pim.coltbl.fm[i].norm() )
    {
      ln = pim.coltbl.fm[i].norm();
      lidx = i;
    }
  }
  //printf( "lidx %i\n", lidx );



  GStack<int> src, des;
  FLOAT3 v0, v1;

  for( i=0; i<pim.coltbl.w; i++ )
    src.push( i );


  des.push( src.remove(lidx) );
  while( src.ns )
  {
    v0 = pim.coltbl.fm[ des.buf[des.ns-1] ];
    v1 = pim.coltbl.fm[ src.buf[0] ];
    ln = (v0-v1).norm();
    lidx = 0;

    for( i=0; i<src.ns; i++ )
    {
      v1 = pim.coltbl.fm[ src.buf[i] ];
      if( ln>(v0-v1).norm() )
      {
        ln = (v0-v1).norm();
        lidx = i;
      }
    }
    des.push( src.remove(lidx) );
  }

  int *rdes = (int*)malloc( des.ns*sizeof(int) );
  for( i=0; i<des.ns; i++ )
    rdes[des.buf[i]] = i;

  GPfm sub_palette;
    sub_palette.load( des.ns, 1 );
    for( i=0; i<des.ns; i++ )
      sub_palette.fm[i] = pim.coltbl.fm[ des.buf[i] ];

  load( pim.w, pim.h, number_of_code );
  coltbl.load( sub_palette.w, sub_palette.h, sub_palette.fm );


  //for( i=0; i<w*h; i++ )
  // im[i] = rdes[pim.im[i]];

  GPfm unfolded_palette;
    unfolded_palette.load( number_of_code/sub_palette.w, sub_palette.w );

  int j;

  for( j=0; j<unfolded_palette.h-1; j++ )
    for( i=0; i<unfolded_palette.w; i++ )
    {
      float s = float(i)/unfolded_palette.w;
      unfolded_palette.pm[j][i] = (1-s)*sub_palette.fm[j] + s*sub_palette.fm[j+1];
    }
  for( i=0; i<unfolded_palette.w; i++ )
    unfolded_palette.pm[j][i] = sub_palette.fm[j];

  for( i=0; i<w*h; i++ )
    im[i] = unfolded_palette.match( img.fm[i] );

  free(rdes);
}

void GPim::getidx( void *index, int bytes_per_element ) const
{
  GBYTE *p = (GBYTE*)index;

  int i, j;
  for( j=0; j<h; j++ )
    for( i=0; i<w; i++, p+=bytes_per_element )
      memcpy( p, &pm[j][i], bytes_per_element );
}

void GPim::flip_vertical()
{
  int *tline, *line0, *line1;
    tline = (int*) malloc( w * sizeof(int) );
    line0 = im;
    line1 = im + w*(h-1);

  int j, jlen = h/2;
  for( j=0; j<jlen; j++, line0+=w, line1-=w )
  {
    memcpy( tline, line0, w*sizeof(int) );
    memcpy( line0, line1, w*sizeof(int) );
    memcpy( line1, tline, w*sizeof(int) );
  }

  free(tline);
}

void GPim::lbg_codeword_update( float *codeword, int n_member, const float *vec, int vsize )
{
  int i;
  for( i=0; i<vsize; i++ )
    codeword[i] += ( vec[i] - codeword[i] ) / n_member;
}

bool GPim::lbg_codebook_converge( const GPf1 &old_codebook, const GPf1 &new_codebook )
{
  float tolerance = .000001f;

  int vsize, n_code;
    vsize   = old_codebook.w;
    n_code  = old_codebook.h;

  float *oword, *nword;

  GPf1 dword;
    dword.load( vsize, 1 );

  int j;

  float se = 0;

  for( j=0; j<n_code; j++ )
  {
    oword = old_codebook.pm[j];
    nword = new_codebook.pm[j];
    vsub( nword, oword, dword.fm, vsize );
    se += vdot( dword.fm, dword.fm, vsize );
  }

  //printf( "%f\n", se );
  if( se<=tolerance )
    return true;
  else
    return false;
}

void GPim::lbg_codebook_initialization( const GPf1 &src, int n_code, GPf1 &codebook )
{
  int n_src, vsize;
    vsize   = src.w;
    n_src   = src.h;

  int *idx, j;

  codebook.load( vsize, n_code );

  idx = (int*) malloc( n_code*sizeof(int) );
  random_index( idx, n_src, n_code );

  for( j=0; j<n_code; j++ )
  {
    //memcpy( codebook.pm[j], src.pm[idx[j]], vsize*sizeof(float) );
    vperturb( src.pm[idx[j]], codebook.pm[j], vsize, .1f );
  }

  free(idx);
}

void GPim::lbg_training( const GPf1 &src, int max_cycle, GPf1 &codebook, GStack<int> *cluster )
{
  srand( (unsigned int)time(0) );

  int n_src, vsize, n_code;
    vsize   = src.w;
    n_src   = src.h;
    n_code  = codebook.h;

  int i, j, k, l;

  GPf1 new_codebook;


  float *vec;

  int min_idx;
  float min_distortion;
  float distortion;

  int *count = (int*) malloc( n_code * sizeof(int) );


  for( k=0; k<max_cycle; k++ )
  {
    //printf( "%i\n", k );
    new_codebook.load( codebook.w, codebook.h );

    memset( count, 0, n_code*sizeof(int) );
    for( i=0; i<n_code; i++ )
      cluster[i].clear();

    for( j=0; j<n_src; j++ )
    {
      vec = src.pm[j];
      min_distortion = FLT_MAX;
      for ( i=0; i<n_code; i++ )
      {
        bool isskip = false;
        float dum;

        distortion = 0;
        for( l=0; l<vsize; l++ )
        {
          dum = vec[l] - codebook.pm[i][l];
          distortion += dum * dum;
          if( distortion>min_distortion )
          {
            isskip=true;
            break;
          }
        }
        if( !isskip )
        {
          min_distortion = distortion;
          min_idx = i;
        }
      }

      cluster[min_idx].push(j);
      count[min_idx]++;
      lbg_codeword_update( new_codebook.pm[min_idx], count[min_idx], vec, vsize );
    }

    {
      for( i=0; i<n_code; i++ )
      {
        if( count[i]==0 )
        {
          int idx;
          random_index( &idx, n_src, 1 );
          vperturb( src.pm[idx], new_codebook.pm[i], vsize, .1f );
        }
      }
    }

    if( lbg_codebook_converge( new_codebook, codebook ) )
    {
      printf( "[Notice] lbg(), codebook converged in %i iterations\n", k+1 );
      goto done;
    }

    codebook.load( new_codebook.w, new_codebook.h, new_codebook.fm );
  }

  printf( "[Warning] lbg(), codebook still not converged in %i iterations\n", max_cycle );

done:
  free( count );
}

GGif::GGif()
{
  memset( this, 0, sizeof(GGif) );
}

GGif::~GGif()
{
  SAFE_FREE(bm);
  SAFE_FREE(pm);
  SAFE_FREE( delay );
}

void GGif::load( int width, int height )
{
  load( width, height, 1 );
}

void GGif::load( int width, int height, int nframe )
{
  SAFE_FREE( bm );
  SAFE_FREE( pm );
  SAFE_FREE( delay );
  
  w = width;
  h = height*nframe;
  n = nframe;
  
  bm = (BYTE3*) malloc( w * h * sizeof(BYTE3) );
  memset( bm, 0, w*h*sizeof(BYTE3) );
  delay = (float*) malloc( n*sizeof(float) );
  memset( delay, 0, n*sizeof(float) );

  pm = (BYTE3**) malloc( h * sizeof(BYTE3*) );
  int j;
  for( j=0; j<h; j++ )
    pm[j] = &bm[j*w];
}

void GGif::load( const char *spath )
{
  load( spath, 0 );
}
void GGif::load( const char *spath, FLOAT3 col )
{
  load( spath, &col );
}

void GGif::load( const char *spath, FLOAT3 *col )
{
  gif_header_info gif_header;
  gif_control_ex gif_control;
  gif_app_ex gif_app; 
  gif_image_descriptor gif_descriptor;
  gif_comment_ex gif_com;

  GStack<BYTE3 **> stkgif_out;  //stack to store gif_out(last output data)
  GStack<float> stkdelay;

  unsigned char gif_end=1;
  BYTE3 *global_ct, *local_ct;  // global/local color table
  unsigned char *des, *psrc;
  unsigned char **gif_index;  //framedata whose size may not equate to the image;
  BYTE3 **gif_out;  //framedata whose size equate to the image;(add backgroud color)
  BYTE3 bgcolor={0};

  unsigned char appblocksize, comblocksize;
  float delaytime;
 
  int framecounter, i, j, k;  //counters

  framecounter=0;
  
  if( !fexist(spath) )
  {
    printf( "[Error] GGif::load, file %s not found.\n", spath );
    exit(-1);
  }  
  
  psrc = (unsigned char *)freadall( spath );
    
  memcpy( &gif_header, psrc, sizeof(gif_header) );  
  psrc+=sizeof(gif_header);  
  
  if( gif_header.gct & (1<<7) )// if the gct flag is 1
  {
    int gctsize = 1 << ((gif_header.gct&0x7)+1);    
    global_ct = (BYTE3 *) malloc( gctsize*sizeof(BYTE3) ); 
    memcpy( global_ct, psrc, gctsize*sizeof(BYTE3) );  psrc+=gctsize*sizeof(BYTE3);
    bgcolor = global_ct[gif_header.bgcolor];
  }
  else
  {
    global_ct = NULL;
  }

  if(col)
    bgcolor = *col*255;
  
  int disposal_method = 2;
  int last_keyframe = 0;

  do
  { 
    while( *psrc==0x21 )
    {
      switch( psrc[1] )
      {
        case 0xff:
          memcpy( &gif_app, psrc, sizeof(gif_app) ); 
          psrc+=sizeof(gif_app);
          memcpy( &appblocksize, psrc, sizeof(appblocksize) ); 
          psrc+=sizeof(unsigned char);
          psrc+=(appblocksize+1)*sizeof(unsigned char);  
        break;

        case 0xfe:
          memcpy( &gif_com, psrc, sizeof(gif_com) ); 
          psrc+=sizeof(gif_com);
          memcpy( &comblocksize, psrc, sizeof(comblocksize) ); psrc+=sizeof(unsigned char);
          psrc+=(comblocksize+1)*sizeof(unsigned char);  
        break;

        case 0xf9:
          memcpy( &gif_control, psrc, sizeof(gif_control) );  psrc+=sizeof(gif_control);
          delaytime = float(gif_control.delay_time)/100;
        break;

        case 0x01:
          printf("[Error] GGif::load, Plain Text Extension exists here. We need to revise the program!\n");
          exit(-1);
        break;
      }
    }

    if( *psrc==0x3b )
      break;

    gif_out = (BYTE3 **) malloc2d( gif_header.width, gif_header.height, sizeof(BYTE3) );
    stkdelay.push(delaytime);

    if( *psrc==0x2c )
    {
      memcpy( &gif_descriptor, psrc, sizeof(gif_descriptor) ); 
      psrc+=sizeof(gif_descriptor);
    }    

    if( gif_descriptor.localflag & (1<<7) )
    {
      int lctsize = 1 << ((gif_descriptor.localflag&0x7)+1);    
      local_ct = (BYTE3 *) malloc( lctsize*sizeof(BYTE3) ); 
      memcpy( local_ct, psrc, lctsize*sizeof(BYTE3) );  psrc+=lctsize*sizeof(BYTE3);
    }
    else
    {
      local_ct = NULL;
    }
        
    des = (unsigned char*) malloc( gif_descriptor.width*gif_descriptor.height*sizeof(unsigned char) );
    
    psrc+=DecodeImageData( psrc, gif_descriptor.min_bit, des, gif_descriptor.width, gif_descriptor.height );
    memcpy( &gif_end, psrc, sizeof(gif_end) );  
    
    gif_index = (unsigned char **)  malloc2d( gif_descriptor.width, gif_descriptor.height, sizeof(unsigned char) );
    
    if( gif_descriptor.localflag & (1<<6) )  // if the interlace flag is "1";
      InverseInterlace( des, gif_index, gif_descriptor.width, gif_descriptor.height);
    else
      memcpy( gif_index[0], des, gif_descriptor.width*gif_descriptor.height*sizeof(unsigned char) );
   
    stkgif_out.push(gif_out);

    {
      switch( disposal_method )
      {
        case 0:
        case 1:
          memcpy( gif_out[0], stkgif_out.buf[framecounter-1][0], gif_header.width*gif_header.height*sizeof(BYTE3) );
          last_keyframe = framecounter-1;
        break;
        case 3:
          memcpy( gif_out[0], stkgif_out.buf[last_keyframe][0], gif_header.width*gif_header.height*sizeof(BYTE3) );
        break;
        case 2:
          for( j=0; j<gif_header.height; j++ )
            for( i=0; i<gif_header.width; i++ )
              gif_out[j][i] = bgcolor;
        break;
        default:
          printf( "[Error] GGif::load: unknown the disposal method!\n" );
          exit(-1);
        break;
      }
    }
    
    frameresize( 
      gif_out, 
      gif_descriptor.left, gif_descriptor.top,
      gif_descriptor.width, gif_descriptor.height,
      local_ct, global_ct, gif_index, 
      gif_control.disposal_method&1, gif_control.color_index
    );

    disposal_method = (gif_control.disposal_method&28)>>2;
    
    SAFE_FREE(local_ct);
    SAFE_FREE(gif_index);
    SAFE_FREE(des);

    framecounter++;
  }while( gif_end!=0x3b );

  if( !framecounter )
  {
    printf("[Error] GGif::load, No image data found!!!\n");
    exit(-1);
  }
 
  load( gif_header.width, gif_header.height, framecounter );
  
  for( k=0; k<framecounter; k++)
  {
    delay[k] = stkdelay.buf[k];
    for( j=0; j<gif_header.height; j++ )
      for( i=0; i<gif_header.width; i++ )
        pm[k*gif_header.height+j][i] = stkgif_out.buf[k][j][i];
  }  
      
  SAFE_FREE(global_ct);
      
  for( i=0; i<framecounter; i++ )
    free(stkgif_out.buf[i]);
}

void GGif::frameresize( 
  BYTE3 **output, 
  int left, int top, int width, int height, 
  const BYTE3 *global_ct, const BYTE3 *local_ct, unsigned char **gif_index,
  bool transparent_bit, unsigned char transparent_index
){
  const BYTE3 *ct;
  int i, j;

  ct = local_ct ? local_ct : global_ct;
  if(ct==0)
  {
    printf( "[Error] GGif::frameresize(), Both local and global color table are not available, we need improvement here!\n" );
    exit(-1);
  }
  if( transparent_bit )
  {
    for( j=0; j<height; j++ )
      for( i=0; i<width; i++)
        if( transparent_index != gif_index[j][i] )
          output[j+top][i+left] = ct[gif_index[j][i]];
  }else
  {
    for( j=0; j<height; j++ )
      for( i=0; i<width; i++)     
        output[j+top][i+left] = ct[gif_index[j][i]];
  }
}

void GGif::InverseInterlace( const unsigned char *input, unsigned char **output, int width, int height )
{
  int i, j, t;
  for( j=0, t=0; j<height; j+=8 )
    for( i=0; i<width; i++, t++ )
      output[j][i] = input[t];
  for( j=4; j<height; j+=8 )
    for( i=0; i<width; i++, t++ )
      output[j][i] = input[t];
  for( j=2; j<height; j+=4 )
    for( i=0; i<width; i++, t++ )
      output[j][i] = input[t];
  for( j=1; j<height; j+=2 )
    for( i=0; i<width; i++, t++ )
      output[j][i] = input[t];
}

int GGif::DecodeImageData( const unsigned char *src, int dataSize, unsigned char *des, int width, int height )
{
  int byte_count = 0;
  int MaxStackSize = 4096;  
  int NullCode = -1;
  int pixelCount = width * height;  //get the image pixel number
  
  int codeSize = dataSize + 1;  //the minimal code size of LZW in gif
  int clearFlag = 1 << dataSize;  //clear flag of LZW;
  int endFlag = clearFlag + 1;  //endflag of LZW;
  int available = endFlag + 1;  //the first available code
  
  int code = NullCode;  //to store current code
  int old_code = NullCode;  //to store the last code
  int code_mask = (1 << codeSize) - 1;  // Maximal code value
  int bits = 0;
  int *prefix;  //to store the prefix set
  int *suffix;  //store the suffix set
  int *pixelStatck;  //store temporary data stream  
  int top = 0;
  int count = 0;  //Bytes numbers which require to be processed;  
  int i = 0;  //pixel numbers attained so far;
  
  int data = 0;  //the value of the current data
  int first = 0;  // the first Byte in a string
  int inCode = NullCode;  // existing code to be transfered to the next prefix
  unsigned char *pixels;
  const unsigned char *buffer;
  
  prefix = (int*) malloc(MaxStackSize*sizeof(int));
  suffix = (int*) malloc(MaxStackSize*sizeof(int));
  pixelStatck = (int*) malloc((MaxStackSize+1)*sizeof(int));

  pixels = des;
  while( src[byte_count] )
    byte_count += src[byte_count]+1;
  byte_count++;
  
  for( code=0; code<clearFlag; code++ )
  {
    prefix[code] = 0;  //the initial prefix is 0
    suffix[code] = (unsigned char) code;  //suffix = original datum = code
  }
  
  buffer = src;
  while( i<pixelCount )
  {
    if( top==0 )    //maximum pixels:pixelCount = width * width
    {
      if( bits<codeSize )
      {
        if( count==0 )  //if the current bit numbers are shorter than the codesize, new data need to be loaded
        {
          //if count=0, we need to read a section in the data stream to analyse if the data is over;          
          count = *(buffer++);
          if( count==0 )
          {
            break;  //cann't load new data, which means the all data have been processed;
          }
        }
        //get the current data value 
        data += (*(buffer++)) << bits;  //bits shift, because the code size is always not 8;
        bits += 8;  //a byte have been processed here, therefore bits=bits+8;
        count--;
        continue;
      }
      //if there are enough bits, do as follows:
      //get the code
      code = data & code_mask;  //get the data whose bits equate to the code size in LZW
      data >>= codeSize;  //bits shift to prepare for the next step
      bits -= codeSize;  //subtract the bits number which have been processed
      
      //the following step are processing according to the code
      if( code>available || code==endFlag ) 
      {
        break;  //when the code reaches the maximal value or the endflag
      }
      if ( code==clearFlag )  //if the code is the clear flag, re-initialize the variables and restart
      {
        codeSize = dataSize + 1;        
        code_mask = (1 << codeSize) - 1;  //re-initialize the maximal code number       
        available = clearFlag + 2;  //re-initialize the next available code        
        old_code = NullCode;  //clear the old code so as to restart
        continue;
      }
      //the following codes are in the proper range where they can be processed correctly
      if( old_code==NullCode )
      {
        //if the current code is NULL, it is the first time the code is attained
        pixelStatck[top++] = suffix[code];  //get a data stream        
        old_code = code;  //current encoding step finished,save the code to old_code        
        first = code;  //the first character is the current code
        continue;
      }
      inCode = code;
      if( code==available )
      {
        //if current code equates to the code to be generate this time
        pixelStatck[top++] = (unsigned char) first;  //The next data byte equates to the first byte of the current string
        code = old_code;  //back to the last code
      }
      
      while( code > clearFlag )  //if current code is larger than the clear flag, the encoding can achieve compression;
      {        
        pixelStatck[top++] = suffix[code];
        code = prefix[code];  //back to the last code
      }
      first = suffix[code];
      if( available>MaxStackSize )
      {
        break;  //if the current code larger than the maximal code, break
      }      
      pixelStatck[top++] = suffix[code];  //get next datum
      prefix[available] = old_code;  //set prefix to the current encoding position
      suffix[available] = first;  //set suffix to the current encoding position
      available++;
      if( available==code_mask+1 && available<MaxStackSize ) 
      {
        codeSize++;  //code size increased
        code_mask = (1 << codeSize) - 1;  //reset the maximal code number
      }
      old_code = inCode;  //restore the old code
    }
    top--;  //back to the last position
    pixels[i++] = (unsigned char) pixelStatck[top];  //get original datum
  }
  
  free(prefix);
  free(suffix);
  free(pixelStatck);
  return byte_count;
}


#define DDSD_CAPS               0x00000001     // default
#define DDSD_HEIGHT             0x00000002     // default
#define DDSD_WIDTH              0x00000004     // default
#define DDSD_PITCH              0x00000008     // For uncompressed formats
#define DDSD_PIXELFORMAT        0x00001000     // default
#define DDSD_MIPMAPCOUNT        0x00020000
#define DDSD_LINEARSIZE         0x00080000     // For compressed formats
#define DDSD_DEPTH              0x00800000     // Volume Textures
#define DDPF_RGB                0x00000040     // DDPIXELFORMAT flags, Uncompressed formats
#define DDPF_ALPHAPIXELS        0x00000001
#define DDPF_FOURCC             0x00000004     // Compressed formats 
#define DDPF_ALPHA              0x00000002
#define DDPF_COMPRESSED         0x00000080
#define DDPF_LUMINANCE          0x00020000
#define DDSCAPS_TEXTURE         0x00001000     // DDSCAPS flags, default
#define DDSCAPS_COMPLEX         0x00000008
#define DDSCAPS_MIPMAP          0x00400000
#define DDSCAPS2_VOLUME         0x00200000
#define DDSCAPS2_CUBEMAP 	      0x00000200
#define GL_COMPRESSED_RGB_S3TC_DXT1_EXT 33776
#define GL_COMPRESSED_RGBA_S3TC_DXT3_EXT 33778
#define GL_COMPRESSED_RGBA_S3TC_DXT5_EXT 33779
const int GDds::GDDS_RGB_DXT1  = GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
const int GDds::GDDS_RGBA_DXT3 = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
const int GDds::GDDS_RGBA_DXT5 = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
GDds::GDds()
{
  memset( this, 0, sizeof(GDds) );
}

GDds::~GDds()
{
  SAFE_FREE(data);
  SAFE_FREE(lm);
  SAFE_FREE(mipmapw);
  SAFE_FREE(mipmaph);
}

int GDds::getmemsize( int klevel ) const
{
  return ((mipmapw[klevel]+3)/4)*((mipmaph[klevel]+3)/4)*(compressmethod==GL_COMPRESSED_RGB_S3TC_DXT1_EXT?8:16);
}

void GDds::load( int width, int height, int fourcc, int n_mip_layer )
{
  SAFE_FREE(data);
  SAFE_FREE(lm);
  SAFE_FREE(mipmapw);
  SAFE_FREE(mipmaph);

  int wblocks, hblocks, i, *tmp, tmpw, tmph;

  w = width;
  h = height;
  compressmethod = fourcc;
  mipmaplevel = n_mip_layer;

  lm = (unsigned char **) malloc( mipmaplevel*sizeof(unsigned char *) );
  mipmapw = (int *) malloc( mipmaplevel*sizeof(int) );
  mipmaph = (int *) malloc( mipmaplevel*sizeof(int) );
  
  tmp = (int *) malloc( mipmaplevel*sizeof(int) );
  tmpw = width;
  tmph = height;
  for( i=0, datasize=0; i<mipmaplevel; i++ )
  {
    wblocks = (tmpw+3)>>2;
    hblocks = (tmph+3)>>2;
    mipmapw[i] = tmpw;
    mipmaph[i] = tmph;
    tmp[i] = datasize;
    if( fourcc==0 )
      datasize += tmpw * tmph * 4;
    else if( fourcc==GL_COMPRESSED_RGB_S3TC_DXT1_EXT )
      datasize += wblocks * hblocks * 8;
    else
      datasize += wblocks * hblocks * 16;
    tmpw /= 2;
    tmph /= 2;
  }

  data = (unsigned char *)malloc( datasize );
  memset( data, 0, datasize );
  for( i=0; i<mipmaplevel; i++ )
    lm[i] = &data[tmp[i]];

  SAFE_FREE(tmp);
}

// http://msdn.microsoft.com/en-us/library/bb943981(VS.85).aspx
void GDds::load( const char *spath )
{
  FILE *f0;
  int fsize=0;
  int dwMagic;
  int fourcc, pos;
  unsigned char dat[124];
  unsigned char datDX10[20];
  DDSURFACEDESC2 header;

  d = iscubemap = pos = 0;

  if( !fexist(spath) )
  {
    printf( "[Error] GDds::load, file %s not found.\n", spath );
    exit(-1);
  }

  f0 = fopen( spath, "rb" );

  fseek( f0, 0, SEEK_END );
  fsize = ftell(f0);
  rewind(f0);

  fread( &dwMagic, sizeof(int), 1, f0 );

  if( dwMagic != ' SDD' )
  {
    printf("[Error] GDds::load, incorrect file format (not .dds)\n" );
    fclose(f0);
    exit(-1);
  }

  fread( dat, sizeof(dat), 1, f0 );  //reading header
  header = *((DDSURFACEDESC2*)dat);


  if(!( header.dwFlags | DDSD_CAPS && header.dwFlags | DDSD_HEIGHT && header.dwFlags | DDSD_WIDTH && header.dwFlags | DDSD_PIXELFORMAT ))
  {
    printf("[Error] GDds::load, Inproper file(header error)\n" );
    fclose(f0);
    exit(-1);
  }

  if( header.dwDepth > 0 && (header.dwFlags & DDSD_DEPTH))
  { 
    d = 1;
    printf("[Error] GDds::load, Unhandled texture(3D texture)\n" );
    fclose(f0);
    exit(-1);
  }

  if (header.ddsCaps.dwCaps2 & DDSCAPS2_CUBEMAP)
  { 
    iscubemap = 1;
    printf("[Error] GDds::load, Unhandled texture(cubemap)\n" );
    fclose(f0);
    exit(-1);
  }

  if( header.ddpfPixelFormat.dwFourCC )
  {
    fourcc = header.ddpfPixelFormat.dwFourCC;
    swapbyte( &fourcc, 4 );
    
    if( fourcc=='DXT1' )
    {
      compressmethod=GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
    }else if( fourcc=='DXT3' )
    {
      compressmethod = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
    }else if( fourcc=='DXT5' )
    {
      compressmethod = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
    }else if( fourcc=='DX10' )
    {
      fread( datDX10, sizeof(datDX10), 1, f0 );   //reading header
      DDS_HEADER_DXT10 &header_dxt10 = *((DDS_HEADER_DXT10*)datDX10);
      switch (header_dxt10.dxgiFormat)
      {
        case 70:
        case 71:
        case 72:
          compressmethod = GL_COMPRESSED_RGB_S3TC_DXT1_EXT;  break;
        case 74:
        case 75:
          compressmethod = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;  break;
        case 77:
        case 78:
          compressmethod = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;  break;
        default:  break;
      }
    }
  }else
  {
    rmask=header.ddpfPixelFormat.dwRBitMask;
    gmask=header.ddpfPixelFormat.dwGBitMask;
    bmask=header.ddpfPixelFormat.dwBBitMask;
    amask=header.ddpfPixelFormat.dwABitMask;
  }

  if( header.dwFlags & DDSD_MIPMAPCOUNT )
  {
    mipmaplevel = header.dwMipMapCount;
  }
  else
  {
    mipmaplevel=1;
  }

  if( header.ddpfPixelFormat.dwFlags & DDPF_ALPHAPIXELS )
  {
    amask=header.ddpfPixelFormat.dwABitMask;
    alphaimage=1;
  }

  load( header.dwWidth, header.dwHeight, compressmethod, mipmaplevel );

  fread( data, fsize-(( fourcc== 'DX10')?148:128), 1, f0);

  fclose(f0);
}

unsigned int GDds::shiftcount(unsigned int mask)
{
  unsigned int shift;
  shift = 0;

  while( mask%2==0 )
  {
    mask>>=1;
    shift++;
  }
  return shift;
}

//Handling one block of RGB data (8 bytes)
// http://msdn.microsoft.com/en-us/library/bb694531(VS.85).aspx
void GDds::doblock( FLOAT3 *block, int compressmethod, int aimage, const void *input )
{
  FLOAT3 color[4];
  int i;
  unsigned int input1, input2, col1, col2;

  input1 = ((unsigned int*)input)[0];
  input2 = ((unsigned int*)input)[1];

  col1 = input1>>16;
  col2 = (input1<<16)>>16;
  
  color[1].x = (input1>>27 )%32 / 31.f;
  color[1].y = (input1>>21 )%64 / 63.f;
  color[1].z = (input1>>16 )%32 / 31.f;

  color[0].x = (input1>>11 )%32 / 31.f;
  color[0].y = (input1>> 5 )%64 / 63.f;
  color[0].z =  input1      %32 / 31.f;

  //RGB interpolation
  if( col1>col2 )
  {
    color[2].x = ( color[0].x + color[1].x )/2.0f;
    color[2].y = ( color[0].y + color[1].y )/2.0f;
    color[2].z = ( color[0].z + color[1].z )/2.0f;

    color[3].x = 0;   //assume black background
    color[3].y = 0;
    color[3].z = 0;
  }else
  {
    color[2].x = ( 2*color[0].x + color[1].x )/3.0f;
    color[2].y = ( 2*color[0].y + color[1].y )/3.0f;
    color[2].z = ( 2*color[0].z + color[1].z )/3.0f;

    color[3].x = ( color[0].x + 2*color[1].x )/3.0f;
    color[3].y = ( color[0].y + 2*color[1].y )/3.0f;
    color[3].z = ( color[0].z + 2*color[1].z )/3.0f;
  }

  for( i=0; i<16; i++ )
    block[i] = color[(input2>>(i*2))%4];
}

float GDds::comparecolor( FLOAT3 color1, FLOAT3 color2 )
{
  float diff;
  diff = 0.0;

  diff += (color1.x>color2.x)? color1.x-color2.x : color2.x-color1.x;
  diff += (color1.y>color2.y)? color1.y-color2.y : color2.y-color1.y;
  diff += (color1.z>color2.z)? color1.z-color2.z : color2.z-color1.z;

  return diff;
}


// Compress a 4x4 block
// http://msdn.microsoft.com/en-us/library/bb694531(VS.85).aspx
// http://developer.download.nvidia.com/compute/cuda/sdk/website/projects/dxtc/doc/cuda_dxtc.pdf
unsigned __int64 GDds::compressblock( int width, const unsigned char *tmp, int x, int y, int wsize, int hsize, int fourcc)
{
  int i, j, k, m, index[16], element, aimage;
  unsigned __int64 output;
  unsigned int vectors, vectortmp;
  
  BYTE3 clr[2];
  FLOAT3 pixel[16], mincolor, maxcolor, color[4], paxis, taxis, dmean;
  float *cr, covariance[3][3], c, d, dotpro[16];

  aimage = 0;
  cr = (float *) malloc( hsize*wsize*3*sizeof(float) );
  element = hsize*wsize;

  for( i=0; i<3; i++ ) 
    for( j=0; j<3; j++ )
      covariance[i][j] = 0;

  dmean = 0;
  for( i=0; i<hsize; i++ )
    for( j=0; j<wsize; j++ )
    {
      pixel[i*4+j].x = float (tmp[((y+i)*width+j+x)*4+2]/255.0);
      pixel[i*4+j].y = float (tmp[((y+i)*width+j+x)*4+1]/255.0);
      pixel[i*4+j].z = float (tmp[((y+i)*width+j+x)*4]/255.0);
      dmean += pixel[i*4+j]; 
    }
  dmean /= float (element);
  
  for( i=0; i<element; i++ )
  {
    cr[i*3]   = pixel[i].x - dmean.x;
    cr[i*3+1] = pixel[i].y - dmean.y;
    cr[i*3+2] = pixel[i].z - dmean.z;
  }

  // covariance matriz = CR transpose * CR, where CR is the color matrix of the 4x4 block 
  for( i=0; i<3; i++ )
    for( j=0; j<3; j++ )
      for( k=0; k<element; k++ )
        covariance[i][j] += cr[k*3+j]*cr[k*3+i];

  // use Power Method to find the principle axis
  paxis = vnormalize( FLOAT3(float(rand()),float(rand()),float(rand())) );
  for( i=0; i<8; i++ )
  {
    taxis.x = covariance[0][0]*paxis.x + covariance[0][1]*paxis.y + covariance[0][2]*paxis.z;
    taxis.y = covariance[1][0]*paxis.x + covariance[1][1]*paxis.y + covariance[1][2]*paxis.z;
    taxis.z = covariance[2][0]*paxis.x + covariance[2][1]*paxis.y + covariance[2][2]*paxis.z;
    paxis = vnormalize( taxis );
  }

  for( i=0; i<element; i++ )
    dotpro[i] = vdot( pixel[i]-dmean, paxis );

  quick_sort( dotpro, index, element );

  mincolor = dmean + paxis*dotpro[index[0]];
  maxcolor = dmean + paxis*dotpro[index[element-1]]; 

  // to ensure mincolor is less than maxcolor in 16bit representation 
  // to indicate the interpolation method

  clr[0].x = (GBYTE)ftoi(mincolor.x,31);
  clr[0].y = (GBYTE)ftoi(mincolor.y,63);
  clr[0].z = (GBYTE)ftoi(mincolor.z,31);
  clr[1].x = (GBYTE)ftoi(maxcolor.x,31);
  clr[1].y = (GBYTE)ftoi(maxcolor.y,63);
  clr[1].z = (GBYTE)ftoi(maxcolor.z,31);
  k = (int (clr[0].x)<<11)+(int (clr[0].y)<<5)+int (clr[0].z);
  m = (int (clr[1].x)<<11)+(int (clr[1].y)<<5)+int (clr[1].z);
  if( k>m )
    swap( mincolor, maxcolor );

  if( aimage && fourcc==GL_COMPRESSED_RGB_S3TC_DXT1_EXT )
  {
    color[0] = mincolor;
    color[1] = maxcolor;

    color[2].x = ( color[0].x + color[1].x )/2;
    color[2].y = ( color[0].y + color[1].y )/2;
    color[2].z = ( color[0].z + color[1].z )/2;

    color[3].x = 0;   //assume black background
    color[3].y = 0;
    color[3].z = 0;
  }else
  {
    color[1] = mincolor;
    color[0] = maxcolor;

    color[2].x = ( 2*color[0].x + color[1].x )/3;
    color[2].y = ( 2*color[0].y + color[1].y )/3;
    color[2].z = ( 2*color[0].z + color[1].z )/3;

    color[3].x = ( color[0].x + 2*color[1].x )/3;
    color[3].y = ( color[0].y + 2*color[1].y )/3;
    color[3].z = ( color[0].z + 2*color[1].z )/3;
  }

  for( i=0; i<2; i++ )
  {
    clr[i].x = (GBYTE)ftoi(color[i].x,31);
    clr[i].y = (GBYTE)ftoi(color[i].y,63);
    clr[i].z = (GBYTE)ftoi(color[i].z,31);
  }

  vectors = 0;
  vectortmp = 0;
  for( i=0; i<16; i++ )
  {
    c = 1000;
    for( j=0; j<4; j++ )
    {
      d = comparecolor(color[j], pixel[i]);
      if( d < c )
      {
        c = d;
        vectortmp = j;
      }
    }
    vectors |= (vectortmp%4)<<(i*2);
  }

  output = 0;
  for( i=0; i<2; i++ )
  {
    output |= unsigned __int64 (clr[i].x%32)<<(i*16+11);
    output |= unsigned __int64 (clr[i].y%64)<<(i*16+5);
    output |= unsigned __int64 (clr[i].z%32)<<(i*16);
  }

  if( clr[0].x!=clr[1].x || clr[0].y!=clr[1].y || clr[0].z!=clr[1].z )
    output |= (unsigned __int64) vectors<<32;

  SAFE_FREE( cr );
  return output;
}

// Decode a dds format data into RGB values
void GDds::decode( GPfm &pfm ) const
{
  int totalwidth, x;

  totalwidth = 0;

  if( mipmaplevel>0 )
  {
    for( x=0; x<mipmaplevel; x++ )
      totalwidth += mipmapw[x];
  }
  else
    totalwidth = w;

  pfm.load( totalwidth, h );

  if( compressmethod==0 )   //uncompressed
  {
    int i, j, k, i0, pos;
    unsigned int input;
    unsigned int rshift, gshift, bshift, ashift, rmax, gmax, bmax, amax;
    unsigned int alpha;

    i0 = 0;
    pos = 0;

    rshift = shiftcount(rmask);
    gshift = shiftcount(gmask);
    bshift = shiftcount(bmask);

    rmax = rmask>>rshift;
    gmax = gmask>>gshift;
    bmax = bmask>>bshift;

    if( alphaimage )
    {
      ashift = shiftcount(amask);
      amax = amask>>ashift;
      alpha = amax;
    }else
    {
      amax=255;
      alpha = amax;
    }


    for( k=0; k<mipmaplevel; k++ )
    {
      for( j=0; j<mipmaph[k]; j++ )
        for( i=0; i<mipmapw[k]; i++ )
        {
          memcpy( &input, &data[pos], sizeof(int));
          pos += sizeof(int);

          if( alphaimage )
            alpha = ( input & amask ) >> ashift;
        
          pfm.pm[j][i+i0].x = (( input & rmask ) >> rshift)/float(rmax) * alpha/float(amax);
          pfm.pm[j][i+i0].y = (( input & gmask ) >> gshift)/float(gmax) * alpha/float(amax);
          pfm.pm[j][i+i0].z = (( input & bmask ) >> bshift)/float(bmax) * alpha/float(amax);
        }

      i0 += mipmapw[k];
    }
  }
  else if( compressmethod==GL_COMPRESSED_RGB_S3TC_DXT1_EXT )  //compressed with DXT1
  {
    int wblocks, hblocks, wsize, hsize, x, z, i, j, k, i0;
    unsigned int *pt;
    GPfm blk;

    pt = (unsigned int*)data;
    i0 = 0;
    blk.load( 4, 4 );

    for( k=0; k<mipmaplevel; k++ )
    {
      wblocks = (mipmapw[k]+3)/4;
      hblocks = (mipmaph[k]+3)/4;

      for( j=0; j<hblocks; j++ )
        for( i=0; i<wblocks; i++, pt+=2 )
        {
          hsize=g_min( 4, mipmaph[k]-j*4 );
          wsize=g_min( 4, mipmapw[k]-i*4 );

          doblock( blk.fm, 1, alphaimage, pt );

          for( x=0; x<hsize; x++ )
            for( z=0; z<wsize; z++ )
              if( x<hsize && z<wsize )    // padding handling
                pfm.pm[j*4+x][i*4+i0+z] = blk.pm[x][z];
        }

      i0 += mipmapw[k];
    }
  }
  else if ( compressmethod==GL_COMPRESSED_RGBA_S3TC_DXT3_EXT )   //compressed with DXT3
  {
    int wblocks, hblocks, wsize, hsize, x, z, i, j, k, i0;
    unsigned __int64 *pt;
    GPfm blk;

    i0 = 0;
    pt = (unsigned __int64*)data;


    for( k=0; k<mipmaplevel; k++ )
    {
      wblocks = (mipmapw[k]+3)/4;
      hblocks = (mipmaph[k]+3)/4;

      blk.load( 4, 4 );

      for( j=0; j<hblocks; j++ )
        for( i=0; i<wblocks; i++, pt+=2)
        {
          hsize=g_min( 4, mipmaph[k]-j*4 );
          wsize=g_min( 4, mipmapw[k]-i*4 );
        
          doblock( blk.fm, 3, alphaimage, pt+1 );
        
          for( x=0 ; x<4; x++)
            for( z=0 ; z<4; z++)
            {
              if( x<hsize && z<wsize )    // padding handling

                pfm.pm[j*4+x][i*4+z+i0] = blk.pm[x][z] * float(
                (signed __int64)((pt[0]>>((x*4+z)*4))%16)
                )  /15.0f;  //explicit alpha
              //alpha multiplication, assume black background
            }
        }
      i0 += mipmapw[k];
    }
  }
  else if ( compressmethod==GL_COMPRESSED_RGBA_S3TC_DXT5_EXT )   //compressed with DXT5
  {
    int wblocks, hblocks, wsize, hsize, x, z, i, j, k, i0;
    unsigned __int64 *pt;
    GPfm blk;
    float alpha[8];

    i0 = 0;
    
    pt = (unsigned __int64*)data;


    for( k=0; k<mipmaplevel; k++ )
    {
      wblocks = (mipmapw[k]+3)/4;
      hblocks = (mipmaph[k]+3)/4;

      blk.load( 4, 4 );

      for( j=0; j<hblocks; j++ )
        for( i=0; i<wblocks; i++, pt+=2)
        {
          hsize=g_min( 4, mipmaph[k]-j*4 );
          wsize=g_min( 4, mipmapw[k]-i*4 );

          alpha[0] = float((signed __int64)(pt[0]%256));
          alpha[1] = float((signed __int64)((pt[0]>>8)%256));

          if( alpha[0] > alpha[1] )   //alpha interpolation
          {
            for( x=2; x<8; x++ )
              alpha[x] = ((8-x)*alpha[0] + (x-1)*alpha[1])/7.0f;
          }else
          {
            for( x=2; x<6; x++ )
              alpha[x] = ((6-x)*alpha[0] + (x-1)*alpha[1])/5.0f;
            alpha[6] = 0;
            alpha[7] = 255;
          }
        
          doblock( blk.fm, 5, alphaimage, pt+1 );
        
          for( x=0 ; x<4; x++)
            for( z=0 ; z<4; z++)
              if( x<hsize && z<wsize )    // handle padding
                pfm.pm[j*4+x][i*4+z+i0] =(blk.pm[x][z]*alpha[(pt[0] >> ((x*4+z)*3+16))%8]/255.0f);  //interpolated alpha
              //alpha multiplication, assume black background
        }
      i0 += mipmapw[k];
    }
  }
}


void GDds::downsample( const unsigned char *src, int w0, int h0, unsigned char *des )
{
  int i, j, s, w1, h1;
  const unsigned char *s0, *s1, *s2, *s3;
  unsigned char *dd;

  for( j=0, dd=des, w1=w0/2, h1=h0/2; j<h1; j++ )
  {
    s0 = &src[j*2*w0*4];
    s1 = s0+4;
    s2 = s0+w0*4;
    s3 = s0+w0*4+4;
    for( i=0; i<w1; i++, s0+=8, s1+=8, s2+=8, s3+=8 )
    for( s=0; s<4; s++, dd++ )
      *dd = unsigned char( (int(s0[s]) + s1[s] + s2[s] + s3[s])/4 );
  }
}




// Loading RGB data into dds format
// Uncompressed data and DXT1 compression are supported
void GDds::load( int width, int height, const FLOAT3 *fm, int fourcc, int genmipmap )
{
  if( genmipmap )
    load( width, height, fourcc, mipmaplevelcount(width, height) );
  else
    load( width, height, fourcc, 1 );

  GDds thislevel, lastlevel;
  GBYTE *tm;
  
  int i, j, k, wblocks, hblocks;
  int m;
  
  if( fourcc==0 )  //without compression
  {    
    for( i=0, tm=data; i<w*h; i++, tm+=4 )
    {
      tm[3] = 255;
      tm[2] = (GBYTE)ftoi(fm[i].x,255);
      tm[1] = (GBYTE)ftoi(fm[i].y,255);
      tm[0] = (GBYTE)ftoi(fm[i].z,255);
    }
    // mipmap generation
    for( k=1; k<mipmaplevel; k++ )
      downsample( lm[k-1], mipmapw[k-1], mipmaph[k-1], lm[k] );
  }else if( fourcc==GL_COMPRESSED_RGB_S3TC_DXT1_EXT )
  {
    unsigned __int64 blockoutput;
    int hsize, wsize;
    unsigned char *tmp1=0, *tmp2=0;
           
    tmp1 = (unsigned char *) malloc( w*h*4 );
    tmp2 = (unsigned char *) malloc( w*h*4 );
    for( i=0, tm=tmp1; i<w*h; i++, tm+=4 )
    {
      tm[3] = 255;
      tm[2] = (GBYTE)ftoi(fm[i].x,255);
      tm[1] = (GBYTE)ftoi(fm[i].y,255);
      tm[0] = (GBYTE)ftoi(fm[i].z,255);
    }
    
    // mipmap generation
    for( k=0; k<mipmaplevel; k++ )
    { 
      if( k>0 )
      {
        downsample( tmp1, mipmapw[k-1], mipmaph[k-1], tmp2 );
        swap( tmp1, tmp2 );
      }

      wblocks = (mipmapw[k]+3)/4;
      hblocks = (mipmaph[k]+3)/4;
      for( j=0; j<hblocks; j++)
      {
        for( i=0; i<wblocks; i++)
        {
          hsize=g_min( 4, mipmaph[k]-j*4 );
          wsize=g_min( 4, mipmapw[k]-i*4 );
          blockoutput = compressblock(mipmapw[k], tmp1, i*4, j*4, wsize, hsize, fourcc );
          for( m=0; m<8; m++ )
            lm[k][(j*wblocks+i)*8+m] = unsigned char( (blockoutput >> (m*8))%256 );
        }
      }
    }

    SAFE_FREE(tmp1);
    SAFE_FREE(tmp2);
  }else
  {
    printf( "[Error] GDds::load, Unsupported compresstion method.\n" );
    exit(-1);
  }
}

int GDds::mipmaplevelcount( int width, int height )
{
  int tmp, level;
  level = 1;
  tmp = (width>height)? width:height;
  while( tmp>1 )
  {
    tmp /= 2;
    level++;
  }
  return level;
}

// http://msdn.microsoft.com/en-us/library/bb943981(VS.85).aspx
void GDds::save( const char *spath ) const
{
  DDSURFACEDESC2 header;
  FILE *f0;

  memset( &header, 0, sizeof(DDSURFACEDESC2));

  header.dwSize = 124;
  header.dwFlags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT;

  if( mipmaplevel>1 )
    header.dwFlags |= DDSD_MIPMAPCOUNT;

  if( compressmethod==0 )
  {
    header.dwFlags |= DDSD_PITCH;
    header.dwPitchOrLinearSize = w*4;
    header.ddpfPixelFormat.dwFlags |= DDPF_RGB;
  }
  else
  {
    header.dwFlags |= DDSD_LINEARSIZE;
    if( compressmethod==GL_COMPRESSED_RGB_S3TC_DXT1_EXT )
      header.dwPitchOrLinearSize = ((w+3)/4) * ((h+3)/4) * 8;
    else
      header.dwPitchOrLinearSize = ((w+3)/4) * ((h+3)/4) * 16;
  }
  if( d )
    header.dwFlags |= DDSD_DEPTH;

  header.dwHeight = h;
  header.dwWidth = w;
  header.dwDepth = d;

  header.dwMipMapCount = (mipmaplevel>1)?mipmaplevel:0;

  header.ddpfPixelFormat.dwSize = 32;
  if( alphaimage==1 )
    header.ddpfPixelFormat.dwFlags |= DDPF_ALPHAPIXELS;

  if( compressmethod==0 )
  {
    header.ddpfPixelFormat.dwFourCC = 0;
    header.ddpfPixelFormat.dwRGBBitCount = 32;
    header.ddpfPixelFormat.dwFlags |= DDPF_ALPHAPIXELS; //saving with alpha data is not implemented

    //only supporting A8R8G8B8
    header.ddpfPixelFormat.dwABitMask = 0xFF000000;
    header.ddpfPixelFormat.dwRBitMask = 0x00FF0000;
    header.ddpfPixelFormat.dwGBitMask = 0x0000FF00;
    header.ddpfPixelFormat.dwBBitMask = 0x000000FF;
  }
  else 
  {
    header.ddpfPixelFormat.dwFlags |= DDPF_FOURCC;
    if( compressmethod==GL_COMPRESSED_RGB_S3TC_DXT1_EXT )
      header.ddpfPixelFormat.dwFourCC = '1TXD';
    else if( compressmethod==GL_COMPRESSED_RGBA_S3TC_DXT3_EXT )
      header.ddpfPixelFormat.dwFourCC = '3TXD';
    else
      header.ddpfPixelFormat.dwFourCC = '5TXD';
    header.ddpfPixelFormat.dwRGBBitCount = 0;
    //header.ddpfPixelFormat.dwFlags |= DDPF_ALPHAPIXELS; //saving with alpha data is not implemented

    //only supporting A8R8G8B8
    header.ddpfPixelFormat.dwABitMask = 0;
    header.ddpfPixelFormat.dwRBitMask = 0;
    header.ddpfPixelFormat.dwGBitMask = 0;
    header.ddpfPixelFormat.dwBBitMask = 0;
  }

  header.ddsCaps.dwCaps1 = 0;   //no mipmap, no cube map , no volume texture (optional)
  header.ddsCaps.dwCaps2 = 0;   //no cubemap, no volume texture (optional)

  if( f0=fopen( spath, "wb" ) )
  { 
    fprintf( f0, "DDS " );
    fwrite( &header, sizeof(DDSURFACEDESC2), 1, f0 );

    fwrite( data, datasize, 1, f0);

    fclose( f0 );
  }else
  {
		printf( "[Error]: GDds::save(), cannot open file %s for write.\n", spath );
	}
} 