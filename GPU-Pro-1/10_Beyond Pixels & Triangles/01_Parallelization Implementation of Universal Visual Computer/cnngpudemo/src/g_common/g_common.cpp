#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>
#include <time.h>


#include "g_common.h"



double nCr(int n, int r)
{
  double ans = 1;
  int il = n-r;
  for(int i=1, j=r+1; i<=il; i++, j++)
    ans *= ((double)j)/i;
  return ans;
}


// mR is taken to be OpenGL version of 4x4 Matrix
// that is assuming row vector.
//
// Singularity at attitude = +- PI/2
//
// heading : [-pi  , pi  ]
// attitude: [-pi/2, pi/2]
// bank    : [-pi  , pi  ]
//
// http://martinb.com/maths/geometry/rotations/euler/index.htm
//
void Matrix2Euler( const float *mR, float *heading, float *attitude, float *bank )
{
  if( mR[1] > 0.998 )   // singularity at north pole
  {
    *heading = atan2f( mR[8], mR[10]);
    *attitude = G_PI/2;
    *bank = 0;
  }else if( mR[1] < -0.998 )   // singularity at south pole
  {
    *heading = atan2f( mR[8], mR[10]);
    *attitude = -G_PI/2;
    *bank = 0;
  }else
  {
    *heading = atan2f( -mR[2], mR[0] );
    *bank = atan2f( -mR[9], mR[5] );
    *attitude = asinf( mR[1] );
 }
}

// mR is taken to be OpenGL version of 4x4 Matrix
// that is assuming row vector.
void Euler2Matrix( float *mR, float heading, float attitude, float bank )
{
  float ch = cosf(heading);    
  float sh = sinf(heading);    
  float ca = cosf(attitude);    
  float sa = sinf(attitude);    
  float cb = cosf(bank);    
  float sb = sinf(bank);

  mR[0] = ch * ca;
  mR[4] = sh*sb - ch*sa*cb;
  mR[8] = ch*sa*sb + sh*cb;
  mR[12] = 0;

  mR[1] = sa;
  mR[5] = ca*cb;
  mR[9] = -ca*sb;
  mR[13] = 0;

  mR[2] = -sh*ca;
  mR[6] = sh*sa*cb + ch*sb;
  mR[10] = -sh*sa*sb + ch*cb;
  mR[14] = 0;

  mR[3] = 0;
  mR[7] = 0;
  mR[11] = 0;
  mR[15] = 1;
}

 void AxisAngle2Matrix( float *m, float angle, float x, float y, float z  )
 {
   float c, s, t;
   
   c = cosf( angle );
   s = sinf( angle );
   t = 1 - c;
   
   memset( m, 0, 16*sizeof(float) );
   
   m[0]  = c + x*x*t;
   m[1]  = x*y*t + z*s;
   m[2]  = x*z*t - y*s;
   
   m[4]  = x*y*t - z*s;
   m[5]  = c + y*y*t;
   m[6]  = y*z*t + x*s;
   
   m[8]  = x*z*t + y*s;
   m[9]  = y*z*t - x*s;
   m[10] = c + z*z*t;
   
   m[15] = 1;

   // float3 axis;
   // float angle;
   // axis = float3( ay.z-az.y, az.x-ax.z, ax.y-ay.x )/2;
   // angle = asin( clamp( length(axis),0,1) )/G_PI;
   // if( ax.x+ay.y+az.z-1 < 0 )
   //   angle = 1-angle;
}

GPath parse_spath( const char *spath )
{
  GPath gp;
    parse_path( spath, gp.dname, gp.fname, gp.ename );
    return gp;
}

GPath* parse_path( const char *_spath, char *dname, char *fname, char *ename)
{
  static GPath g_path;

  if( dname==NULL )
  {
    dname = g_path.dname;
    fname = g_path.fname;
    ename = g_path.ename;
  }

  char spath[256];
  strcpy( spath, _spath );
  int slen = strlen( spath );
  int i;
  for( i=0; i<slen; i++ )
    if( spath[i] == '\\' ) 
      spath[i]='/';


  const char *t_fname = strrchr( spath, '/' );

  int dlen, flen;
  if( t_fname )
  {
    dlen = strlen(spath) - strlen(t_fname)+1;
    t_fname++;
  }else
  {
    dlen = 0;
    t_fname = spath;
  }
  memcpy( dname, spath, dlen*sizeof(char) );
  dname[dlen] = '\0';

  if( slen == dlen )
  {
    fname[0] = '\0';
    ename[0] = '\0';
    return &g_path;
  }

  const char *t_ename = strrchr( t_fname, '.' );

  if( t_ename )
  {
    flen = strlen(t_fname) - strlen(t_ename);
    t_ename++;

    if( strcmp( t_ename, "" ) == 0 )
    {
      printf( "[Error] : parse path error, invalid path \"%s\"  .\n", spath );
      exit(-1);
    }
  }else
  {
    flen = strlen(t_fname);
    t_ename = "";
  }

  strcpy( ename, t_ename );

  memcpy( fname, t_fname, flen*sizeof(char) );
  fname[flen] = '\0';

  if( strcmp( fname, "" ) == 0 )
  {
    printf( "[Error] : parse path error, invalid path \"%s\"  .\n", spath );
    exit(-1);
  }

  //if( strcmp(dname, "" )==0 )
  //  strcpy( dname, "./");

  return &g_path;
}

void gen_zmap( int *zmap, int n_dim )
{
  int *t_zmap = zmap;

  int s_zrow, t_zrow;
  int x=0, y=0;    //(x,y)=(0,0)
  int tx=1, ty=-1; //(tx,ty)=(0,0)

  for( s_zrow=0; s_zrow<n_dim; s_zrow++ )
  {
    for( t_zrow=0; t_zrow<s_zrow; t_zrow++, x+=tx, y+=ty )
      *t_zmap++ = y*n_dim + x;
      *t_zmap++ = y*n_dim + x;

    (s_zrow%2)? y+=1 : x+=1;
    tx=-tx; ty=-ty;
  }

  for( s_zrow=n_dim-2, x+=tx, y+=ty; s_zrow>=0; s_zrow-- )
  {
    for( t_zrow=0; t_zrow<s_zrow; t_zrow++, x+=tx, y+=ty )
      *t_zmap++ = y*n_dim + x;
      *t_zmap++ = y*n_dim + x;

    (s_zrow%2)? x+=1 : y+=1;
    tx=-tx; ty=-ty;
  }

}

float g_psnr( float mse, float peak )
{
  return 10*logf( peak*peak/mse )/logf(10);
}

float g_fps( void (*func)(void), int n_frame )
{
  clock_t start, finish;
  int i;
  float fps;

  printf( "Performing benchmark, please wait" );
    start = clock();
    for( i=0; i<n_frame; i++ )
    {
      if( (i+1)%(n_frame/10)==0 )
        printf(".");

      func();

    }
    printf( "done\n" );
    finish = clock();

  fps = float(n_frame)/(finish-start)*CLOCKS_PER_SEC;
  return fps;
}


float Secant( float x0, float x1, float (*f)(float) )
{
  if( x0==x1 )
    return x1;
  
  float x2 = x1 - f(x1)*(x1-x0)/(f(x1)-f(x0));
  return Secant( x1, x2, f );
}

float Bisect( float x0, float x1, float (*f)(float) )
{
  float y, x;

  int s0, s1;
  s0 = G_SIGN(f(x0));
  s1 = G_SIGN(f(x1));
  if( s0 == s1 )
  {
    printf( "[Error] Bisect, both end points are of the same sign %f %f.\n", x0, x1 );
    exit(-1);
  }

  while( fabs(x0-x1)>.00001 )
  {
    x = (x0+x1)/2;
    y = f(x);

    if( s0 == G_SIGN(y) )
      x0 = x;
    else
      x1 = x;
  }

  return x;
}

void swapbyte( void *src, int n )
{
  int i,j,n2;
  unsigned char a;
  unsigned char *s0;
  s0 = (unsigned char*)src;

  n2 = n/2;
  for( i=0, j=n-1; i<n2; i++, j-- )
  {
    a = s0[i];
    s0[i] = s0[j];
    s0[j] = a;
  }
}

bool fexist( const char *spath )
{
  FILE *f0 = fopen( spath, "rb" );
  if( f0==NULL )
    return false;
  fclose(f0);
  return true;
}

void* freadall( const char *spath )
{
  if( !fexist(spath) )
  {
    printf( "[Error] : freadall(), \"%s\" not found.\n", spath );
    exit(-1);
  }

  unsigned char *dat;
  int pend;

  FILE *f0 = fopen( spath, "rb" );
    fseek( f0, 0, SEEK_END );
    pend = ftell(f0);
    fseek( f0, 0, SEEK_SET );
    dat = (unsigned char*) malloc( (pend+1)*sizeof(unsigned char) );
    fread( dat, sizeof(unsigned char), pend, f0 );
    dat[pend]=0;
  fclose(f0);

  return dat;
}


void bubble_sort( const float *val, int *idx, int n, bool ascending )
{
  int i,j;

  for( i=0; i<n; i++ )
    idx[i] = i;

  for( i=0; i<n-1; i++ )
    for( j=0; j<n-1-i; j++ )
      if( val[idx[j+1]] < val[idx[j]] )
        swap( idx[j], idx[j+1] );

  if( !ascending )
    for( i=0; i<n/2; i++ )
      swap( idx[i], idx[n-1-i] );
}

void rank_sort( const float *val, int *idx, int n, bool ascending )
{
  int *rank;
    rank = (int*) malloc( n * sizeof(int) );
    memset(rank, 0, n * sizeof(int) );

  int i, j;

  for( j=0; j<n; j++ )
    for( i=0; i<n; i++ )
      rank[j] += ( val[j] > val[i] );

  for( i=0; i<n; i++ )
    idx[rank[i]] = i;

  if( !ascending )
    for( i=0; i<n/2; i++ )
      swap( idx[i], idx[n-1-i] );

  free(rank);
}


void partial_quicksort( const float *val, int *idxm, int n, int m, bool ascending )
{
  int i, *idx;

  idx = (int*) malloc( n*sizeof(int) );

  for( i=0; i<n; i++ )
    idx[i] = i;

  {
    GStack<int> gs;
    int low, high, l, h;
    float sv;

    gs.push( 0, n-1 );
    while( gs.ns )
    {
      gs.pop( high, low );
      sv = val[idx[high]];
      for( l=low, h=low; h<high; h++ )
        if( (val[idx[h]] <= sv) == ascending )
          swap( idx[l++], idx[h] );
        swap( idx[l], idx[h] );

      if( low < l-1 )
        gs.push( low, l-1  );
      if( l+1 < high && l+1 < m )
        gs.push( l+1, high );
    }
  }

  for( i=0; i<m; i++ )
    idxm[i] = idx[i];

  free(idx);
}

void* malloc2d( int w, int h, int size )
{
  int j;
  void **a = (void**) malloc( h*sizeof(void*) + w*h*size );
  for( j=0; j<h; j++ )
    a[j] = ((char *)(a+h)) + j*w*size;
  return a;

  //int j;
  //void **a = (void**) malloc( h*sizeof(void*) );
  //for( j=0; j<h; j++ )
  //  a[j] = malloc( w*size );
  //return a;
}

void* malloc3d( int w, int h, int d, int size )
{
  int j, k;
  void ***a = (void***) malloc( d*sizeof(void**) + h*d*sizeof(void*) + w*h*d*size );
  for( k=0; k<d; k++ )
    a[k] = ((void**)(a+d)) + k*h;
  for( k=0; k<d; k++ )
    for( j=0; j<h; j++ )
      a[k][j] = ((char*)(a+d+h*d)) + (k*h+j)*w*size;
  return a;
}

GPermutation::GPermutation(){ memset( this, 0, sizeof(GPermutation) ); } 
GPermutation::~GPermutation(){ SAFE_FREE(buf); SAFE_FREE(p); } 
void GPermutation::load( int *dat, int n_dat ) 
{ 
  SAFE_FREE(buf); 
  SAFE_FREE(p); 
  n = n_dat; 
  buf = (int*) malloc( n * sizeof(int) ); 
  p = (int*) malloc( n * sizeof(int) ); 
  
  for( i=0; i<n; i++ ) 
  { 
    buf[i] = dat[i]; 
    p[i] = n-i; 
  } 
  i = n-1; 
} 

bool GPermutation::next() 
{ 
  if( i>0 ) 
  { 
    p[i]--;  i--;  j = n-1; 
    do{  swap(buf[i], buf[j]);  j--;  i++;  }while( j>i ); 
    i = n-1; 
    while( !p[i] ){  p[i] = n - i;  i--;  } 
    return true; 
  }else 
    return false; 
} 

char* replace_char( char *str, char a, char b )
{
  int i, l;
  for( i=0, l=strlen( str ); i<l; i++ )
    if( str[i] == a ) 
      str[i] = b;
  return str;
}

bool getbit( unsigned char c, int i )
{
  // i==[0..7], return bit 0 to bit 7

  if ( i>7 || i<0 )
  {
    printf( "[Error] getbit(): i=%i\n", i );
    exit(-1);
  }

  return (c>>i&1)>0;
}

void setbit( unsigned char &c, int i, bool value )
{

// i==[0..7], return bit 0 to bit 7
  if ( i>7 || i<0 )
  {
    printf( "[Error] setbit(): i=%i\n", i );
    exit(-1);
  }
  c = c - (c & 1<<i);
  if( value==true )
    c += 1<<i;
}

void random_index( int *idx, int n )
{
  int i, j, *tmp, n_tmp, pos;

  tmp = (int*) malloc( n*sizeof(int) );

  n_tmp = n;
  for( i=0; i<n; i++ )
    tmp[i] = i;

  for( j=0; j<n; j++ )
  {
    pos = ( rand()*(RAND_MAX+1)+rand() ) % n_tmp;
    idx[j] = tmp[pos];

    n_tmp = n_tmp-1;
    for( i=pos; i<n_tmp; i++ )
      tmp[i] = tmp[i+1];
  }

  free(tmp);
}

void random_index( int *idx, int n, int m )
{
  int i, j, *tmp, n_tmp, pos;

  tmp = (int*) malloc( n*sizeof(int) );

  n_tmp = n;
  for( i=0; i<n; i++ )
    tmp[i] = i;

  for( j=0; j<m; j++ )
  {
    pos = ( rand()*(RAND_MAX+1)+rand() ) % n_tmp;
    idx[j] = tmp[pos];

    n_tmp = n_tmp-1;
    for( i=pos; i<n_tmp; i++ )
      tmp[i] = tmp[i+1];
  }

  free(tmp);
}

bool g_ferr( const float &f )
{
  if( (*((unsigned int*)&f) & 0x7f800000) == 0x7f800000 )
    return true;
  else
    return false;
}

int ftoi( float x, int xmax )
{
  return g_clamp( g_round(x*xmax), 0, xmax );
}


/*
class GLink
{
  public:

    GLink( GLink *prev_link, GLink *next_link, void *value );
    GLink *prev;
    GLink *next;
    void *val;
};

template <typename T>
class GLinkLst
{
  public:

    GLinkLst(){ memset(this,0,sizeof(GLinkLst)); }
    ~GLinkLst(){ clear(); }

    void clear()
    {
      GLink *curr, *prev;
      curr = head;
      while( curr )
      {
        prev = curr;
        curr = curr->next;
        delete prev;
      }
      memset(this,0,sizeof(GLinkLst));
    }

    void add( T &val )
    {
      if( !head )
      {
        head = new GLink( NULL, NULL, (void*)&val );
        tail = head;
      }else
      {
        tail->next = new GLink( tail, NULL, (void*)&val );
        tail = tail->next;
      }
      ns++;
    }

    void remove( GLink *&lnk )
    {
      if( lnk==NULL )
      {
        printf( "[Error] GLinkLst::remove(), cannont remove NULL link.\n" );
        exit(-1);
      }

      if( head==NULL )
      {
        printf( "[Error] GLinkLst::remove(), nothing to be removed.\n" );
        exit(-1);
      }

      if( lnk==head )
        head = lnk->next;
      if( lnk==tail )
        tail = lnk->prev;
      if( lnk->prev )
        lnk->prev->next = lnk->next;
      if( lnk->next )
        lnk->next->prev = lnk->prev;

      ns--;

      GLink* next = lnk->next;
      delete lnk;
      lnk = next;
    }

    void insert( GLink *lnk, T &val )
    {
      if( lnk==head )
        head = new GLink( NULL, head, (void*)&val );
      else
      {
        GLink* tmp = new GLink( lnk->prev, lnk, (void*)&val );
        lnk->prev->next = tmp;
        lnk->prev = tmp;
      }
      ns++;
    }

    T& eval( const GLink *lnk )
    {
      return *((T*)lnk->val);
    }

    T& remove( int idx )
    {
      int i;
      GLink *curr;
      for( i=0, curr=head; i<idx; i++, curr=curr->next );
      T &tmp = eval(curr);
      remove(curr);
      return tmp;
    }


    GLink *head;
    GLink *tail;
    int ns;
};
GLink::GLink( GLink *prev_link, GLink *next_link, void *value )
{ 
  prev = prev_link;
  next = next_link;
  val  = value;
}

*/
