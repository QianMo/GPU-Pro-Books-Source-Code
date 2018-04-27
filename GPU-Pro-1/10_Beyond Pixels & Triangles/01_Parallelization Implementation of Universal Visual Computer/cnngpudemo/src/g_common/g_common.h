#ifndef G_COMMON_H
#define G_COMMON_H

#include <stdio.h>
#include <memory.h>

#define	G_PI 3.14159265358979323846f

#define SAFE_FREE(p)  { if(p){ free(p);  (p)=NULL;} }
#define SAFE_FCLOSE(p) { if(p){ fclose(p);  (p)=NULL;} }
#define SAFE_DELETE(p){ if(p){ delete (p); (p)=NULL;} }
#define SAFE_DELETE_ARRAY(p){ if(p){ delete[] (p); (p)=NULL;} }
#define ROUND(f) ((int)( (f)>0 ? (f)+.5 : (f)-.5 ))
#define G_CLAMP(x,a,b)( ((x)<(a))? (a) : (((x)<(b))?(x):(b))  )
#define G_MAX(a,b)            (((a) > (b)) ? (a) : (b))
#define G_MIN(a,b)            (((a) < (b)) ? (a) : (b))
#define G_SIGN(a) ( (a)>0?1:-1 )
#define G_PROCESS_KILL(p) (TerminateProcess( GetCurrentProcess(), p ))


typedef struct _GPath
{
  char dname[256];
  char fname[256];
  char ename[256];
}GPath;

GPath parse_spath( const char *spath );

double nCr(int n, int r);
void Euler2Matrix( float *mR, float heading, float attitude, float bank );
void Matrix2Euler( const float *mR, float *heading, float *attitude, float *bank );
void AxisAngle2Matrix( float *m, float angle, float x, float y, float z  );
GPath* parse_path( const char *spath, char *dname=NULL, char *fname=NULL, char *ename=NULL);
void gen_zmap( int *zmap, int n_dim );

float g_psnr( float mse, float peak );
float g_fps( void (*func)(void), int n_frame=100 );
float Secant( float x0, float x1, float (*f)(float) );
float Bisect( float x0, float x1, float (*f)(float) );
void swapbyte( void *src, int n );

void bubble_sort( const float *val, int *idx, int n, bool ascending = true );
void rank_sort(   const float *val, int *idx, int n, bool ascending = true );
void partial_quicksort( const float *val, int *idx, int n, int m, bool ascending = true );

bool fexist( const char *spath );
void* freadall( const char *spath );
bool g_ferr( const float &f );
int ftoi( float x, int xmax );

void random_index( int *idx, int n );
void random_index( int *idx, int n, int m );

void* malloc2d( int w, int h, int size );
void* malloc3d( int w, int h, int d, int size );
char* replace_char( char *str, char a, char b );

bool getbit( unsigned char c, int i );
void setbit( unsigned char &c, int i, bool value );


template <typename T> inline void swap( T &a, T &b ){ T c;  c = a;  a = b;  b = c; }
template <typename T> T g_clamp( const T &x, const T &a, const T &b){  return x<a ? a : (x<b?x:b);  }
template <typename T> T g_lerp( const T &a, const T &b, float s )
{
  s = g_clamp(s,0.f,1.f);
  return (1-s)*a+s*b;  
}
template <typename T> T g_max( const T &a, const T &b ){ return a>b?a:b; }
template <typename T> T g_min( const T &a, const T &b ){ return a<b?a:b; }
template <typename T> T g_step( const T &a, const T &b ){ return a<=b; }
template <typename T> int g_round( const T &f ){ return int(f>0 ? f+.5 : f-.5); }


template <typename T> 
class GStack
{
  public:
    int ns;
    T *buf;

    GStack(){ load(); }
    GStack( int blk_size ){ load(blk_size); }
    GStack( const GStack &a ){ load(); *this=a; }

    void push( const T &a, const T &b )
    {
      push(a);
      push(b);
    }
    void push( const T a )
    {
        T *newblk;
      if( ns+1 > n_buf )
      {
        n_buf += m_buf;

        //buf = (T*) realloc( buf, n_buf * sizeof(T) );
        newblk = new T[n_buf];

        if( newblk==0 )
        {
          int xxx = 1;
        }
        int i;
        for( i=0; i<ns; i++ )
          newblk[i] = buf[i];
        SAFE_DELETE_ARRAY(buf);
        buf=newblk;
      }
      buf[ns]=a;
      ns++;
    }
    void pop( T &a, T &b ){ a=buf[--ns]; b=buf[--ns]; }
    void pop( T &a ){ a=buf[--ns]; }
    void reset(){ ns=0; } 
    void clear(){ SAFE_DELETE_ARRAY(buf);  n_buf=0; ns=0; } 
    ~GStack(){ clear(); }

    T remove( int idx )
    {
      T a = buf[idx];
      ns--;
      int i;
      for( i=idx; i<ns; i++ )
        buf[i] = buf[i+1];
      return a;
    }

    void insert( int idx, const T &a )
    {
      push(a);
      int i;
      for( i=ns-1; i>idx; i-- )
        buf[i] = buf[i-1];
      buf[idx] = a;
    }

    GStack& operator=( const GStack &a )
    {
      if( this==&a )
        return *this;
      clear();
      load();
      int i;
      for( i=0; i<a.ns; i++ )
        push( a.buf[i] );
      return *this;
    }

  private:
    int n_buf, m_buf;
    void load( int blk_size ){ memset(this,0,sizeof(GStack)); m_buf=blk_size; }
    void load(){ load(256); }
};

template <typename T> 
void quick_sort( const T *val, int *idx, int n, bool ascending=true )
{
  if( n<=0 )
    return;

  GStack<int> gs;
  int low, high, l, h;
  T sv;

  for( l=0; l<n; l++ )
    idx[l] = l;

  gs.push( 0, n-1 );
  while( gs.ns )
  {
    gs.pop( high, low );
    sv = val[idx[high]];

    if(ascending)
    {
      for( l=low, h=low; h<high; h++ )
        if( val[idx[h]]<sv || (val[idx[h]]==sv && idx[h]<idx[high]) )
          swap( idx[l++], idx[h] );
    }else
    {
      for( l=low, h=low; h<high; h++ )
        if( val[idx[h]]>sv || (val[idx[h]]==sv && idx[h]<idx[high]) )
          swap( idx[l++], idx[h] );
    }
      swap( idx[l], idx[h] );
      
    if( low < l-1 )
      gs.push( low, l-1  );
    if( l+1 < high )
      gs.push( l+1, high );
  }
}

template <typename T> 
void quick_sort( T *val, int n, bool ascending=true )
{
  T *sval;
  int *idx;
  int i;

  sval = (T*) malloc( n*sizeof(T) );
  idx  = (int*) malloc( n*sizeof(int) );
  quick_sort( val, idx, n, ascending );
  for( i=0; i<n; i++ )
    sval[i] = val[idx[i]];
  memcpy( val, sval, n*sizeof(T) );

  free(sval);
  free(idx);
}

template <typename T> 
class GMemParser
{
  public:
    int *dim, ns;
 
    GMemParser(){ memset(this,0,sizeof(GMemParser) ); }
    ~GMemParser(){ if(dim) free(dim); }
 
    // contiguous dimension first and end with 0 { w, h, d, ..., 0 }
    void ride( void *dat, ... )
    {
      int i, *n;
      if(dim) free(dim);
      fm=(T*)dat;
      n=(int*)(&dat+1);
      for( i=0, ns=0; n[i]>0; i++ )
        ns++;
      dim = (int*) malloc( ns*sizeof(int) );
      memcpy( dim, n, ns*sizeof(int) );
    }
 
    // contiguous dimension first { i, j, k, ... }
    T &lm( int x, ... )
    {
      int *n = &x;
      int i, idx;
      idx = n[ns-1];
      for( i=ns-2; i>=0; i-- )
        idx = idx*dim[i]+n[i];
      return *(fm+idx);
    }
 
  private:
    T *fm;
};

class GPermutation 
{ 
  public: 
     int *buf, n; 

     GPermutation(); 
     ~GPermutation(); 
     void load( int *dat, int n_dat ); 
     bool next(); 

  private: 
     int i, j, *p; 
}; 

#endif
