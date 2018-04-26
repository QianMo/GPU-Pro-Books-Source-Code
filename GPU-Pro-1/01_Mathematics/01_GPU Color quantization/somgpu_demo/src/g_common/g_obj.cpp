// Alan Au, 19-10-2005, aualan@hotmail.com
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <windows.h>
#include <GL/gl.h>

#include "g_common.h"
#include "g_obj.h"


GObj::GObj(){  memset( this, 0, sizeof(GObj) );  }
GObj::~GObj()
{  
  clear();
}

void GObj::clear()
{
  if( tex0 )
    glDeleteTextures( 1, &tex0 );
  if( material )
    for( int i=0; i<n_material; i++ )
      if( material[i].texid )
        glDeleteTextures( 1, &material[i].texid );

  vertex.clear();
  normal.clear();
  texc.clear();

  tangent.clear();
  binormal.clear();

  SAFE_FREE( face_nidx );
  SAFE_FREE( face_vidx );
  SAFE_FREE( face_tidx );
  SAFE_FREE( face_eidx );
  SAFE_FREE( material );
  SAFE_DELETE_ARRAY( mgroup );
  SAFE_DELETE_ARRAY( sgroup );
  SAFE_FREE( s_edge );

  default_group.clear();

  memset( this, 0, sizeof(GObj) );
}

void GObj::reversewinding()
{
  int i, tmp;
  
  // reverse winding of faces/triangles
  for( i=0; i<n_face; i++ )
  {
    if( face_vidx )
    {
      tmp         = face_vidx[i*3+0];
      face_vidx[i*3+0] = face_vidx[i*3+2];
      face_vidx[i*3+2] = tmp;
    }
    
    if( face_tidx )
    {
      tmp = face_tidx[i*3+0];
      face_tidx[i*3+0] = face_tidx[i*3+2];
      face_tidx[i*3+2] = tmp;
    }

    if( face_nidx )
    {
      tmp = face_nidx[i*3+0];
      face_nidx[i*3+0] = face_nidx[i*3+2];
      face_nidx[i*3+2] = tmp;
    }
  }
}

void GObj::scale( float scale )
{
  int i;
  for( i=0; i<n_vertex; i++ )
    vertex.fm[i] *= scale;
}

void GObj::rotate_euler( float rx, float ry, float rz ) // Rxyz
{
  rx = rx * G_PI / 180;
  ry = ry * G_PI / 180;
  rz = rz * G_PI / 180;

  float mx[16] = 
  {
    1,       0,        0, 0,
    0, cosf(rx), -sinf(rx), 0, 
    0, sinf(rx),  cosf(rx), 0, 
    0, 0, 0, 1
  };

  float my[16] = 
  {
     cosf(ry), 0,  sinf(ry), 0,
          0,  1,        0, 0,
    -sinf(ry), 0,  cosf(ry), 0,
    0, 0, 0, 1
  };

  float mz[16] = 
  {
    cosf(rz), -sinf(rz), 0, 0,
    sinf(rz),  cosf(rz), 0, 0,
          0,        0, 1, 0,
    0, 0, 0, 1
  };

  int i;
  for( i=0; i<n_vertex; i++ )
  {
    vertex.fm[i] = vertex.fm[i].rmul( mx );
    vertex.fm[i] = vertex.fm[i].rmul( my );
    vertex.fm[i] = vertex.fm[i].rmul( mz );
  }

  for( i=0; i<n_normal; i++ )
  {
    normal.fm[i] = normal.fm[i].rmul( mx );
    normal.fm[i] = normal.fm[i].rmul( my );
    normal.fm[i] = normal.fm[i].rmul( mz );
  }
}

void GObj::rotate( GQuat q )
{
  float m[16];
  q.matrix(m);

  int i;

  for( i=0; i<vertex.w; i++ )
    vertex.fm[i] = vertex.fm[i].rmul(m);
  for( i=0; i<normal.w; i++ )
    normal.fm[i] = normal.fm[i].rmul(m);
  for( i=0; i<tangent.w; i++ )
    tangent.fm[i] = tangent.fm[i].rmul(m);
  for( i=0; i<binormal.w; i++ )
    binormal.fm[i] = binormal.fm[i].rmul(m);
}

void GObj::rotate( float angle, float x, float y, float z )
{
  rotate( GQuat( FLOAT3(x, y, z), -angle*G_PI/180 ) );
}

void GObj::translate( float x, float y, float z )
{
  int i;
  FLOAT3 a( x, y, z );
  for( i=0; i<n_vertex; i++ )
    vertex.fm[i] += a;
}

FLOAT4 GObj::unitize()
{
  float scale;
  FLOAT3 m, n, d;
    m = vertex.vmax();
    n = vertex.vmin();
    d = m - n;
    scale = 2 / G_MAX( d.x, G_MAX( d.y, d.z ));

  FLOAT3 center;
    center = (m+n)/2;

  vertex.add( -center );
  vertex.mul( scale );
  return FLOAT4( -center.x, -center.y, -center.z, scale );
}

void GObj::load_vertex( int number_of_vertex )
{
  n_vertex = number_of_vertex;
  vertex.load( n_vertex, 1 );
}

void GObj::load_normal( int number_of_normal )
{
  n_normal = number_of_normal;
  normal.load( n_normal, 1 );

  tangent.clear();
  binormal.clear();
}

void GObj::load_texc( int number_of_texc )
{
  n_texc = number_of_texc;
  texc.load( n_texc, 1 );
}

void GObj::load_mgroup( int number_of_group )
{
  SAFE_DELETE_ARRAY( mgroup );
  n_mgroup = number_of_group;

  if( n_mgroup )
  {
    mgroup = new Group[n_mgroup];
  }
}

void GObj::load_sgroup( int number_of_group )
{
  SAFE_DELETE_ARRAY( sgroup );
  n_sgroup = number_of_group;

  if( n_sgroup )
  {
    sgroup = new Group[n_sgroup];
  }
}

void GObj::load_face( int number_of_face )
{
  SAFE_FREE( face_nidx );
  SAFE_FREE( face_vidx );
  SAFE_FREE( face_tidx );

  n_face = number_of_face;
  face_nidx = (int*) malloc( n_face * 3 * sizeof(int) );
  face_vidx = (int*) malloc( n_face * 3 * sizeof(int) );
  face_tidx = (int*) malloc( n_face * 3 * sizeof(int) );

  memset( face_nidx, 0, n_face * 3 * sizeof(int) );
  memset( face_vidx, 0, n_face * 3 * sizeof(int) );
  memset( face_tidx, 0, n_face * 3 * sizeof(int) );
}

void GObj::loadx( const char *spath )
{
  if( !fexist(spath) )
  {
    printf( "[Error]: GObj::loadx(), File %s not found.\n", spath );
    exit(-1);
  }
  
  int number_of_vertex=0;
  int maxid_of_vertex=0;
  int number_of_face=0;
  {
    int idx;
    char str[256];
    char tag0[256];

    FILE *f0 = fopen( spath, "rb" );
    while( fgets( str, 256, f0 ) )
    {
      tag0[0] = 0;
      sscanf( str, "%s %i", tag0, &idx );
      if( strcmp( "Vertex", tag0 )==0 )
      {
        if( maxid_of_vertex<idx )
          maxid_of_vertex=idx;
        number_of_vertex++;
      }
      if( strcmp( "Face", tag0 )==0 )
        number_of_face++;
    }
    fclose(f0);
  }
  

  int vidx, vid;
  float vx, vy, vz;
  float nx, ny, nz;
  int fidx, fid;
  int v0, v1, v2;
  int *vidmap;
  char str[256];
  char tag0[256];

  load_face( number_of_face );
  load_normal( number_of_vertex );
  load_vertex( number_of_vertex );
  vidmap = (int*) malloc( (maxid_of_vertex+1) * sizeof(int) );
  memset( vidmap, 0, (maxid_of_vertex+1) * sizeof(int) );
  fidx = 0;
  vidx = 0;

  FILE *f0 = fopen( spath, "rb" );
  while( fgets( str, 256, f0 ) )
  {
    if( sscanf( str, "%s ", tag0 ) )
    {
      if( strcmp( "Vertex", tag0 )==0 )
      {
        //Vertex 4949  129.363 222.93 687.266 {normal=(-0.826753 -0.560178 -0.0517436)}
        sscanf( str, "Vertex %i %f %f %f {normal=(%f %f %f)}", &vid,  &vx, &vy, &vz,   &nx, &ny, &nz );
        vidmap[vid] = vidx;
        vertex.fm[vidx] = FLOAT3( vx, vy, -vz );
        normal.fm[vidx] = FLOAT3( nx, ny, -nz );
        vidx++;
      }
    
      if( strcmp( "Face", tag0 )==0 )
      {
        //Face 1  2165 111 526
        sscanf( str, "Face %i %i %i %i", &fid, &v0, &v1, &v2 );
        fid--;
  
        face_vidx[ 3*fidx+0 ] = vidmap[v0];
        face_vidx[ 3*fidx+1 ] = vidmap[v1];
        face_vidx[ 3*fidx+2 ] = vidmap[v2];
        face_nidx[ 3*fidx+0 ] = vidmap[v0];
        face_nidx[ 3*fidx+1 ] = vidmap[v1];
        face_nidx[ 3*fidx+2 ] = vidmap[v2];

        default_group.add( fidx );
        fidx++;
      }
    }
  }
  default_group.type = GOBJ_V_N;
  fclose(f0);

  reversewinding();
  printf_info();
  strcpy( obj_path, spath );
  SAFE_FREE(vidmap);
}

void GObj::save_mtl( const char *spath ) const
{
  int i;
  FILE *f0 = fopen( spath, "wt" );
  for( i=0; i<n_material; i++ )
  {
    fprintf( f0, "newmtl %s\n", material[i].name );
    fprintf( f0, "Ka %f %f %f\n", material[i].ambient.x, material[i].ambient.y, material[i].ambient.z );
    fprintf( f0, "Kd %f %f %f\n", material[i].diffuse.x, material[i].diffuse.y, material[i].diffuse.z );
    if( material[i].illum != 1 )
    {
      fprintf( f0, "Ks %f %f %f\n", material[i].specular.x, material[i].specular.y, material[i].specular.z );
      fprintf( f0, "illum %i\n", material[i].illum );
      fprintf( f0, "Ns %.0f\n", material[i].shininess * 1000 / 128 );
    }
    if( strcmp( material[i].texname, "" )!=0 )
      fprintf( f0, "map_Kd %s\n", material[i].texname );
    fprintf( f0, "\n" );
  }
  fclose(f0);
}

void GObj::save( const char *spath ) const
{
  if( material )
  {
    GPath gp = parse_spath(spath);
    char tpath[256];
    sprintf( tpath, "%s%s.mtl", gp.dname, gp.fname );
    save_mtl( tpath );
  }

  const Group *grp;
  int ngrp;
  int i, j;

  if( mgroup )
  {
    grp = mgroup;
    ngrp = n_mgroup;
  }else if( sgroup )
  {
    grp = sgroup;
    ngrp = n_sgroup;
  }else
  {
    grp = &default_group;
    ngrp = 1;
  }


  FILE* f0 = fopen( spath, "wt" );

    for( i=0; i<n_vertex; i++ )
      fprintf( f0, "v %f %f %f\n", vertex.fm[i].x, vertex.fm[i].y, vertex.fm[i].z );

    for( i=0; i<n_texc; i++ )
      fprintf( f0, "vt %f %f %f\n", texc.fm[i].x, texc.fm[i].y, texc.fm[i].z );

    for( i=0; i<n_normal; i++ )
      fprintf( f0, "vn %f %f %f\n", normal.fm[i].x, normal.fm[i].y, normal.fm[i].z );

  int *grpi, grpi0;
  const Group *g;

  grpi = (int*) malloc( n_face*sizeof(int) );
  for( i=0; i<n_face; i++ )
    grpi[i] = -1;
  for( j=0; j<ngrp; j++ )
    for( i=0; i<grp[j].n_face; i++ )
      grpi[ grp[j].tri[i] ] = j;
  grpi0 = -1;

  for( i=0; i<n_face; i++ )
  {
    if( grpi[i]<0 )
      continue;

    if( grpi0!=grpi[i] )
    {
      grpi0=grpi[i];
      if( mgroup )
        fprintf( f0, "usemtl %s\n", grp[grpi0].name );
      else if( sgroup )
        fprintf( f0, "g %s\n", grp[grpi0].name );
      g = &grp[grpi0];
    }

    switch( g->type )
    {
      case GOBJ_V__:
          fprintf( f0, "f %i %i %i\n", 
            face_vidx[ i*3+0 ]+1, face_vidx[ i*3+1 ]+1, face_vidx[ i*3+2 ]+1 );
      break;
      case GOBJ_VT_:
          fprintf( f0, "f %i/%i %i/%i %i/%i\n", 
            face_vidx[ i*3+0 ]+1, face_tidx[ i*3+0 ]+1,
            face_vidx[ i*3+1 ]+1, face_tidx[ i*3+1 ]+1,
            face_vidx[ i*3+2 ]+1, face_tidx[ i*3+2 ]+1
          );
      break;
      case GOBJ_V_N:
          fprintf( f0, "f %i//%i %i//%i %i//%i\n", 
            face_vidx[ i*3+0 ]+1, face_nidx[ i*3+0 ]+1,
            face_vidx[ i*3+1 ]+1, face_nidx[ i*3+1 ]+1,
            face_vidx[ i*3+2 ]+1, face_nidx[ i*3+2 ]+1
          );
      break;
      case GOBJ_VTN:
          fprintf( f0, "f %i/%i/%i %i/%i/%i %i/%i/%i\n", 
            face_vidx[ i*3+0 ]+1, face_tidx[ i*3+0 ]+1, face_nidx[ i*3+0 ]+1,
            face_vidx[ i*3+1 ]+1, face_tidx[ i*3+1 ]+1, face_nidx[ i*3+1 ]+1,
            face_vidx[ i*3+2 ]+1, face_tidx[ i*3+2 ]+1, face_nidx[ i*3+2 ]+1
          );
      break;
    }
  }

  free(grpi);

  fclose( f0 );
}

void GObj::load( const char *spath )
{
  clear();

  char *str, tag[256], *str0;
  int type;


  GPath *gp = parse_path( spath );
  if( strcmp( gp->ename, "m" )==NULL )
  {
    loadx( spath );
    return;
  }

  int m_buf;
  {
    unsigned char c;
    int n_buf;
    FILE* f0 = fopen( spath, "rb" );
    if( f0==NULL )
    {
      printf( "[Error]: GObj::load(), File %s not found.\n", spath );
      exit(-1);
    }

    n_buf = 0;
    m_buf = 0;
    while( fread( &c, sizeof(char), 1, f0 ) )
    {
      if( c == '\n' )
      {
        n_buf++;
        if( n_buf > m_buf )
          m_buf = n_buf;
        n_buf = 0;
      }else
      {
        n_buf++;
      }
    }
    fclose(f0);
    m_buf = m_buf+2;
  }
  str = (char*) malloc( m_buf * sizeof(char) );


  int number_of_vertex = 0;
  int number_of_face   = 0;
  int number_of_texc   = 0;
  int number_of_normal = 0;
  {
    FILE* f0 = fopen( spath, "rb" );

      str[m_buf-2] = 0;

      int mgrp_n_face=0;
      int sgrp_n_face=0;
      while( fgets( str, m_buf, f0 ) )
      {
        if( str[m_buf-2]!=0 )
        {
          printf( "[Error] GObj::load(), line buffer overrun.\n" );
          exit(-1);
        }

        str0 = str;
        while( *str0 == ' ' || *str0=='\t' )
          str0++;

        type = str0[0]*256 + str0[1];
        switch( type )
        {
          case 'v ':
            number_of_vertex++;
            break;
          case 'vn':
            number_of_normal++;
            break;
          case 'vt':
            number_of_texc++;
            break;
          case 'mt':
            {
              GPath *gp = parse_path( spath );
              if( sscanf( str0, "mtllib %s", tag ) )
              {
                sprintf( mtl_path, "%s%s", gp->dname, tag );
                load_mtl( mtl_path );
              }
            }
            break;
          case 'f ':
            number_of_face++;
            mgrp_n_face++;
            sgrp_n_face++;

            char *token;
            token = strtok( str0, " \r\n" );
            token = strtok( NULL, " \r\n" );
            token = strtok( NULL, " \r\n" );
            token = strtok( NULL, " \r\n" );
            while( ( token = strtok( NULL, " \r\n" ) ) != NULL )
              number_of_face++;
            break;
        }
      }
    fclose( f0 );
  }



  load_face( number_of_face );
  load_vertex( number_of_vertex );
  load_normal( number_of_normal );
  load_texc( number_of_texc );



  GStack<Group*> msg, ssg;
  Group *mg, *sg;
  mg = new Group();
  sg = new Group();
  msg.push( mg );
  ssg.push( sg );


  FLOAT3 *v = vertex.fm;
  FLOAT3 *n = normal.fm;
  FLOAT3 *vt = texc.fm;

  int V[65536], N[65536], T[65536];
  int vi, ni, ti;

  int face_pos;
    face_pos = 0;

  FILE *f0 = fopen( spath, "rb" );
    while( fgets( str, m_buf, f0 ) )
    {

      str0 = str;
      while( *str0 == ' ' || *str0=='\t' )
        str0++;

      type = str0[0]*256 + str0[1];
 
      switch( type )
      {
        case 'v ':
          sscanf( str0, "v %f %f %f", &v->x, &v->y, &v->z );
          v++;
          break;
        case 'vn':
          sscanf( str0, "vn %f %f %f", &n->x, &n->y, &n->z );
          n++;
          break;

        case 'vt':
          sscanf( str0, "vt %f %f %f", &vt->x, &vt->y, &vt->z );
          vt++;
          break;

        case 'us':
          sscanf( str0, "usemtl %s", tag );
          if( mg->n_face==0 )
          {
            msg.pop(mg);
            SAFE_DELETE(mg);
          }

          {
            int i;
            for( i=0; i<msg.ns; i++ )
            {
              if( strcmp( msg.buf[i]->name, tag )==0 )
              {
                mg=msg.buf[i];
                break;
              }
            }
            if( i==msg.ns )
            {
              mg = new Group();
              strcpy( mg->name, tag );
              msg.push( mg );
            }
          }
          //mg->idx = face_pos;
          break;

        case 'g ':
          sscanf( str0, "g %s", tag );
          if( sg->n_face==0 )
          {
            ssg.pop(sg);
            SAFE_DELETE(sg);
          }

          {
            int i;
            for( i=0; i<ssg.ns; i++ )
            {
              if( strcmp( ssg.buf[i]->name, tag )==0 )
              {
                sg=ssg.buf[i];
                break;
              }
            }
            if( i==ssg.ns )
            {
              sg = new Group();
              strcpy( sg->name, tag );
              ssg.push( sg );
            }
          }
          //sg->idx = face_pos;
          break;

        case 'f ':
        {
          int i, l;
          int face_type;
          
         if( sscanf( str0, "f %i//%i", &vi, &ni )==2 )
            face_type = GOBJ_V_N;
          else if( sscanf( str0, "f %i/%i/%i", &vi, &ti, &ni )==3 )
            face_type = GOBJ_VTN;
          else if( sscanf( str0, "f %i/%i", &vi, &ti )==2 )
            face_type = GOBJ_VT_;
          else if( sscanf( str0, "f %i", &vi )==1 )
            face_type = GOBJ_V__;
          else
            face_type = -1;

          l = 0;
          char *token;
          token = strtok( str0, " \r\n" );
          token = strtok( NULL, " \r\n" );
          while( token != NULL )
          {
            switch( face_type )
            {
              case GOBJ_V__:
                sscanf( token, "%i", &V[l] );  V[l]--;
                break;
              case GOBJ_VT_:
                sscanf( token, "%i/%i", &V[l], &T[l] );  V[l]--; T[l]--;
                break;
              case GOBJ_V_N:
                sscanf( token, "%i//%i", &V[l], &N[l] );  V[l]--; N[l]--;
                break;
              case GOBJ_VTN:
                sscanf( token, "%i/%i/%i", &V[l], &T[l], &N[l] );  V[l]--; T[l]--; N[l]--;
                break;
              default:
                printf( "[Error] GObj::load(), obj file syntax error\n" );
                exit(-1);
                break;
            }
            token = strtok( NULL, " \r\n" );
            l++;

            if( l>=65536 )
            {
              printf( "[Error] GObj::load(), index buffer overrun.\n" );
              exit(-1);
            }
          }
          l = l-2;


          int *fv, *ft, *fn, *vv, *tt, *nn;
            fv = &face_vidx[3*face_pos]; vv = V+1;
            ft = &face_tidx[3*face_pos]; tt = T+1;
            fn = &face_nidx[3*face_pos]; nn = N+1;
          for( i=0; i<l; i++, face_pos++, fv+=3, ft+=3, fn+=3, vv++, tt++, nn++ )
          {
            fv[0] = V[0];   ft[0] = T[0];   fn[0] = N[0];
            fv[1] = vv[0];  ft[1] = tt[0];  fn[1] = nn[0];
            fv[2] = vv[1];  ft[2] = tt[1];  fn[2] = nn[1];
            mg->add( face_pos );
            sg->add( face_pos );
            default_group.add( face_pos );
          }

            mg->type = mg->type ? mg->type & face_type : face_type;
            sg->type = sg->type ? sg->type & face_type : face_type;
            default_group.type = default_group.type ? default_group.type & face_type : face_type;
        }
        break;
      }
    }
  fclose( f0 );


  if( mg->n_face==0 )
  {
    msg.pop(mg);
    SAFE_DELETE(mg);
  }
  if( sg->n_face==0 )
  {
    ssg.pop(sg);
    SAFE_DELETE(sg);
  }

  if( msg.ns )
  {
    if( msg.ns==1 && msg.buf[0]->name[0]==0 )
    {
      msg.pop(mg);
      SAFE_DELETE(mg);
    }
    int i;
    load_mgroup( msg.ns );
    for( i=0; i<msg.ns; i++ )
    {
      memcpy( &mgroup[i], msg.buf[i], sizeof(Group) );
      msg.buf[i]->tri=0;
      delete msg.buf[i];
    }
  }

  if( ssg.ns )
  {
    if( ssg.ns==1 && ssg.buf[0]->name[0]==0 )
    {
      ssg.pop(sg);
      SAFE_DELETE(sg);
    }
    int i;
    load_sgroup( ssg.ns );
    for( i=0; i<ssg.ns; i++ )
    {
      memcpy( &sgroup[i], ssg.buf[i], sizeof(Group) );
      ssg.buf[i]->tri=0;
      delete ssg.buf[i];
    }
  }

  if( n_material==0 )
  {
    GPath *gp = parse_path( spath );
    sprintf( mtl_path, "%s%s.mtl", gp->dname, gp->fname );
    load_mtl( mtl_path );
  }

  if( n_texc )
  {
    GPath *gp = parse_path( spath );
    GPfm pfm;
    char ext[][4] = { "pfm", "ppm", "bmp", "png", "tga", "dds", "gif", "jpg" };
    int i;

    for( i=0; i<sizeof(ext)/sizeof(ext[0]); i++ )
    {
      sprintf( tex_path, "%s%s.%s", gp->dname, gp->fname, ext[i] );
      if( fexist( tex_path ) )
      {
        pfm.load( tex_path );
        pfm.resample( 512, 512 );
        //pfm.flip_vertical();
        break;
      }
    }

    if( pfm.w==0 || pfm.h==0 )
      pfm.load( 64, 64, 1,1,1 );

    glGenTextures( 1, &tex0 );
    glBindTexture( GL_TEXTURE_2D, tex0 );
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, pfm.w, pfm.h, 0, GL_RGB, GL_FLOAT, pfm.fm );
  }

  strcpy( obj_path, spath );
  //printf_info();

  free( str );
}

/*
void GObj::load( const char *spath )
{
  clear();
  GPath *gp = parse_path( spath );
  if( strcmp( gp->ename, "m" )==NULL )
  {
    loadx( spath );
    return;
  }


  char *dat;
  GStack<char*> ldat;
  FLOAT3 *v, *n, *vt;
  GStack<int> vu, nu, tu;

  char *str0;
  char *token;
  int type;
  int face_type;
  int face_pos;
  int i, j, l;
  int  vi,  ti,  ni;
  int *fv, *ft, *fn;
  int *vv, *tt, *nn;

  {
    if( !fexist(spath) )
    {
      printf( "[Error]: GObj::load(), file %s not found.\n", spath );
      exit(-1);
    }

    int nsize;
    {
      FILE* f0 = fopen( spath, "rb" );
        fseek( f0, 0, SEEK_END );
        nsize = ftell( f0 );
      fclose(f0);
    }

    dat = (char*) malloc( (nsize+1)*sizeof(char) );
    {
      FILE* f0 = fopen( spath, "rb" );
        fread( dat, sizeof(char), nsize, f0 );
      fclose(f0);
      dat[nsize] = 0;
    }

    for( i=0; i<nsize; i++ )
      if( dat[i]==0 )
        dat[i] = '\n';

    replace_char( dat, '\t', ' ' );
    replace_char( dat, '\r', '\n' );

    token = strtok( dat, "\n" );
    while( token != NULL )
    {
      while( *token==' ' )
        token++;
      ldat.push( token );
      token = strtok( NULL, "\r\n" );
    }

    //printf( "ldat.ns %i\n", ldat.ns );
  }


  int number_of_vertex = 0;
  int number_of_face   = 0;
  int number_of_texc   = 0;
  int number_of_normal = 0;
  int number_of_mgroup = 0;
  int number_of_sgroup = 0;
  {
      int mgrp_n_face=0;
      int sgrp_n_face=0;

      for( i=0; i<ldat.ns; i++ )
      {
        str0 = ldat.buf[i];

        type = str0[0]*256 + str0[1];
        switch( type )
        {
          case 'v ':
            number_of_vertex++;
            break;
          case 'vn':
            number_of_normal++;
            break;
          case 'vt':
            number_of_texc++;
            break;
          case 'us':
            number_of_mgroup = number_of_mgroup ? ( mgrp_n_face ? number_of_mgroup+1 : number_of_mgroup ) : 1;
            mgrp_n_face = 0;
            break;
          case 'g ':
            number_of_sgroup = number_of_sgroup ? ( sgrp_n_face ? number_of_sgroup+1 : number_of_sgroup ) : 1;
            sgrp_n_face = 0;
            break;
          case 'mt':
            {
              char tag[256];
              GPath *gp = parse_path( spath );
              if( sscanf( str0, "mtllib %s", tag ) )
              {
                sprintf( mtl_path, "%s%s", gp->dname, tag );
                load_mtl( mtl_path );
              }
            }
            break;
          case 'f ':
            number_of_face++;
            mgrp_n_face++;
            sgrp_n_face++;

            char *tmp = (char*) malloc( (strlen(str0)+1)*sizeof(char) );
            strcpy( tmp, str0 );

            char *token;
            token = strtok( tmp, " " );
            token = strtok( NULL, " " );
            token = strtok( NULL, " " );
            token = strtok( NULL, " " );
            while( ( token = strtok( NULL, " " ) ) != NULL )
              number_of_face++;

            free(tmp);
            break;
        }
      }

  }



  load_face( number_of_face );
  load_vertex( number_of_vertex );
  load_normal( number_of_normal );
  load_texc( number_of_texc );
  load_mgroup( number_of_mgroup+1 );
  load_sgroup( number_of_sgroup+1 );


  Group *mg, *sg;
    mg = mgroup;
    sg = sgroup;

        

  v = vertex.fm;
  n = normal.fm;
  vt = texc.fm;
  face_pos = 0;
  for( i=0; i<ldat.ns; i++ )
  {
    str0 = ldat.buf[i];
    type = str0[0]*256 + str0[1];
 
    switch( type )
    {
      case 'v ':
        sscanf( str0, "v %f %f %f", &v->x, &v->y, &v->z );
        v++;
        break;
      case 'vn':
        sscanf( str0, "vn %f %f %f", &n->x, &n->y, &n->z );
        n++;
        break;

      case 'vt':
        sscanf( str0, "vt %f %f %f", &vt->x, &vt->y, &vt->z );
        vt++;
        break;

      case 'us':
        mg = mg->n_face ? mg+1 : mg;
        sscanf( str0, "usemtl %s", mg->name );
        //mg->idx = face_pos;
        break;

      case 'g ':
        sg = sg->n_face ? sg+1 : sg;
        sscanf( str0, "g %s", sg->name );
        //sg->idx = face_pos;
        break;

      case 'f ':
      {
        vu.reset();
        tu.reset();
        nu.reset();

        if( sscanf( str0, "f %i//%i", &vi, &ni )==2 )
          face_type = GOBJ_V_N;
        else if( sscanf( str0, "f %i/%i/%i", &vi, &ti, &ni )==3 )
          face_type = GOBJ_VTN;
        else if( sscanf( str0, "f %i/%i", &vi, &ti )==2 )
          face_type = GOBJ_VT_;
        else if( sscanf( str0, "f %i", &vi )==1 )
          face_type = GOBJ_V__;
        else
          face_type = -1;

        token = strtok( str0, " " );
        token = strtok( NULL, " " );
        while( token != NULL )
        {
          vi=0;  ti=0;  ni=0;
          switch( face_type )
          {
            case GOBJ_V__:
              sscanf( token, "%i", &vi );  vi--;
              break;
            case GOBJ_VT_:
              sscanf( token, "%i/%i", &vi, &ti );  vi--; ti--;
              break;
            case GOBJ_V_N:
              sscanf( token, "%i//%i", &vi, &ni );  vi--; ni--;
              break;
            case GOBJ_VTN:
              sscanf( token, "%i/%i/%i", &vi, &ti, &ni );  vi--; ti--; ni--;
              break;
            default:
              printf( "[Error] GObj::load(), obj file syntax error\n" );
              exit(-1);
              break;
          }

          vu.push(vi);
          tu.push(ti);
          nu.push(ni);
          token = strtok( NULL, " " );
        }

        l = vu.ns-2;
        fv = &face_vidx[3*face_pos]; vv = vu.buf+1;
        ft = &face_tidx[3*face_pos]; tt = tu.buf+1;
        fn = &face_nidx[3*face_pos]; nn = nu.buf+1;
        for( j=0; j<l; j++, face_pos++, fv+=3, ft+=3, fn+=3, vv++, tt++, nn++ )
        {
          fv[0] = vu.buf[0];   ft[0] = tu.buf[0];   fn[0] = nu.buf[0];
          fv[1] = vv[0];  ft[1] = tt[0];  fn[1] = nn[0];
          fv[2] = vv[1];  ft[2] = tt[1];  fn[2] = nn[1];
          mg->add( face_pos );
          sg->add( face_pos );
          default_group.add( face_pos );
        }

        mg->type = mg->type ? mg->type & face_type : face_type;
        sg->type = sg->type ? sg->type & face_type : face_type;
        default_group.type = default_group.type ? default_group.type & face_type : face_type;
      }
      break;
    }
  }




  if( sgroup[n_sgroup-1].n_face == 0 ) n_sgroup--;
  if( mgroup[n_mgroup-1].n_face == 0 ) n_mgroup--;
  if( n_sgroup == 1 && sgroup[0].name[0]=='\0' ) load_sgroup( 0 );
  if( n_mgroup == 1 && mgroup[0].name[0]=='\0' ) load_mgroup( 0 );



  if( n_material==0 )
  {
    GPath *gp = parse_path( spath );

    char tpath[256];
    sprintf( tpath, "%s%s.mtl", gp->dname, gp->fname );
    load_mtl( tpath );
  }

  if( n_texc )
  {
    GPath *gp = parse_path( spath );
    GPfm pfm;
    char ext[][4] = { "pfm", "ppm", "bmp", "png", "jpg", "tga" };
    int i;

    for( i=0; i<sizeof(ext)/sizeof(ext[0]); i++ )
    {
      sprintf( tex_path, "%s%s.%s", gp->dname, gp->fname, ext[i] );
      if( fexist( tex_path ) )
      {
        pfm.load( tex_path );
        pfm.scale( 512, 512 );
        pfm.flip_vertical();
        break;
      }
    }

    if( pfm.w==0 || pfm.h==0 )
      pfm.load( 64, 64, 1,1,1 );

    glGenTextures( 1, &tex0 );
    glBindTexture( GL_TEXTURE_2D, tex0 );
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, pfm.w, pfm.h, 0, GL_RGB, GL_FLOAT, pfm.fm );
  }

  strcpy( obj_path, spath );
  //printf_info();


  free(dat);
}
*/

void GObj::printf_info()
{
  GPath *gp;
    printf( "OBJ:\n" );
    if( strcmp( obj_path, "" )!=0 )
    {
      gp = parse_path( obj_path );
      printf( "  src_path : %s.%s ", gp->fname, gp->ename );
    }
    printf( "[ v %i; ", n_vertex );
    if( n_texc )
      printf( "vt %i; ", n_texc );
    if( n_normal )
      printf( "vn %i; ", n_normal );
    printf( "f %i; ", n_face );
    printf( "]\n" );



  if( strcmp( obj_path, "" )!=0 )
  {
    bool mtl_defined = false;
    bool mtl_found   = false;
    if( strcmp( mtl_path, "" )!=0 )
    {
      mtl_defined = true;
      gp = parse_path( mtl_path );

      if( fexist(mtl_path) )
      {
        mtl_found = true;
        printf( "  mtl_path : %s.%s\n", gp->fname, gp->ename );
      }
    }

    if( !mtl_found )
    {
      gp = parse_path( obj_path );
      char tpath[256];
      sprintf( tpath, "%s%s.mtl", gp->dname, gp->fname );

      if( fexist(tpath) )
      {
        printf( "  mtl_path : %s.mtl\n", gp->fname );
        mtl_found = true;
      }
    }

    if( mtl_defined && !mtl_found )
    {      
      gp = parse_path( mtl_path );
      printf( "  mtl_path : %s.%s <missing>\n", gp->fname, gp->ename );
    }
  }

  if( strcmp( tex_path, "" )!=0 )
  {
    gp = parse_path( tex_path );
    if( fexist(tex_path) )
      printf( "  tex_path : %s.%s", gp->fname, gp->ename );
    else
      printf( "  tex_path : %s.%s<missing>", gp->fname, gp->ename );

    if( material )
    {
      char texpath[256];
      GPath gp = parse_spath( mtl_path );
      int i;

      for( i=0; i<n_material; i++ )
      {
        if( strcmp( material[i].texname, "" )!=0 )
        {
          sprintf( texpath, "%s%s", gp.dname, material[i].texname );

          if( fexist(texpath) )
            printf( " %s", material[i].texname );
          else
            printf( " %s<missing>", material[i].texname );
        }
      }
    }
    printf( "\n" );
  }


  if( n_sgroup )
    printf( "  number of scene groups    : %i\n", n_sgroup );
  if( n_mgroup )
    printf( "  number of material groups : %i\n", n_mgroup );
  if( n_material )
    printf( "  number of materials       : %i\n", n_material );
}


void GObj::load_mtl( const char *spath )
{
  char str[256], *str0, tag[256]="";

  SAFE_FREE( material );
  n_material = 0;
  {
    FILE *f0 = fopen( spath, "rb" );
      if( f0==NULL )
        return;
      while( fgets( str, 256, f0 ) )
      {
        str0 = str;
        while( *str0 == ' ' || *str0=='\t' )
          str0++;
        if( sscanf( str0, "%s", tag ) == 1 )
          if( strcmp( tag, "newmtl" )==0 )
            n_material++;
      }
    fclose(f0);
  }

  Material default_mtl;
    memset( &default_mtl, 0, sizeof(Material) );
    default_mtl.ambient   = FLOAT4( .2f, .2f, .2f, 1 );
    default_mtl.diffuse   = FLOAT4( .8f, .8f, .8f, 1 );
    default_mtl.specular  = FLOAT4( 1, 1, 1, 1 );
    default_mtl.ward      = FLOAT4( 1,1,0,0 );
    default_mtl.shininess = 0.0;
    default_mtl.illum     = 1;
    default_mtl.texid     = 0;

  material = (Material*) malloc( n_material * sizeof(Material) );
  {
    int i;
    for( i=0; i<n_material; i++ )
      memcpy( &material[i], &default_mtl, sizeof(Material) );

    Material *mtl;
      mtl = NULL;
    FILE *f0 = fopen( spath, "rb" );
      while( fgets( str, 256, f0 ) )
      {
        str0 = str;
        while( *str0 == ' ' || *str0=='\t' )
          str0++;

        if( sscanf( str0, "%s", tag )==1 )
        {
          if( strcmp( tag, "newmtl" )==0 )
          {
            mtl = mtl ? mtl+1 : material;
            sscanf( str0, "newmtl %s", mtl->name );
          }else if( strcmp( tag, "Kd" )==0 )
          {
            sscanf( str0, "Kd %f %f %f", &mtl->diffuse.x, &mtl->diffuse.y, &mtl->diffuse.z );
          }else if( strcmp( tag, "Ka" )==0 )
          {
            sscanf( str0, "Ka %f %f %f", &mtl->ambient.x, &mtl->ambient.y, &mtl->ambient.z );
          }else if( strcmp( tag, "Ks" )==0 )
          {
            sscanf( str0, "Ks %f %f %f", &mtl->specular.x, &mtl->specular.y, &mtl->specular.z );
          }else if( strcmp( tag, "Ns" )==0 )
          {
            sscanf( str0, "Ns %f", &mtl->shininess );
            mtl->shininess = mtl->shininess * 128 / 1000;
          }else if( strcmp( tag, "illum" )==0 )
          {
            sscanf( str0, "illum %i", &mtl->illum );
          }else if( strcmp( tag, "ward" )==0 )
          {
            sscanf( str0, "ward %f %f", &mtl->ward.x, &mtl->ward.y );
            mtl->ward.w = 1;
          }else if( strcmp( tag, "map_Kd" )==0 )
          {
            char texpath[256];
            sscanf( str0, "map_Kd %s", mtl->texname );
            GPath gp = parse_spath( spath );
            sprintf( texpath, "%s%s", gp.dname, mtl->texname );

            if( fexist(texpath) )
            {
              GPfm pfm;
              pfm.load( texpath );
              pfm.resample( 512, 512 );
              //pfm.flip_vertical();

              if( mtl->texid )
                glDeleteTextures( 1, &mtl->texid );
              glGenTextures( 1, &mtl->texid );
              glBindTexture( GL_TEXTURE_2D, mtl->texid );
              glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
              glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
              glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, pfm.w, pfm.h, 0, GL_RGB, GL_FLOAT, pfm.fm );
            }

          }
        }
      }

    fclose(f0);
  }

}

GObj::Material GObj::getmtl( const char *id ) const
{
  int i;
  for( i=0; i<n_material; i++ )
  {
    if( strcmp( material[i].name, id )==0 )
      return material[i];
  }

  for( i=0; i<n_material; i++ )
  {
    if( strcmp( material[i].name, "" )==0 )
      return material[i];
  }

  Material default_mtl;
    memset( &default_mtl, 0, sizeof(Material) );
    default_mtl.ambient   = FLOAT4( .2f, .2f, .2f, 1 );
    default_mtl.diffuse   = FLOAT4( .8f, .8f, .8f, 1 );
    default_mtl.specular  = FLOAT4( 1, 1, 1, 1 );
    default_mtl.shininess = 0.0;
    default_mtl.illum     = 1;
    default_mtl.texid     = 0;
  return default_mtl;
}

void GObj::usemtl( const Material &mtl ) const
{
  float zero[4] = {0, 0 , 0 , 1};
    glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT,   (float*)&mtl.ambient   );

    if( mtl.illum>=1 )
      glMaterialfv( GL_FRONT_AND_BACK, GL_DIFFUSE,   (float*)&mtl.diffuse   );
    else
      glMaterialfv( GL_FRONT_AND_BACK, GL_DIFFUSE,   zero  );

    if( mtl.illum==2 )
      glMaterialfv( GL_FRONT_AND_BACK, GL_SPECULAR,  (float*)&mtl.specular  );
    else
      glMaterialfv( GL_FRONT_AND_BACK, GL_SPECULAR,  zero  );

    glMaterialf(  GL_FRONT_AND_BACK, GL_SHININESS, mtl.shininess );
}

void GObj::draw_group( const Group &grp, int draw_option ) const
{
  if( grp.n_face==0 )
    return;

  Material mtl;
    mtl = getmtl( grp.name );
    usemtl( mtl );

  int i;
  int t0;

    //if( (grp.type & draw_option)==GOBJ_VTN && n_texc && (mtl.texid || tex0) && n_normal )
    if( (grp.type & draw_option)==GOBJ_VTN && n_texc && n_normal )
    {
      if( mtl.texid )
        glBindTexture( GL_TEXTURE_2D, mtl.texid );
      else if( tex0 )
        glBindTexture( GL_TEXTURE_2D, tex0 );
      if(mtl.texid || tex0)
        glEnable( GL_TEXTURE_2D );

      glBegin( GL_TRIANGLES );
      for( i=0; i<grp.n_face; i++ )
      {
        t0 = grp.tri[i];
        glNormal3fv( (float*)&normal.fm[ face_nidx[t0*3  ] ] );
        glTexCoord2fv( (float*)&texc.fm[ face_tidx[t0*3  ] ] );
        glVertex3fv( (float*)&vertex.fm[ face_vidx[t0*3  ] ] );
        glNormal3fv( (float*)&normal.fm[ face_nidx[t0*3+1] ] );
        glTexCoord2fv( (float*)&texc.fm[ face_tidx[t0*3+1] ] );
        glVertex3fv( (float*)&vertex.fm[ face_vidx[t0*3+1] ] );
        glNormal3fv( (float*)&normal.fm[ face_nidx[t0*3+2] ] );
        glTexCoord2fv( (float*)&texc.fm[ face_tidx[t0*3+2] ] );
        glVertex3fv( (float*)&vertex.fm[ face_vidx[t0*3+2] ] );
      }
      glEnd();

      if(mtl.texid || tex0)
      glDisable( GL_TEXTURE_2D );

    }else if( (grp.type & draw_option)==GOBJ_V_N && n_normal )
    {
      glBegin( GL_TRIANGLES );
      for( i=0; i<grp.n_face; i++ )
      {
        t0 = grp.tri[i];
        glNormal3fv( (float*)&normal.fm[ face_nidx[t0*3  ] ] );
        glVertex3fv( (float*)&vertex.fm[ face_vidx[t0*3  ] ] );
        glNormal3fv( (float*)&normal.fm[ face_nidx[t0*3+1] ] );
        glVertex3fv( (float*)&vertex.fm[ face_vidx[t0*3+1] ] );
        glNormal3fv( (float*)&normal.fm[ face_nidx[t0*3+2] ] );
        glVertex3fv( (float*)&vertex.fm[ face_vidx[t0*3+2] ] );
      }
      glEnd();
    //}else if( (grp.type & draw_option)==GOBJ_VT_ && n_texc && (mtl.texid || tex0) )
    }else if( (grp.type & draw_option)==GOBJ_VT_ && n_texc )
    {
      if( mtl.texid )
        glBindTexture( GL_TEXTURE_2D, mtl.texid );
      else if( tex0 )
        glBindTexture( GL_TEXTURE_2D, tex0 );
      if(mtl.texid || tex0) 
        glEnable( GL_TEXTURE_2D );

      glBegin( GL_TRIANGLES );
      glNormal3f( 0,0,1 );
      for( i=0; i<grp.n_face; i++ )
      {
        t0 = grp.tri[i];
        glTexCoord2fv( (float*)&texc.fm[ face_tidx[t0*3  ] ] );
        glVertex3fv( (float*)&vertex.fm[ face_vidx[t0*3  ] ] );
        glTexCoord2fv( (float*)&texc.fm[ face_tidx[t0*3+1] ] );
        glVertex3fv( (float*)&vertex.fm[ face_vidx[t0*3+1] ] );
        glTexCoord2fv( (float*)&texc.fm[ face_tidx[t0*3+2] ] );
        glVertex3fv( (float*)&vertex.fm[ face_vidx[t0*3+2] ] );
      }
      glEnd();

      if(mtl.texid || tex0) 
      glDisable( GL_TEXTURE_2D );

    }else
    {
      glBegin( GL_TRIANGLES );
      glNormal3f( 0,0,1 );
      for( i=0; i<grp.n_face; i++ )
      {
        t0 = grp.tri[i];
        glVertex3fv( (float*)&vertex.fm[ face_vidx[t0*3  ] ] );
        glVertex3fv( (float*)&vertex.fm[ face_vidx[t0*3+1] ] );
        glVertex3fv( (float*)&vertex.fm[ face_vidx[t0*3+2] ] );
      }
      glEnd();
    }
}

void GObj::draw( int draw_option ) const
{ 
  int j;

  glPushAttrib( GL_ENABLE_BIT );

  if( n_mgroup && !(favour_sgroup && sgroup) )
  {
    for( j=0; j<n_mgroup; j++ )
      draw_group( mgroup[j], draw_option );
  }else if( n_sgroup )
  {
    for( j=0; j<n_sgroup; j++ )
      draw_group( sgroup[j], draw_option );
  }else
  {
    draw_group( default_group, draw_option );
  }

  glPopAttrib();
}

GLuint GObj::list()
{
  GLuint list;
  list = glGenLists(1);
  glNewList(list, GL_COMPILE);
  draw();
  glEndList();
  
  return list;
}

double GObj::calculate_face_area( int ti )
{
  double p0[3] = { vertex.fm[ face_vidx[ti*3+0] ].x, vertex.fm[ face_vidx[ti*3+0] ].y, vertex.fm[ face_vidx[ti*3+0] ].z };
  double p1[3] = { vertex.fm[ face_vidx[ti*3+1] ].x, vertex.fm[ face_vidx[ti*3+1] ].y, vertex.fm[ face_vidx[ti*3+1] ].z };
  double p2[3] = { vertex.fm[ face_vidx[ti*3+2] ].x, vertex.fm[ face_vidx[ti*3+2] ].y, vertex.fm[ face_vidx[ti*3+2] ].z };
  double l0[3] = { p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2] };
  double l1[3] = { p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2] };
  double a0[3] = { l0[1]*l1[2]-l1[1]*l0[2], -l0[0]*l1[2]+l1[0]*l0[2], l0[0]*l1[1]-l1[0]*l0[1] };
  double area = sqrt( a0[0]*a0[0]+a0[1]*a0[1]+a0[2]*a0[2]  )/2;
  return area;
}

void GObj::calculate_face_normal()
{
  load_normal( n_face );

  int i;
  FLOAT3 u, v, n;

  for( i=0; i<n_face; i++ )
  {
    u = vertex.fm[ face_vidx[i*3+1] ] - vertex.fm[ face_vidx[i*3+0] ];
    v = vertex.fm[ face_vidx[i*3+2] ] - vertex.fm[ face_vidx[i*3+0] ];

    //vnormalize( &u );
    //vnormalize( &v );

    n = vcross( u, v );
    vnormalize( &n );

    normal.fm[i] = n;

    face_nidx[i*3+0] = i;
    face_nidx[i*3+1] = i;
    face_nidx[i*3+2] = i;
  }

  for( i=0; i<n_mgroup; i++ )
    mgroup[i].type |= GOBJ_NORMAL;
  for( i=0; i<n_sgroup; i++ )
    sgroup[i].type |= GOBJ_NORMAL;
  default_group.type |= GOBJ_NORMAL;
}

void GObj::calculate_access( int *&a0, int **&b0, const int *access, int n_access )
{
  int j;
  int **c0, al, *tmp;
  
  a0 = (int*) malloc( n_access * sizeof(int) );
  memset( a0, 0, n_access * sizeof(int) );
  for( j=0; j<3*n_face; j++ )
    a0[ access[j] ]++;
  
  al = 0;
  for( j=0; j<n_access; j++ )
    al+=a0[j];
  
  b0 = (int**) malloc( n_access * sizeof(int*) + al * sizeof(int) );
  c0 = (int**) malloc( n_access * sizeof(int*) );
  
  tmp = (int*)(b0+n_access);
  for( j=0; j<n_access; j++ )
  {
    b0[j] = tmp;
    c0[j] = tmp;
    tmp += a0[j];
  }
  
  for( j=0; j<3*n_face; j++ )
  {
    *(c0[ access[j] ]) = j;
    c0[ access[j] ]++;
  }

  free(c0);
}

void GObj::unitize_normal()
{
  int i;
  for( i=0; i<n_normal; i++ )
     vnormalize( &normal.fm[i] );
}

void GObj::texfilter( unsigned int filter_type )
{
  if( tex0 )
  {
    glBindTexture( GL_TEXTURE_2D, tex0 );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter_type);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter_type);
  }
  if( material )
    for( int i=0; i<n_material; i++ )
      if( material[i].texid )
      {
        glBindTexture( GL_TEXTURE_2D, material[i].texid );
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter_type);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter_type);
      }
}

void GObj::calculate_edge_info()
{
  SAFE_FREE( s_edge );
  SAFE_FREE( face_eidx );

  int u[3];
  bool found;
  int i, j, k, ei ;
  int s,t;

  s_edge = (GEdge*) malloc( 3*n_face * sizeof(GEdge) );
  face_eidx = (int*) malloc( 3*n_face * sizeof(int) );
  n_edge = 0;

  GStack<int> **v2e;
  v2e = new GStack<int>*[n_vertex];
  for( i=0; i<n_vertex; i++ )
    v2e[i] = new GStack<int>(16);

  for( i=0; i<3*n_face; i++ )
  {
    s_edge[i].n1 = -1;
    s_edge[i].t1 = -1;
  }

  for( j=0; j<n_face; j++ )
  {
    u[0] = face_vidx[3*j+0];
    u[1] = face_vidx[3*j+1];
    u[2] = face_vidx[3*j+2];

    for( k=0; k<3; k++ )
    {
      s = u[k];
      t = u[(k+1)%3];
      found = false;

      /*
      for( i=0; i<n_edge; i++ )
      {
        if( s_edge[i].t1==-1 && s_edge[i].v0==t && s_edge[i].v1==s )
        {
          found = true;
          break;
        }
      }
      */
      for( ei=0; ei<v2e[t]->ns; ei++ )
      {
        i = v2e[t]->buf[ei];
        if( s_edge[i].v1==s )
        {
          v2e[t]->remove(ei);
          found = true;
          break;
        }
      }
    
      if( !found )
      {
        s_edge[n_edge].v0 = s;
        s_edge[n_edge].v1 = t;
        s_edge[n_edge].n0 = face_nidx[3*j+0];
        s_edge[n_edge].t0 = j;
        face_eidx[3*j+k] = n_edge;
        v2e[s_edge[n_edge].v0]->push(n_edge);
        n_edge++;
      }else
      {
        s_edge[i].n1 = face_nidx[3*j+0];
        s_edge[i].t1 = j;
        face_eidx[3*j+k] = i;
      }
    }
  }

  closed = true;
  for( i=0; i<n_edge; i++ )
  {
    if( s_edge[i].n1==-1 )
    {
      //printf( "[Warning] GObj::calculate_edge_info(), not all edges are shared by two triangles.\n" );
      closed = false;
      break;
    }
  }
  s_edge = (GEdge*)realloc( s_edge, n_edge * sizeof(GEdge) );


  for( i=0; i<n_vertex; i++ )
    delete v2e[i];
  delete[] v2e;
}

#include<Windows.h>

void GObj::clean_redundant_faces()
{
  int i, j;
  Group **mgrp = (Group**) malloc( n_face * sizeof(Group*) );
  Group **sgrp = (Group**) malloc( n_face * sizeof(Group*) );
  memset( mgrp, 0, n_face * sizeof(Group*) );
  memset( sgrp, 0, n_face * sizeof(Group*) );

  int ncleaned;

  for( j=0; j<n_mgroup; j++ )
  {
    Group &grp = mgroup[j];
    for( i=0; i<grp.n_face; i++ )
      mgrp[ grp.tri[i] ] = &mgroup[j];
  }

  for( j=0; j<n_sgroup; j++ )
  {
    Group &grp = sgroup[j];
    for( i=0; i<grp.n_face; i++ )
      sgrp[ grp.tri[i] ] = &sgroup[j];
  }

  ncleaned=0;
  for( i=0; i<n_face; i++ )
  {
    if( 
      ( face_vidx[3*i+0] == face_vidx[3*i+1] ) || 
      ( face_vidx[3*i+1] == face_vidx[3*i+2] ) || 
      ( face_vidx[3*i+2] == face_vidx[3*i+0] ) ||
      calculate_face_area(i)<FLT_EPSILON
    ){
      if( n_mgroup && mgrp[i] ) mgrp[i]->remove( i );
      if( n_sgroup && sgrp[i]) sgrp[i]->remove( i );
      default_group.remove( i );
      ncleaned++;
    }
  }

  printf( "%i number of faces are removed.\n", ncleaned );

  free( mgrp );
  free( sgrp );
}

void GObj::weld()
{
  GPfm vmap;
    vmap.load( n_vertex, 1 );

  FLOAT3 d;
  int n_vmap = 0;
  bool vfound, ifound;
  int i, j, idx;
  int *v;
  int n;

  for( j=0; j<n_vertex; j++ )
  {
    ifound = false;
    v = face_vidx;
    n = 3*n_face;
    for( i=0; i<n; i++, v++ )
    {
      if( *v == j )
      {
        ifound = true;
        break;
      }
    }

    if( ifound )
    {
      vfound = false;
      for( i=0; i<n_vmap; i++ )
      {
        d = vertex.fm[j] - vmap.fm[i];
        if( d.norm() < 10*FLT_EPSILON )
        {
          vfound = true;
          break;
        }
      }

      if( vfound )
      {
        idx = i;
      }else
      {
        idx = n_vmap;
        vmap.fm[n_vmap++] = vertex.fm[j];
      }

      v = face_vidx;
      n = 3*n_face;
      for( i=0; i<n; i++, v++ )
      {
        if( *v == j )
          *v = idx;
      }
    }
  }

  //printf( "%i out of %i vertices were removed\n", n_vertex - n_vmap, n_vertex );

  n_vertex = n_vmap;
  vertex.load( n_vmap, 1, vmap.fm );
}

float GObj::calculate_smoothness()
{
  float smoothness;
  FLOAT3 v0, v1, v2;
  FLOAT3 n0, n1, n2;
  FLOAT3 fn;
  int i;

  smoothness = 0;
  for( i=0; i<n_face; i++ )
  {
    n0 = normal.fm[ face_nidx[3*i+0] ];
    n1 = normal.fm[ face_nidx[3*i+1] ];
    n2 = normal.fm[ face_nidx[3*i+2] ];

    v0 = vertex.fm[ face_vidx[3*i+0] ];
    v1 = vertex.fm[ face_vidx[3*i+1] ];
    v2 = vertex.fm[ face_vidx[3*i+2] ];

    fn  = vnormalize( vcross( v1-v0, v2-v0 ) );

    smoothness += acosf(G_CLAMP(vdot(n0,fn),-1,1));
    smoothness += acosf(G_CLAMP(vdot(n1,fn),-1,1));
    smoothness += acosf(G_CLAMP(vdot(n2,fn),-1,1));
  }
  smoothness /= n_face * 3;

  return smoothness;
}

FLOAT3 GObj::get_face_normal( int face_index )
{
  FLOAT3 v0, v1, v2;
  FLOAT3 fn;
    v0 = vertex.fm[ face_vidx[3*face_index+0] ];
    v1 = vertex.fm[ face_vidx[3*face_index+1] ];
    v2 = vertex.fm[ face_vidx[3*face_index+2] ];

    fn  = vnormalize( vcross( v1-v0, v2-v0 ) );
  return fn;
}

FLOAT3 GObj::get_face_normal( FLOAT3 v0, FLOAT3 v1, FLOAT3 v2 )
{
  return vnormalize( vcross( v1-v0, v2-v0 ) );
}

void GObj::force_single_sided()
{
  int i, j, k;

  weld();
  calculate_edge_info();

  bool *edone = (bool*) malloc( n_edge * sizeof(bool) );
  bool *tdone = (bool*) malloc( n_face * sizeof(bool) );
  memset( edone, 0, n_edge * sizeof(bool) );
  memset( tdone, 0, n_face * sizeof(bool) );

  Group **mgrp = (Group**) malloc( n_face * sizeof(Group*) );
  Group **sgrp = (Group**) malloc( n_face * sizeof(Group*) );

  for( j=0; j<n_mgroup; j++ )
  {
    Group &grp = mgroup[j];
    for( i=0; i<grp.n_face; i++ )
      mgrp[ grp.tri[i] ] = &mgroup[j];
  }

  for( j=0; j<n_sgroup; j++ )
  {
    Group &grp = sgroup[j];
    for( i=0; i<grp.n_face; i++ )
      sgrp[ grp.tri[i] ] = &sgroup[j];
  }

  GPfm nmap;
    nmap.load( 3 * n_face * 2, 1 );
    nmap.draw( normal, 0,0 );

  int n_nmap;
    n_nmap = n_normal;

  FLOAT3 fn, vn;
  int eidx;
  int t0, t1;
  int e;
  int *pist, *_pist, *qist;
  int *eist, n_eist;
  int *tist, n_tist;
    _pist = (int*) malloc( n_edge * sizeof(int) );
    eist = (int*) malloc( n_edge * sizeof(int) );
    tist = (int*) malloc( n_face * sizeof(int) );

  for( j=0; j<n_edge; j++ )
  {
    if( !edone[j] )
    {
      pist = _pist;
      qist = _pist;
      n_eist = 0;
      n_tist = 0;

      edone[j] = true;
      *pist++ = j;  

      while( qist != pist )
      {
        eidx = *qist++;
        t0 = s_edge[eidx].t0;
        t1 = s_edge[eidx].t1;
        
        if( t0!=-1 )
        {
          for( k=0; k<3; k++ )
          {
            e = face_eidx[3*t0+k];
            if( !edone[e] )
            {
              edone[e] = true;
              *pist++ = e;  
            }
          }
          if( !tdone[t0] )
          {
            tdone[t0] = true;
            tist[n_tist++] = t0;
          }
        }else
        {
          eist[n_eist++] = eidx;
        }

        if( t1!=-1 )
        {
          for( k=0; k<3; k++ )
          {
            e = face_eidx[3*t1+k];
            if( !edone[e] )
            {
              edone[e] = true;
              *pist++ = e;  
            }
          }
          if( !tdone[t1] )
          {
            tdone[t1] = true;
            tist[n_tist++] = t1;
          }
        }else
        {
          eist[n_eist++] = eidx;
        }
      }

      if( n_eist == 0 )
      {
        //printf( "close part found\n" );
      }
      if( n_eist )
      {
      
        face_vidx = (int*)realloc( face_vidx, (n_face+n_tist) * 3*sizeof(int) );
        face_tidx = (int*)realloc( face_tidx, (n_face+n_tist) * 3*sizeof(int) );
        face_nidx = (int*)realloc( face_nidx, (n_face+n_tist) * 3*sizeof(int) );
        for( i=0; i<n_tist; i++ )
        {
          t0 = tist[i];
        
          nmap.fm[n_nmap+0] = -normal.fm[ face_nidx[3*t0+0] ];
          nmap.fm[n_nmap+1] = -normal.fm[ face_nidx[3*t0+1] ];
          nmap.fm[n_nmap+2] = -normal.fm[ face_nidx[3*t0+2] ];

          face_vidx[ 3*n_face+0 ] = face_vidx[3*t0+0];
          face_vidx[ 3*n_face+1 ] = face_vidx[3*t0+2];
          face_vidx[ 3*n_face+2 ] = face_vidx[3*t0+1];
          face_tidx[ 3*n_face+0 ] = face_tidx[3*t0+0];
          face_tidx[ 3*n_face+1 ] = face_tidx[3*t0+2];
          face_tidx[ 3*n_face+2 ] = face_tidx[3*t0+1];
          face_nidx[ 3*n_face+0 ] = n_nmap+0;
          face_nidx[ 3*n_face+1 ] = n_nmap+2;
          face_nidx[ 3*n_face+2 ] = n_nmap+1;

          n_nmap += 3;

          if( n_mgroup ) mgrp[t0]->add( n_face );
          if( n_sgroup ) sgrp[t0]->add( n_face );
          default_group.add( n_face );
        
          n_face = n_face + 1;
        }
      }
    }
  }

  normal.load( n_nmap, 1, nmap.fm );
  n_normal = n_nmap;


  free( edone );
  free( tdone );
  free( _pist );
  free( eist );
  free( tist );
  free( mgrp );
  free( sgrp );
}


void Group::add( int face_idx )
{
  if( n_face+1 > n_buf )
  {
    n_buf += 256;
    tri = (int*) realloc( tri, n_buf * sizeof(int) );
  }

  tri[n_face++] = face_idx;
}

void Group::remove( int face_idx )
{
  int i;

  for( i=0; i<n_face; i++ )
  {
    if( tri[i] == face_idx )
    {
      tri[i] = tri[n_face-1];
      n_face--;
    }
  }
}

void Group::clear()
{
  SAFE_DELETE( tri );
  memset( this, 0, sizeof(Group) );
}

Group::Group()
{
  memset( this, 0, sizeof(Group) );
}

Group::~Group()
{
  SAFE_DELETE( tri );
}

void GObj::merge( const GObj &b ){ obj_merge(b); }
void GObj::obj_merge( const GObj &b )
{
  obj_merge( *this, b );
}

void GObj::obj_merge( const GObj &a, const GObj &b )
{
  if( &a==this || &b==this )
  {
    GObj obj;
      obj.obj_merge( a, b );
      this->obj_merge( obj, GObj() );
      return;
  }

  if( a.default_group.type!=b.default_group.type && a.n_face && b.n_face )
  {
    printf( "[Error] GObj::obj_merge, Type mismatch\n" );
    exit(-1);
  }
  int i;
    
  GObj &c = *this;
  c.clear();

  c.default_group.type = a.n_face ? a.default_group.type : b.default_group.type;
  c.load_face( a.n_face + b.n_face );

  if( c.default_group.type & GOBJ_VERTEX )
  {
    c.load_vertex( a.n_vertex + b.n_vertex );
    memcpy( c.vertex.fm, a.vertex.fm, a.n_vertex * sizeof(FLOAT3) );
    memcpy( c.vertex.fm+a.n_vertex, b.vertex.fm, b.n_vertex * sizeof(FLOAT3) );
    memcpy( c.face_vidx, a.face_vidx, a.n_face*3*sizeof(int) );
    for( i=0; i<b.n_face; i++ )
    {
      c.face_vidx[3*a.n_face + 3*i+0] = b.face_vidx[3*i+0] + a.n_vertex;
      c.face_vidx[3*a.n_face + 3*i+1] = b.face_vidx[3*i+1] + a.n_vertex;
      c.face_vidx[3*a.n_face + 3*i+2] = b.face_vidx[3*i+2] + a.n_vertex;
    }
  }

  if( c.default_group.type & GOBJ_NORMAL )
  {
    c.load_normal( a.n_normal + b.n_normal );
    memcpy( c.normal.fm, a.normal.fm, a.n_normal * sizeof(FLOAT3) );
    memcpy( c.normal.fm+a.n_normal, b.normal.fm, b.n_normal * sizeof(FLOAT3) );
    memcpy( c.face_nidx, a.face_nidx, a.n_face*3*sizeof(int) );
    for( i=0; i<b.n_face; i++ )
    {
      c.face_nidx[3*a.n_face + 3*i+0] = b.face_nidx[3*i+0] + a.n_normal;
      c.face_nidx[3*a.n_face + 3*i+1] = b.face_nidx[3*i+1] + a.n_normal;
      c.face_nidx[3*a.n_face + 3*i+2] = b.face_nidx[3*i+2] + a.n_normal;
    }
  }

  if( c.default_group.type & GOBJ_TEXCOR )
  {
    c.load_texc( a.n_texc + b.n_texc );
    memcpy( c.texc.fm, a.texc.fm, a.n_texc * sizeof(FLOAT3) );
    memcpy( c.texc.fm+a.n_texc, b.texc.fm, b.n_texc * sizeof(FLOAT3) );
    memcpy( c.face_tidx, a.face_tidx, a.n_face*3*sizeof(int) );
    for( i=0; i<b.n_face; i++ )
    {
      c.face_tidx[3*a.n_face + 3*i+0] = b.face_tidx[3*i+0] + a.n_texc;
      c.face_tidx[3*a.n_face + 3*i+1] = b.face_tidx[3*i+1] + a.n_texc;
      c.face_tidx[3*a.n_face + 3*i+2] = b.face_tidx[3*i+2] + a.n_texc;
    }
  }

  for( i=0; i<c.n_face; i++ )
    default_group.add( i );
}

void GObj::calculate_face_tangent( GPfm &face_tangent )
{

  face_tangent.load( n_face, 1 );

  const float tol = 1e-6f;

  int i;

  FLOAT3 v0, v1, v2;
  FLOAT3 t0, t1, t2;
  FLOAT3 u, v, p;

  FLOAT3 xn, yn, zn;
  FLOAT3 fn;

  if( default_group.type & GOBJ_TEXCOR )
  {
    for( i=0; i<n_face; i++ )
    {
      v0 = vertex.fm[ face_vidx[3*i+0] ];
      v1 = vertex.fm[ face_vidx[3*i+1] ];
      v2 = vertex.fm[ face_vidx[3*i+2] ];

      t0 = texc.fm[ face_tidx[3*i+0] ];
      t1 = texc.fm[ face_tidx[3*i+1] ];
      t2 = texc.fm[ face_tidx[3*i+2] ];

      u = FLOAT3( v1.x-v0.x, t1.x-t0.x, t1.y-t0.y );
      v = FLOAT3( v2.x-v0.x, t2.x-t0.x, t2.y-t0.y );
      p = vnormalize( vcross(u,v) );
      if ( p.x < tol )
        p.x = tol * G_SIGN(p.x);
      zn.x = -p.y/p.x;
      xn.x = -p.z/p.x;

      u = FLOAT3( v1.y-v0.y, t1.x-t0.x, t1.y-t0.y );
      v = FLOAT3( v2.y-v0.y, t2.x-t0.x, t2.y-t0.y );
      p = vnormalize( vcross(u,v) );
      if ( p.x < tol )
        p.x = tol * G_SIGN(p.x);
      zn.y = -p.y/p.x;
      xn.y = -p.z/p.x;

      u = FLOAT3( v1.z-v0.z, t1.x-t0.x, t1.y-t0.y );
      v = FLOAT3( v2.z-v0.z, t2.x-t0.x, t2.y-t0.y );
      p = vnormalize( vcross(u,v) );
      if ( p.x < tol )
        p.x = tol * G_SIGN(p.x);
      zn.z = -p.y/p.x;
      xn.z = -p.z/p.x;

      zn = vnormalize(zn);
      xn = vnormalize(xn);
      yn = vnormalize( vcross(zn,xn) );

      if( zn.norm() ==0 )
      {
        if( n_normal )
          yn = normal.fm[ face_nidx[3*i+0] ];
        else
          yn = get_face_normal(i);

        if( xn.norm()==0 )
        {
          if( vcross(yn, FLOAT3(0,0,1)).norm()!=0 )
            zn = vnormalize( vcross(yn, FLOAT3(0,0,1)) );
          else
            zn = vnormalize( vcross(yn, FLOAT3(0,1,0)) );
        }else
        {
          xn = xn - vdot(xn,yn)*yn;
          xn = vnormalize(xn);
          zn = vcross( xn, yn );
        }
      }

      face_tangent.fm[i] = zn;
    }
  }else
  {
    printf( "[Warning] tangents are calculated without texture coordinates.\n" );

    FLOAT3 n0, n1, n2;
    for( i=0; i<n_face; i++ )
    {
      n0 = normal.fm[ face_nidx[3*i+0] ];
      n1 = normal.fm[ face_nidx[3*i+1] ];
      n2 = normal.fm[ face_nidx[3*i+2] ];
      yn = vnormalize( n0+n1+n2 );
      zn = vnormalize( vcross(yn, FLOAT3(0,0,.97987f) ));

      if( g_ferr(zn.x) || g_ferr(zn.y) || g_ferr(zn.z) )
      {
        printf( 
          "[Warning] GObj::calculate_face_tangent(), failed "
          "to calculate face tangent because of some "
          "face normal are too close to reference direction.\n" );
        face_tangent.fm[i] = FLOAT3(1,0,0);
      }else
        face_tangent.fm[i] = zn;
    }
  }
}

void GObj::weld_normal()
{
  calculate_vertex_normal( .25 );
}

void GObj::calculate_vertex_normal( float angle )
{
  if( (default_group.type & GOBJ_NORMAL)==0 )
  {
    printf( "[Error] GObj::calculate_vertex_normal, default group has no normal.\n" );
    exit(-1);
  }

  float cos_angle, bdot;
    cos_angle = cosf(angle * G_PI / 180.0f);

  int *a0, **b0;
    calculate_access( a0, b0, face_vidx, n_vertex );

  int i, j, k, il, k0;
  FLOAT3 n;
  GStack<FLOAT3> gs;
  k0=0;
  for( j=0; j<n_vertex; j++ )
  {

    il=a0[j];
    for( i=0; i<il; i++ )
    {
      n = normal.fm[ face_nidx[ b0[j][i] ] ];

      for( k=k0; k<gs.ns; k++ )
      {
        bdot = vdot( n, gs.buf[k] );
        bdot = G_CLAMP( bdot, -1, 1 );
        if( bdot >= cos_angle )
        {
          face_nidx[ b0[j][i] ] = k;
          gs.buf[k] += n;
          break;
        }
      }

      if( k==gs.ns )
      {
        face_nidx[ b0[j][i] ] = k;
        gs.push(n);
      }
    }

    for( k=k0; k<gs.ns; k++ )
      vnormalize( &gs.buf[k] );

    k0 = gs.ns;
  }

  load_normal( gs.ns );
  normal.load( gs.ns, 1, gs.buf );

  free(a0);
  free(b0);
}

void GObj::calculate_tangent_space()
{
  int *a0, **b0;
  calculate_access( a0, b0, face_nidx, n_normal );


  GPfm face_tangent;
  calculate_face_tangent( face_tangent );
  tangent.load( n_normal, 1 );
  binormal.load( n_normal, 1 );

  int i, j, l;
  FLOAT3 xn, yn, zn;
  for( j=0; j<n_normal; j++ )
  {
    yn = normal.fm[j];
    
    l = a0[j];
    zn = 0;
    for( i=0; i<l; i++ )
      zn += face_tangent.fm[ b0[j][i]/3 ];
    
    zn =  zn - vdot(zn,yn)*yn;
    if( zn.norm() < .00001 )
    {
      zn = face_tangent.fm[ b0[j][0]/3 ];
      zn = zn - vdot(zn,yn)*yn;
    }

    zn = vnormalize( zn );
    xn = vcross( yn, zn );
    
    binormal.fm[j] = xn;
    tangent.fm[j] = zn;

  }

  free(a0);
  free(b0);
}

void GObj::save_edge_info( const char *spath )
{
  FILE *f0 = fopen( spath, "wb" );
  if(f0)
  {
    fwrite( &n_edge, sizeof(int), 1, f0 );
    fwrite( &closed, sizeof(bool), 1, f0 );
    fwrite( face_eidx, sizeof(int), 3*n_face, f0 );
    fwrite( s_edge, sizeof(GEdge), n_edge, f0 );
  fclose(f0);

  }else
  {
    printf( "[Warning] : unable to open %s for writting.\n", spath );
  }
}

void GObj::load_edge_info( const char *spath )
{
  FILE *f0 = fopen( spath, "rb" );
    fread( &n_edge, sizeof(int), 1, f0 );
    fread( &closed, sizeof(bool), 1, f0 );

    face_eidx = (int*) malloc( 3*n_face * sizeof(int) );
    s_edge = (GEdge*) malloc( n_edge * sizeof(GEdge) );

    fread( face_eidx, sizeof(int), 3*n_face, f0 );
    fread( s_edge, sizeof(GEdge), n_edge, f0 );
  fclose(f0);
}

void GObj::unfold_in_normal_access()
{
  int *a0, **b0;
  int j;
  GPfm new_vertex;

  calculate_access( a0, b0, face_nidx, n_normal );
    new_vertex.load( n_normal, 1 );

  for( j=0; j<n_normal; j++ )
    new_vertex.fm[j] = vertex.fm[ face_vidx[ b0[j][0] ] ];

  load_vertex( n_normal );
  vertex.load( new_vertex.w, new_vertex.h, new_vertex.fm );
  memcpy( face_vidx, face_nidx, n_face*3*sizeof(int) );

  free(a0);
  free(b0);
}

void GObj::draw_point( int vertex_index )
{
  glBegin( GL_POINTS );
    glVertex3fv( (float*)&vertex.fm[vertex_index] );
  glEnd();
}

void GObj::draw_point_set( const GStack<int> &point_set )
{
  int i;
  glBegin( GL_POINTS );
  for( i=0; i<point_set.ns; i++ )
    glVertex3fv( (float*)&vertex.fm[ point_set.buf[i] ] );
  glEnd();
}

void GObj::draw_edge( int eidx )
{
  glBegin( GL_LINES );
    glVertex3fv( (float*)&vertex.fm[s_edge[eidx].v0] );
    glVertex3fv( (float*)&vertex.fm[s_edge[eidx].v1] );
  glEnd();
}

void GObj::draw_face( int tidx )
{
  glBegin( GL_TRIANGLES );
    glVertex3fv( (float*)&vertex.fm[ face_vidx[3*tidx+0] ] );
    glVertex3fv( (float*)&vertex.fm[ face_vidx[3*tidx+1] ] );
    glVertex3fv( (float*)&vertex.fm[ face_vidx[3*tidx+2] ] );
  glEnd();
}

/*
void GObj::force_single_sided()
{
  int orginal_n_face = n_face;
  int i, j, k;

  weld();
  calculate_edge_info();

  bool *edone = (bool*) malloc( n_edge * sizeof(bool) );
  bool *tdone = (bool*) malloc( n_face * sizeof(bool) );
  memset( edone, 0, n_edge * sizeof(bool) );
  memset( tdone, 0, n_face * sizeof(bool) );

  GPfm nmap;
    nmap.load( 3 * n_face * 2, 1 );
    nmap.draw( normal, 0,0 );

  int n_nmap;
    n_nmap = n_normal;

  FLOAT3 fn, vn;
  int eidx;
  int t0, t1;
  int e;
  int *pist, *_pist, *qist;
  int *eist, n_eist;
  int *tist, n_tist;
    _pist = (int*) malloc( n_edge * sizeof(int) );
    eist = (int*) malloc( n_edge * sizeof(int) );  n_eist = 0;
    tist = (int*) malloc( n_face * sizeof(int) );  n_tist = 0;

  for( j=0; j<n_edge; j++ )
  {
    if( !edone[j] )
    {
      pist = _pist;
      qist = _pist;
      n_eist = 0;
      n_tist = 0;

      edone[j] = true;
      *pist++ = j;  

      while( qist != pist )
      {
        eidx = *qist++;
        t0 = s_edge[eidx].t0;
        t1 = s_edge[eidx].t1;
        
        if( t0!=-1 )
        {
          for( k=0; k<3; k++ )
          {
            e = face_eidx[3*t0+k];
            if( !edone[e] )
            {
              edone[e] = true;
              *pist++ = e;  
            }
          }
          if( !tdone[t0] )
          {
            tdone[t0] = true;
            tist[n_tist++] = t0;
          }
        }else
        {
          eist[n_eist++] = eidx;
        }

        if( t1!=-1 )
        {
          for( k=0; k<3; k++ )
          {
            e = face_eidx[3*t1+k];
            if( !edone[e] )
            {
              edone[e] = true;
              *pist++ = e;  
            }
          }
          if( !tdone[t1] )
          {
            tdone[t1] = true;
            tist[n_tist++] = t1;
            fn  = get_face_normal( t1 );
          }
        }else
        {
          eist[n_eist++] = eidx;
        }
      }

      if( n_eist == 0 )
      {
        //printf( "close part found\n" );
      }
      if( n_eist )
      {
        face_vidx = (int*)realloc( face_vidx, (n_face+n_tist+2*n_eist) * 3*sizeof(int) );
        face_nidx = (int*)realloc( face_nidx, (n_face+n_tist+2*n_eist) * 3*sizeof(int) );
        face_tidx = (int*)realloc( face_tidx, (n_face+n_tist+2*n_eist) * 3*sizeof(int) );
        for( i=0; i<n_tist; i++ )
        {
          t0 = tist[i];
        
          face_vidx[ 3*n_face+0 ] = face_vidx[3*t0+0];
          face_vidx[ 3*n_face+1 ] = face_vidx[3*t0+2];
          face_vidx[ 3*n_face+2 ] = face_vidx[3*t0+1];

          if( default_group.type & GOBJ_TEXCOR )
          {
            face_tidx[ 3*n_face+0 ] = face_tidx[3*t0+0];
            face_tidx[ 3*n_face+1 ] = face_tidx[3*t0+2];
            face_tidx[ 3*n_face+2 ] = face_tidx[3*t0+1];
          }

          if( default_group.type & GOBJ_NORMAL )
          {
            nmap.fm[n_nmap+0] = -normal.fm[ face_nidx[3*t0+0] ];
            nmap.fm[n_nmap+1] = -normal.fm[ face_nidx[3*t0+2] ];
            nmap.fm[n_nmap+2] = -normal.fm[ face_nidx[3*t0+1] ];
            face_nidx[ 3*n_face+0 ] = n_nmap+0;
            face_nidx[ 3*n_face+1 ] = n_nmap+1;
            face_nidx[ 3*n_face+2 ] = n_nmap+2;
            n_nmap += 3;
          }
          n_face = n_face + 1;
        }
      }
    }
  }

  normal.load( nmap.w, nmap.h, nmap.fm );
  n_normal = n_nmap;

  if( n_mgroup )
  {
    mgroup = (Group*) realloc( mgroup, (n_mgroup+1) * sizeof( Group ) );
    mgroup[n_mgroup].name[0] = 0;
    mgroup[n_mgroup].type = default_group.type;
    mgroup[n_mgroup].idx    = orginal_n_face;
    mgroup[n_mgroup].n_face = n_face - orginal_n_face;
    n_mgroup = n_mgroup+1;
  }

  if( n_sgroup )
  {
    sgroup = (Group*) realloc( sgroup, (n_sgroup+1) * sizeof( Group ) );
    sgroup[n_sgroup].name[0] = 0;
    sgroup[n_sgroup].type = default_group.type;
    sgroup[n_sgroup].idx    = orginal_n_face;
    sgroup[n_sgroup].n_face = n_face - orginal_n_face;
    n_sgroup = n_sgroup+1;
  }
  n_mgroup = 0;
  n_sgroup = 0;

  default_group.n_face = n_face;

  free( edone );
  free( tdone );
  free( _pist );
  free( eist );
  free( tist );
}
*/
