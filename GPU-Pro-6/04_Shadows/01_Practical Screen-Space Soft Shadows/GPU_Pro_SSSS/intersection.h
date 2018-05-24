#ifndef intersection_h
#define intersection_h

#include "mymath/mymath.h"
#include <vector>

//based on the fast, constant time multi-dispatcher design pattern
template<class lhs, 
class rhs = lhs, 
class ret = void,
class cbk = ret (*)(lhs, rhs)>
class dispatcher
{
  typedef std::vector<cbk> matrix;
  matrix callbacks;
  int elements;
public:
  dispatcher() : elements(0){}

  void set_elements( int num )
  {
    elements = num;
    callbacks.resize( num * num );

    for( int c = 0; c < num * num; ++c )
      callbacks[c] = 0;
  }

  template<class _lhs, class _rhs>
  void add(cbk func)
  {
    int idx_lhs = _lhs::get_class_idx();
    int idx_rhs = _rhs::get_class_idx();

    callbacks[idx_lhs * elements + idx_rhs] = func;
  }

  ret go(lhs _lhs, rhs _rhs)
  {
    int idx_lhs = _lhs->get_class_index();
    int idx_rhs = _rhs->get_class_index();

    assert( idx_lhs >= 0 || idx_rhs >=  0 || idx_lhs < elements || idx_rhs < elements );
    assert( callbacks[idx_lhs * elements + idx_rhs] != 0 );

    return callbacks[idx_lhs * elements + idx_rhs](_lhs, _rhs);
  }
};

//forward declarations
class sphere;
class plane;
class aabb;
class frustum;
class ray;
class triangle;

//generic abstract shape class
//needed so that any shape can intersect any other shape
class shape
{
  static dispatcher<shape*, shape*, bool> _is_on_right_side;
  static dispatcher<shape*, shape*, bool> _is_inside;
  static dispatcher<shape*, shape*, bool> _intersects;
  static bool is_setup;
public:
  static void set_up_intersection();

  bool is_on_right_side(shape* s)
  {
    assert(is_setup);
    return _is_on_right_side.go(this, s);
  }

  bool is_inside(shape* s)
  {
    assert(is_setup);
    return _is_inside.go(this, s);
  }

  bool intersects(shape* s)
  {
    assert(is_setup);
    return _intersects.go(this, s);
  }

  virtual int get_class_index() = 0;
};

dispatcher<shape*, shape*, bool> shape::_is_on_right_side;
dispatcher<shape*, shape*, bool> shape::_is_inside;
dispatcher<shape*, shape*, bool> shape::_intersects;
bool shape::is_setup = false;

class ray : public shape
{
  public:
    //define a ray by origin and direction
    mm::vec3 origin, direction;
    
    static int get_class_idx()
    {
      static int idx = 0;
      return idx;
    }
    
    int get_class_index()
    {
      return get_class_idx();
    }
    
    ray( mm::vec3 o = mm::vec3(), mm::vec3 d = mm::vec3(0, 0, -1) ) : origin( o ), direction( d ) {}
};

class triangle : public shape
{
  public:
    //define a triangle by three points
    mm::vec3 k, l, m;
    
    static int get_class_idx()
    {
      static int idx = 1;
      return idx;
    }
    
    int get_class_index()
    {
      return get_class_idx();
    }
    
    triangle( mm::vec3 kk = mm::vec3(), mm::vec3 ll = mm::vec3(), mm::vec3 mm = mm::vec3() ) : k( kk ), l( ll ), m( mm ) {}
};

class sphere : public shape
{
public:
  //define a sphere by a center and a radius
  mm::vec3 center;
  float radius;

  static int get_class_idx()
  {
    static int idx = 2;
    return idx;
  }

  int get_class_index()
  {
    return get_class_idx();
  }

  sphere( mm::vec3 c = mm::vec3(), float r = float() ) : center( c ), radius( r ) {}
};

class plane : public shape
{
public:
  //define a plane by a normal and a point
  mm::vec3 normal, point;
  float d; //cache -(normal dot point)

  static int get_class_idx()
  {
    static int idx = 3;
    return idx;
  }

  int get_class_index()
  {
    return get_class_idx();
  }

  //define a plane by 3 points
  void set_up( const mm::vec3& a, const mm::vec3& b, const mm::vec3& c )
  {
    mm::vec3 tmp1, tmp2;

    tmp1 = a - b;
    tmp2 = c - b;

    normal = mm::normalize( mm::cross( tmp2, tmp1 ) );
    point = a;
    d = -mm::dot( normal, point );
  }

  //signed distance
  float distance( const mm::vec3& p )
  {
    return d + mm::dot( normal, p );
  }

  //define a plane by a normal and a point
  plane( const mm::vec3& n = mm::vec3(), const mm::vec3& p = mm::vec3() ) : normal( n ), point( p )
  {
    d = -mm::dot( n, p );
  }

  plane( const mm::vec3& a, const mm::vec3& b, const mm::vec3& c )
  {
    set_up( a, b, c );
  }
};

class aabb : public shape
{
public:
  mm::vec3 pos; //center of the aabb
  mm::vec3 extents; //half-width/height of the aabb
  mm::vec3 min, max; //minimum/maximum apex of the aabb

  static int get_class_idx()
  {
    static int idx = 4;
    return idx;
  }

  int get_class_index()
  {
    return get_class_idx();
  }

  //returns the vertices of the triangles of the aabb in counter clockwise order
  std::vector<mm::vec3> get_vertices()
  {
    std::vector<mm::vec3> v;

    //left
    v.push_back( mm::vec3( min.x, max.yz ) );
    v.push_back( mm::vec3( min.x, max.y, min.z ) );
    v.push_back( mm::vec3( min.xyz ) );

    v.push_back( mm::vec3( min.xyz ) );
    v.push_back( mm::vec3( min.xy, max.z ) );
    v.push_back( mm::vec3( min.x, max.yz ) );

    //front
    v.push_back( mm::vec3( min.xy, max.z ) );
    v.push_back( mm::vec3( max.x, min.y, max.z ) );
    v.push_back( mm::vec3( max.xyz ) );

    v.push_back( mm::vec3( max.xyz ) );
    v.push_back( mm::vec3( min.x, max.yz ) );
    v.push_back( mm::vec3( min.xy, max.z ) );

    //right
    v.push_back( mm::vec3( max.xy, min.z ) );
    v.push_back( mm::vec3( max.xyz ) );
    v.push_back( mm::vec3( max.x, min.y, max.z ) );

    v.push_back( mm::vec3( max.x, min.y, max.z ) );
    v.push_back( mm::vec3( max.x, min.yz ) );
    v.push_back( mm::vec3( max.xy, min.z ) );

    //back
    v.push_back( mm::vec3( max.xy, min.z ) );
    v.push_back( mm::vec3( max.x, min.yz ) );
    v.push_back( mm::vec3( min.xyz ) );

    v.push_back( mm::vec3( min.xyz ) );
    v.push_back( mm::vec3( min.x, max.y, min.z ) );
    v.push_back( mm::vec3( max.xy, min.z ) );

    //top
    v.push_back( mm::vec3( min.x, max.y, min.z ) );
    v.push_back( mm::vec3( min.x, max.yz ) );
    v.push_back( mm::vec3( max.xyz ) );

    v.push_back( mm::vec3( max.xyz ) );
    v.push_back( mm::vec3( max.xy, min.z ) );
    v.push_back( mm::vec3( min.x, max.y, min.z ) );

    //bottom
    v.push_back( mm::vec3( max.x, min.y, max.z ) );
    v.push_back( mm::vec3( min.xy, max.z ) );
    v.push_back( mm::vec3( min.xyz ) );

    v.push_back( mm::vec3( min.xyz ) );
    v.push_back( mm::vec3( max.x, min.yz ) );
    v.push_back( mm::vec3( max.x, min.y, max.z ) );

    return v;
  }

  mm::vec3 get_pos_vertex( const mm::vec3& n )
  {
    mm::vec3 res = min;

    if( n.x >= 0 )
      res.x = max.x;

    if( n.y >= 0 )
      res.y = max.y;

    if( n.z >= 0 )
      res.z = max.z;

    return res;
  }

  void expand( const mm::vec3& p )
  {
    min = mm::min( min, p );
    max = mm::max( max, p );
    extents = abs( max - min ) / 2.0f;
    pos = min + extents;
  }

  mm::vec3 get_neg_vertex( const mm::vec3& n )
  {
    mm::vec3 res = max;

    if( n.x >= 0 )
      res.x = min.x;

    if( n.y >= 0 )
      res.y = min.y;

    if( n.z >= 0 )
      res.z = min.z;

    return res;
  }

  void reset_minmax()
  {
    min = mm::vec3( FLT_MAX );
    max = mm::vec3( -FLT_MAX );
  }

  aabb()
  {
    reset_minmax();
  }

  aabb( const mm::vec3& p, const mm::vec3& e ) : pos( p ), extents( e )
  {
    min = pos - extents;
    max = pos + extents;
  }
};

//haxx
#ifdef _WIN32
#undef FAR
#endif

class frustum : public shape
{
public:
  shape* planes[6];
  vec3 points[8];

  enum which_plane
  {
    TOP = 0, BOTTOM, LEFT, RIGHT, NEAR, FAR
  };

  enum which_point
  {
    NTL = 0, NTR, NBL, NBR, FTL, FTR, FBL, FBR
  };

  static int get_class_idx()
  {
    static int idx = 5;
    return idx;
  }

  int get_class_index()
  {
    return get_class_idx();
  }

  frustum()
  {
    for( int c = 0; c < 6; ++c )
      planes[c] = new plane();
  }

  void set_up( const mm::camera<float>& cam )
  {
    mm::vec3 nc = cam.pos - cam.view_dir * cam.get_frame()->near_ll.z;
    mm::vec3 fc = cam.pos - cam.view_dir * cam.get_frame()->far_ll.z;

    mm::vec3 right = -mm::normalize( mm::cross( cam.up_vector, cam.view_dir ) );

    float nw = cam.get_frame()->near_lr.x - cam.get_frame()->near_ll.x;
    float nh = cam.get_frame()->near_ul.y - cam.get_frame()->near_ll.y;

    float fw = cam.get_frame()->far_lr.x - cam.get_frame()->far_ll.x;
    float fh = cam.get_frame()->far_ul.y - cam.get_frame()->far_ll.y;

    //near top left
    mm::vec3 ntl = nc + cam.up_vector * nh - right * nw;
    mm::vec3 ntr = nc + cam.up_vector * nh + right * nw;
    mm::vec3 nbl = nc - cam.up_vector * nh - right * nw;
    mm::vec3 nbr = nc - cam.up_vector * nh + right * nw;

    mm::vec3 ftl = fc + cam.up_vector * fh - right * fw;
    mm::vec3 ftr = fc + cam.up_vector * fh + right * fw;
    mm::vec3 fbl = fc - cam.up_vector * fh - right * fw;
    mm::vec3 fbr = fc - cam.up_vector * fh + right * fw;

    points[NTL] = ntl;
    points[NTR] = ntr;
    points[NBL] = nbl;
    points[NBR] = nbr;

    points[FTL] = ftl;
    points[FTR] = ftr;
    points[FBL] = fbl;
    points[FBR] = fbr;

    static_cast<plane*>( planes[TOP] )->set_up( ntr, ntl, ftl );
    static_cast<plane*>( planes[BOTTOM] )->set_up( nbl, nbr, fbr );
    static_cast<plane*>( planes[LEFT] )->set_up( ntl, nbl, fbl );
    static_cast<plane*>( planes[RIGHT] )->set_up( nbr, ntr, fbr );
    static_cast<plane*>( planes[NEAR] )->set_up( ntl, ntr, nbr );
    static_cast<plane*>( planes[FAR] )->set_up( ftr, ftl, fbl );
  }

  std::vector<mm::vec3> get_vertices()
  {
	std::vector<mm::vec3> v;

    //top
    v.push_back( points[NTL] );
    v.push_back( points[NTR] );
    v.push_back( points[FTR] );

    v.push_back( points[NTL] );
    v.push_back( points[FTR] );
    v.push_back( points[FTL] );

    //bottom
    v.push_back( points[NBL] );
    v.push_back( points[FBL] );
    v.push_back( points[FBR] );

    v.push_back( points[NBL] );
    v.push_back( points[FBR] );
    v.push_back( points[NBR] );

	//left
    v.push_back( points[NTL] );
    v.push_back( points[FTL] );
    v.push_back( points[NBL] );

    v.push_back( points[NBL] );
    v.push_back( points[FTL] );
    v.push_back( points[FBL] );
    
    //right
    v.push_back( points[NTR] );
    v.push_back( points[NBR] );
    v.push_back( points[FTR] );

    v.push_back( points[NBR] );
    v.push_back( points[FBR] );
    v.push_back( points[FTR] );

    //near
    v.push_back( points[NBL] );
    v.push_back( points[NTR] );
    v.push_back( points[NTL] );

    v.push_back( points[NBL] );
    v.push_back( points[NBR] );
    v.push_back( points[NTR] );

    //far
    v.push_back( points[FTL] );
    v.push_back( points[FTR] );
    v.push_back( points[FBR] );

    v.push_back( points[FTR] );
    v.push_back( points[FBR] );
    v.push_back( points[FBL] );

	return v;
  }
};

namespace inner
{
  //only tells if the sphere is on the right side of the plane!
  static bool is_on_right_side_sp( shape* aa, shape* bb )
  {
    auto a = static_cast<sphere*>(aa);
    auto b = static_cast<plane*>(bb);

    float dist = b->distance( a->center );
    //dist + radius == how far is the sphere from the plane (usable when we want to do lod using the near plane)

    if( dist < -a->radius )
      return false;

    return true;
  }

  static bool is_on_right_side_ps( shape* aa, shape* bb )
  {
    return is_on_right_side_sp(bb, aa);
  }

  //only tells if the sphere is on the right side of the plane!
  static bool is_on_right_side_ap( shape* aa, shape* bb )
  {
    auto a = static_cast<aabb*>(aa);
    auto b = static_cast<plane*>(bb);

    if( b->distance( a->get_pos_vertex( b->normal ) ) < 0 )
      return false;

    return true;
  }

  static bool is_on_right_side_pa( shape* aa, shape* bb )
  {
    return is_on_right_side_ap(bb, aa);
  }

  static bool intersect_ss( shape* aa, shape* bb )
  {
    auto a = static_cast<sphere*>(aa);
    auto b = static_cast<sphere*>(bb);

    mm::vec3 diff = a->center - b->center;
    float dist = mm::dot( diff, diff );

    float rad_sum = b->radius + a->radius;

    if( dist > rad_sum * rad_sum ) //squared distance check
      return false;

    return true;
  }

  //checks if a sphere intersects a plane
  static bool intersect_sp( shape* aa, shape* bb )
  {
    auto a = static_cast<sphere*>(aa);
    auto b = static_cast<plane*>(bb);

    float dist = b->distance( a->center );

    if( abs( dist ) <= a->radius )
      return true;

    return false;
  }

  static bool intersect_ps( shape* aa, shape* bb )
  {
    return intersect_sp(bb, aa);
  }

  static bool intersect_pp( shape* aa, shape* bb )
  {
    auto a = static_cast<plane*>(aa);
    auto b = static_cast<plane*>(bb);

    mm::vec3 vector = mm::cross( a->normal, b->normal );

    //if the cross product yields a null vector
    //then the angle is either 0 or 180
    //that is the two normals are parallel
    // sin(alpha) = 0
    // ==> |a| * |b| * sin(alpha) = 0
    if( mm::equal( vector, mm::vec3( 0 ) ) )
      return false;

    return true;
  }

  static bool intersect_aa( shape* aa, shape* bb )
  {
    auto a = static_cast<aabb*>(aa);
    auto b = static_cast<aabb*>(bb);

    mm::vec3 t = b->pos - a->pos;

    if( abs( t.x ) > ( a->extents.x + b->extents.x ) )
      return false;

    if( abs( t.y ) > ( a->extents.y + b->extents.y ) )
      return false;

    if( abs( t.z ) > ( a->extents.z + b->extents.z ) )
      return false;

    return true;
  }

  static bool intersect_as( shape* aa, shape* bb )
  {
    auto a = static_cast<aabb*>(aa);
    auto b = static_cast<sphere*>(bb);

    //square distance check between spheres and aabbs
    mm::vec3 vec = b->center - mm::clamp( a->pos + (b->center - a->pos), a->min, a->max );

    float sqlength = mm::dot( vec, vec );

    if( sqlength > b->radius * b->radius )
      return false;

    return true;
  }

  static bool intersect_sa( shape* aa, shape* bb )
  {
    return intersect_as(bb, aa);
  }

  static bool intersect_ap( shape* aa, shape* bb )
  {
    auto a = static_cast<aabb*>(aa);
    auto b = static_cast<plane*>(bb);

    mm::vec3 p = a->get_pos_vertex( b->normal );
    mm::vec3 n = a->get_neg_vertex( b->normal );

    float dist_p = b->distance( p );
    float dist_n = b->distance( n );

    if( ( dist_n > 0 && dist_p > 0 ) ||
      ( dist_n < 0 && dist_p < 0 ) )
      return false;

    return true;
  }

  static bool intersect_pa( shape* aa, shape* bb )
  {
    return intersect_ap(bb, aa);
  }

  static bool intersect_fs( shape* aa, shape* bb )
  {
    auto a = static_cast<frustum*>(aa);

    for( int c = 0; c < 6; ++c )
    {
      if( !is_on_right_side_ps( a->planes[c], bb ) )
        return false;
    }

    return true;
  }

  static bool intersect_sf( shape* aa, shape* bb )
  {
    return intersect_fs(bb, aa);
  }

  static bool intersect_fa( shape* aa, shape* bb )
  {
    auto a = static_cast<frustum*>(aa);
    auto b = static_cast<aabb*>(bb);

    bool res = true;
    for( int c = 0; c < 6; ++c )
    {
      if( !bb->is_on_right_side(a->planes[c]) )
      {
        res = false;
        break;
      }
    }
    return res;
  }

  static bool intersect_af( shape* aa, shape* bb )
  {
    return intersect_fa(bb, aa);
  }
  
  template <typename T>
  static void sort3(T& t1, T& t2, T& t3)
  {
    if (t1 > t2) {
      T tmp = t2;
      t2 = t1;
      t1 = tmp;
    }
    
    if (t2 > t3) {
      T tmp = t3;
      t3 = t2;
      t2 = tmp;
    }
    
    if (t1 > t2) {
      T tmp = t2;
      t2 = t1;
      t1 = tmp;
    }
  }

  //TODO test this
  static bool intersect_ff( shape* aa, shape* bb )
  {
    auto a = static_cast<frustum*>(aa);
    auto b = static_cast<frustum*>(bb);

    /* PLANES-VERTICES CHECK (uncomment if it speeds up this function) */
	// this code does an early out, if one of frustum A's planes has all of
	// frustum B's vertices on the "wrong" side (sufficient criteria for non-intersection)
    /*bool res = true;
    for( int c = 0; c < 6; ++c )
    {
	  bool res_pt = true;
	  for( int d = 0; d < 8; ++d ) {
		res_pt = res_pt && (static_cast<plane*>(a->planes[c])->distance(b->points[d]) >= 0);
	    if( !res_pt )
        {
          break;
        }
	  }
	  if( res_pt )
	  {
		  return false;
	  }
    }*/
	/* END OF PLANES-VERTICES CHECK */

	auto tris_a = a->get_vertices();
	auto tris_b = b->get_vertices();

	for( int c = 0; c < 6; ++c )
    {
	  for( int d = 0; d < 6; ++d )
      {
		auto plane_a = static_cast<plane*>(a->planes[c]);
		auto plane_b = static_cast<plane*>(b->planes[d]);

		// Compute direction of line common to both planes (0 == coplanar)
		auto intersection_ray_dir_nonorm = mm::cross( plane_a->normal, plane_b->normal );
		
		// Early out if one triangle is on a single side of the other's plane
		mm::vec3 side_a( plane_a->distance( tris_b[3*d+0] ),
						 plane_a->distance( tris_b[3*d+1] ),
						 plane_a->distance( tris_b[3*d+2] ) );
		
		// Early out if the triangles are in different parallel planes
		if ( mm::equal( intersection_ray_dir_nonorm, mm::vec3( 0 ) )
		     && ( fabs( side_a.x ) > 0.0001f ) )
		{
		  continue;
		}
		
		// Early out if a triangle is on one side of the other's plane
		if( mm::greaterThanEqual(side_a, mm::vec3( 0 ))
		    || mm::lessThanEqual(side_a, mm::vec3( 0 )) )
		{
		  continue;
		}
		
		mm::vec3 side_b( plane_b->distance( tris_a[3*d+0] ),
						 plane_b->distance( tris_a[3*d+1] ),
						 plane_b->distance( tris_a[3*d+2] ) );
		
		// Check the other way around too (the previous check does not catch all cases)
		if( mm::greaterThanEqual(side_b, mm::vec3( 0 ))
		    || mm::lessThanEqual(side_b, mm::vec3( 0 )) )
		{
		  continue;
		}
		
		// At this point we know that there's a line that crosses both triangles
		auto intersection_ray_dir = mm::normalize( intersection_ray_dir_nonorm );
        
        float dv01 = mm::dot( tris_a[3*d+0], intersection_ray_dir );
        float dv02 = mm::dot( tris_a[3*d+1], intersection_ray_dir );
        float dv03 = mm::dot( tris_a[3*d+2], intersection_ray_dir );
        
        float dv11 = mm::dot( tris_b[3*d+0], intersection_ray_dir );
        float dv12 = mm::dot( tris_b[3*d+1], intersection_ray_dir );
        float dv13 = mm::dot( tris_b[3*d+2], intersection_ray_dir );
        
        sort3(dv01, dv02, dv03);
        sort3(dv11, dv12, dv13);
        
        if( (dv01 + dv02) <= (dv12 + dv13) && (dv02 + dv03) >= (dv11 + dv12) )
        {
          return true;
        }
        
	  }
	}

    return false;
  }

  //is a inside b?
  static bool is_inside_sa( shape* aa, shape* bb )
  {
    auto a = static_cast<sphere*>(aa);
    auto b = static_cast<aabb*>(bb);

    mm::vec3 spheremax = a->center + a->radius;
    mm::vec3 spheremin = a->center - a->radius;

    if( mm::lessThanEqual(spheremax, b->max) && mm::greaterThanEqual(spheremin, b->min) )
      return true;

    return false;
  }

  //is a inside b?
  static bool is_inside_as( shape* aa, shape* bb )
  {
    auto a = static_cast<aabb*>(aa);
    auto b = static_cast<sphere*>(bb);

    mm::vec3 minvec = a->min - b->center;
    mm::vec3 maxvec = a->max - b->center;
    float sqrmaxlength = mm::dot(maxvec, maxvec);
    float sqrminlength = mm::dot(minvec, minvec);
    float sqrradius = b->radius * b->radius;

    if( sqrmaxlength <= sqrradius && sqrminlength <= sqrradius )
      return true;

    return false;
  }

  //is a inside b?
  static bool is_inside_aa( shape* aa, shape* bb )
  {
    auto a = static_cast<aabb*>(aa);
    auto b = static_cast<aabb*>(bb);

    if( mm::greaterThanEqual( a->min, b->min ) && mm::lessThanEqual( a->max, b->max ) )
      return true;

    return false;
  }

  //is a inside b?
  static bool is_inside_ss( shape* aa, shape* bb )
  {
    auto a = static_cast<sphere*>(aa);
    auto b = static_cast<sphere*>(bb);

    mm::vec3 spheredist = b->center - a->center;

    if( mm::dot(spheredist, spheredist) <= b->radius * b->radius )
      return true;

    return false;
  }

  static bool intersect_rt( shape* aa, shape* bb )
  {
    auto a = static_cast<ray*>(aa);
    auto b = static_cast<triangle*>(bb);

    mm::vec3 e = b->k - b->m;
    mm::vec3 f = b->l - b->m;

    mm::vec3 g = a->origin - b->m;

    //apply barycentric triangle math
    float t = 1.0f / dot( cross( a->direction, f ), e );
    vec3 tkl = t * vec3( dot( cross( g, e ), f ),
                         dot( cross( a->direction, f ), g ),
                         dot( cross( g, e ), a->direction ) );

    //barycentric coordinate check
    //if between 0...1 the point is inside
    return tkl.y > 0 && tkl.z > 0 && ( tkl.y + tkl.z ) < 1;
  }

  static bool intersect_tr( shape* aa, shape* bb )
  {
    return intersect_rt( bb, aa );
  }

  //TODO test it
  static bool intersect_rs( shape* aa, shape* bb )
  {
    auto a = static_cast<ray*>(aa);
    auto b = static_cast<sphere*>(bb);

    mm::vec3 diff = b->center - a->origin;
    float dist = dot( diff, diff ) - ( b->radius * b->radius );

    if( dist <= 0 )
      return true;

    float dist2 = dot( a->direction, diff );

    if( dist2 >= 0 )
      return false;

    return (dist2 * dist2 - dist) >= 0;
  }

  static bool intersect_sr( shape* aa, shape* bb )
  {
    return intersect_rs( bb, aa );
  }

  //TODO make it work
  static bool intersect_ra( shape* aa, shape* bb )
  {
    auto a = static_cast<ray*>(aa);
    auto b = static_cast<aabb*>(bb);

    float tmin = -FLT_MAX;
    float tmax = FLT_MAX;

    for( int c = 0; c < 3; ++c )
    {
      if( a->direction[c] != 0 )
      {
        float inv_dir = 1.0f / a->direction[c];
        float tx1 = (b->min[c] - a->origin[c]) * inv_dir;
        float tx2 = (b->max[c] - a->origin[c]) * inv_dir;
        tmin = max(tmin, min(tx1, tx2));
        tmax = min(tmax, max(tx1, tx2));

        if( tmin > tmax || tmax < 0 )
          return false;
      }
      else
      {
        if( a->origin[c] < b->min[c] && a->origin[c] > b->max[c] )
          return false;
      }
    }

    return true;
  }

  static bool intersect_ar( shape* aa, shape* bb )
  {
    return intersect_ra( bb, aa );
  }

  //TODO test it
  static bool intersect_rp( shape* aa, shape* bb )
  {
    auto a = static_cast<ray*>(aa);
    auto b = static_cast<plane*>(bb);

    float denom = dot( b->normal, a->direction );

    if( denom > 0 )
    {
      vec3 ray_to_plane = b->point - a->origin;
      float t = dot(ray_to_plane, b->normal) / denom;
      return t >= 0;
    }

    return false;
  }

  static bool intersect_pr( shape* aa, shape* bb )
  {
    return intersect_rp( bb, aa );
  }
}

void shape::set_up_intersection()
{
  //order doesnt matter
  _is_on_right_side.set_elements( 6 );
  _is_on_right_side.add<sphere, plane>(inner::is_on_right_side_sp);
  _is_on_right_side.add<aabb, plane>(inner::is_on_right_side_ap);
  _is_on_right_side.add<plane, sphere>(inner::is_on_right_side_ps);
  _is_on_right_side.add<plane, aabb>(inner::is_on_right_side_pa);

  /////////////
  _intersects.set_elements( 6 );
  _intersects.add<aabb, aabb>(inner::intersect_aa);
  _intersects.add<aabb, sphere>(inner::intersect_as);
  _intersects.add<aabb, ray>(inner::intersect_ar);
  _intersects.add<aabb, frustum>(inner::intersect_af);
  _intersects.add<aabb, plane>(inner::intersect_ap);

  _intersects.add<plane, aabb>(inner::intersect_pa);
  _intersects.add<plane, sphere>(inner::intersect_ps);
  _intersects.add<plane, ray>(inner::intersect_pr);
  _intersects.add<plane, plane>(inner::intersect_pp);

  _intersects.add<sphere, aabb>(inner::intersect_sa);
  _intersects.add<sphere, sphere>(inner::intersect_ss);
  _intersects.add<sphere, ray>(inner::intersect_sr);
  _intersects.add<sphere, frustum>(inner::intersect_sf);
  _intersects.add<sphere, plane>(inner::intersect_sp);
  
  _intersects.add<frustum, aabb>(inner::intersect_fa);
  _intersects.add<frustum, sphere>(inner::intersect_fs);
  _intersects.add<frustum, frustum>(inner::intersect_ff);
  
  _intersects.add<ray, aabb>(inner::intersect_ra);
  _intersects.add<ray, sphere>(inner::intersect_rs);
  _intersects.add<ray, triangle>(inner::intersect_rt);
  _intersects.add<ray, plane>(inner::intersect_rp);
  
  _intersects.add<triangle, ray>(inner::intersect_tr);

  //order matters
  _is_inside.set_elements( 6 );
  _is_inside.add<aabb, aabb>(inner::is_inside_aa);
  _is_inside.add<aabb, sphere>(inner::is_inside_as);
  _is_inside.add<sphere, aabb>(inner::is_inside_sa);
  _is_inside.add<sphere, sphere>(inner::is_inside_ss);

  //usage
  //is_on_the_right_side.go(new aabb(), new sphere());

  is_setup = true;
}

#endif
