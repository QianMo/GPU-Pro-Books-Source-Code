// -------------------------------------------------
/*

Sylvain Lefebvre, Samuel Hornus - 2013

This file contains the base implementation of the A-buffer techniques
described in our GPUPro paper.

NOTES:
- atomicMax 64bits is emulated through a loop and a 64 bits atomicCAS.
  This can be directly replaced by a native atomicMax64 when available.
- On Kepler and Fermi architectures a race-condition occurs sometimes 
  in loops with atomic operations. For this reason, a number of tricks 
  had to be used to  escape the race condition. All of these incur 
  performance penalty, but will hopefully become unncessary in the near 
  future. The tricks vary between Kepler and Fermi.
  Some loops are written in an unatural manner to circumvent these issues.
  Similarly, the iteration counters in loops are only used due to race
  condition issues ('iter' variables).
- Depth is currently encoded on 24 bits. This stems from the packing of
  the data in 64 bits records when using hashing.

IMPORTANT: 

This implementation has been carefully crafted to run with driver 320.49 
on both Fermi (GTX480) and Kepler (Titan). Due to issues with race 
conditions (see above) we cannot guarantee this code will run on future 
driver versions until race conditions are fully resolved in the 
driver/hardware. 

Please visit http://www.antexel.com/research/gpupro5 for updates.

*/
// -------------------------------------------------

#define MAX_DEPTH              (1u<<24u) // limited to 24 bits
#define MAX_ITER                2048     // never reached on normal operation

#ifdef Kepler
#define BARRIER memoryBarrier()
#else
#define BARRIER // barriers seem unnecessary on our GTX480 ...
                // in case of trouble put them back in place.
#endif

// -------------------------------------------------

coherent uniform uint64_t *u_Records; // all fragment records (<depth|pointer> pairs)
coherent uniform uint32_t *u_Counts;  // auxiliary counters
coherent uniform uint32_t *u_RGBA;    // RGBA color of fragments
coherent uniform uint32_t *u_Rec32;   // 32 bits version of u_Records (points to same)

// global counter for linear allocation
layout (binding = 0, offset = 0) uniform atomic_uint u_Counter;

uniform uint  u_NumRecords;
uniform uint  u_ScreenSz;
uniform uint  u_HashSz;

uniform uvec2 u_Offsets[256]; 

// --------------------------------------------

vec4 RGBA(uint32_t rgba)
{
  return vec4( float((rgba>>24u)&255u),float((rgba>>16u)&255u),float((rgba>>8u)&255u),float(rgba&255u) ) / 255.0;
}

// -------------------------------------------------

uint64_t atomicMax64(coherent uint64_t *ptr,uint64_t val)
{
  uint64_t cur = *ptr;
  bool done = (val <= cur);
  while ( ! done ) {
    uint64_t prev = cur;
    cur = atomicCompSwap( ptr, cur, val );
    if (cur == prev) {
       return cur;
       // done = true;
    } else {
       done = (val <= cur);
    }
  }
  return cur;
}

// -------------------------------------------------

#define DEPTH(a)      ( uint32_t( (uint64_t(a)>>uint64_t(32)) ) )
#define PTR(a)        ( uint32_t(a) )
#define PACK(d,p)     ( (uint64_t(d)<<uint64_t(32)) | (uint64_t(p)) )
#define EMPTY(a)      ( (a) == 0u )

// -------------------------------------------------

#define PG_SIZE  4u  /// good performance/memory compromise
#define LOCK     (1u<<31u)
#define UNLOCK   (LOCK-1u)

uint32_t allocate_paged(uint32_t pix)
{
  uint32_t next = 0;
  bool     done = false;
  // lock counter
  while ( ! done ) { /*carefully written to avoid race condition*/
    if ( (atomicOr( u_Counts+pix , LOCK) & LOCK) == 0 ) { 
      done = true;
      // update page if needed
      if ((u_Counts[ pix ] & UNLOCK) % PG_SIZE == 0) {
        // allocate new page
        uint32_t  page  = atomicCounterIncrement( u_Counter ) * PG_SIZE;
        u_Counts[ pix ] = (LOCK | page);
      }
      // get next
      next = u_Counts[ pix ] & UNLOCK;
      // increment
      u_Counts[ pix ] = (next + 1u) | LOCK;
      // make sure memory is updated before releasing the lock
      BARRIER;
      // release lock
      atomicAnd( u_Counts+pix , UNLOCK );
    }
  }
  return next;
}

// -------------------------------------------------

uint32_t allocate(uint32_t inspos,uint32_t depth)
{
#ifdef AllocNaive
  uint32_t ptr  = (u_ScreenSz * u_ScreenSz)/*skip heads*/ + atomicCounterIncrement( u_Counter );
#else
  uint32_t ptr  = (u_ScreenSz * u_ScreenSz)/*skip heads*/ + allocate_paged( uint32_t(inspos) );
#endif
  return ptr;
}

// -------------------------------------------------

bool insert_prelin_max64(uint32_t depth,uvec2 ij,uint32_t data)
{
  int iter = 0;

  uint32_t  inspos     = (ij.x + ij.y * u_ScreenSz); // start at head
  
  uint32_t  ptr        = allocate( inspos, depth );

  // init allocated record to 0
  u_Records[ ptr ] = 0;
  BARRIER; // make sure init is visible to all
  // store data
  u_RGBA   [ ptr ] = data;
  
  // pack depth and pointer
  uint64_t  record     = PACK(depth,ptr);

  // insertion loop
#ifdef Kepler
  // On Kepler this algorithm had to be rewritten as a single loop (atomicMax64 is currently
  // emulated by a loop and an atomicCAS64). This should no longer be necessary once atomicMax64
  // is natively supported (as it is in CUDA), and when race condition issues are resolved.
  uint64_t atpos = u_Records[ inspos ];
  bool     max_ok;
  bool     done = false;
  do {
    // max64
    max_ok = true; // true if the max at u_Records[ inspos ] is up to date
    if ( record > atpos ) {
      uint64_t prev = atpos;
      atpos  = atomicCompSwap( u_Records + inspos, atpos, record );
      max_ok = (atpos == prev); // check if max update succeeded
    }
    if (max_ok) {
      if (atpos == 0) {
        done = true;
      } else {
        // decide what to do
        inspos = PTR( atpos>record ? atpos:record ); // max
        record =    ( atpos<record ? atpos:record ); // min
        atpos  = u_Records[ inspos ];
      }
    }
  } while (!done 
            && iter ++ < MAX_ITER && atomicAdd(u_Counts + (ij.x + ij.y * u_ScreenSz),0) > 0 );
            //                                   always true: race condition avoidance ^^^
#else
  uint64_t atpos;
  while ( (atpos = atomicMax64( u_Records + inspos, record )) > 0 
       && iter ++ < MAX_ITER ) {
    if ( atpos > record ) {
      // record at inspos is greater, go to next
      inspos = PTR(atpos); 
    } else {
      // we inserted! update record itself
      inspos = PTR(record);
      record = atpos;
    }
  }
#endif
  
  return iter < MAX_ITER;
}

// -------------------------------------------------

bool insert_prelin_cas32( uint32_t depth,uvec2 ij,uint32_t data )
{
  uint32_t  head       = (ij.x + ij.y * u_ScreenSz); // start at head
  
  uint32_t  ptr        = allocate( head, depth );

  // store data
  u_RGBA[ ptr ]        = data;

  // cast to 32 bits pointers
  ptr    = ptr  << 1u;
  head   = head << 1u;
  
  // store depth in record
  u_Rec32 [ ptr + 0 ]  = depth;
  // u_Rec32 [ ptr + 1 ]  = (1u<<30u); ///// DEBUG 
  BARRIER; // make sure init is visible to all

  // prev on head
  uint32_t prev = head;
  uint32_t cur  = u_Rec32[ prev + 1u ];
  int iter = 0;
  while (iter ++ < MAX_ITER) {
    bool insert_here = false;
    if (cur == 0) {
      insert_here = true;
    } else if (depth > u_Rec32[ cur + 0u ]) {
      insert_here = true;
    }
    if (insert_here) {
      // attempt insertion
      u_Rec32 [ ptr + 1u ] = cur; // next of new record is cur
      BARRIER; // make sure pointer is updated before insertion
      uint32_t res = atomicCompSwap( u_Rec32 + (prev + 1u) , cur, ptr );
      if (res == cur) {
        break; // done!
      } else {
        // could not insert! retry from same place in list
        cur = res;
      }
    } else {
      // advance in list
      prev = cur;
      cur  = u_Rec32[ prev + 1u ];
    }
  }
  return iter < MAX_ITER;  
}

// -------------------------------------------------

bool insert_postlin(uint32_t depth,uvec2 ij,uint32_t data)
{
  uint64_t  inspos     = uint64_t(ij.x + ij.y * u_ScreenSz); // start at head
  uint32_t  ptr        = allocate( uint32_t(inspos), depth );

  // store data
  u_RGBA   [ ptr ] = data;
  // insert  
  uint64_t  record = PACK(depth,ptr);
  uint64_t  other  = atomicExchange( u_Records + inspos, record );
  u_Records[ ptr ] = other;

  return true;
}

// -------------------------------------------------

#define HA_MAX_DEPTH_1         ((MAX_DEPTH)-1u)
#define HA_KEY(a,depth) ( \
                            (uint32_t(a     /*&uint32_t(   255)*/       )<<uint32_t(24)) | /* age   on  8 bits, MSB */ \
                            (uint32_t(depth /*&uint32_t( MAX_DEPTH-1u)*/)                ) /* depth on 22 bits      */ \
								        )
#define HA_PACK_DATA(a,depth,data)  ( ( uint64_t(HA_KEY(a,depth)) << uint64_t(32) ) + uint64_t(data) )
#define HA_AGE(k)                    uint32_t( (uint32_t(k)>>uint32_t(24))/*  & uint32_t(      255)*/ )
#define HA_DEPTH(k)                  uint32_t( (uint32_t(k)              )    & uint32_t( HA_MAX_DEPTH_1 ) )
#define HA_INC_AGE_64                (uint64_t(1) << uint64_t(24+32))
#define HA_AGE_MASK_64               ((uint64_t(0x00ffffffu)<<uint64_t(32))+ 0xffffffffu )
#define HA_WRITE_AGE_64(K,a)         (( K & HA_AGE_MASK_64 ) + (uint64_t(a) << uint64_t(32+24)))

uint64_t Saddr(uvec2 l) { return uint64_t( l.x + l.y * u_ScreenSz ); }
uint64_t Haddr(uvec2 l) { return uint64_t( l.x + l.y * u_HashSz   ); }

// -------------------------------------------------

bool insert_postopen( uint32_t depth,uvec2 ij,uint32_t data )
{
  // limit depth
  depth             = depth & uint32_t( HA_MAX_DEPTH_1 );

  uint  age         = u_Counts[ Saddr(ij) ] + 1u;
  uint64_t key_info = HA_PACK_DATA(age,depth,data);

  int   iter        = 0;
  while (iter++ < MAX_ITER) {
    
    uvec2    l   = ( ij + u_Offsets[age] );
    uint64_t h   = Haddr( l % uvec2(u_HashSz) );
    uint64_t old = atomicMax64( u_Records + h , key_info );

    // -> decide what to do next
    if (old < key_info) {
      // key_info was inserted!
      // -> update max age table
      atomicMax( u_Counts + Saddr(ij) , age );
      if (old == 0) { 
        // -> stop on success
        break;
      } else {
        // -> evict
        uint32_t old_age = uint32_t( old >> uint64_t(32+24) );
        // recompute coordinates from offset
        ij       = uvec2( l + uvec2(u_HashSz) - u_Offsets[old_age] ) % uvec2(u_HashSz);
        // reinsert evicted key
        age      = u_Counts[ Saddr(ij) ]; // teleport age
        key_info = HA_WRITE_AGE_64( old , age );
      }
    }
    // else, failed: try next age

    age ++;
    key_info = key_info + HA_INC_AGE_64;

 }
 return iter < MAX_ITER;
}

// -------------------------------------------------

bool insert_preopen( uint32_t depth,uvec2 ij,uint32_t data )
{ 
  // limit depth
  depth             = depth & uint32_t( HA_MAX_DEPTH_1 );

  uint  age         = 1u;
  uint64_t key_info = HA_PACK_DATA(age,depth,data);

#ifdef EarlyCulling
  float accum = 0.0; // accumulated opacity
#endif

  int   iter = 0;
  while ((iter ++ < MAX_ITER) && atomicAdd(u_Counts + Saddr(ij),0) < 256 /*always true: race condition avoidance*/ ) {

    uvec2    l   = ( ij + u_Offsets[age] );
    uint64_t h   = Haddr( l % uvec2(u_HashSz) );
    
    uint64_t old = atomicMax64( u_Records + h , key_info );
    
    // -> decide what to do next
    if (old < key_info) {
      // key_info was inserted!
      // -> update max age table
      atomicMax( u_Counts + Saddr(ij) , age );
      if (old == 0) {
        // -> stop on success
        break;
      } else {
        // -> evict
        age      = uint32_t( old >> uint64_t(32+24) );
        // recompute coordinates from offset
        ij       = uvec2( (l + uvec2(u_HashSz)) - u_Offsets[age] ) % uvec2(u_HashSz);
        // reinsert evicted key
        key_info = old;
#ifdef EarlyCulling
        accum    = 0.0; // under-estimates opacity (conservative)
#endif
      }
    }
    // else, failed: try next age
#ifdef EarlyCulling
    else if (uint32_t( old >> uint64_t(32+24) ) == age) {
      // -> check whether opaque
      if (accum > 0.95) { // 5% error allowed
        break;
      }
      // accumulate opacity from traversed record
      // -> retreive color
      uint32_t rgba = uint32_t(old);
      // -> accumulate
      accum    += (1.0-accum) * float(rgba&255u)/255.0;
    }
#endif
    
    age ++;
    key_info = key_info + HA_INC_AGE_64;

 }
 return (iter < MAX_ITER);
}

// -------------------------------------------------
