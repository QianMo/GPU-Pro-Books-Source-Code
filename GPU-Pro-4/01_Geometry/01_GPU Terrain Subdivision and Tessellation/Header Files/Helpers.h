//-------------------------------------------------------------------------------------------------
// File: Helpers.h
// Author: Ben Mistal
// Copyright 2010-2012 Mistal Research, Inc.
//-------------------------------------------------------------------------------------------------
#ifndef PI
#define PI 3.141592654f
#endif

#define Radians( x ) ( x / 180.0f * PI )

#define SAFE_RELEASE( pRef ) if ( pRef ) { pRef->Release(); pRef = NULL; }
#define SAFE_DELETE( pMem ) if ( pMem ) { delete pMem; pMem = NULL; }
#define SAFE_FREE( pMem ) if ( pMem ) { free( pMem ); pMem = NULL; }

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif
