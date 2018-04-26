/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#ifndef _DEFINES_H_
#define _DEFINES_H_

typedef unsigned int    uint;
typedef unsigned char   uchar;

static const int STR_LENGTH_MAX = 255;

#define  NUM_ELEMS(ARRAY) (sizeof(ARRAY)/(sizeof(ARRAY[0])))

// not so nice macro to generate descriptions of the lists
#define  MAKE_LIST(ARRAY)  (LIST_DESC(ARRAY, NUM_ELEMS(ARRAY)))

// stores array of item names and its size
struct LIST_DESC {
   LIST_DESC(const char** newItems, int newSize) : itemNames(newItems), itemsSize(newSize) { }

   const char**   itemNames;
   int            itemsSize;
};

// no fancy math code here - only data (as everything's been calculated at the export stage)
union MATRIX_STORE 
{   
   float x[16];
   float components[4][4];
};

// used to transfer generated geometry between different subsystems
struct GEOMETRY_DESC {     
   unsigned int   positions, normals;
   unsigned int   numElemets;
};

// The number of threads to use for triangle generation (limited by shared memory size)
#define NTHREADS 32


#endif
