////////////////////////////////////////////////////////////////////////
//
//	Note : this file is included as part of the Smaller Animals Software
//	JpegFile package. Though this file has not been modified from it's 
//	original IJG 6a form, it is not the responsibility on the Independent
//	JPEG Group to answer questions regarding this code.
//	
//	Any questions you have about this code should be addressed to :
//
//	CHRISDL@PAGESZ.NET	- the distributor of this package.
//
//	Remember, by including this code in the JpegFile package, Smaller 
//	Animals Software assumes all responsibilities for answering questions
//	about it. If we (SA Software) can't answer your questions ourselves, we 
//	will direct you to people who can.
//
//	Thanks, CDL.
//
////////////////////////////////////////////////////////////////////////
/*
 * jchuff.h
 *
 * Copyright (C) 1991-1996, Thomas G. Lane.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * This file contains declarations for Huffman entropy encoding routines
 * that are shared between the sequential encoder (jchuff.c) and the
 * progressive encoder (jcphuff.c).  No other modules need to see these.
 */

/* Derived data constructed for each Huffman table */

typedef struct {
  unsigned int ehufco[256];	/* code for each symbol */
  char ehufsi[256];		/* length of code for each symbol */
  /* If no code has been allocated for a symbol S, ehufsi[S] contains 0 */
} c_derived_tbl;

/* Short forms of external names for systems with brain-damaged linkers. */

#ifdef NEED_SHORT_EXTERNAL_NAMES
#define jpeg_make_c_derived_tbl	jMkCDerived
#define jpeg_gen_optimal_table	jGenOptTbl
#endif /* NEED_SHORT_EXTERNAL_NAMES */

/* Expand a Huffman table definition into the derived format */
EXTERN(void) jpeg_make_c_derived_tbl
	JPP((j_compress_ptr cinfo, JHUFF_TBL * htbl, c_derived_tbl ** pdtbl));

/* Generate an optimal table definition given the specified counts */
EXTERN(void) jpeg_gen_optimal_table
	JPP((j_compress_ptr cinfo, JHUFF_TBL * htbl, long freq[]));
