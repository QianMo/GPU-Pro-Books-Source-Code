/*
 * libxml.h: internal header only used during the compilation of libxml
 *
 * See COPYRIGHT for the status of this software
 *
 * Author: breese@users.sourceforge.net
 */

#ifndef _FBXSDK__XML_LIBXML_H__
#define _FBXSDK__XML_LIBXML_H__

#ifndef NO_LARGEFILE_SOURCE
#ifndef _LARGEFILE_SOURCE
#define _LARGEFILE_SOURCE
#endif
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif
#endif


#if defined(macintosh)
    #define HAVE_CTYPE_H
    #define HAVE_MATH_H
#else
    #include <libxml/config.h>
#endif

#include <libxml/xmlversion.h>

// cheat to compile without warnings in 64 bits
#include <karch/arch.h>
#ifdef KARCH_ARCH_64
    #define _XMLLONG_ long long
#else
    #define _XMLLONG_ long
#endif

#ifndef WITH_TRIO
#include <stdio.h>
#else
/**
 * TRIO_REPLACE_STDIO:
 *
 * This macro is defined if teh trio string formatting functions are to
 * be used instead of the default stdio ones.
 */
#define TRIO_REPLACE_STDIO
#include "trio.h"
#endif

#include <libxml_nsbegin.h>

/*
 * Internal variable indicating if a callback has been registered for
 * node creation/destruction. It avoids spending a lot of time in locking
 * function while checking if the callback exists.
 */
extern int __xmlRegisterCallbacks;
/*
 * internal error reporting routines, shared but not partof the API.
 */
void __xmlIOErr(int domain, int code, const char *extra);
void __xmlLoaderErr(void *ctx, const char *msg, const char *filename);

#include <libxml_nsend.h>

#endif /* ! _FBXSDK__XML_LIBXML_H__ */
