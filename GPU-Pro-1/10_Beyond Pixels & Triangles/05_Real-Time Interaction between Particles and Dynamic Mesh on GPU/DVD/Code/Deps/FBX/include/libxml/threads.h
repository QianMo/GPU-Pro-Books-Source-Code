/**
 * Summary: interfaces for thread handling
 * Description: set of generic threading related routines
 *              should work with pthreads, Windows native or TLS threads
 *
 * Copy: See Copyright for the status of this software.
 *
 * Author: Daniel Veillard
 */

#ifndef _FBXSDK__XML_THREADS_H__
#define _FBXSDK__XML_THREADS_H__

#include <libxml/xmlversion.h>

#include <libxml_nsbegin.h>

/*
 * xmlMutex are a simple mutual exception locks.
 */
typedef struct _xmlMutex xmlMutex;
typedef xmlMutex *xmlMutexPtr;

/*
 * xmlRMutex are reentrant mutual exception locks.
 */
typedef struct _xmlRMutex xmlRMutex;
typedef xmlRMutex *xmlRMutexPtr;

// PJT: globals.h is dependent on this file. So don't make this
// file dependent on globals.h too.
// #include <libxml/globals.h>

XMLPUBFUN xmlMutexPtr XMLCALL		
			xmlNewMutex	(void);
XMLPUBFUN void XMLCALL			
			xmlMutexLock	(xmlMutexPtr tok);
XMLPUBFUN void XMLCALL			
			xmlMutexUnlock	(xmlMutexPtr tok);
XMLPUBFUN void XMLCALL			
			xmlFreeMutex	(xmlMutexPtr tok);

XMLPUBFUN xmlRMutexPtr XMLCALL		
			xmlNewRMutex	(void);
XMLPUBFUN void XMLCALL			
			xmlRMutexLock	(xmlRMutexPtr tok);
XMLPUBFUN void XMLCALL			
			xmlRMutexUnlock	(xmlRMutexPtr tok);
XMLPUBFUN void XMLCALL			
			xmlFreeRMutex	(xmlRMutexPtr tok);

/*
 * Library wide APIs.
 */
XMLPUBFUN void XMLCALL			
			xmlInitThreads	(void);
XMLPUBFUN void XMLCALL			
			xmlLockLibrary	(void);
XMLPUBFUN void XMLCALL			
			xmlUnlockLibrary(void);
XMLPUBFUN int XMLCALL			
			xmlGetThreadId	(void);
XMLPUBFUN int XMLCALL			
			xmlIsMainThread	(void);
XMLPUBFUN void XMLCALL			
			xmlCleanupThreads(void);
XMLPUBFUN xmlGlobalStatePtr XMLCALL	
			xmlGetGlobalState(void);

#include <libxml_nsend.h>

#endif /* _FBXSDK__XML_THREADS_H__ */
