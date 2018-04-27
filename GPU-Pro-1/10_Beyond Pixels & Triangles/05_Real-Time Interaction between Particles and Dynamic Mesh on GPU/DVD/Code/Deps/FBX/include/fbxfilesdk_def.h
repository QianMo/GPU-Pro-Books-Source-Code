/*!  \file fbxfilesdk_def.h
 */

/**************************************************************************************

 Copyright © 2001 - 2008 Autodesk, Inc. and/or its licensors.
 All Rights Reserved.

 The coded instructions, statements, computer programs, and/or related material 
 (collectively the "Data") in these files contain unpublished information 
 proprietary to Autodesk, Inc. and/or its licensors, which is protected by 
 Canada and United States of America federal copyright law and by international 
 treaties. 
 
 The Data may not be disclosed or distributed to third parties, in whole or in
 part, without the prior written consent of Autodesk, Inc. ("Autodesk").

 THE DATA IS PROVIDED "AS IS" AND WITHOUT WARRANTY.
 ALL WARRANTIES ARE EXPRESSLY EXCLUDED AND DISCLAIMED. AUTODESK MAKES NO
 WARRANTY OF ANY KIND WITH RESPECT TO THE DATA, EXPRESS, IMPLIED OR ARISING
 BY CUSTOM OR TRADE USAGE, AND DISCLAIMS ANY IMPLIED WARRANTIES OF TITLE, 
 NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE OR USE. 
 WITHOUT LIMITING THE FOREGOING, AUTODESK DOES NOT WARRANT THAT THE OPERATION
 OF THE DATA WILL BE UNINTERRUPTED OR ERROR FREE. 
 
 IN NO EVENT SHALL AUTODESK, ITS AFFILIATES, PARENT COMPANIES, LICENSORS
 OR SUPPLIERS ("AUTODESK GROUP") BE LIABLE FOR ANY LOSSES, DAMAGES OR EXPENSES
 OF ANY KIND (INCLUDING WITHOUT LIMITATION PUNITIVE OR MULTIPLE DAMAGES OR OTHER
 SPECIAL, DIRECT, INDIRECT, EXEMPLARY, INCIDENTAL, LOSS OF PROFITS, REVENUE
 OR DATA, COST OF COVER OR CONSEQUENTIAL LOSSES OR DAMAGES OF ANY KIND),
 HOWEVER CAUSED, AND REGARDLESS OF THE THEORY OF LIABILITY, WHETHER DERIVED
 FROM CONTRACT, TORT (INCLUDING, BUT NOT LIMITED TO, NEGLIGENCE), OR OTHERWISE,
 ARISING OUT OF OR RELATING TO THE DATA OR ITS USE OR ANY OTHER PERFORMANCE,
 WHETHER OR NOT AUTODESK HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH LOSS
 OR DAMAGE. 

**************************************************************************************/
#ifndef _FBXFILESDK_H
#define _FBXFILESDK_H

#define FBXFILESDK_NAMESPACE_USE

#if defined(__MACOS__) && !defined(__MACH__)
    // Namespaces can't be activated in CFM compilation
    // because some function signature exceed 255 characters.
    #ifdef FBXFILESDK_NAMESPACE_USE
        #undef FBXFILESDK_NAMESPACE_USE
    #endif
#endif


#ifdef FBXFILESDK_NAMESPACE_USE
    #ifdef USE_FBXPLUGIN_NAMESPACE //Defined in jamrules.jam
        #define FBXFILESDK_NAMESPACE        fbxplugins_200901

        #define _3DSFTK_NAMESPACE_USE
        #define _3DSFTK_NAMESPACE           fbxplugins_200901

        #define KBASELIB_NAMESPACE_USE
        #define KBASELIB_NAMESPACE          fbxplugins_200901

        #define KCHARACTERDEF_NAMESPACE_USE
        #define KCHARACTERDEF_NAMESPACE     fbxplugins_200901

        #define KFBXOBJECT_NAMESPACE_USE
        #define KFBXOBJECT_NAMESPACE        fbxplugins_200901

        #define KFCURVE_NAMESPACE_USE
        #define KFCURVE_NAMESPACE           fbxplugins_200901

        #define LIBTIFF_NAMESPACE_USE
        #define LIBTIFF_NAMESPACE           fbxplugins_200901

        #define LIBXML_NAMESPACE_USE
        #define LIBXML_NAMESPACE            fbxplugins_200901

        #define MINIZIP_NAMESPACE_USE
        #define MINIZIP_NAMESPACE           fbxplugins_200901

        #define ZLIB_NAMESPACE_USE
        #define ZLIB_NAMESPACE              fbxplugins_200901

        #define BASE_NAMESPACE_USE
        #define BASE_NAMESPACE              fbxplugins_200901

        #define AWCACHE_NAMESPACE_USE
        #define AWCACHE_NAMESPACE           fbxplugins_200901

        #define IFF_NAMESPACE_USE
        #define IFF_NAMESPACE               fbxplugins_200901

        #define COLLADA_NAMESPACE_USE
        #define COLLADA_NAMESPACE           fbxplugins_200901
    #else
        #define FBXFILESDK_NAMESPACE        fbxsdk_200901

        #define _3DSFTK_NAMESPACE_USE
        #define _3DSFTK_NAMESPACE           fbxsdk_200901

        #define KBASELIB_NAMESPACE_USE
        #define KBASELIB_NAMESPACE          fbxsdk_200901

        #define KCHARACTERDEF_NAMESPACE_USE
        #define KCHARACTERDEF_NAMESPACE     fbxsdk_200901

        #define KFBXOBJECT_NAMESPACE_USE
        #define KFBXOBJECT_NAMESPACE        fbxsdk_200901

        #define KFCURVE_NAMESPACE_USE
        #define KFCURVE_NAMESPACE           fbxsdk_200901

        #define LIBTIFF_NAMESPACE_USE
        #define LIBTIFF_NAMESPACE           fbxsdk_200901

        #define LIBXML_NAMESPACE_USE
        #define LIBXML_NAMESPACE            fbxsdk_200901

        #define MINIZIP_NAMESPACE_USE
        #define MINIZIP_NAMESPACE           fbxsdk_200901

        #define ZLIB_NAMESPACE_USE
        #define ZLIB_NAMESPACE              fbxsdk_200901

        #define BASE_NAMESPACE_USE
        #define BASE_NAMESPACE              fbxsdk_200901

        #define AWCACHE_NAMESPACE_USE
        #define AWCACHE_NAMESPACE           fbxsdk_200901

        #define IFF_NAMESPACE_USE
        #define IFF_NAMESPACE               fbxsdk_200901

        #define COLLADA_NAMESPACE_USE
        #define COLLADA_NAMESPACE           fbxsdk_200901
    #endif
#else
    #define FBXFILESDK_NAMESPACE
#endif


#endif // _FBXFILESDK_H
