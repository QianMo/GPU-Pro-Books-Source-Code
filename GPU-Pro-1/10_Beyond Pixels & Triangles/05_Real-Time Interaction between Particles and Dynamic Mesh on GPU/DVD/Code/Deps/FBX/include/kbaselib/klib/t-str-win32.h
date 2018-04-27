/*!  \file t-str-win32.h
 */
 
#ifndef _FBXSDK_T_STR_WIN32_H_
#define _FBXSDK_T_STR_WIN32_H_

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

#include <kaydara.h>

#ifdef K_UNICODE

    #include <wchar.h>
    #include <stdlib.h>
    #include <stdarg.h>
	#include <io.h>

    #define t_strcpy   wcscpy
    #define t_strncpy  wcsncpy
    #define t_strcat   wcscat
    #define t_strcmp   wcscmp
    #define t_stricmp  wcsicmp
    #define t_strncmp  wcsncmp
    #define t_strnicmp wcsnicmp
    #define t_strlen   wcslen
    #define t_strchr   wcschr
    #define t_strrchr  wcsrchr
    #define t_strstr   wcsstr
    #define t_strtok   wcstok

    #define t_printf   wprintf
    #define t_sprintf  swprintf
    #define t_fprintf  fwprintf
    #define t_vsprintf vswprintf
    #define t_vprintf  vwprintf

    #define t_scanf    wscanf
    #define t_sscanf   swscanf
    #define t_fscanf   fwscanf

    #define t_toupper  towupper
    #define t_tolower  towlower

    #define t_atoi     _wtoi
    #define t_atol     _wtol
    inline double t_atof( kWChar* pStr )  // There's no UNICODE version of atof.
    {
        char lStr[1024];
        wcstombs( lStr, pStr, 1024 );
        return atof( lStr );
    }
    #define t_itoa     _itow

    #define t_isalnum  iswalnum
    #define t_isalpha  iswalpha
    #define t_isdigit  iswdigit
    #define t_isspace  iswspace

    #define t_fopen    _wfopen
    #define t_remove   _wremove
    #define t_rename   _wrename
    #define t_getcwd   _wgetcwd

    #define t_fgets    fgetws
    #define t_fputs    fputws
    #define t_fputc    fputwc

	#define t_ftruncate _chsize
	#define t_fileno    _fileno

    #define t_getenv   _wgetenv

	// Win 32 only funcs
    #define t_fullpath  _wfullpath
    #define t_splitpath _wsplitpath
    #define t_makepath  _wmakepath
    #define t_access    _waccess
    #define t_mkdir     _wmkdir
    #define t_stat      _wstat
    #define t_stat64    _wstati64
    #define t_spawnv    _wspawnv
    #define t_strlwr    _wcslwr
    #define t_ctime     _wctime  // Not ANSI.

#else  // K_UNICODE

    #include <string.h>
    #include <stdlib.h>
    #include <stdio.h>
    #include <ctype.h>
	#include <io.h>

    #define t_strcpy   strcpy
    #define t_strncpy  strncpy
    #define t_strcat   strcat
    #define t_strcmp   strcmp
    #define t_stricmp  stricmp
    #define t_strncmp  strncmp
    #define t_strnicmp strnicmp
    #define t_strlen   strlen
    #define t_strchr   strchr
    #define t_strrchr  strrchr
    #define t_strstr   strstr
    #define t_strtok   strtok

    #define t_printf   printf
    #define t_sprintf  sprintf
    #define t_fprintf  fprintf
    #define t_vsprintf vsprintf
    #define t_vprintf  vprintf

    #define t_scanf    scanf
    #define t_sscanf   sscanf
    #define t_fscanf   fscanf

    #define t_toupper  toupper
    #define t_tolower  tolower

    #define t_atoi     atoi
    #define t_atol     atol
    #define t_atof     atof
    #define t_itoa     itoa

    #define t_isalnum  isalnum
    #define t_isalpha  isalpha
    #define t_isdigit  isdigit
    #define t_isspace  isspace

    #define t_fopen    fopen
    #define t_remove   remove
    #define t_rename   rename
    #define t_getcwd   getcwd

    #define t_fgets    fgets
    #define t_fputs    fputs
    #define t_fputc    fputc

	#define t_ftruncate _chsize
	#define t_fileno    _fileno

    #define t_getenv   getenv

	// Win 32 only funcs
    #define t_fullpath  _fullpath
    #define t_splitpath _splitpath
    #define t_makepath  _makepath
    #define t_access    _access
    #define t_mkdir     _mkdir
    #define t_stat      _stat
    #define t_stat64    _stati64
    #define t_spawnv    _spawnv
    #define t_strlwr    _strlwr
    #define t_ctime     ctime  // ANSI, but the wide version is WIN only.

#endif // K_UNICODE

#endif // _FBXSDK_T_STR_WIN32_H_

