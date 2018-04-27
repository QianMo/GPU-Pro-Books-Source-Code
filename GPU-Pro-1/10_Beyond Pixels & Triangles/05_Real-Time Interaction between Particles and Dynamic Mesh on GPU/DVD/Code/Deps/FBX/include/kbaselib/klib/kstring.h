/*!  \file kstring.h
 */

#ifndef _FBXSDK_KSTRING_H_
#define _FBXSDK_KSTRING_H_

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
#include <string.h>

// FBX includes
#include <kbaselib_h.h>
#include <klib/kscopedptr.h>


#if defined(KARCH_ENV_WIN32)

// Conversions between WideChar, UTF8 and ANSI
    #define UTF8_to_WIDECHAR( inUTF8,outWideChar ) \
        wchar_t outWideChar[1024]; \
        MultiByteToWideChar( CP_UTF8, 0, inUTF8, -1, outWideChar, 1024 );

    #define WIDECHAR_to_UTF8( inWideChar,outUTF8 ) \
        char outUTF8[1024]; \
        WideCharToMultiByte( CP_UTF8,0,inWideChar,-1,outUTF8,1024,NULL,NULL);

    #define WIDECHAR_to_ANSI( inWideChar,outANSI ) \
        char outANSI[1024]; \
        WideCharToMultiByte( CP_ACP,0,inWideChar,-1,outANSI,1024,NULL,NULL);

    #define ANSI_to_WIDECHAR( inANSI,outWideChar ) \
        wchar_t outWideChar[1024]; \
        MultiByteToWideChar( CP_ACP, 0, inANSI, -1, outWideChar, 1024 );

    #define ANSI_to_UTF8( inANSI,outUTF8 ) { \
        wchar_t tempWideChar[1024]; \
        MultiByteToWideChar( CP_ACP, 0, inANSI, -1, tempWideChar, 1024); \
        WideCharToMultiByte( CP_UTF8, 0, tempWideChar, -1, outUTF8, 1024, NULL, NULL); \
    }

    #define UTF8_to_ANSI( inUTF8,outANSI, dummy, dummy1 ) { \
        wchar_t tempWideChar[1024]; \
        MultiByteToWideChar( CP_UTF8, 0, inUTF8, -1, tempWideChar, 1024); \
        WideCharToMultiByte( CP_ACP, 0, tempWideChar, -1, outANSI, 1024, NULL, NULL); \
    }
#endif



#include <kbaselib_nsbegin.h>
class KMemoryAllocator;

#if !defined(KARCH_ENV_WIN32)
    KBASELIB_DLL void UTF8_to_ANSI( const char* pSrc, char* pDest, size_t pDestSize, char pSubstitute = '.' );
#endif

//  #define UseKStringStatic
    #ifdef UseKStringStatic
        #define KStringStaticSize 64
    #endif

    /** Utility class to manipulate strings.
    * \nosubgrouping
    */
    class KBASELIB_DLL KString
    {
        public:
            /**
            * \name Constructors and Destructor
            */
            //@{
                //! Create an instance.
                static KString      *Create();
                //! Create an instance if not already allocated ( is null )
                static KString      *Create(KString *pString);
                //! Destroy an allocated version of the string
                void                Destroy();
                //! Destroy the allocated space if empty
                static KString* DestroyIfEmpty(KString *pString);
                //! Destroy the allocated space if empty
                static KString* StringOrEmpty(KString *pString);
            //@}

        public:
            /**
            * \name Constructors and Destructor
            */
            //@{

                //! Default constructor.
                KString();

                //! Copy constructor.
                KString(const KString& pStr);

                //! String constructor.
                KString(const char* pStr);

                //! Character constructor.
                KString(char pChar, size_t pNbRepeat = 1);

                //! String constructor with maximum length.
                KString(const char* pCharPtr, size_t pLength);

                //! Int constructor.
                KString(const int pValue);

                //! Float constructor.
                KString(const float pValue);

                //! Double constructor.
                KString(const double pValue);

                //! Destructor.
                ~KString();
            //@}

            /**
            * \name Instance Validation.
            */
            //@{
                //! Return \c true if string is valid.
                inline bool IsOK() const;

                //! Invalidate string.
                KString &Invalidate();

                //! Get string length like "C" strlen().
                inline size_t GetLen() const;

                //! Return \c true if string is of length 0.
                inline bool IsEmpty() const;

                //! Discard the content of the object.
                KString  &Empty();
            //@}

            /**
            * \name Buffer Access
            */
            //@{

                //! Access by reference.
                char& operator[](int pIndex);

                //! Access by copy.
                char operator[](int pIndex) const;

                //! Cast operator.
                inline operator const char*() const;

                //! Non-const buffer access.
                inline char* Buffer();

                //! const buffer access.
                inline const char* Buffer()const;

            //@}

            /**
            * \name Assignment Operators
            */
            //@{

                //! KString assignment operator.
                const KString& operator=(const KString& pStr);

                //! Character assignment operator.
                const KString& operator=(char pChar);

                //! String assignment operator.
                const KString& operator=(const char* pStr);

                //! Int assignment operator.
                const KString& operator=(int  pValue);

                //! Float assignment operator.
                const KString& operator=(float pValue);

                //! Double assignment operator.
                const KString& operator=(double pValue);

                //! String assignment function.
                const KString& Copy(size_t pLen, const char* pStr);

                // Swap the contents of two KString objects; no allocation is performed.
                void Swap(KString&);
            //@}

            /**
            * \name Append and Concatenation
            */
            //@{

                //! Append as "C" strcat().
                const KString& Append(const char* pStr);

                //! Append as "C" strncat().
                const KString& AppendN(const char* pStr, size_t pLength);

                //! KString append.
                const KString& operator+=(const KString& pKStr);

                //! Character append.
                const KString& operator+=(char pChar);

                //! String append.
                const KString& operator+=(const char* pStr);

                //! Int append.
                const KString& operator+=(int pValue);

                //! Float append.
                const KString& operator+=(float pValue);

                //! KString concatenation.
                friend KBASELIB_DLL KString operator+(const KString& pKStr1, const KString& pKStr2);

                //! Character concatenation.
                friend KBASELIB_DLL KString operator+(const KString& pKStr, char pChar);

                //! String concatenation.
                friend KBASELIB_DLL KString operator+(const KString& pKStr, const char* pStr);

                //! Int concatenation.
                friend KBASELIB_DLL KString operator+(const KString& pKStr, int pValue);

                //! Float concatenation.
                friend KBASELIB_DLL KString operator+(const KString& pKStr, float pValue);
            //@}

            /**
            * \name String Comparison
            */
            //@{

                //! Compare as "C" strcmp().
                int Compare(const char * pStr) const;

                //! Compare as "C" stricmp().
                int CompareNoCase( const char * pStr ) const;

                //! Equality operator.
                bool operator==(const KString &pStr) const;

                //! Inequality operator.
                bool operator!=(const KString &pStr) const;

                //! Inferior to operator.
                bool operator< (const KString &pStr) const;

                //! Inferior or equal to operator.
                bool operator<=(const KString &pStr) const;

                //! Superior or equal to operator.
                bool operator>=(const KString &pStr) const;

                //! Superior to operator.
                bool operator> (const KString &pStr) const;

                //! Equality operator.
                bool operator==(const char *pStr) const;

                //! Inequality operator.
                bool operator!=(const char *pStr) const;

                //! Inferior to operator.
                bool operator< (const char *pStr) const;

                //! Inferior or equal to operator.
                bool operator<=(const char *pStr) const;

                //! Superior or equal to operator.
                bool operator>=(const char *pStr) const;

                //! Superior to operator.
                bool operator> (const char *pStr) const;
            //@}

            /**
            * \name Substring Extraction
            */
            //@{

                //! Extract middle string for a given length.
                KString Mid(size_t pFirst, size_t pCount) const;

                //! Extract middle string up to the end.
                KString Mid(size_t pFirst) const;

                //! Extract left string.
                KString Left(size_t pCount) const;

                //! Extract right string.
                KString Right(size_t pCount) const;

            //@}

            /**
            * \name Padding
            */
            //@{

				/** \enum PaddingType      Padding types.
				  * - \e eRight
				  * - \e eLeft
				  * - \e eBoth
				  */
                enum PaddingType 
				{ 
					eRight,
					eLeft, 
					eBoth
				};

                //! Add padding characters.
                KString Pad(PaddingType pPadding, size_t pLen, char pCar = ' ') const;

                //! Remove padding characters.
                KString UnPad(PaddingType pPadding) const;

            //@}

            /**
            * \name Conversion
            */
            //@{
                //! Uppercase conversion
                KString Upper() const;

                //! Lowercase conversion
                KString Lower() const;

                //! Reverse conversion
                KString Reverse() const;

                //! Convert to Unix, changes \\r\\n characters for a single \\n
                KString ConvertToUnix() const;

                //! Convert to Windows, changes \\n character for both \\r\\n
                KString ConvertToWindows() const;
            //@}

            /**
            * \name Search
            */
            //@{
                /** Look for a single character match, like "C" strchr().
                * \return Index or -1 if not found.
                */
                int Find(char pChar, size_t pStartPosition = 0) const;

                /** Look for a substring match, like "C" strstr().
                * \return Starting index or -1 if not found.
                */
                int Find(const char* pStrSub, size_t pStartPosition = 0) const;

                /** Look for a single character match, like "C" strrchr().
                * \return Index or -1 if not found.
                */
                int ReverseFind(char pChar) const;

                /** Look for a single character match, like "C" strpbrk().
                * \return Index or -1 if not found.
                */
                int FindOneOf(const char * pStrCharSet, size_t pStartPosition = 0) const;

                /** Replace a substring.
                * \return \c true if substring found and replaced.
                */
                bool FindAndReplace(const char* pFind, const char* pReplaceBy, size_t pStartPosition = 0);

                /** Replace a character.
                * \return \c true if character found and replaced.
                */
                bool ReplaceAll( char pFind, char pReplaceBy );

            //@}

            /**
            * \name Token Extraction
            */
            //@{
                //! Get number of tokens.
                int GetTokenCount(const char* pSpans) const;

                //! Get token at given index.
                KString GetToken(int pTokenIndex, const char* pSpans) const;
            //@}

            /**
            * \name Memory Pool Management
            */
            //@{
            public:
                static KMemoryAllocator* AllocatorGet();
                static void AllocatorPurge();
                static void  AllocatorRelease();
            //@}

        ///////////////////////////////////////////////////////////////////////////////
        //  WARNING!
        //  Anything beyond these lines may not be documented accurately and is
        //  subject to change without notice.
        ///////////////////////////////////////////////////////////////////////////////
        #ifndef DOXYGEN_SHOULD_SKIP_THIS
        private:
            void Init();
            void FreeBuffer();
            void FreeBuffer(char *&pOldData);

            bool AllocCopy(KString& pDest, size_t pCopyLen, size_t pCopyIndex) const;
            bool AllocBuffer(size_t pLen);
            bool AllocBuffer(size_t pLen, char *&pOldData);

            bool AssignCopy(size_t pSrcLen, const char * pSrcData);
            bool ConcatCopy(size_t pSrc1Len, const char * pSrc1Data,
                            size_t pSrc2Len, const char * pSrc2Data);
            bool ConcatInPlace(size_t pSrcLen, const char * pSrcData);

            bool IsIn(char pChar, const char* pString) const;

            static KScopedPtr<KMemoryAllocator> mMemoryPool;

        // Lengths/sizes in characters.
        // Note: an extra character is always allocated.
        private:
            char*       mData; // Actual string (zero terminated).
            size_t      mDataLen; // Does not include terminating 0.
            size_t      mAllocLen; // Does not include terminating 0.
            #ifdef UseKStringStatic
                char    mStaticData[KStringStaticSize]; // Actual static string (zero terminated) used as long as the string is smaller than KStringStaticSize
            #endif

            static char ksNilString[1];    // Used when strlen == 0;
            static char ksErrorString[1];  // Used on allocation errors, so

        #endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

    };

    //! KString concatenation.
    KBASELIB_DLL KString operator+(const KString& pKStr1, const KString& pKStr2);

    //! Character concatenation.
    KBASELIB_DLL KString operator+(const KString& pKStr, char pChar);

    //! String concatenation.
    KBASELIB_DLL KString operator+(const KString& pKStr, const char* pStr);

    //! Int concatenation.
    KBASELIB_DLL KString operator+(const KString& pKStr, int pValue);

    //! Float concatenation.
    KBASELIB_DLL KString operator+(const KString& pKStr, float pValue);


    // Strip the first and last character of a string assuming they are quotation marks.
    inline void StripQuotationMarks(KString& str)
    {
        str = str.Mid( 1, str.GetLen() - 2 );
    }

    inline void RemoveChar( KString& pString, char lToRemove )
    {
        size_t lPos = pString.ReverseFind( lToRemove );

        while( lPos >= 0 )
        {
            pString = pString.Left( lPos ) + pString.Mid( lPos + 1 );
            lPos = pString.ReverseFind( lToRemove );
        }
    }

    // ---- inline definition ----------------------------------------------
    inline bool KString::IsOK()const{ return mData != ksErrorString; }
    inline size_t KString::GetLen() const { return mDataLen; }
    inline bool KString::IsEmpty()const { return( GetLen() == 0 ); }

    inline char *KString::Buffer() { return mData; }
    inline const char *KString::Buffer()const{ return mData; }
    inline KString::operator const char *() const{ return mData; }

    /** Functor class suitable for use in KMap.
     */
    class KStringCompare
    {
        public:
            inline int operator()(KString const &pKeyA, KString const &pKeyB) const
            {
                return (pKeyA < pKeyB) ? -1 : ((pKeyB < pKeyA) ? 1 : 0);
            }
    };

    class KCharCompare
    {
        public:
            inline int operator()(char const* pKeyA, char const* pKeyB) const
            {
                return strcmp( pKeyA, pKeyB );
            }
    };

#include <kbaselib_nsend.h>

#endif // #define _FBXSDK_KSTRING_H_

