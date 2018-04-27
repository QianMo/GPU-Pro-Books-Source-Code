/*!  \file kerror.h
 */

#ifndef _FBXSDK_KERROR_H_
#define _FBXSDK_KERROR_H_

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
#include <kbaselib_h.h>

#include <klib/kstring.h>

#include <kbaselib_nsbegin.h>

    //! Error class.
    class KBASELIB_DLL KError
    {
    public:

        /** Constructor.
         */
        KError();

        /** Constructor.
        *   \param pStringArray The error string table in use.
        *   \param pErrorCount Number of elements in the error string table.
        */
        KError(char* pStringArray [], int pErrorCount);

        //! Destructor.
        ~KError();

        /** Get the size of the error string table.
        *   \return Number of elements in the error string table.
        */
        int GetErrorCount() const;

        /** Get the error message string.
        *   \param pIndex Error index.
        *   \return Error string.
        */
        char* GetErrorString(int pIndex) const;

        /** Set the last error ID and the last error string.
        *   \param pIndex Error index.
        * \param pString Error string.
        *   \remarks This method will also set the last error string to the default
        *   string value contained in the error string table for this error ID.
        */
        void SetLastError(int pIndex, const char* pString);

        /** Set the last error index.
        *   \param pIndex Error index.
        *   \remarks This method will also set the last error string to the default
        *   string value contained in the error string table for this error index.
        */
        void SetLastErrorID(int pIndex);

        /** Return the last error index.
        *   \return The last error index or -1 if none is set.
        */
        int GetLastErrorID() const;

        /** Get the message string associated with the last error.
        *   \return Error string or empty string if none is set.
        */
        const char* GetLastErrorString() const;

        /** Set the message string associated with the last error.
        *   \param pString Error string.
        *   This method should be called after KError::SetLastErrorID()
        * in order to customize the error string.
        */
        void SetLastErrorString(const char * pString);

        //! Reset the last error.
        void ClearLastError();

    ///////////////////////////////////////////////////////////////////////////////
    //
    //  WARNING!
    //
    //  Anything beyond these lines may not be documented accurately and is
    //  subject to change without notice.
    //
    ///////////////////////////////////////////////////////////////////////////////

    #ifndef DOXYGEN_SHOULD_SKIP_THIS

    private:

        int mLastErrorID;
        int mErrorCount;

        KString mLastErrorString;
        char** mStringArray;

    #endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

    };
#include <kbaselib_nsend.h>

#endif // #ifndef _FBXSDK_KERROR_H_


