/*!  \file kname.h
 */

#ifndef _FBXSDK_KNAME_H_
#define _FBXSDK_KNAME_H_

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
#include <klib/karrayul.h>

#include <kbaselib_nsbegin.h>

    /** Name class.
    *	Provides two stings, current and an initial name, for reversible
    * renaming.  This is especially useful for name clashing, renaming
    *	strategies and merging back to a former 3D scene using the initial
    *	names.
    */
    class KBASELIB_DLL KName
    {
    public:
    	
	    /** Constructor.
        * \param pInitialName Name string used to initialize both members (initialName and currentName)
        * of this class.
        */
        KName(char const* pInitialName = "");

	    //! Copy constructor.
        KName(KName const& pName);

        //!Destructor
	    ~KName();

	    /** Set initial name.
        * \param pInitialName New string for the initial name.
	    *	\remarks The current name will also be changed to this value.
	    */
	    void SetInitialName(char const* pInitialName);

	    /** Get initial name.
        * \return Pointer to the InitialName string buffer.
        */
	    char const* GetInitialName() const;

	    /** Set current name.
        * \param pNewName New string for the current name.
        * \remarks The initial name is not affected.
        */
	    void SetCurrentName(char const* pNewName);

	    /** Get current name.
        * \return Pointert to the CurrentName string buffer.
        */
	    char const* GetCurrentName() const;

	    /** Set the namespace.
        * \param pNameSpace New string for the namespace.
        * \remarks The initial name is not affected.
        */
	    void SetNameSpace(char const* pNameSpace);

	    /** Get the namespace.
        * \return Pointert to the CurrentName string buffer.
        */
	    char const* GetNameSpace() const;

	    /** Check if the current name and internal name match.
        * \return \c true if current name isn't identical to initial name.
	    */
	    bool IsRenamed() const;

    	
	    //! Assignment operator
	    KName& operator= (KName const& pName);

    ///////////////////////////////////////////////////////////////////////////////
    //
    //  WARNING!
    //
    //	Anything beyond these lines may not be documented accurately and is 
    // 	subject to change without notice.
    //
    ///////////////////////////////////////////////////////////////////////////////

    #ifndef DOXYGEN_SHOULD_SKIP_THIS

		/** Get the namespaces in a string pointer array format.
        * \return KArrayTemplate<KString*> .
        */
	    KArrayTemplate<KString*> GetNameSpaceArray(char identifier);

    private:

	    KString mInitialName;
	    KString mCurrentName;
		KString mNameSpace;

    #endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

    };

#include <kbaselib_nsend.h>

#endif // #define _FBXSDK_KNAME_H_
