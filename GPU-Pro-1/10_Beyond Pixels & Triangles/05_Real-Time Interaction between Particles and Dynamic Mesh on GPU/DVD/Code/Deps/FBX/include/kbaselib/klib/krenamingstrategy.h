/*!  \file krenamingstrategy.h
 */

#ifndef _FBXSDK_KRENAMINGSTRATEGY_H_
#define _FBXSDK_KRENAMINGSTRATEGY_H_

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

#include <klib/karrayul.h>
#include <klib/kname.h>

#include <kbaselib_nsbegin.h>

    /** Renaming strategy mechanism.
    *	Base class describing how the renaming process is handled.
    *	This class is intented to be derived into a specialised 
    *	renaming class.
    *
    *	Basicaly, the Rename is called everytime a new element is added to
    *   an entity.  the strategy keep
    */
    class KBASELIB_DLL KRenamingStrategy
    {
    public:	
    	
	    //! Constructor.
	    KRenamingStrategy();

	    //! Destructor.
	    virtual ~KRenamingStrategy ();

	    //! Empty all memories about given names
	    virtual void Clear() = 0;

	    /** Rename.
	    *	\param pName
	    *	\return how the operation went.
	    */
	    virtual bool Rename(KName& pName) = 0;

	    /** Spawn mechanism.  
	    *	Create a dynamic renaming strategy instance of the same type
	    *	the child class.
	    *	\return new KRenamingStrategy;	
	    */
	    virtual KRenamingStrategy* Clone() = 0;

    };


    /** Usual renaming numbering renaming strategy.
    *	This renaming strategy will be used by the FBXSDK if no other is specified.
    */
    class KBASELIB_DLL KNumberRenamingStrategy : public KRenamingStrategy
    {
    public:	
    	
	    //! Constructor.
	    KNumberRenamingStrategy();

	    //! Destructor.
	    virtual ~KNumberRenamingStrategy ();

	    //! Empty all memories about given names
	    virtual void Clear();

	    /** Rename.
	    *	\param pName
	    *	\return how the operation went.
	    */
	    virtual bool Rename(KName& pName);

	    /** Spawn mechanism.  
	    *	Create a dynamic renaming strategy instance of the same type
	    *	the child class.
	    *	\return new KNumberRenamingStrategy;	
	    */
	    virtual KRenamingStrategy* Clone();

    private:
    	
	    struct NameCell
	    {
		    NameCell(char const* pName) :
			    mName(pName),
			    mInstanceCount(0)
		    {
		    }
    			
		    KString mName;
		    int mInstanceCount;		
	    };

	    KArrayTemplate<NameCell*> mNameArray;
    };
#include <kbaselib_nsend.h>

#endif
