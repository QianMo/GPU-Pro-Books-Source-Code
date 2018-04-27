/*!  \file kfbxusernotification.h
 */

#ifndef _FBXSDK_USER_NOTIFICATION_H_
#define _FBXSDK_USER_NOTIFICATION_H_

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

#include <kaydaradef.h>
#ifndef KFBX_DLL 
	#define KFBX_DLL K_DLLIMPORT
#endif

#include <stdio.h>
#include <klib/karrayul.h>
#include <klib/kstring.h>
#include <klib/kset.h>

#ifndef MB_FBXSDK
#include <kbaselib_nsuse.h>
#endif

#include <fbxfilesdk_nsbegin.h>

class KFbxLogFile;
class KFbxMessageEmitter;
class KFbxNode;
class KFbxUserNotificationFilteredIterator;
class KFbxSdkManager;

/** \brief This class defines one entry object held by the KFbxUserNotification class.
  * \nosubgrouping
  * Direct manipulation of this object should not be required. At most, access to
  * its members can be granted for querying purposes.
  */
class KFBX_DLL AccumulatorEntry
{
public:
	enum AEClass {
		eAE_ERROR=1,
		eAE_WARNING=2,
		eAE_INFO=4,
		eAE_ANY=7 //! cannot be used as a class ID
	};

	/** Constructor.
	  *	\param pAEClass     Specify the category for this entry.
	  *	\param pName        Identifies this entry (more than one object can have the same name).
      *	\param pDescr       The description of the entry. This is the common message. The details
	  *	                    are added separately by the KFbxUserNotification classes.
	  *	\param pDetail      A list of detail string that will be copied into the local array.
	  * \param pMuteState   Whether this entry is muted.
	  *	\remarks            By default the object is muted so it does not get processed by the lowlevel
	  *                     output routines of the UserNotification accumulator. The entry gets activated 
	  *                     (unmuted) by the calls to AddDetail() in the accumulator.		        
      */
	AccumulatorEntry(AEClass pAEClass, const KString& pName, const KString& pDescr, 
					 KString pDetail="", bool pMuteState=true);

	//! Copy Constructor
	AccumulatorEntry(const AccumulatorEntry& pAE, bool pSkipDetails);

	//! Destructor.
	~AccumulatorEntry();

	//! Returns the class of this entry.
	AEClass GetClass() const;

	//! Returns the name of this entry.
	KString GetName() const;

	//! Returns the description of this entry.
	KString	GetDescription() const;

	//! Returns the number of details stored.
	int GetDetailsCount() const;

	//! Returns a pointer to one specific detail string (or NULL if the id is invalid).
	const KString* GetDetail(int id) const;

	//! Returns True if this entry is muted.
	bool IsMuted() const;

private:
	friend class KFbxUserNotification;
	KArrayTemplate<KString*>& GetDetails();
	void Mute(bool pState);

	bool    mMute;
	AEClass mAEClass;
	KString mName;
	KString mDescr;
	KArrayTemplate<KString*> mDetails;
};


/** This class accumulates user notifications and sends them to any device opened
  * by the derived classes. If this class is not derived, the data can only be
  * sent to a log file. To send data to a log file, it must be opened before attempting
  * to send data, otherwise, the messages will be lost.
  */
class KFBX_DLL KFbxUserNotification
{
public:
	static KFbxUserNotification* Create(KFbxSdkManager* pManager, 
										const KString& pLogFileName, 
										const KString& pSessionDescription);

	static void Destroy(KFbxSdkManager* pManager);

	/** Instanciate a KFbxUserNotification but leave it uninitialized. The caller must
	  * explicitly call InitAccumulator to initialize it and ClearAccumulator when finished
	  * using it.
	  * \param pManager
	  * \param pLogFileName            Name of the log file that will be open in the directory 
	  *                                defined by the GetLogFilePath method.
	  * \remarks                       If pLogFileName is an empty string the logfile does not get created and any
	  *                                output sent to it is lost.
	  * \param pSessionDescription     This string is used to separate session logs in the file.
	  * \remarks                       If the specified logfile already exists, messages are appended to it. This
	  *                                class never deletes the log file. Derived classes may delete the log file 
	  *                                before opening (it must be done in the constructor because the log file is 
	  *                                opened in the InitAccumulator) or at the end of the processing in the 
	  *                                PostTerminate method.
	  */
	KFbxUserNotification(KFbxSdkManager* pManager,
                         KString const& pLogFileName, 
						 KString const& pSessionDescription);

	virtual ~KFbxUserNotification();

	/** This method must be called before using the Accumulator. It opens the log file and
	  * calls AccumulatorInit followed by OpenExtraDevices. Failing to call this method
	  * will prevent other actions except ClearAccumulator, GetLogFileName and GetLogFilePath.
	  */
	void InitAccumulator();

	/** This method must be called when the Accumulator is no longer needed. It calls 
	  * CloseExtraDevices, followed by the AccumulatorClear, and then closes the log file.
	  */
	void ClearAccumulator();

	enum AEid {
		eBINDPOSE_INVALIDOBJECT = 0x0000,
		eBINDPOSE_INVALIDROOT,
		eBINDPOSE_NOTALLANCESTORS_NODES,
		eBINDPOSE_NOTALLDEFORMING_NODES,
		eBINDPOSE_NOTALLANCESTORS_DEFNODES,
		eBINDPOSE_RELATIVEMATRIX,
		eFILE_IO_NOTIFICATION, // this is generic for reader and writer to log notifications.
        eFILE_IO_NOTIFICATION_MATERIAL,
		eAE_START_ID // Starting ID for any Accumulator entry added by derived classes.
	};

	/**
	  * \name Accumulator Management
	  */
	//@{
	/** Adds one entry into the accumulator.
	  * \param pID          This entry unique ID.
	  *	\param pName        This entry name.
	  *	\param pDescr       The description of this entry.
	  * \param pClass       The category of this entry.
	  * \returns            The ID of the newly allocated entry. This ID is pEntryId.
	  */
	int AddEntry(const int pID, const KString& pName, const KString& pDescr, AccumulatorEntry::AEClass pClass=AccumulatorEntry::eAE_WARNING);

	/** Completes the accumulator entry (there can be more that one detail for each entry) and implicitly defines
	  * the sequence of events. Each call to this method is internally recorded, making it possible to output each
	  * notification in the order they have been defined. Also, when a detail is added to an entry, it is automatically unmuted
	  * so it can be sent to the devices (muted AccumulatorEntry objects are not processed).
	  * \param pEntryId     The entry index (as returned by AddEntry).
	  * \return             The id of the detail in the recorded sequence of events. This Id should be used when the call to
	  *                     Output has the eSEQUENCED_DETAILS set as a source. If an error occurs, the returned value is -1
	  */
	int AddDetail(int pEntryId);

	/** Completes the accumulator entry (there can be more that one detail for each entry) and implicitly defines
	  * the sequence of events. Each call to this method is internally recorded, making it possible to output each
	  * notification in the order they have been defined. Also, when a detail is added to an entry, it is automatically unmuted
	  * so it can be sent to the devices (muted AccumulatorEntry objects are not processed).
	  * \param pEntryId     The entry index (as returned by AddEntry).
	  * \param pString      The detail string to add to the entry.
	  * \return             The id of the detail in the recorded sequence of events. This Id should be used when the call to
	  *                     Output has the eSEQUENCED_DETAILS set as a source. If an error occurs, the returned value is -1
	  */
	int AddDetail(int pEntryId, KString pString);

	/** Completes the accumulator entry (there can be more that one detail for each entry) and implicitly defines
	  * the sequence of events. Each call to this method is internally recorded, making it possible to output each
	  * notification in the order they have been defined. Also, when a detail is added to an entry, it is automatically unmuted
	  * so it can be sent to the devices (muted AccumulatorEntry objects are not processed).
	  * \param pEntryId     The entry index (as returned by AddEntry).
	  * \param pNode        The node to add to the entry.
	  * \return             The id of the detail in the recorded sequence of events. This Id should be used when the call to
	  *                     Output has the eSEQUENCED_DETAILS set as a source. If an error occurs, the returned value is -1
	  */
	int AddDetail(int pEntryId, KFbxNode* pNode);

	//! Returns the number of AccumulatorEntries currently stored in this accumulator.
	int  GetNbEntries() const;

	/** Get the specified AccumulatorEntry.
	  * \param pEntryId     ID of the entry to retrieve.
	  * \return             Pointer to the specified entry, otherwise \c NULL if either the id is invalid or the Accumulator
	  *                     is not properly initialized.
	  */
	const AccumulatorEntry* GetEntry(int pEntryId) const;

	/** Get the AccumulatorEntry at the specified index.
	  * \param pEntryIndex     index of the entry to retrieve.
	  * \return                Pointer to the specified entry, otherwise \c NULL if either the index is invalid or the Accumulator
	  *                        is not properly initialized..
	  */
	const AccumulatorEntry* GetEntryAt(int pEntryIndex) const;

	//! Returns the number of Details recorded so far in this accumulator.
	int GetNbDetails() const;

	/** Get the specified detail.
	  * \param pDetailId     Index of the detail. This is the idth detail of type pClass as inserted
	  *                      when the AddDetail 
	  * \param pAE           Pointer to the AccumulatorEntry object that contains the requested detail.
	  *                      The returned valued can be NULL if an error occured.
	  * \return              The index of the detail to be used when calling the GetDetail of the AccumulatorEntry.
	  * \remarks             A value of -1 is acceptable and means that the AccumulatorEntry has no details. However,
	  *                      if pAE is NULL, the return value is meaningless.
	  */
	int GetDetail(int pDetailId, const AccumulatorEntry*& pAE) const;

	//@}

	/**
	  * \name Accumulator Output
	  */
	//@{
	enum OutputSource {
		eACCUMULATOR_ENTRY,
		eSEQUENCED_DETAILS
	};

	/** Send the accumulator entries to the output devices.
	  * This method needs to be explicitly called by the program that uses this
	  * class. 
	  * \param pOutSrc               Specify which data has to be sent to the output devices. Set to SEQUENCED_DETAILS
	  *                              to send the Details in the recorded order. Set to ACCUMULATOR_ENTRY to send
	  *                              each entry with its details regardless of the order in which the events occurred.
	  * \param pIndex                If this parameter >= 0, only send the specified entry/detail index to the output devices.
	  *                              Otherwise send all of them.
	  * \param pExtraDevicesOnly     If this parameter is True, the output is not sent to the log file.
	  * \remark                      The pExtraDevicesOnly parameter is ignored if the log file has been disabled.
	  */
	bool Output(OutputSource pOutSrc=eACCUMULATOR_ENTRY, int pIndex = -1, bool pExtraDevicesOnly = false);

	/** Send the accumulator entry to the output devices.
	  * \param pId		             Send the entry/detail that matching pIdx to the output devices,
	  *                              otherwise send all of them.
	  * \param pOutSrc               Specify which data has to be sent to the output devices. Set to SEQUENCED_DETAILS
	  *                              to send the Details in the recorded order. Set to ACCUMULATOR_ENTRY to send
	  *                              each entry with its details regardless of the order in which the events occurred..
	  * \param pExtraDevicesOnly     If this parameter is True, the output is not sent to the log file.
	  */	  
	bool OutputById(AEid pId, OutputSource pOutSrc=eACCUMULATOR_ENTRY, bool pExtraDevicesOnly = false);

	/** Send an immediate entry to the output devices.
	  * This metohod bypasses the accumulator by sending the entry directly to the output devices
	  * and discarding it right after. The internal accumulator lists are left unchanged by this call.
	  *	\param pName                 This entry name.
	  *	\param pDescr                The description of this entry.
	  * \param pClass                The category of this entry.
	  * \param pExtraDevicesOnly     If this parameter is True, the output is not sent to the log file.
	  * \remarks                     The pExtraDevicesOnly parameter is ignored if the log file has been disabled.
	  */
	bool Output(const KString& pName, const KString& pDescr, AccumulatorEntry::AEClass pClass, bool pExtraDevicesOnly = false);

	/** Sends the content of the iterator to the output devices.
	  * This metohod bypasses the accumulator by sending each entry in the iterator directly to 
	  * the output devices. The internal accumulator lists are left unchanged by this call.
	  *	\param pAEFIter              The Filtered AccumulatorEntry iterator object.
	  * \param pExtraDevicesOnly     If this parameter is True, the output is not sent to the log file.
	  * \remarks                     The pExtraDevicesOnly parameter is ignored if the log file has been disabled.
	  */
	bool Output(KFbxUserNotificationFilteredIterator& pAEFIter, bool pExtraDevicesOnly = false);

	/**
	  * \name Utilities
	  */
	//@{
	/** Returns the absolute path to the log file. If this method is not overridden in a derived class, it
      *  returns the TEMP directory.
      * \param pPath     The returned path.
	  */
	virtual void GetLogFilePath(KString& pPath);
	
	/** Returns the log file name 
	  */	
	inline KString GetLogFileName() { return mLogFileName; }
	//@}

protected:
	class AESequence
	{
	public:
		AESequence(AccumulatorEntry* pAE, int pDetailId) :
			mAE(pAE),
			mDetailId(pDetailId)
			{
			};

		AccumulatorEntry* AE() { return mAE; }
		int DetailId() { return mDetailId; }

	private:
		AccumulatorEntry* mAE;
		int mDetailId;
	};

	friend class KFbxUserNotificationFilteredIterator;

	/** Allow a derived class to finalize processing AFTER the log file handle has been
	  * deleted. This may be required if the logfile needs to be moved or shown.
	  * \returns     True if the object is properly cleaned.
	  */
	virtual bool PostTerminate();

	/** Allow the implementation class to perform accumulator initializations before 
	  * the Extra devices are opened. By default this method does nothing.
	  */
	virtual void AccumulatorInit();

	/** Allow the implementation class to perform accumulator clear after the Extra devices are
	  * closed. By default this method does nothing.
	  */
	virtual void AccumulatorClear();

	/** Allow the implementation class to opens its output devices (called by InitAccumulator).
	  * By default this method does nothing.
	  */
	virtual void OpenExtraDevices();

	/** Allow the implementation class to send all the accumulator entries to the devices.
	  * By default this method loop trough all the elements of the received array and
	  * call the SendToExtraDevices method with the appropriate AccumulatorEntry element and id.
	  * \return     \c true if successful, \c false otherwise.
	  */
	virtual bool SendToExtraDevices(bool pOutputNow, KArrayTemplate<AccumulatorEntry*>& pEntries);
	virtual bool SendToExtraDevices(bool pOutputNow, KArrayTemplate<AESequence*>& pAESequence);

	/** Allow the implementation class to send one accumulator entry to the devices.
	  * By default this method does nothing. 
	  * \return      \c true if successful, \c false otherwise.
	  * \remarks     Derived methods should check for the IsMuted() state to decide if the accumulator
	  *              entry should get through or get discaded. See AddDetail for more details.
	  */
	virtual bool SendToExtraDevices(bool pOutputNow, const AccumulatorEntry* pAccEntry, int pDetailId = -1);

	
	/** Allow the implementation class to close it's output devices (called in the ClearAccumulator)
	  * By default this method does nothing.
	  */
	virtual void CloseExtraDevices();

	//! Clears the Accumulator list.
	void ResetAccumulator();

	//! Clears the Sequence list.
	void ResetSequence();

	//! Send the pIdth element of the accumulator or sequence list to the log file.
	void SendToLog(OutputSource pOutSrc, int pId);
	void SendToLog(const AccumulatorEntry* pAccEntry, int pDetailId = -1);

private:
	KString mLogFileName;
	KString mSessionDescription;
#ifndef K_FBXSDK
    KFbxLogFile* mLogFile;
	KFbxMessageEmitter* mLog;
#else
	KString* mLogFile;
	KString* mLog;
#endif

	bool mProperlyInitialized;
	bool mProperlyCleaned;

	KSet mAccuHT;                             // The set establish a relationship between an AccumulatorEntry and it's ID
	KArrayTemplate<AccumulatorEntry*> mAccu;  // The array defines the order the AccumulatorEntry objects have been 
											  // added to the accumulator (calls to AddEntry)
											  // Both structures share the same pointers.
	KArrayTemplate<AESequence*> mAESequence;
    KFbxSdkManager*             mSdkManager;
};

#if 0
/** \brief This class sends accumulated messages to a file handler specified by a string.
  * If the string argument is "StdOut" or "StdErr" (case insensitive), the standard C stdout/stderr devices are used. 
  * Similarly, "cout" and "cerr" can be used for standard out/error. Otherwise, the string argument is assumed to be 
  * a full filename and is used to open a text file for write. This class does not creates a log file by default.
  */
class KFBX_DLL KFbxUserNotificationFILE : public KFbxUserNotification
{
public:

	KFbxUserNotificationFILE(KString pFileDevice, KString pLogFileName="", KString pSessionDescription="");
	virtual ~KFbxUserNotificationFILE();

	virtual void OpenExtraDevices();
	virtual bool SendToExtraDevices(bool pOutputNow, KArrayTemplate<AccumulatorEntry*>& pEntries);
	virtual bool SendToExtraDevices(bool pOutputNow, KArrayTemplate<AESequence*>& pAESequence);
	virtual bool SendToExtraDevices(bool pOutputNow, const AccumulatorEntry* pAccEntry, int pDetailId = -1);
	virtual void CloseExtraDevices();

private:
	KString mFileDevice;
	FILE* mFP;
	int   mUseStream;
};
#endif
/** This class iterates through the accumulated messages depending on the configuration
  * flags (filter). The iterator keeps a local copy of the data extracted from the
  * accumulator.
  */
class KFBX_DLL KFbxUserNotificationFilteredIterator
{
public:
	/** Constructor. 
	  * \param pAccumulator     This reference is only used during construction for retrieving
	  *                         the data required to fill the iterator.
	  * \param pFilterClass     The bitwise combination of the AEClass identifiers. An AccumulatorEntry
	  *                         element is copyed from the accumulator if its Class matches one of the
	  *	                        bits of this flag.
	  * \param pSrc	            Specify which data format is extracted from the accumulator.
	  * \param pNoDetail	    This parameter is used ONLY if pSrc == eACCUMULATOR_ENTRY and, if set to
	  *                         false, the details of the AccumulatorEntry are also sent to the output
	  *						    devices. If left to its default value, only the description of the
	  *						    AccumulatorEntry is sent.
	  */
	KFbxUserNotificationFilteredIterator(KFbxUserNotification& pAccumulator, 
			int pFilterClass,
			KFbxUserNotification::OutputSource pSrc = KFbxUserNotification::eSEQUENCED_DETAILS,
			bool pNoDetail = true);

	virtual ~KFbxUserNotificationFilteredIterator();

	//! Returns the number of elements contained in this iterator.
	int  GetNbItems() const;

	//! Put the iterator in its reset state.
	void Reset();

	/** Get this iterator's first item. 
	  * \return     NULL if the iterator is empty.
	  */
	AccumulatorEntry* const First();

	/** Get this iterator's previous item.
	  * \return     NULL if the iterator reached the beginning (or is empty).
	  * \remarks    This method will also return NULL if it is called before
	  *             or immediately after a call to First() and reset the iterator to
	  *             its reset state (meaning that a call to First() is mandatory
	  *             to be able to iterate again).
      */
	AccumulatorEntry* const Previous();

	/** Get this iterator's next item.
	  * \return     NULL if the iterator reached the end (or is empty).
	  * \remark     This method will also return NULL if it is called while 
	  *             the iterator is in its reset state (called before
	  *             First() or after a preceding call to Previous() reached 
	  *             beyond the beginning).
      */
	AccumulatorEntry* const Next();

protected:
	// Called in the constructor.
	virtual void BuildFilteredList(KFbxUserNotification& pAccumulator);

	int									mIterator;
	int									mFilterClass;
	bool								mNoDetail;
	KFbxUserNotification::OutputSource	mAccuSrcData;
	KArrayTemplate<AccumulatorEntry*>	mFilteredAE;
};


#include <fbxfilesdk_nsend.h>

#endif // #define _FBXSDK_USER_NOTIFICATION_H_


