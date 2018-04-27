/*!  \file kfbximageconverter.h
 */

#ifndef _FBXSDK_IMAGE_CONVERTER_H_
#define _FBXSDK_IMAGE_CONVERTER_H_

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

#include <kaydara.h>

#include <klib/kstring.h>

#ifndef MB_FBXSDK
#include <kbaselib_nsuse.h>
#endif
#include <kbaselib_forward.h>
#include <kfbxplugins/kfbxobject.h>

#include <fbxfilesdk_nsbegin.h>

class KFbxSdkManager;

const int ColorSpaceRGB = 0;
const int ColorSpaceYUV = 1;
const int FileToBuffer  = 0;
const int BufferToFile  = 1;

/** image converter buffer
  *\nosubgrouping
  */
class KFBX_DLL ImageConverterBuffer
{
private:
	int     mWidth;			// nb of horizontal pixels
	int     mHeight;		// nb of vertical pixels
	int     mColorSpace;	// either RGB or YUV
	int     mPixelSize;		// 3 or 4 (case of RGBA)
	kByte  *mData;			// pointer to the array of pixels

	bool    mOriginalFormat;	// true if the data in this object has 
	KString mOriginalFileName;	// not been converted in any way

	bool    mValid;
	bool    mUseDataBuffer;    // if false, the data to use is an external file on disk
	                           // otherwise the data is the allocated memory buffer.

public:
	/**
	  * \name Constructor and Destructor. 
	  */
    //@{

    //!Constructor.
	ImageConverterBuffer();
	//!Destructor.
	~ImageConverterBuffer();
	//@}

	/** Check if this object is correctly initialized.
	  * \return     \c true if the object has been initialized with acceptable values.
	  */
	bool    IsValid()				{ return mValid; }

	/** Tells where the data to use is located.
	  * \return     \c true if the data is in the allocated memory buffer. \c false if the data is from an external file on disk.
	  */
	bool    UseDataBuffer()			{ return mUseDataBuffer; }

	/** Get the width of the image.
	  * \return     The number of horizontal pixels.
	  */
	int     GetWidth()				{ return mWidth; }

	/** Get the height of the image.
	  * \return     The number of vertical pixels.
	  */
	int     GetHeight()				{ return mHeight; }

	/** Get the color space of the image.
	  * \return     Either ColorSpaceRGB or ColorSpaceYUV.
	  */
	int     GetColorSpace()			{ return mColorSpace; }

	/** Get the number of bytes per pixel.
	  * \return     Either 3 for RGB images, or 4 for RGBA images.
	  */
	char    GetPixelSize()			{ return (char) mPixelSize; }

	/** Get access to the image data.
	  * \return     Pointer to the array of pixels.
	  */
	kByte*  GetData()				{ return mData; }

	/** Tells if the image has not been converted from its original format.
	  * \return     \c true if the image is stored in its original format, \c false if the image has been converted.
	  */
	bool    GetOriginalFormat()		{ return mOriginalFormat; }

	/** Get the original filename of the image file before conversion.
	  * \return      The original filename.
	  * \remarks     When a conversion to another format occurs, the converted image is given a different filename. 
      *              The original filename can be stored in the FBX file so that the original file can be extracted
	  *              from the converted image (also stored in the FBX file). 
	  */
	KString GetOriginalFileName()	{ return mOriginalFileName; }

	/** Initialize the object.
	  * \param pWidth             Image width.
	  * \param pHeight            Image height.
	  * \param pUseDataBuffer     Set to \c true if the image buffer needs to be allocated.
	  * \param pColorSpace        The image color space.
	  * \param pPixelSize         The number of bytes per pixel.
	  * \remarks                  The total number of bytes allocated (if the pUseDataBuffer is \c true) is:
	  *                           total = pWidth * pHeight * pPixelSize
	  */
	void Init(int pWidth, int pHeight, bool pUseDataBuffer, int pColorSpace = ColorSpaceRGB, char pPixelSize = 4);

	/** Set the original format flag.
	  * \param pState     The value of the original format flag.
	  */
	void SetOriginalFormat(bool pState);

	/** Set the original filename string.
	  * \param pFilename     The filename to use.
	  */
	void SetOriginalFileName(KString pFilename);
};

// This function should return 0 if successful and any non zero value otherwise. And should
// init the pBuffer with the correct values.
typedef int (*ImageConverterFunction)(int pDirection, KString& pFileName, ImageConverterBuffer& pBuffer);


//! Provides a placeholder of functions to convert from a file format to "raw" data and vice et versa.
class KFBX_DLL KFbxImageConverter : public KFbxObject
{
	KFBXOBJECT_DECLARE(KFbxImageConverter,KFbxObject);

public:

	/** Register a user converter function into the system.
	  * \param pFileExt     The image file extension the registered function can handle.
	  * \param pFct         The function that can convert the image file.
	  * \remarks            If the function can handle multiple image files, each file extension
	  *                     has to be registered individually (the same function can be used more than once
	  *                     in the RegisterConverterFunction).
	  */
	void RegisterConverterFunction(KString pFileExt, ImageConverterFunction pFct);

	/** Removes a user converter function from the system.
	  * \param pFct     The function to be removed from the list of converters.
	  */
	void UnregisterConverterFunction(ImageConverterFunction pFct);

	/** Perform the actual conversion.
	  * \param pDirection     Either FileToBuffer (0) or BufferToFile (1).
	  * \param pFileName      The destination filename (can be changed by the ImageConverterFunction).
	  * \param pBuffer        The data placeholder.
	  * \return               \c true if the conversion is successful, \c false otherwise.
	  */
	bool Convert(int pDirection, KString& pFileName, ImageConverterBuffer& pBuffer);

///////////////////////////////////////////////////////////////////////////////
//
//  WARNING!
//
//	Anything beyond these lines may not be documented accurately and is 
// 	subject to change without notice.
//
///////////////////////////////////////////////////////////////////////////////
#ifndef DOXYGEN_SHOULD_SKIP_THIS

protected:

	virtual KFbxObject* Clone( KFbxObject* pContainer, KFbxObject::ECloneType pCloneType ) const;
	KFbxImageConverter(KFbxSdkManager& pManager,char const *pName);
	~KFbxImageConverter();

	KFbxImageConverter& operator=( const KFbxImageConverter& pOther );

	KCharPtrSet*    mConverterFunctions;
	KFbxSdkManager* mManager;

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

};

// global utilities
KFBX_DLL bool IsCompressedTif(KString pFile);
KFBX_DLL bool ReadFromTif(KString pFile, ImageConverterBuffer& pBuffer, bool deleteFile);
KFBX_DLL bool WriteToTif(KString pFile, bool pCompressed, ImageConverterBuffer& pBuffer);

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_IMAGE_CONVERTER_H_


