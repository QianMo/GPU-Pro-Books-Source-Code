/*!  \file kfbxmatrix.h
 */

#ifndef _FBXSDK_MATRIX_H_
#define _FBXSDK_MATRIX_H_

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

#include <kfbxplugins/kfbxtypes.h>
#include <kfbxmath/kfbxvector4.h>

#include <fbxfilesdk_nsbegin.h>

class KFbxQuaternion;
class KFbxXMatrix;


/**	FBX SDK matrix class.
  * \nosubgrouping
  */
class KFBX_DLL KFbxMatrix : public fbxDouble44
{

public:

	/**
	  * \name Constructors and Destructor
	  */
	//@{

	//! Constructor. Constructs an identity matrix.
	KFbxMatrix ();

	//! Copy constructor.
	KFbxMatrix (const KFbxMatrix& pM);

	/** Constructor.
	  *	\param pT     Translation vector.
	  *	\param pR     Euler rotation vector.
	  *	\param pS     Scale vector.
	  */
	KFbxMatrix (KFbxVector4& pT,
		        KFbxVector4& pR,
				KFbxVector4& pS);

	/** Constructor.
	  *	\param pT     Translation vector.
	  *	\param pQ     Quaternion.
	  *	\param pS     Scale vector.
	  */
	KFbxMatrix (KFbxVector4& pT,
		        KFbxQuaternion& pQ,
				KFbxVector4& pS);

	/** Constructor.
	  * \param pM     Affine matrix
	  */
	KFbxMatrix (const KFbxXMatrix& pM);
		
	//! Destructor.
	~KFbxMatrix ();
		
	//@}

	/**
	  * \name Access
	  */
	//@{

	/** Retrieve matrix element.
	  *	\param pY     Row index.
	  *	\param pX     Column index.
	  * \return       Value at element [ pX, pY ] of the matrix.
	  */
	double Get(int pY, int pX);

	/** Extract a row vector.
	  *	\param pY     Row index.
	  * \return       The row vector.
	  */
	KFbxVector4 GetRow(int pY);

	/** Extract a column vector.
	  *	\param pX      Column index.
	  * \return        The column vector.
	  */
	KFbxVector4 GetColumn(int pX);

	/** Set matrix element.
	  *	\param pY          Row index.
	  *	\param pX          Column index.
	  *	\param pValue      New component value.
	  */
	void Set(int pY, int pX, double pValue);

	//! Set matrix to identity.
	void SetIdentity();
	
	/** Set matrix.
	  *	\param pT     Translation vector.
	  *	\param pR     Euler rotation vector.
	  *	\param pS     Scale vector.
	  */
	void SetTRS(KFbxVector4& pT,
		        KFbxVector4& pR,
				KFbxVector4& pS);

	/** Set matrix.
	  *	\param pT     Translation vector.
	  *	\param pQ     Quaternion.
	  *	\param pS     Scale vector.
	  */
	void SetTQS(KFbxVector4& pT,
		        KFbxQuaternion& pQ,
				KFbxVector4& pS);

	/** Set a matrix row.
	  *	\param pY       Row index.
	  *	\param pRow	    Row vector.
	  */
	void SetRow(int pY, KFbxVector4& pRow);

	/** Set a matrix column.
	  *	\param pX           Column index.
	  *	\param pColumn      Column vector.
	  */
	void SetColumn(int pX, KFbxVector4& pColumn);

	/** Assignment operator.
	  *	\param pMatrix     Source matrix.
	  */
	KFbxMatrix& operator=(const KFbxMatrix& pMatrix);
	
	//@}

	/**
	  * \name Matrix Operations
	  */
	//@{	

	/**	Unary minus operator.
	  * \return     A matrix where each element is multiplied by -1.
	  */
	KFbxMatrix operator-();
	
	/** Add two matrices together.
	  * \param pMatrix    A matrix.
	  * \return           The result of this matrix + pMatrix.
	  */
	KFbxMatrix operator+(const KFbxMatrix& pMatrix) const;

	/** Subtract a matrix from another matrix.
	  * \param pMatrix     A matrix.
	  * \return            The result of this matrix - pMatrix.
	  */
	KFbxMatrix operator-(const KFbxMatrix& pMatrix) const;

	/** Multiply two matrices.
	  * \param pMatrix     A matrix.
	  * \return            The result of this matrix * pMatrix.
	  */
	KFbxMatrix operator*(const KFbxMatrix& pMatrix) const;

	/** Add two matrices together.
	  * \param pMatrix     A matrix.
	  * \return            The result of this matrix + pMatrix, replacing this matrix.
	  */
	KFbxMatrix& operator+=(KFbxMatrix& pMatrix);

	/** Subtract a matrix from another matrix.
	  * \param pMatrix     A matrix.
	  * \return            The result of this matrix - pMatrix, replacing this matrix.
	  */
	KFbxMatrix& operator-=(KFbxMatrix& pMatrix);

	/** Multiply two matrices.
	  * \param pMatrix     A matrix.
	  * \return            The result of this matrix * pMatrix, replacing this matrix.
	  */
	KFbxMatrix& operator*=(KFbxMatrix& pMatrix);

	/** Calculate the matrix transpose.
	  * \return     This matrix transposed.
	  */
	KFbxMatrix Transpose();

	//@}

	/**
	  * \name Vector Operations
	  */
	//@{	

    /** Multiply this matrix by pVector, the w component is normalized to 1.
    * \param pVector     A vector.
    * \return            The result of this matrix * pVector.
    */
    KFbxVector4 MultNormalize(const KFbxVector4& pVector) const;

    //@}

	/**
	  * \name Boolean Operations
	  */
	//@{

	/**	Equivalence operator.
	  * \param pM     The matrix to be compared against this matrix.
	  * \return       \c true if the two matrices are equal (each element is within a 1.0e-6 tolerance), \c false otherwise.
	  */
	bool operator==(KFbxMatrix& pM);
	bool operator==(KFbxMatrix const& pM) const;

	/**	Equivalence operator.
	  * \param pM     The affine matrix to be compared against this matrix.
	  * \return       \c true if the two matrices are equal (each element is within a 1.0e-6 tolerance), \c false otherwise
	  */
	bool operator==(KFbxXMatrix& pM);
	bool operator==(KFbxXMatrix const& pM) const;

	/**	Non-equivalence operator.
	  * \param pM     The matrix to be compared against this matrix.
	  * \return       \c false if the two matrices are equal (each element is within a 1.0e-6 tolerance), \c true otherwise.
	  */
	bool operator!=(KFbxMatrix& pM);
	bool operator!=(KFbxMatrix const& pM) const;

	/**	Non-equivalence operator.
	  * \param pM     The affine matrix to be compared against this matrix.
	  * \return       \c false if the two matrices are equal (each element is within a 1.0e-6 tolerance), \c true otherwise
	  */
	bool operator!=(KFbxXMatrix& pM);
	bool operator!=(KFbxXMatrix const& pM) const;

	
	//@}

	/**
	  * \name Casting
	  */
	//@{
	
	//! Cast the vector in a double pointer.
	operator double* ();

	typedef const double(kDouble44)[4][4] ;

	inline kDouble44 & Double44() const { return *((kDouble44 *)&mData); }

	//@}

	// Matrix data.
//	double mData[4][4];

};

inline EFbxType FbxTypeOf( KFbxMatrix	const &pItem )	{ return eDOUBLE44; }


#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_MATRIX_H_


