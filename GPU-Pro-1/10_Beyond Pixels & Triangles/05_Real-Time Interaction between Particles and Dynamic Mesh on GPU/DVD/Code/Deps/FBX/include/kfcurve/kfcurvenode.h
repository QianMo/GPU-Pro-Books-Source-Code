/*!  \file kfcurvenode.h
 */

#ifndef _FBXSDK_KFCURVE_NODE_H_
#define _FBXSDK_KFCURVE_NODE_H_

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

#include <kfcurve/kfcurve_h.h>

#include <klib/karrayul.h>
#include <klib/kerror.h>
#include <klib/kstring.h>
#include <object/e/keventbase.h>

#include <kfcurve/kfcurve.h>

#ifndef K_PLUGIN
	#include <kfcurve/i/ifcurveeti.h>
	#include <object/i/iobject.h>
	#include <object/i/ifbobjectholder.h>
#endif

K_FORWARD (IConnector) 
K_FORWARD (IData) 

#ifndef MB_FBXSDK
#include <kbaselib_nsuse.h>
#endif
#include <kbaselib_forward.h>

#include <kfcurve/kfcurve_nsbegin.h>

	enum 
	{	
		KFCN_IN = 0, 
		KFCN_OUT = 1, 
		KFCN_INOUT_MASK = 1, 

		KFCN_HIDDEN = 0, 
		KFCN_SHOWN = 2, 

		KFCN_CONTAINER_NODE = 0, 
		KFCN_REFERENCER_NODE = 8, 
		KFCN_REFERENCER_MASK = 8, 

		KFCN_TAKE_NODE = 16, 
		KFCN_FIXED_NODE = 0, 
		KFCN_TAKE_MASK = 16,

		KFCN_EXPENDED_NODE = 32, 
		KFCN_NOTEXPENDED_NODE = 0, 
		KFCN_EXPENDED_MASK = 32,

		KFCN_MULTILAYER = 64,
		KFCN_MONOLAYER	= 0,
		KFCN_MULTILAYER_MASK = 64,

		KFCN_ALWAYS_CANDIDATE = 128,
		KFCN_ALWAYS_CANDIDATE_TESTED = 256
	};

	//
	// Picking Types define
	//
	enum 
	{
		KCP_KEY = 1, 
		KCP_LTANGENT, 
		KCP_RTANGENT
	};

	//
	//	Node/Layer info
	//
	enum
	{
		KFCL_NOLAYER,
		KFCL_ADD,		// For translation and standard others
		KFCL_EULER_XYZ,	// For rotation
		KFCL_MULT,		// For scaling, mainly
		KFCL_QUAT		// For quaternion rotation
	};

	#define KFCURVENODE_TRANSFORM               "Transform"
		#define KFCURVENODE_T						"T"
			#define KFCURVENODE_T_X						"X"
			#define KFCURVENODE_T_Y						"Y"
			#define KFCURVENODE_T_Z						"Z"
		#define KFCURVENODE_R						"R"
			#define KFCURVENODE_R_X						"X"
			#define KFCURVENODE_R_Y						"Y"
			#define KFCURVENODE_R_Z						"Z"
			#define KFCURVENODE_R_W						"W"
		#define KFCURVENODE_S						"S"
			#define KFCURVENODE_S_X						"X"
			#define KFCURVENODE_S_Y						"Y"
			#define KFCURVENODE_S_Z						"Z"
	#define KFCURVENODE_LCL_T                   "Lcl Translation"
	#define KFCURVENODE_LCL_R                   "Lcl Rotation"
	#define KFCURVENODE_LCL_S                   "Lcl Scaling"
	#define KFCURVENODE_VISIBILITY				"Visibility"

	// Addition for the optical marker node attribute.
	#define KFCURVENODE_OCCLUSION				"Occlusion"

	// Addition for the IK effector marker node attribute.
	#define KFCURVENODE_IK_REACH_TRANSLATION    "IK Reach Translation"
	#define KFCURVENODE_IK_REACH_ROTATION       "IK Reach Rotation"

	// Addition for the light node attribute.
	#define KFCURVENODE_LIGHT_CONEANGLE			"Cone angle"
	#define KFCURVENODE_LIGHT_FOG				"Fog"
	#define KFCURVENODE_LIGHT_INTENSITY			"Intensity"
	#define KFCURVENODE_COLOR				    "Color"
	#define KFCURVENODE_COLOR_RED			        "X"
	#define KFCURVENODE_COLOR_GREEN		            "Y"
	#define KFCURVENODE_COLOR_BLUE		            "Z"

	// Addition for the camera node attribute.
	#define KFCURVENODE_CAMERA_FIELDOFVIEW		"FieldOfView"
	#define KFCURVENODE_CAMERA_FIELDOFVIEWX		"FieldOfViewX"
	#define KFCURVENODE_CAMERA_FIELDOFVIEWY		"FieldOfViewY"
	#define KFCURVENODE_CAMERA_FOCALLENGTH		"FocalLength"
	#define KFCURVENODE_CAMERA_OPTICALCENTERX	"OpticalCenterX"
	#define KFCURVENODE_CAMERA_OPTICALCENTERY	"OpticalCenterY"
	#define KFCURVENODE_CAMERA_ROLL				"Roll"
	#define KFCURVENODE_CAMERA_TURNTABLE		"TurnTable"
	#define KFCURVENODE_BACKGROUND_COLOR        "BackgroundColor"
	#define KFCURVENODE_BACKGROUND_COLOR_RED        "X"
	#define KFCURVENODE_BACKGROUND_COLOR_GREEN      "Y"
	#define KFCURVENODE_BACKGROUND_COLOR_BLUE       "Z"

	// Addition for the camera switcher node attribute.
	#define KFCURVENODE_CAMERA_INDEX            "Camera Index"

	// Addition for the texture.
	#define KFCURVENODE_TEXTURE_TRANSLATION     "Translation"
	#define KFCURVENODE_TEXTURE_TRANSLATION_X   "X"
	#define KFCURVENODE_TEXTURE_TRANSLATION_Y   "Y"
	#define KFCURVENODE_TEXTURE_TRANSLATION_Z   "Z"
	#define KFCURVENODE_TEXTURE_ROTATION        "Rotation"
	#define KFCURVENODE_TEXTURE_ROTATION_X      "X"
	#define KFCURVENODE_TEXTURE_ROTATION_Y      "Y"
	#define KFCURVENODE_TEXTURE_ROTATION_Z      "Z"
	#define KFCURVENODE_TEXTURE_SCALING         "Scaling"
	#define KFCURVENODE_TEXTURE_SCALING_X       "X"
	#define KFCURVENODE_TEXTURE_SCALING_Y       "Y"
	#define KFCURVENODE_TEXTURE_SCALING_Z       "Z"
	#define KFCURVENODE_TEXTURE_ALPHA           "Alpha"

	// Addition for the material.
	#define KFCURVENODE_MATERIAL_EMISSIVE       "Emissive"
	#define KFCURVENODE_MATERIAL_EMISSIVE_R     "X"
	#define KFCURVENODE_MATERIAL_EMISSIVE_G     "Y"
	#define KFCURVENODE_MATERIAL_EMISSIVE_B     "Z"
	#define KFCURVENODE_MATERIAL_AMBIENT        "Ambient"
	#define KFCURVENODE_MATERIAL_AMBIENT_R      "X"
	#define KFCURVENODE_MATERIAL_AMBIENT_G      "Y"
	#define KFCURVENODE_MATERIAL_AMBIENT_B      "Z"
	#define KFCURVENODE_MATERIAL_DIFFUSE        "Diffuse"
	#define KFCURVENODE_MATERIAL_DIFFUSE_R      "X"
	#define KFCURVENODE_MATERIAL_DIFFUSE_G      "Y"
	#define KFCURVENODE_MATERIAL_DIFFUSE_B      "Z"
	#define KFCURVENODE_MATERIAL_SPECULAR       "Specular"
	#define KFCURVENODE_MATERIAL_SPECULAR_R     "X"
	#define KFCURVENODE_MATERIAL_SPECULAR_G     "Y"
	#define KFCURVENODE_MATERIAL_SPECULAR_B     "Z"
	#define KFCURVENODE_MATERIAL_OPACITY        "Opacity"
	#define KFCURVENODE_MATERIAL_REFLECTIVITY   "Reflectivity"
	#define KFCURVENODE_MATERIAL_SHININESS      "Shininess"

	// Addition for the generic vector property.
	#define KFCURVENODE_USER_PROPERTY_VECTOR_X  "X"
	#define KFCURVENODE_USER_PROPERTY_VECTOR_Y  "Y"
	#define KFCURVENODE_USER_PROPERTY_VECTOR_Z  "Z"

	// Addition for the generic color property.
	#define KFCURVENODE_USER_PROPERTY_COLOR_R  "X"
	#define KFCURVENODE_USER_PROPERTY_COLOR_G  "Y"
	#define KFCURVENODE_USER_PROPERTY_COLOR_B  "Z"


	// Addition of generic matrix
	#define KFCURVENODE_USER_PROPERTY_MATRIX "Matrix"
	#define KFCURVENODE_USER_PROPERTY_MATRIX_0 "0"
	#define KFCURVENODE_USER_PROPERTY_MATRIX_1 "1"
	#define KFCURVENODE_USER_PROPERTY_MATRIX_2 "2"
	#define KFCURVENODE_USER_PROPERTY_MATRIX_3 "3"
	#define KFCURVENODE_USER_PROPERTY_MATRIX_4 "4"
	#define KFCURVENODE_USER_PROPERTY_MATRIX_5 "5"
	#define KFCURVENODE_USER_PROPERTY_MATRIX_6 "6"
	#define KFCURVENODE_USER_PROPERTY_MATRIX_7 "7" 
	#define KFCURVENODE_USER_PROPERTY_MATRIX_8 "8"
	#define KFCURVENODE_USER_PROPERTY_MATRIX_9 "9"
	#define KFCURVENODE_USER_PROPERTY_MATRIX_A "A"
	#define KFCURVENODE_USER_PROPERTY_MATRIX_B "B"
	#define KFCURVENODE_USER_PROPERTY_MATRIX_C "C"
	#define KFCURVENODE_USER_PROPERTY_MATRIX_D "D"
	#define KFCURVENODE_USER_PROPERTY_MATRIX_E "E"
	#define KFCURVENODE_USER_PROPERTY_MATRIX_F "F"

	// Default array size for 1 minute of animation (30fps) 
	#define SELECTION_ARRAY_SIZE	30*60+1
	//#define SELECTION_ARRAY_SIZE 1024*1024

	#define KFCURVENODE_CANDIDATE_MAX_COUNT 4


	typedef int  kFCurveNodeType;

	K_FORWARD (KFCurveNode);
	typedef HKFCurveNode HIKFCurveNode;
	typedef KFCurveNode IKFCurveNode;

	#define IKFCurveNodeID 43763634

	enum EKFCurveNodeEventType
	{
		KFCURVENODEEVENT_NONE = 0, 
		KFCURVENODEEVENT_SELECTION = 1, 
		KFCURVENODEEVENT_DESTROY = 2, 
		KFCURVENODEEVENT_FCURVE = 4, 
		KFCURVENODEEVENT_TIMEWARP = 8, 
		KFCURVENODEEVENT_CTRLCURVE = 16, 
		KFCURVENODEEVENT_DATAUPDATED = 32
	};

	struct KExternalTimingInformation
	{
		KTime mLclOffset;
		KTime mDuration;
	};


	// Curve node event class.
	class KFCURVE_DLL KFCurveNodeEvent : public KEventBase
	{
	public:
		/* Nature (s) of this node event.
		*	It is a bit field.  Value defined in the enum stated before 
		*	the class can composited. 
		* stored in mType
		*/
		// Event count.
		int mEventCount;

		// Data.
		void* mData;

		// Clear the curve node event object.
		inline void Clear () 
		{
			mType=KFCURVENODEEVENT_NONE; 
			mEventCount = 0; 
			mData = NULL; 
		}

		/* Add an event of type pWhat.
		*	\param pWhat	Use EKFCurveNodeEventType to identify what pWhat means.
		*/
		inline void Add (int pWhat) 
		{
			mType |= pWhat;
			mEventCount++;
		}

		// Set the data contained by the node event.
		inline void SetData (void* pData) {mData = pData;}
	};


	struct KFCurveNodeCallback 
	{
  		KFCurveNodeEvent mEvent;
		KArrayUL mCallbackFunctions;   // no delete on object must use array ul
		KArrayUL mCallbackObjects;	   // no delete on object must use array ul
		bool mCallbackEnable;
	};

	typedef void (*kFCurveNodeCallback) (KFCurveNode* pFCurve, KFCurveNodeEvent* FCurveNodeEvent, void* pObject) ;

	typedef class KArrayTemplate< HKFCurveNode > KArrayTemplateFCurveNode;	

	// Return codes for function KFCurveNode::IsKey().
	enum 
	{ 
		ISKEY_NONE,
		ISKEY_SOME, 
		ISKEY_ALL 
	};


	class KFCURVE_DLL KFCurveNodeCandidateState
	{
	public:
		KFCurveNodeCandidateState();
		~KFCurveNodeCandidateState();

		void Dump(int pLevel = 0);

		void SetCandidateTotalTime      ( KTime &pCandidateTime ) { mCandidateTotalTime = pCandidateTime; };
		void SetCandidateSpecificTime   ( KTime &pCandidateTime ) { mCandidateSpecificTime = pCandidateTime; };

	protected:
		friend class KFCurveNode;
		
		int mSpecificCandidateMethod;
		int mTotalCandidateMethod;

		double mCandidateTotalValue[KFCURVENODE_CANDIDATE_MAX_COUNT];
		KTime	mCandidateTotalTime;

		double mCandidateSpecificValue;
		double mFCurveValue;
		KTime mCandidateSpecificTime;

		KFCurveNodeCandidateState *mSubStates[KFCURVENODE_CANDIDATE_MAX_COUNT];
		KFCurveNodeCandidateState *mLayerState;
		int						   mLayerID;

	};


	/**	This class provides the structure to build a curve node hierarchy.
	* \nosubgrouping
	*	It is a composite class that contains curve node tree management 
	* services in itself. Cyclic graphs are forbidden in a curve node 
	* hierarchy. 
	*
	* \par 
	* The content of a curve node is in its function curve, which is an 
	* instance of class KFCurve. A function curve can't be shared among 
	* curve nodes. 
	* \nosubgrouping
	*/
	#ifdef K_PLUGIN
		class KFCURVE_DLL KFCurveNode
	#else 
		class KFCURVE_DLL KFCurveNode : public IFBObjectHolder
	#endif
	{

	public:

		/**
		* \name Constructors and Destructor
		*/
		//@{

  		/** Constructor.
		* \param pNodeName Curve node name.
		* \param pTimeWarpName Curve node.
		* \param pDataType Curve node type.
		* \param pLayerType LayerType.
		* \param pLayerID LayerID.
		* \remarks No function curve is created in the new curve node.
		*/
		KFCurveNode 
		(
			char* pNodeName, 
			char* pTimeWarpName, 
			HKDataType pDataType, 
			int	 pLayerType = KFCL_NOLAYER,
			int	 pLayerID = 0
		) ;

		/** Template constructor.
		* \param pTemplateCurveNode Template curve node. 
		* \remarks This is not a copy constructor. This constructor makes the
		* created curve node share a structure that defines the node name, node 
		* display name and node type name. No function curve is created or 
		* copied in the new curve node.
		*/
		KFCurveNode(HKFCurveNode pTemplateCurveNode);

		//! Destructor.
		virtual ~KFCurveNode () ;

		#ifdef K_PLUGIN
			void Destroy (int Local=0) ;
		#else
			IObject_Declare (Implementation) 
		#endif

		//@}

		/**
		* \name FCurve Creation
		*/
		//@{

		/** Create the FCurve for this node.
		* This function is called recursively for all children curve nodes.
		*/
		void CreateFCurve( );

		/** Test to know if the FCurve is created.
		* This function is called recursively for all children curve nodes.
		* It returns false as soon as one child does not have its FCurve.
		*/
		bool FCurveCreated( );

		//@}

		/**
		* \name Cloning and Copy Functions
		*/
		//@{

		/** Return a duplicate of this node.
		* This function is called recursively for all children curve nodes.
		* The function curve is copied in the new curve node.
		*	\param pKeepAttached If \c true, the created object will share a structure 
		* that defines the node name, node display name and node type name. Otherwise,
		* this structure is copied.
		* \remarks This function is equal to a call to KFCurveNode::CloneTemplate() 
		* with parameter \c pCloneFCurves set to \c true.
		* \return A partial or complete copy of the current object.
		*/
		HKFCurveNode Clone(bool pKeepAttached);

		/** Return a duplicate of this node.
		* This function is called recursively for all children curve nodes.
		*	\param pKeepAttached If \c true, the created object will share a structure 
		* that defines the node name, node display name and node type name. Otherwise,
		* this structure is copied.
		*	\param pCloneFCurves If \c true and the current curve node has a function curve, 
		* the function curve is copied in the new curve node. 
		* \param pCreateCurves If \c true, \c pCloneFCurve is set to \c false and the 
		* current curve node has a function curve, an empty function curve is created 
		* in the new curve node.
		* \param pLayerID LayerID.
		* \return A partial or complete copy of the current object.
		*/
		HKFCurveNode CloneTemplate(bool pKeepAttached=false, bool pCloneFCurves=false, bool pCreateCurves=false, int pLayerID = -1);

		/**	Copy the function curve contained in the given curve node.
		* \param pSource Given curve node to copy the function curve from.
		* \param pTransferCurve If \c true, curve data is transferred instead of being copied.
		* \remarks This function is called recursively. The first child 
		* curve node of \c pSource is copied in the first child curve node 
		* of the current object and so on.
		*/
		void CopyFrom(HKFCurveNode pSource, bool pTransferCurve = false);

		/** Create a copy of the current curve node.
		* The structure defining the node name, node display name and node 
		* type name is copied, not shared. Only the animation keys in a 
		* given time range are copied.
		* \param pStart Start time for range of animation keys copied.
		* \param pStop Stop time for range of animation keys copied.
		* \return A partial or complete copy of the current object.
		*/
		HKFCurveNode Copy(KTime pStart=KTIME_MINUS_INFINITE, KTime pStop=KTIME_INFINITE);

		//@}

		/**
		* \name Node Identification
		*/
		//@{

		//! Retrieve the node's name.
		char* GetName () ;

		//! Retrieve the node's timewrap name.
		char* GetTimeWarpName() ;

		//! Retrieve the node type name.
		char* GetTypeName() ;

		//@}

		/**
		* \name Function Curve Management
		*/
		//@{

		/** Get pointer to the node's function curve.
		* \return Pointer to the curve or \c NULL pointer if there's none. 
		*/
		HKFCurve FCurveGet();

		/** Set the function curve of this curve node.
		*	\param pCurve New function curve.
		*	\param pDestroyOldCurve Set to \c true to delete the function 
		* curve already contained in the curve node.
		*	\return Function curve previously contained in the curve node
		* if there was any and if parameter \c pDestroyOldCurve is set 
		* to \c false. Otherwise, \c NULL is returned.
		*/
		KFCurve* FCurveSet (KFCurve* pCurve, bool pDestroyOldCurve = false) ;

		/** Replace the function curve of this curve node.
		*	Previous function curve is automatically destroyed.
		*	\param pCurve New function curve.
		* \remarks This function is equal to a call to KFCurveNode::FCurveSet() 
		* with parameter \c pDestroyOldCurve set to \c true.
		*/
		void FCurveReplace (KFCurve* pCurve) ;

		/** Return translation, rotation and scaling curves of the children of this curve node.
		* This function is meant to be called on the root KFCurveNode object found in class
		* KFbxTakeNode.
		*	\param pT Array to receive X, Y and Z translation function curves. 
		*	\param pR Array to receive X, Y and Z rotation function curves. 
		*	\param pS Array to receive X, Y and Z scaling function curves. 
		*	\return \c true if all function curves have been found, \c false otherwise.
		*/
		bool GetTransformCurves (KFCurve* pT[3], KFCurve* pR[3], KFCurve* pS[3]) ;

		//@}

		/**
		* \name Curve Node Management
		*/
		//@{

		/** Clear this node.
		* Delete the function curve and all the children curve nodes.
		*/
		virtual void Clear () ;

		//! Add a child curve node to this node.
		virtual int Add(HKFCurveNode pCurveNode);
		
		/** Remove a child curve node by index.
		* \param pIndex Index of child curve node to be removed.
		*/
		virtual void Remove (int pIndex) ;
		
		/** Remove and delete a child curve node by index.
		* \param pIndex Index of child curve node to be deleted.
		*/
		virtual void Delete (int pIndex) ;
		
		//! Get children curve nodes count.
		virtual int GetCount () ;
		
		/** Get pointer to child curve node by index.
		* \param pIndex Index of child curve node.
		* \return Pointer to child curve node or \c NULL is index is out of bounds.
		*/
		virtual HKFCurveNode Get(int pIndex) ;
		
		/** Ask if a given curve node is a child of the current curve node.
		* \param pCurveNode Curve node searched.
		* \param pRecursive Set to \c true to search recursively. Set to \c false
		* to search only in the immediate children curve nodes.
		* \return \c true if the curve node searched is found, \c false otherwise.
		*/
		virtual bool IsChild(KFCurveNode* pCurveNode, bool pRecursive = false) ;

		/** Get index of child curve node by name.
		* \param pName Name of searched child curve node.
		*	\return Index or -1 if a child curve node with this name could not be found. 
		* \remarks This function is not recursive.
		*/
		virtual int Find (char* pName) ;

		/** Get index of child curve node by pointer.
		* \param pNode Pointer to searched child curve node.
		*	\return Index or -1 if a child curve node with this pointer could not be found. 
		* \remarks This function is not recursive.
		*/
		virtual int Find(HKFCurveNode pNode);

		/** Recursively find the first child curve node by name.
		* \param pName Name of searched child curve node.
		* \return Pointer to child curve node or \c NULL if this name could not be found. 
		*/
		HKFCurveNode FindRecursive(char* pName);

		/** Find a child node by name or create it if it can not be found.
		* \param pName Name of child node to find.
		*	\param pFindOrCreateCurve If \c true and the function curve does not exist, create 
		* one in the curve node found or created.
		* \remarks This function is not recursive.
		*/
		HKFCurveNode FindOrCreate(char* pName, bool pFindOrCreateCurve = true) ;
		
		//! Set the parent curve node.
		virtual void SetParent(HKFCurveNode pParent) {mParent = pParent;}
		
		/** Get the parent curve node.
		*	\return Pointer to the parent curve node or \c NULL if there is none. 
		*/
		virtual HKFCurveNode GetParent() {return mParent;}

		//@}

		/**
		* \name Key Management
		*/
		//@{

		/** Get the number of animation keys.
		* Return the total number of keys contained in the function curves 
		* included in the current node and its children curve nodes.
		*/
		int KeyGetCount(bool pRecursiveInLayers = false);

		/** Get the number of animation keys.
		* \param pCurveCount returns the number of fcurves held by this node
		* \param pTotalCount returns the number of keys on all the fcurves held by this node
		* \param pMinCount returns the minimum number of keys on an fcurve held by this node
		* \param pMaxCount returns the maximum number of keys on an fcurve held by this node
		*/
		void KeyGetCount(int& pCurveCount, int& pTotalCount, int& pMinCount, int& pMaxCount);

		// Get the fcurve that has the earliest key. If many fcurves
		// has a key at the earliest time, returns one of the fcurves.
		void GetCurveWithFirstKey(KFCurve*& lCurveWithFirstKey, bool *pCurveMask /* = NULL */, bool pRecursiveInLayers);
		// Get the fcurve that has the latest key. If many fcurves has
		// a key at the latest time, returns one of the fcurves.
		void GetCurveWithLastKey(KFCurve*& lCurveWithLastKey, bool *pCurveMask /* = NULL */, bool pRecursiveInLayers);

		// Get the first key time in the children specified by the curve mask (NULL = all children)
		// If there is no key, return KTIME_MINUS_INFINITE.
		KTime GetFirstKeyTime(bool *pCurveMask = NULL, bool pRecursiveInLayers = false);
		// Get the last key time in the children specified by the curve mask (NULL = all children)
		// If there is no key, return KTIME_INFINITE.
		KTime GetLastKeyTime(bool *pCurveMask = NULL, bool pRecursiveInLayers = false);

		/** Recursively add a set of keys in the curve node tree.
		* Depth first key adding in the function curves included in the current 
		* node and its children curve nodes.
		* \param pTime Time to add the keys.
		* \param pValue Array containing all the key values. 
		* This array must contain as many elements as the number of function 
		* curves included in the current node and its children curve nodes.
		* \return Value of parameter \c pValue added with an offset equal to the
		* number of affected function curves times \c sizeof(kDouble).
		*/
		double* KeyAdd (KTime pTime, double* pValue) ;

		/** Recursively append a set of keys in the curve node tree.
		* Depth first key appending in the function curves included in the current 
		* node and its children curve nodes.
		* \param pTime Time set in the appended keys. Make sure this time has a 
		* greater value than any of the last keys in the affected function curves.
		* \param pValue Array containing all the key values. 
		* This array must contain as many elements as the number of function 
		* curves included in the current node and its children curve nodes.
		* \param pCheckForContinuousRotation Flag, when set to true we will check for continuous rotation values.
		* This is like an unroll filter on the fly. Relevant only for rotation fcurve nodes.
		* \return Value of parameter \c pValue added with an offset equal to the
		* number of function curves affected times \c sizeof(kDouble).
		*/
		double* KeyAppend (KTime pTime, double* pValue, bool pCheckForContinuousRotation = false) ;

		/** Find out start and stop time of the animation for a curve node and recursively in its children.
		*	\param pStart Reference to store start time. 
		* \c pStart is overwritten only if the start time found is lower than \c pStart value.
		* Initialize to KTIME_INFINITE to make sure the start time is overwritten in any case.
		*	\param pStop Reference to store end time.
		* \c pStop is overwritten only if the stop time found is higher than \c pStop value.
		* Initialize to KTIME_MINUS_INFINITE to make sure the stop time is overwritten in any case.
		*	\return \c true if at least one key has been found in all the function 
		* curves scanned.
		*/
		bool GetAnimationInterval (KTime& pStart, KTime& pStop) ;

		/** Find out start and stop time of the animation for a curve node and recursively in it's children.
		*	\param pStart Reference to receive the smaller key time, set to KTIME_INFINITE if no key could be found.
		*	\param pStop Reference to receive the larger key time, set to KTIME_MINUS_INFINITE if no key could be found.
		*/
		void GetTimeSpan		  (KTime& pStart, KTime& pStop) ;	

		/** Recursively check if keys exist at a given time in the curve node tree.
		* Check if a key exists at the given time in the function curves included 
		* in the current node and its children curve nodes.
		* \param pTime Given time to check key occurence.
		* \param pLast Function curve index to speed up search. If this 
		* function is called in a loop, initialize this value to 0 and let it 
		* be updated by each call.
		* \param pCurveMask allows you to use only specific children in the CurveNode. NULL means all of them.
		* \param pKeyIndexTolerance allows the test to be less strict when matching the time with a key index.
		*	\param	pMin	Minimal Value to Consider the Key.
		*	\param	pMax	Maximal Value to Consider the Key.
		*	\return \c ISKEY_NONE if no function curve has a key at the given time,
		*	\c ISKEY_SOME if some function curves have a key at the given time or
		*	\c ISKEY_ALL if all function curves have a key at the given time.
		*/
		int IsKey (KTime& pTime, int *pLast, bool* pCurveMask = NULL, double pKeyIndexTolerance = 0.0, double pMin = -K_DOUBLE_MAX,  double pMax = K_DOUBLE_MAX );

		/** Recursively delete keys at a given time in the curve node tree.
		* Delete keys at the given time in the function curves included 
		* in the current node and its children curve nodes.
		* \param pTime Given time to delete keys.
		* \param pCurveMask allows you to use only specific children in the CurveNode. NULL means all of them.
		* \param pKeyIndexTolerance allows to be less strict when matching the time with a key index.
		*	\return \c true if at least one key has been found at the given 
		* time and deleted.
		*/
		bool DeleteKey (KTime& pTime, bool* pCurveMask = NULL, double pKeyIndexTolerance = 0.0 );

		/** Recursively find the closest later time at which a key can be found compared to a given time.
		* Find the closest but later time at which a key can be found 
		* compared to a given time in the function curves included in 
		* the current node and its children curve nodes.
		* \param pTime Given time to search the next key time.
		* \param pCurveMask allows you to use only specific children in the CurveNode. NULL means all of them.
		* \param pRecursiveInLayers
		*	\return Next key time or KTIME_INFINITE if there exists no key
		* with a time larger than the given time.
		*/
		KTime GetNextKeyTime (KTime pTime, bool* pCurveMask = NULL, bool pRecursiveInLayers = false);

		/** Recursively find the closest ealier time at which a key can be found compared to a given time.
		* Find the closest earlier time at which a key can be found 
		* compared to a given time in the function curves included in 
		* the current node and its children curve nodes.
		* \param pTime Given time to search the previous key time.
		* \param pCurveMask allows you to use only specific children in the CurveNode. NULL means all of them.
		*	\return Previous key time or KTIME_MINUS_INFINITE if there exists no key
		* with a time smaller than the given time.
		*/
		KTime GetPrevKeyTime (KTime pTime, bool* pCurveMask = NULL );

		/** Recursively evaluate the curve node tree.
		* Depth first evaluation of the function curves included in the current 
		* node and its children curve nodes.
		* \param pValue Array to receive all the function curve evaluations. 
		* This array must be long enough to contain as many elements as the
		* number of function curves included in the current node and its 
		* children curve nodes.
		* \param pTime Time of evaluation.
		* \param pLast Function curve index to speed up evaluation. If this 
		* function is called in a loop, initialize this value to 0 and let it 
		* be updated by each call.
		* \return Value of parameter \c pValue added with an offset equal to the
		* number of function curves evaluated times \c sizeof(double).
		*/
		double* Evaluate (double* pValue, KTime pTime, kFCurveIndex* pLast=NULL) ;
		
		/** Recursively get the default values of the curve node tree.
		* Depth first access to the default values of the function curves 
		* included in the current node and its children curve nodes.
		* \param pValue Array to receive all the function curve default values. 
		* This array must be long enough to contain as many elements as the
		* number of function curves included in the current node and its 
		* children curve nodes.
		* \return Value of parameter \c pValue added with an offset equal to the
		* number of function curves accessed times \c sizeof(double).
		*/
		double* GetValue (double* pValue) ;
		
		/** Recursively set the default values of the curve node tree.
		* Depth first setting of the default values of the function curves 
		* included in the current node and its children curve nodes.
		* \param pValue Array containing all the function curve default values. 
		* This array must contain as many elements as the number of function 
		* curves included in the current node and its children curve nodes.
		* \return Value of parameter \c pValue added with an offset equal to the
		* number of function curves set times \c sizeof(double).
		*/
		double* SetValue (double* pValue) ;

		/** Delete all the keys found within a given time range.
		* Delete all the keys found within a given time range in the function 
		* curves included in the current node and its children curve nodes.
		* \param pStart Beginning of time range.
		* \param pStop End of time range.
		*/
		void Delete(KTime pStart=KTIME_MINUS_INFINITE, KTime pStop=KTIME_INFINITE);

		/** Replace all the keys found within a given time range.
		* Replace all the keys found within a given time range in the function 
		* curves included in the current node and its children curve nodes.
		* \param pSource Source node tree containing the replacement keys. The 
		* curve node tree below \c pSource must be identical to the curve node 
		* tree below the current node.
		* \param pStart Beginning of time range.
		* \param pStop End of time range.
		* \param pUseGivenSpan false = original behavior where the time of the first and last key was used
		* \param pKeyStartEndOnNoKey Inserts a key at the beginning and at the end of the range if there is no key to insert.
		* \param pTimeSpanOffset
		*/
	void Replace(HKFCurveNode pSource, KTime pStart = KTIME_MINUS_INFINITE, KTime pStop = KTIME_INFINITE, bool pUseGivenSpan = false, bool pKeyStartEndOnNoKey = true, KTime pTimeSpanOffset = KTIME_ZERO );

		//@}

	///////////////////////////////////////////////////////////////////////////////
	//
	//  WARNING!
	//
	//	Anything beyond these lines may not be documented accurately and is 
	// 	subject to change without notice.
	//
	///////////////////////////////////////////////////////////////////////////////

	#ifndef DOXYGEN_SHOULD_SKIP_THIS

		/**
		* \name Color
		*/
		//@{

		//! Get the color of the curve node.
		float* GetColor();

		//! Set The color of the curve node.
		void SetColor (float pColor[3]) ;

		//@}

		/**
		* \name Referencing
		*/
		//@{

		//! Increment and return the number of references.
		inline int IncReferenceCount () {return ++mReferenceCount;}

		//! Decrement and return the number of references.
		inline int DecReferenceCount () {mReferenceCount--; return mReferenceCount;}

		//! Return the number of references.
		inline int GetReferenceCount () {return mReferenceCount;}

		//@}

		/**
		* \name Take Type
		*/
		//@{

		//! Get the take type.
		int GetTakeType () ;

		//! Set the take type.
		void SetTakeType (int pType) ;

		//@}

		/**
		* \name Selection
		*/
		//@{

		//! Get the child node currently selected.
		bool GetVisibility () ;

		//! Set the selected child node.
		void SetVisibility (bool pVisible, bool pRecursive=false, bool pRecurseLayer = false, int lChildIndex = -1) ;

		//@}

		/**
		* \name Data Node Management
		*/
		//@{

		/** Set the data node.
		* \param pDataNode	Data node to set.
		*	\param pRecursive	\c true will set this data node to all children node. 
		*/
		void DataNodeSet (HKFCurveNode pDataNode, bool pRecursive=true) ;

		/** Retrieve the data node.
		*	\return Pointer to the node or NULL pointer if there is none. 
		*/
		HKFCurveNode DataNodeGet () ;

		/** Set to FCurve the Post or Pre Extrapolation Data.  If pRespectUserLock is on, no change is applied when in UserLock
		*	\return success or not. 
		*/
		bool SetPreExtrapolation(kUInt pNewPreExtrapolation, bool pRespectUserLock=true );
		bool SetPreExtrapolationCount(kUInt pNewPreExtrapolationCount, bool pRespectUserLock=true);
		bool SetPostExtrapolation(kUInt pNewPreExtrapolation, bool pRespectUserLock=true);
		bool SetPostExtrapolationCount(kUInt pNewPreExtrapolationCount, bool pRespectUserLock=true);

		//@}

		/**
		* \name Container Type
		*/
		//@{

		//! Get the container type.
		int GetContainerType () ;

		//! Set the container type.
		void SetContainerType (int pInOutType) ;

		//@}

		/**
		* \name I/O Type Management 
		*/
		//@{

		//! Get the IO Type.
		int GetIOType () ;

		//! Set the IO Type.
		void SetIOType (int pInOutType) ;

		//@}

		/**
		* \name Function Curve Editor Options 
		*/
		//@{

		//! If the node is expended in FCurveEditor
		int IsExpended () ;

		//! Sets if the node is expended in FCurveEditor.
		void SetExpended (bool pExpended) ;

		//@}

		/**
		* \name Layer Options 
		*/
		//@{

		//! If the node can have layers
		int IsMultiLayer() ;

		//! Sets if the node is expended in FCurveEditor.
		void SetMultiLayer (bool pMultiLayer) ;

		//! Get the fcurve node for the specified layer.
		HKFCurveNode GetLayerNode(int pLayerID);

		//! Extract Keys in the Specified Range
	#ifndef K_PLUGIN
		void ExtractKeysIndex( KArraykInt &pArray, KTimeSpan pTimeSpan, double pMinValue =  -K_DOUBLE_MAX, double pMaxValue =  K_DOUBLE_MAX);
	#endif

		//@}

		/**
		* \name Analysis
		*/
		//@{

		/*! Check if keys may come from a plot operation.
		* \param pThresholdPeriod If the keys period is larger than this threshold, the function will return false.
		*/
		bool LookLikeSampledData(KTime pThresholdPeriod);

		//@}

		/**
		* \name Udpate
		*/
		//@{

		//! Update id.
		int GetUpdateId () ;

		//! Update id.
		int GetValuesUpdateId ();

		//! Node update id.
		int GetNodeUpdateId () ;


		//@}

		/**
		* \name Callback
		*/
		//@{
		
		//!
		bool CallbackEnable (bool pEnable) ;
		
		//!
		bool CallbackIsEnable () ;
		
		//!
		void CallbackClear () ;
		
		//!
		void CallbackAddEvent (int pWhat) ;
		
		//!
		void CallbackRegister (kFCurveNodeCallback, void* pObject) ;
		
		//!
		void CallbackUnregister (kFCurveNodeCallback, void* pObject) ;

		//@}

	#ifndef K_PLUGIN
		/**
		* \name External Timing Information (for transportation only: optional, not used internally, and not persisted).
		*/
		//@{

		//! Time interface attached to this node.
		void SetETI(IKFCurveETI * pFCurveETI);
		IKFCurveETI * GetETI() const;

		//! Helpers for time conversions etc.
		KTime ETINodeTime2GlobalTime(KTime const & pTime);
		KTime ETIGlobalTime2NodeTime(KTime const & pTime);
		KTime ETIGetDuration();
		KTime ETIGetTrimOffset();
		KTime ETIGetGlobalOffset();
		double ETIGetScaleFactor();
		bool ETIGetColor(KgeRGBVector & pColor);
		bool ETIIsGlobalTimeInSpan(KTime const & pTime, bool pIncludeStop = true);
		typedef KTime (IKFCurveETI::*TETITimeFetcher)();
		KTime ETIGetSpecifiedTime(TETITimeFetcher pFunc);
		bool ETIIsSelectedDuration();
	#endif

		//@}

		bool	FbxStore (KFbx* pFbx, bool pOnlyDefaults = false, bool pReplaceLclTRSName = false, bool pSkipTimeWarpName = false, char* pFieldName = "Channel", bool pColor = true, bool pIsVersion5 = false ) ;
		bool	FbxRetrieve (KFbx* pFbx, bool pOnlyDefaults = false, bool pCreateIfNotFound = true,bool pColor = false);
		bool	FbxInternalRetrieve (KFbx* pFbx, bool pOnlyDefaults = false, bool pCreateIfNotFound = true,bool pColor = false) ;
		bool    FbxTimeWarpNameRetrieve(KFbx* pFbx);
		bool    FbxLayerAndTimeWarpRetrieve(KFbx* pFbx, bool pOnlyDefaults = false, bool pCreateIfNotFound = true,bool pColor = false) ;

		//@{
		/**
		* \name AntiGimble
		*/

		bool mUseQuaternion;
		inline bool GetUseQuaternion(){return mUseQuaternion;}; 
		void SetUseQuaternion(bool pVal); 
		private:
			void RotationNodeEvaluate(double* pValue, KTime pTime, kFCurveIndex* pLast,bool pEvaluateCandidate	);

	//@}

	public:
		//@{
		/**
		* \name Undo state management
		*/

		void GetCandidateState(KFCurveNodeCandidateState& pState);
		void SetCandidateState(const KFCurveNodeCandidateState& pState, bool pDestroyMissingLayers = false );

		int		GetCandidateSpecificMethod();
		int		GetCandidateTotalMethod();
		KTime	GetCandidateTotalTime();
		double*	GetCandidateTotalValue();
		int		GetCandidateTotalValueSize();

		void	SetCandidateSpecificMethod(int pMethod);
		void	SetCandidateTotalMethod(int pMethod);
		void	SetCandidateTotalTime(KTime pTime);
		void	SetCandidateTotalValue(double* pValue);

	//@}

	void GetQuaternionInterpolationBezierPoints(KTime pCurrentTime, KgeQuaternion &pQ0, KgeQuaternion &lQ1, KgeQuaternion &lQ2, KgeQuaternion &lQ3);

	protected:
		struct NameInfo 
		{
			KString		mName;
			KString		mTimeWarpName;
			HKDataType	mDataType;
			int		mRefCount;
		}*mNameInfo;

		void	SetNameInfo( NameInfo* pNameInfo );

		int mType;

		int mReferenceCount;
		KArrayTemplate<HKFCurveNode> mNodes;

		HIKFCurve	 mFCurve;
		HKFCurveNode mTimeWarp;
		HKFCurveNode mLayer;		// Ctrl Curve (Animation layering) 
		HKFCurveNode mObjectLayer;	// Pointer to Layer 0
		HKFCurveNode mTemplate;
		HKFCurveNode mDataNode;
		HKFCurveNode mParent;

		int mLayerType;
		int mLayerID;

		int mUpdateId;

		public:
		int mMBRotationOrder;
		inline int GetRotationOrder(){return mMBRotationOrder;}; 
		void SetRotationOrder(int pRotationOrder);

		//! Get layer Type ( = Mult, Add or Rotational node )
		int GetLayerType() { return mLayerType; }

		protected:
		//
		//	Candidates in nodes (Max is KFCURVENODE_CANDIDATE_MAX_COUNT)
		//
		
		void FromTotalCandidateToSpecificCandidate(int pLayerID);

		enum {eUseFCurve = 1, eUseTotal = 2};
		int mSpecificCandidateMethod;  // Used as a bit field
		int mTotalCandidateMethod;     // Used as an enum
										// To my knowledge the possible state are       
										// No validcandidate							mCandidateTotalTime == INFINITE and no FCurve candidate for the current time
										// Total candidate is valid and fcuve invalid   mTotalCandidateMethod ==  eUseTotal and mSpecificCandidateMethod == eUseTotal
										// FCurve is valid and Total invalid			mTotalCandidateMethod ==  eUseFCurve and mSpecificCandidateMethod == eUseFCurve
										// Total valid and FCurve computed				mTotalCandidateMethod ==  eUseTotal and mSpecificCandidateMethod == eUseTotal | eUseFCurve

										// FCurve valid and total computed				mTotalCandidateMethod ==  eUseTotal and mSpecificCandidateMethod == eUseFCurve
										// FCurve valid (and computed from total) and total computed	mTotalCandidateMethod ==  eUseTotal and mSpecificCandidateMethod == eUseTotal | eUseFCurve
						
					
										

		double mCandidateTotalValue[KFCURVENODE_CANDIDATE_MAX_COUNT];	// For all layers at once (only meaningful on ObjectLayer )
		KTime	mCandidateTotalTime;

		//
		//	Callbacks
		//
		
		KFCurveNodeCallback* mCallback;
		int mUserLock;


	#ifndef K_PLUGIN
		//
		//  External Timing Information (for transportation only: optional, not used internally, and not persisted).
		//
		IKFCurveETI * mETI;
	#endif

	private:
		void KeyGetCountRecursive(int& pCurveCount, int& pTotalCount, int& pMinCount, int& pMaxCount);
		bool GetAnimationIntervalRecursive (KTime& pStart, KTime& pStop) ;
		void IsKeyRecursive( KTime& pTime, int &pKeyCount, int &pKeyFound, int *pLast, bool *pCurveMask, double pKeyIndexTolerance = 0.0, double pMin = -K_DOUBLE_MAX,  double pMax = K_DOUBLE_MAX );

		friend KFCURVE_DLL HKFCurveNode KFCurveNodeCreate (KFbx* pFbx, HKFCurveNode pParent, bool pOnlyDefaults, bool pCreateIfNotFound, bool pColor );

	public:
	#ifndef K_FBXSDK_INTERFACE
			#include <kfcurve/kfcurvenodeinhouse.h>
	#endif	// K_FBXSDK_INTERFACE

	#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

	};


	/**	Create a curve node.
	* \param pNodeName Curve node name.
	* \param pTimeWarpName Curve node timewarp name.
	* \param pTypeName Curve node type name.
	* \param pLayerType LayerType.
	* \param pLayerID LayerID.
	* \return Pointer to allocated curve node.
	* \remarks No function curve is created in the new curve node.
	*/
	KFCURVE_DLL HKFCurveNode KFCurveNodeCreate(char* pNodeName, char* pTimeWarpName, char* pTypeName, int pLayerType = KFCL_NOLAYER, int pLayerID = 0);
	KFCURVE_DLL HKFCurveNode KFCurveNodeCreate(char* pNodeName, HKDataType pDataType=NULL, int pLayerType = KFCL_NOLAYER, int pLayerID = 0);


	/**	Create a curve node from a template.
	* \param pTemplateCurveNode Template curve node. 
	* \return Pointer to allocated curve node.
	* \remarks The created curve node shares with the template curve node 
	* a structure that defines the node name, node display name and node 
	* type name. No function curve is created in the new curve node.
	*/
	KFCURVE_DLL HKFCurveNode KFCurveNodeCreate(HKFCurveNode pTemplateCurveNode);

	// Create a curve node, FBX SDK internal use only.
	KFCURVE_DLL HKFCurveNode KFCurveNodeCreate(KFbx* pFbx, bool pOnlyDefaults = false, bool pColor = false);

	// Create a curve node, FBX SDK internal use only.
	KFCURVE_DLL HKFCurveNode KFCurveNodeCreate(KFbx* pFbx, HKFCurveNode pParent, bool pOnlyDefaults = false, bool pCreateIfNotFound = true, bool pColor = false);

	// Create a layer node, FBX SDK internal use only.
	KFCURVE_DLL HKFCurveNode KFCurveNodeLayerCreate (KFbx* pFbx,HKFCurveNode pParent, bool pOnlyDefaults = false );

	// Create a curve node from a datatype, FBX SDK internal use only.
	KFCURVE_DLL HKFCurveNode FCurveNodeCreateFromDataType( HKDataType pDataType,char* pNodeName);

#include <kfcurve/kfcurve_nsend.h>

#endif // #ifndef _FBXSDK_KFCURVE_NODE_H_


