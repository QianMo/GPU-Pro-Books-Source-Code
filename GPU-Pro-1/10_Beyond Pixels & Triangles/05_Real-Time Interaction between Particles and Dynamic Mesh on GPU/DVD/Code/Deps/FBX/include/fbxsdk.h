/*!  \file fbxsdk.h
 */

#ifndef _FbxSdk_h
#define _FbxSdk_h

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

/**
  * \mainpage FBX SDK Reference
  * <p>
  * \section welcome Welcome to the FBX SDK Reference
  * The FBX SDK Reference contains reference information on every header file, 
  * namespace, class, method, enum, typedef, variable, and other C++ elements 
  * that comprise the FBX software development kit (SDK).
  * <p>
  * The FBX SDK Reference is organized into the following sections:
  * <ul><li>Class List: an alphabetical list of FBX SDK classes
  *     <li>Class Hierarchy: a textual representation of the FBX SDK class structure
  *     <li>Graphical Class Hierarchy: a graphical representation of the FBX SDK class structure
  *     <li>File List: an alphabetical list of all documented header files</ul>
  * <p>
  * \section otherdocumentation Other Documentation
  * Apart from this reference guide, an FBX SDK Programming Guide and many FBX 
  * SDK examples are also provided.
  * <p>
  * \section aboutFBXSDK About the FBX SDK
  * The FBX SDK is a C++ software development kit (SDK) that lets you import 
  * and export 3D scenes using the Autodesk FBX file format. The FBX SDK 
  * reads FBX files created with FiLMBOX version 2.5 and later and writes FBX 
  * files compatible with MotionBuilder version 6.0 and up. 
  */

#ifndef K_FBXSDK_INTERNALPLUGINS
    #ifndef K_PLUGIN
        #define K_PLUGIN
    #endif

    #ifndef K_FBXSDK
        #define K_FBXSDK
    #endif

	#ifdef MB_FBXSDK
		#undef K_DLLIMPORT
		#undef K_DLLEXPORT
        
		// import and export definitions
		#if (defined(_MSC_VER) || (defined(__GNUC__) && defined(_WIN32))) && !defined(MB_FBXSDK)
			#define K_DLLIMPORT __declspec()
			#define K_DLLEXPORT __declspec()
		#else 
			#define K_DLLIMPORT
			#define K_DLLEXPORT
		#endif
        
		#define _K_KAYDARADEF_H
		#include <fbxfilesdk/components/kbaselib/karch/arch.h>
        
		#define KFBX_DLL
		#define KBASELIB_DLL
		#define KFCURVE_DLL
		#define KFBX_DLL
	#else
		#ifndef K_FBXSDK_INTERFACE
			#define K_FBXSDK_INTERFACE
		#endif
	#endif /* MB_FBXSDK */
#endif

#ifndef KFBX_DLL 
    #define KFBX_DLL K_DLLIMPORT
#endif

#include <fbxfilesdk_def.h>
#include <kaydara.h>

// io
#include <kfbxio/kfbxio.h>
#include <kfbxio/kfbxexporter.h>
#include <kfbxio/kfbximporter.h>
#include <kfbxio/kfbxstreamoptionsfbx.h>
#include <kfbxio/kfbxstreamoptionsdxf.h>
#include <kfbxio/kfbxstreamoptions3ds.h>
#include <kfbxio/kfbxstreamoptionscollada.h>
#include <kfbxio/kfbxstreamoptionsobj.h>

// math
#include <kfbxmath/kfbxmatrix.h>
#include <kfbxmath/kfbxquaternion.h>
#include <kfbxmath/kfbxvector2.h>
#include <kfbxmath/kfbxvector4.h>
#include <kfbxmath/kfbxxmatrix.h>

// plugins
#include <kfbxplugins/kfbxsdkmanager.h>
#include <kfbxplugins/kfbxmemoryallocator.h>
#include <kfbxplugins/kfbxscene.h>
#include <kfbxplugins/kfbxgloballightsettings.h>
#include <kfbxplugins/kfbxglobalcamerasettings.h>
#include <kfbxplugins/kfbxglobaltimesettings.h>
#include <kfbxplugins/kfbxcolor.h>
#include <kfbxplugins/kfbxnode.h>
#include <kfbxplugins/kfbxnodeattribute.h>
#include <kfbxplugins/kfbxnodeiterator.h>
#include <kfbxplugins/kfbxmarker.h>
#include <kfbxplugins/kfbxcamera.h>
#include <kfbxplugins/kfbxcameraswitcher.h>
#include <kfbxplugins/kfbxlight.h>
#include <kfbxplugins/kfbxopticalreference.h>
#include <kfbxplugins/kfbxskeleton.h>
#include <kfbxplugins/kfbxgeometry.h>
#include <kfbxplugins/kfbxgeometrybase.h>
#include <kfbxplugins/kfbxgeometryconverter.h>
#include <kfbxplugins/kfbxgeometryweightedmap.h>
#include <kfbxplugins/kfbxmesh.h>
#include <kfbxplugins/kfbxnurb.h>
#include <kfbxplugins/kfbxpatch.h>
#include <kfbxplugins/kfbxtexture.h>
#include <kfbxplugins/kfbxsurfacematerial.h>
#include <kfbxplugins/kfbxsurfacelambert.h>
#include <kfbxplugins/kfbxsurfacephong.h>
#include <kfbxplugins/kfbxdeformer.h>
#include <kfbxplugins/kfbxskin.h>
#include <kfbxplugins/kfbxsubdeformer.h>
#include <kfbxplugins/kfbxcluster.h>
#include <kfbxplugins/kfbxshape.h>
#include <kfbxplugins/kfbxtakenode.h>
#include <kfbxplugins/kfbxtakeinfo.h>
#include <kfbxplugins/kfbxpose.h>
#include <kfbxplugins/kfbxnull.h>
#include <kfbxplugins/kfbxthumbnail.h>
#include <fbxcore/fbxcollection/kfbxdocumentinfo.h>
#include <kfbxplugins/kfbxproperty.h>
#include <kfbxplugins/kfbxuserproperty.h>
#include <kfbxplugins/kfbxutilities.h>
#include <kfbxplugins/kfbxvideo.h>
#include <kfbxplugins/kfbxgenericnode.h>
#include <kfbxplugins/kfbxconstraint.h>
#include <kfbxplugins/kfbxconstraintaim.h>
#include <kfbxplugins/kfbxconstraintparent.h>
#include <kfbxplugins/kfbxconstraintposition.h>
#include <kfbxplugins/kfbxconstraintrotation.h>
#include <kfbxplugins/kfbxconstraintscale.h>
#include <kfbxplugins/kfbxconstraintsinglechainik.h>
#include <kfbxplugins/kfbxweightedmapping.h>
#include <kfbxplugins/kfbxnurbscurve.h>
#include <kfbxplugins/kfbxtrimnurbssurface.h>
#include <kfbxplugins/kfbxnurbssurface.h>
#include <kfbxplugins/kfbxiopluginregistry.h>
#include <kfbxplugins/kfbxcache.h>
#include <kfbxplugins/kfbxshadingnode.h>
#include <kfbxplugins/kfbxvertexcachedeformer.h>
#include <kfbxplugins/kfbxrootnodeutility.h>
#include <kfbxplugins/kfbxbindingtable.h>
#include <kfbxplugins/kfbxbindingtableentry.h>
#include <kfbxplugins/kfbxlibrary.h>
#include <kfbxplugins/kfbxlayeredtexture.h>
#include <kfbxplugins/kfbxclonemanager.h>
#include <kfbxplugins/kfbxobjectmetadata.h>
#include <kfbxplugins/kfbxperipheral.h>
#include <kfbxplugins/kfbxproceduralgeometry.h>
#include <kfbxplugins/kfbxmanipulators.h>

#include <fbxcore/kfbxquery.h>
#include <fbxcore/fbxcollection/kfbxpropertymap.h>

#include <fbxcore/fbxxref/fbxxref.h>

#include <kfbxcharacter/kfbxcharacter.h>
#include <kfbxcharacter/kfbxcharacterpose.h>
#include <kfbxcharacter/kfbxcontrolset.h>

#include <kfcurve/kfcurve.h>
#include <kfcurve/kfcurvenode.h>
#include <kfcurve/kfcurveutils.h>

#include <klib/karrayul.h>
#include <klib/kerror.h>
#include <klib/kmemory.h>
#include <klib/kset.h>
#include <klib/kstring.h>
#include <klib/ktime.h>
#include <klib/kscopedptr.h>
#include <klib/kdynamicarray.h>
#include <klib/kmap.h>

// Events
#include <kfbxevents/includes.h>

// Mp
#include <kfbxmp/kfbxmutex.h>

#ifndef MB_FBXSDK
	using namespace FBXFILESDK_NAMESPACE;
#else
	#ifndef K_FBXSDK_INTERNALPLUGINS
    	#undef K_DLLIMPORT
	    #undef K_DLLEXPORT
    	#undef _K_KAYDARADEF_H
	#endif
	#include <fbxfilesdk/components/kbaselib/kaydaradef.h>
#endif

#endif /* _FBXSDK_H_ */

