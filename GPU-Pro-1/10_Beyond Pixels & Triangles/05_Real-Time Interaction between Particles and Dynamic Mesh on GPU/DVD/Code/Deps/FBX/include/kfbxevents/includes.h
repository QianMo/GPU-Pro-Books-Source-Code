// Local includes
#include <kfbxevents/kfbxemitter.h>
#include <kfbxevents/kfbxevents.h>
#include <kfbxevents/kfbxlistener.h>
#include <kfbxevents/kfbxeventhandler.h>

// FBX namespace begin
#include <fbxfilesdk_nsbegin.h>

// FBX users must call this macro in synchronize event 
// type ids between DLLs and applications.
#define INITIALZE_FBX_PUBLIC_EVENTS \
    int ____FBXReservedIndex = 0x4000000;\
    KFbxQueryEvent<int>*_01_(0);_01_=_01_;KFbxQueryEvent<int>::ForceTypeId(++____FBXReservedIndex); \
    KFbxQueryEvent<float>*_02_(0);_02_=_02_;KFbxQueryEvent<float>::ForceTypeId(++____FBXReservedIndex); \
    KFbxQueryEvent<double>*_03_(0);_03_=_03_;KFbxQueryEvent<double>::ForceTypeId(++____FBXReservedIndex); \
    KFbxQueryEvent<KFbxSdkManager>*_04_(0);_04_=_04_;KFbxQueryEvent<KFbxSdkManager>::ForceTypeId(++____FBXReservedIndex); \
    KFbxQueryEvent<KFbxObject>*_05_(0);_05_=_05_;KFbxQueryEvent<KFbxObject>::ForceTypeId(++____FBXReservedIndex); \
    KFbxQueryEvent<KFbxDocument>*_06_(0);_06_=_06_;KFbxQueryEvent<KFbxDocument>::ForceTypeId(++____FBXReservedIndex); \
    KFbxQueryEvent<KFbxLibrary>*_07_(0);_07_=_07_;KFbxQueryEvent<KFbxLibrary>::ForceTypeId(++____FBXReservedIndex); \
    KFbxQueryEvent<KFbxImporter>*_08_(0);_08_=_08_;KFbxQueryEvent<KFbxImporter>::ForceTypeId(++____FBXReservedIndex); \
    KFbxQueryEvent<KFbxExporter>*_09_(0);_09_=_09_;KFbxQueryEvent<KFbxExporter>::ForceTypeId(++____FBXReservedIndex); 
    // MUST add new public event types HERE  

namespace kfbxevents
{
    inline void RegisterTypes(KFbxSdkManager& pSDKManager)
    { 
        // No type registration required.
    }
}

// FBX namespace end
#include <fbxfilesdk_nsend.h>

