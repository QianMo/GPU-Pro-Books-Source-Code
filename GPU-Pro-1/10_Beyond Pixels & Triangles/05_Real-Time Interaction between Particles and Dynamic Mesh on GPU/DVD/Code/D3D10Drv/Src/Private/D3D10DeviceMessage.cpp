#include "Precompiled.h"
#include "D3D10DeviceMessage.h"

namespace Mod
{

	D3D10DeviceMessage::D3D10DeviceMessage( D3D10_MESSAGE_ID messageId ) :
	mMessageID( messageId )
	{
	}

	//------------------------------------------------------------------------

	D3D10DeviceMessage::~D3D10DeviceMessage()
	{

	}

	//------------------------------------------------------------------------

	D3D10_MESSAGE_ID
	D3D10DeviceMessage::GetValue() const
	{
		return mMessageID;
	}

}