#ifndef D3D10DRV_D3D10DEVICEMESSAGE_H_INCLUDED
#define D3D10DRV_D3D10DEVICEMESSAGE_H_INCLUDED

#include "Wrap3D\Src\DeviceMessage.h"

namespace Mod
{

	class D3D10DeviceMessage : public DeviceMessage
	{
		// types
	public:
		typedef DeviceMessage Base;

		// construction/ destruction
	public:
		D3D10DeviceMessage( D3D10_MESSAGE_ID messageId );
		virtual ~D3D10DeviceMessage();

		// manipulation/ access
	public:
		D3D10_MESSAGE_ID	GetValue() const;

		// data
	private:
		D3D10_MESSAGE_ID	mMessageID;

	};

}

#endif