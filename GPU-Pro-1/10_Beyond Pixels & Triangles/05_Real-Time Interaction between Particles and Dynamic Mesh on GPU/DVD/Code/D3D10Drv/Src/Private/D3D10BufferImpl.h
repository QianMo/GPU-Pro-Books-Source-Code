#ifndef D3D10DRV_D3D10BUFFERIMPL_H_INCLUDED
#define D3D10DRV_D3D10BUFFERIMPL_H_INCLUDED

#include "D3D10Buffer.h"

namespace Mod
{

	// nothing special, maybe smthing will arise later

	template<typename Config>
	class D3D10BufferImpl : public D3D10Buffer
	{
		// types
	public:
		typedef Config								Config;
		typedef Parent								Base;
		typedef D3D10BufferImpl<Config>				Parent; // we're this parent
		typedef typename Config::Child				Child;
		typedef typename Config::BufConfigType		BufConfigType;

		// construction/ destruction
	public:
		D3D10BufferImpl( const BufConfigType& cfg, UINT32 bindFlags, ID3D10Device* dev );
		virtual ~D3D10BufferImpl() = 0;	
	};

}

#endif