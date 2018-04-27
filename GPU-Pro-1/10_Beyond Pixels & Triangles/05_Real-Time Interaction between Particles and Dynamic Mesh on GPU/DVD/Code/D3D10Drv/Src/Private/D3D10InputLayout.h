#ifndef D3D10DRV_D3D10INPUTLAYOUT_H_INCLUDED
#define D3D10DRV_D3D10INPUTLAYOUT_H_INCLUDED

#include "Wrap3D/Src/InputLayout.h"

namespace Mod
{

	class D3D10InputLayout : public InputLayout
	{
		// types
	public:

		// construction/ destruction
	public:
		explicit D3D10InputLayout( const InputLayoutConfig& cfg, ID3D10Device* dev );
		~D3D10InputLayout();

	public:
		void Bind( ID3D10Device* dev ) const;

		// data
	private:
		ComPtr< ID3D10InputLayout > mResource;
	};

}

#endif