#ifndef D3D10DRV_D3D10SRV_H_INCLUDED
#define D3D10DRV_D3D10SRV_H_INCLUDED

namespace Mod
{

	class D3D10SRView
	{
		// types
	public:
		typedef ComPtr<ID3D10ShaderResourceView> SRViewPtr;

		// construction/ destruction
	public:
		explicit D3D10SRView( SRViewPtr::PtrType srv );
		~D3D10SRView();

		template <typename BindStockType>
		void BindTo( BindStockType& bindStock, UINT32 bindSlot ) const;

		// data
	private:
		SRViewPtr mSRView;
	};

	//------------------------------------------------------------------------

	template <typename BindStockType>
	void D3D10SRView::BindTo( BindStockType& bindStock, UINT32 bindSlot ) const
	{
		bindStock.Attach( bindSlot, &*mSRView );
	}
}

#endif