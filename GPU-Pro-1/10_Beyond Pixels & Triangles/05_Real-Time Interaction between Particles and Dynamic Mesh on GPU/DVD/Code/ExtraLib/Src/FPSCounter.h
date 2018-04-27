#ifndef EXTRALIB_FPSCounter_H_INCLUDED
#define EXTRALIB_FPSCounter_H_INCLUDED

#include "Forw.h"

#include "BlankExpImp.h"

#define MD_NAMESPACE FPSCounterNS
#include "ConfigurableImpl.h"

namespace Mod
{

	class FPSCounter : public FPSCounterNS::ConfigurableImpl<FPSCounterConfig>
	{
		// types
	public:

		// constructors / destructors
	public:
		explicit FPSCounter( const FPSCounterConfig& cfg );
		~FPSCounter();
	
		// manipulation/ access
	public:
		void	Update( bool update_timer = false );
		float	GetValue() const;

		// data
	private:
		float	mLastMeasurment;
		float	mValue;
		float	mAccum;
		UINT32	mAccumCount;

	};
}

#endif