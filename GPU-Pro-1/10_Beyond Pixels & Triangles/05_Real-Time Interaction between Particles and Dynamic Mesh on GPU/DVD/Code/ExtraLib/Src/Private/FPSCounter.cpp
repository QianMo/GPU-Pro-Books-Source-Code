#include "Precompiled.h"

#include "WrapSys/Src/Timer.h"

#include "FPSCounterConfig.h"
#include "FPSCounter.h"

#define MD_NAMESPACE FPSCounterNS
#include "ConfigurableImpl.cpp.h"

namespace Mod
{
	FPSCounter::FPSCounter( const FPSCounterConfig& cfg ) :
	Parent( cfg ),
	mLastMeasurment( 0 ),
	mValue( 0 ),
	mAccum( 0 ),
	mAccumCount( 0 )
	{
	}

	//------------------------------------------------------------------------

	FPSCounter::~FPSCounter() 
	{
	}

	//------------------------------------------------------------------------

	void
	FPSCounter::Update( bool update_timer /*= false*/ )
	{
		const ConfigType& cfg = GetConfig();

		if( update_timer )
		{
			cfg.timer->Update();
		}

		float delta		= cfg.timer->GetDeltaSecs();
		mLastMeasurment += delta;

		if( mLastMeasurment >= cfg.avarageSpan )
		{
			mValue = mAccum / ( mAccumCount + 1 );

			mAccum			= 0;
			mAccumCount		= 0;
			mLastMeasurment = 0;
		}
		else
		{
			if( delta > std::numeric_limits<float>::epsilon() )
			{
				float fps = 1 / delta;
				mAccum		+= fps;
				mAccumCount	++;
			}
		}
	}

	//------------------------------------------------------------------------

	float
	FPSCounter::GetValue() const
	{
		return mValue;
	}

}