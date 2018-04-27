#include "WrapSys/Src/File.h"

#include "Common/Src/FileHelpers.h"

#include "AnimationTrack.h"

namespace Mod
{
	//------------------------------------------------------------------------

	template <typename Curve>
	void ReadCurveFromFile( Curve& oCurve, const RFilePtr& file )
	{
		UINT32 numKeys;
		file->Read(numKeys);
		for(UINT32 i=0;i<numKeys;i++)
		{
			typename Curve::ItemType	value;
			typename Curve::Time		time;
			file->Read(time);
			file->Read(value);
			oCurve.AddKey(value, time);
		}
	}

	//------------------------------------------------------------------------

	template <typename AnimTrack>
	AnimTrack ReadAnimTrackFromFile( const RFilePtr& file )
	{
		typename AnimTrack::TCurve pcurve;
		typename AnimTrack::RCurve rcurve;
		ReadCurveFromFile(pcurve, file);
		ReadCurveFromFile(rcurve, file);

		return AnimTrack(pcurve, rcurve);	
	}

	//------------------------------------------------------------------------
}

