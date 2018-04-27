#include "Precompiled.h"
#include "SplitPath.h"

namespace Mod
{

	SplitPath::SplitPath(const WCHAR* path) // note implicit for convenience
	{
		Split(path);
	}

	//------------------------------------------------------------------------

	SplitPath::SplitPath( const String& path )
	{
		Split( path.c_str() );
	}

	//------------------------------------------------------------------------

	SplitPath::~SplitPath()
	{

	}

	//------------------------------------------------------------------------

	void
	SplitPath::Split(const WCHAR* path)
	{
		MD_FERROR_ON_FALSE( wcslen(path) < 512 );

		WCHAR drive[512];
		WCHAR dir[512];
		WCHAR fname[512];
		WCHAR ext[512];

		_wsplitpath( path, drive, dir, fname, ext );

		mDrive	= drive;
		mDir	= dir;
		mFName	= fname;
		mExt	= ext;
	}

	//------------------------------------------------------------------------

	void
	SplitPath::ToLowerCase()
	{
		ToLower( mDrive );
		ToLower( mDir	);
		ToLower( mFName	);
		ToLower( mExt	);
	}

	//------------------------------------------------------------------------

	const String&
	SplitPath::Drive() const
	{
		return mDrive;
	}

	//------------------------------------------------------------------------

	const String&
	SplitPath::Dir()	const
	{
		return mDir;
	}

	//------------------------------------------------------------------------

	const String&
	SplitPath::FName() const
	{
		return mFName;
	}

	//------------------------------------------------------------------------

	const String&
	SplitPath::Ext()	const
	{
		return mExt;
	}

	//------------------------------------------------------------------------

	String
	SplitPath::RawExt()	const
	{
		if( mExt.empty() )
		{
			return String();
		}
		else
		{
			return String( mExt.begin() + 1, mExt.end() );
		}
	}

	//------------------------------------------------------------------------

	String
	SplitPath::StrippedExt() const
	{
		if( mExt.empty() )
			return String();
		else
		{
			String res = mExt;

			if( res[0] == '.' )
			{
				res.erase( res.begin() );
			}

			return res;
		}
	}

	//------------------------------------------------------------------------

	String
	SplitPath::Path() const
	{
		return mDrive + mDir;
	}

	//------------------------------------------------------------------------

	String
	SplitPath::NoExt() const
	{
		return Path() + mFName;
	}

	//------------------------------------------------------------------------
}