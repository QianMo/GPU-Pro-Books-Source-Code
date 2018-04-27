#ifndef COMMON_SPLITPATH_H_INCLUDED
#define COMMON_SPLITPATH_H_INCLUDED

namespace Mod
{
	class SplitPath
	{
		// construction/destruction
	public:
		SplitPath( const WCHAR* path ); // note implicit for convenience
		SplitPath( const String& path );
		~SplitPath();

		// manipulation/ access
	public:
		void		Split(const WCHAR* path);
		void		ToLowerCase();
		
		const String&	Drive()		const;
		const String&	Dir()		const;
		const String&	FName()		const;
		const String&	Ext()		const;
		String			RawExt()	const;
		
		String		StrippedExt() const; // no dot

		String		Path() const;
		String		NoExt() const; // all but extension

		// data
	private:
		String mDrive;
		String mDir;
		String mFName;
		String mExt;
	};

}

#endif