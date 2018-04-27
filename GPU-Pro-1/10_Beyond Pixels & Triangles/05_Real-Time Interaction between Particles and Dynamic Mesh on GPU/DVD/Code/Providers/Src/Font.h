#ifndef PROVIDERS_Font_H_INCLUDED
#define PROVIDERS_Font_H_INCLUDED

#include "Forw.h"

#include "ExportDefs.h"

#define MD_NAMESPACE FontNS
#include "ConfigurableImpl.h"

namespace Mod
{

	class Font : public FontNS::ConfigurableImpl<FontConfig>
	{
		// types
	public:
		typedef Parent Base;
		typedef Font Parent;

		struct TextParams
		{
			enum Align
			{
				LEFT_TOP,
				CENTRE_TOP
			};

			EXP_IMP TextParams();

			AnsiString	text;

			Align		align;

			float		x;
			float		y;
			float		size;
		};

		struct FontParams
		{
			float			defSize;
			Math::float2	spacing;
		};

		typedef Types< TextParams > :: Vec TextParamsVec;

		// constructors / destructors
	public:
		explicit Font( const FontConfig& cfg );
		virtual ~Font();
	
		// manipulation/ access
	public:
		EXP_IMP void BeginRender();
		EXP_IMP void Write( const TextParams& params );
		EXP_IMP void EndRender();

		// child access
	protected:
		TextParamsVec&			textParamsVec();
		const FontParams&		fontParams() const;

		// polymorphism
	private:
		virtual void BeginRenderImpl() = 0;
		virtual void EndRenderImpl() = 0;

		// data
	private:
		TextParamsVec	mTextParamsVec;
		bool			mInRender;

		FontParams		mFontParams;
	};
}

#endif