#ifndef DEBUGLIB_DEBUG_H_INCLUDED
#define DEBUGLIB_DEBUG_H_INCLUDED

namespace Mod
{
	void FatalError( const String& message );
	void DebugMessage( const String& message );

#ifdef _DEBUG
	#define MD_FERROR_ADD_DEBUG_BREAK MD_ASSERT_FALSE( L"<NONE>. About to terminate due to fatal error!" );
#else
	#define MD_FERROR_ADD_DEBUG_BREAK
#endif

	#define MD_FERROR1(expr) { MD_FERROR_ADD_DEBUG_BREAK Mod::FatalError(expr); }
	#define MD_FERROR(expr) MD_FERROR1(expr)
	#define MD_FERROR_ON_TRUE2(expr,line,file) if( (expr) ) MD_FERROR( L"Fatal error occurred ( " L#expr L" ) at line " L#line L" of file " L#file )

	#define MD_FERROR_ON_TRUE1(expr,line,file) MD_FERROR_ON_TRUE2(expr,line,file)

	#define MD_FERROR_ON_FALSE(expr) MD_FERROR_ON_TRUE1(!(expr),__LINE__, __FILE__)
	#define MD_FERROR_ON_TRUE(expr) MD_FERROR_ON_TRUE1(expr,__LINE__, __FILE__)

	//------------------------------------------------------------------------

	#define MD_FERROR1_MSG(expr,msg) { MD_FERROR_ADD_DEBUG_BREAK Mod::FatalError(expr + (msg) ); }
	#define MD_FERROR_MSG(expr,msg) MD_FERROR1_MSG(expr,msg)

	#define MD_FERROR_ON_TRUE2_MSG(expr,line,file,msg) if( (expr) ) MD_FERROR_MSG( L"Fatal error occurred ( " L#expr L" ) at line " L#line L" of file " L#file L". ", msg )

	#define MD_FERROR_ON_TRUE1_MSG(expr,line,file,msg) MD_FERROR_ON_TRUE2_MSG(expr,line,file,msg)

	#define MD_FERROR_ON_FALSE_MSG(expr,msg) MD_FERROR_ON_TRUE1_MSG(!(expr),__LINE__, __FILE__,msg)
}

#endif