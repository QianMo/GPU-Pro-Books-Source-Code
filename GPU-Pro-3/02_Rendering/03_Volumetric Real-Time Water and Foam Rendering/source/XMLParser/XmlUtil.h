// http://www.codeproject.com/cpp/stlxmlparser.asp

#if !defined(XmlUtil_H)
#define XmlUtil_H


/////////////////////////////////////////////////////////////////////////
// define tag markers

#define idTagLeft	"<"
#define idTagRight	">"
#define idTagEnd	"</"
#define idTagNoData "/>"


#define idTagLeftLength		1
#define idTagRightLength	1
#define idTagEndLength		2


#include <string>
using namespace std;



//////////////////////////////////////////////////////////////////////////////////
// XmlUtil
//
// Purpose:		provides xml utility methods

class XmlUtil
{

	// tag helper methods
	static string getStartTag ( string & text )
	{
		string tag = idTagLeft;
		tag += text;
		tag += idTagRight;

		return string(tag);
	}

	// static helper methods
	static string getEndTag ( string & text )
	{
		string tag = idTagEnd;
		tag += text;
		tag += idTagRight;

		return string(tag);
	}
/*
	static string getStartTag ( LPCTSTR text )
	{
		string tag = idTagLeft;
		tag += text;
		tag += idTagRight;

		return string(tag);
	}

	static string getEndTag ( LPCTSTR text )
	{
		string tag = idTagEnd;
		tag += text;
		tag += idTagRight;

		return string(tag);
	}
*/
};


#endif