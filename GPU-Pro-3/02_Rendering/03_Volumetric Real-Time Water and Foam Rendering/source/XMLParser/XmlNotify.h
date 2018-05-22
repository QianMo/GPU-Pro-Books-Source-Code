// http://www.codeproject.com/cpp/stlxmlparser.asp

#if !defined(XmlNotify_H)
#define XmlNotify_H

/////////////////////////////////////////////////////////////////////////////////
// XmlNotify
//
// Purpose:		abstract class used to notify about xml tags found.

class XmlNotify
{
public:


	XmlNotify () {}

	// notify methods
	virtual void foundNode		( string & name, string & attributes ) = 0;
	virtual void foundElement	( string & name, string & value, string & attributes ) = 0;

	virtual void startElement	( string & name, string & value, string & attributes ) = 0;
	virtual void endElement		( string & name, string & value, string & attributes ) = 0;

	
};

#endif