// http://www.codeproject.com/cpp/stlxmlparser.asp

#if !defined(XmlStream_H)
#define XmlStream_H

#include "XmlParser.h"
#include "XmlNotify.h"

#include <string>
#include <iostream>
using namespace std;



//////////////////////////////////////////////////////////////////////////////
// XmlStream
//
// Purpose: stores a string that contain start,end tag delimited value


class XmlStream :
	public string
{
	XmlNotify *			_subscriber;	// notification subscriber
public:

	XmlStream () :

		string (),
		_subscriber(NULL)

	{}

	virtual ~XmlStream ()
	{ 
		release(); 
	}

	// release resources
	bool create ()
	{
		return true;
	}

	bool create ( char * buffer, long len )
	{
		if ( buffer && len > 0 )
		{
			assign( buffer, len );
			return true;
		}
		else
			return false;
	}

	void release ()
	{
		erase( begin(), end() );
	}

	// notify methods
	void foundNode		( string & name, string & attributes );
	void foundElement	( string & name, string & value, string & attributes );

	void startElement	( string & name, string & value, string & attributes );
	void endElement		( string & name, string & value, string & attributes );

	// save/load stream
	bool save		( char * buffer );
	bool load		( char * buffer );

	// parse the current buffer
	bool parse			();
	bool parse			( char * buffer, long parseLength );
	bool parseNodes		( XmlParser & parser, char * buffer, long parseLength );

	// get/set subscriber
	bool hasSubscriber ()
	{
		if ( _subscriber )
			return true;
		else
			return false;
	}

	XmlNotify * getSubscriber ()
	{
		return _subscriber;
	}

	void setSubscriber ( XmlNotify & set )
	{
		_subscriber = &set;
	}


	// get ref to tag stream
	XmlStream & getTagStream ()
	{ return *this; }


	// get string ref
	string & str()
	{ return (string &) *this; }
};



#endif