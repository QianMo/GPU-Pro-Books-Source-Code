// http://www.codeproject.com/cpp/stlxmlparser.asp

#include "stdafx.h"
#include "XmlStream.h"
#include "XmlParser.h"


#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

// parser id for debugging
long XmlParser::_parseid = 0;


// show if display debug info
const bool showDebugInfo = false;

// save/load stream
bool XmlStream::save ( char * buffer )
{
	// show success
	bool success = true;
	try
	{
		// save stream to buffer
		memcpy( buffer,(char *) c_str(), size() );
	}
	catch(...)
	{
		success = false;
	}

	return success;
}

bool XmlStream::load ( char * buffer )
{
	// if invalid show failed
	if ( !buffer )
		return false;

	// show success
	bool success = true;
	try
	{
		// load stream from buffer
		str() = buffer;
	}
	catch(...)
	{
		success = false;
	}

	return success;
}




// notify methods
void XmlStream::foundNode ( string & name, string & attributes )
{
	// if no name stop
	if ( name.empty() )
		return;

	if ( showDebugInfo )
	{
		// debug info
		cout << "Found Node: " << endl;
		cout << "name: " << name << endl;

		if ( !attributes.empty() )
			cout << "attributes: " << attributes << endl;

		cout << endl;
	}
	
	// tell subscriber
	if ( _subscriber )
		_subscriber->foundNode(name,attributes);
}

void XmlStream::foundElement ( string & name, string & value, string & attributes )
{
	// if no name stop
	if ( name.empty() )
		return;

	// debug info
	if ( showDebugInfo )
	{
		cout << "Found Element: " << endl;
		cout << "name: " << name << endl;

		if ( !value.empty() )
			cout << "value: " << value << endl;

		if ( !attributes.empty() )
			cout << "attributes: " << attributes << endl;

		cout << endl;
	}
	
	// tell subscriber
	if ( _subscriber )
		_subscriber->foundElement(name,value,attributes);
}

void XmlStream::startElement ( string & name, string & value, string & attributes )
{
	if ( _subscriber )
		_subscriber->startElement(name,value,attributes);
}

void XmlStream::endElement ( string & name, string & value, string & attributes )
{
	if ( _subscriber )
		_subscriber->endElement(name,value,attributes);
}



bool XmlStream::parseNodes ( XmlParser & parser, char * buffer, long parseLength )
{
	// #DGH note
	// need to address a null node within another node
	// i.e.
	// <Contacts>
	//		<Contact/>
	// </Contacts>
	// in this instance Contact will be reported as an
	// element 

	// debug info
	string s(buffer);
	string::iterator buf = s.begin();
	if ( showDebugInfo )
	{
		cout << "parse node: " << parser._id << endl;
		cout << endl;
	}

	// while parse success, note for the first parser
	// this set the internal state also
	while ( parser.parse(buffer,parseLength) )
	{
		// if the value has a tag marker
		if ( parser.valueHasTag() )
		{
			// show found node
			string   name		 = parser.getName();
			string	 attributes  = parser.getAttributes();

			foundNode( name, attributes );

			// get the parse state
			long valueLength;
			char * valueBuffer =
			parser.getValueState(valueLength);

			// if parsing the node fails
			XmlParser parserNode;
			if ( !parseNodes(parserNode,valueBuffer,valueLength) )
				return false;

			// if new parse cur position is in the last parser
			// last tag position we are done with the node
			char * curPos     = parserNode.getCurPos();
			char * lastCurPos = parser.getLastTagPos();
			if ( curPos >= lastCurPos )
			{
				break;
			}
		}
		else
		{
			// show found element
			string   name		 = parser.getName();
			string   value		 = parser.getValue();
			string	 attributes  = parser.getAttributes();

			foundElement(name,value,attributes);
		}

		// get new parse state
		buffer =
		parser.getParseState(parseLength);
	}

	return true;
}



// parse the buffer

bool XmlStream::parse ( char * buffer, long length )
{
	// if invalid stop
	if ( !buffer || length <= 0 )
		return false;

	// debug info
	assign( buffer, length );

	// debug info
	if ( showDebugInfo )
	{
		cout << "----- parse document -----" << endl;
		cout << buffer << endl;
		cout << endl;
		cout << "----- start parsing -----" << endl;
		cout << endl;
	}

	// declare parser
	XmlParser parser;

	// parse nodes
	bool docParsed = parseNodes(parser,buffer,length);

	return docParsed;
}


