//****************************************************************************//
// streamsource.h                                                            //
// Copyright (C) 2001-2003 Bruno 'Beosil' Heidelberger                       //
//****************************************************************************//
// This library is free software; you can redistribute it and/or modify it    //
// under the terms of the GNU Lesser General Public License as published by   //
// the Free Software Foundation; either version 2.1 of the License, or (at    //
// your option) any later version.                                            //
//****************************************************************************//

#ifndef CAL_STREAMSOURCE_H
#define CAL_STREAMSOURCE_H

//****************************************************************************//
// Includes                                                                   //
//****************************************************************************//

#include "cal3d/global.h"
#include "cal3d/datasource.h"
#include <istream>

/**
 * CalStreamSource class.
 *
 * This is an object designed to represent a source of Cal3d data as coming from
 * a standard input stream.
 */


class CAL3D_API CalStreamSource : public CalDataSource
{
public:
   CalStreamSource(std::istream& inputStream);
   virtual ~CalStreamSource();

   virtual bool ok() const;
   virtual void setError() const;
   virtual bool readBytes(void* pBuffer, int length);
   virtual bool readFloat(float& value);
   virtual bool readInteger(int& value);
   virtual bool readString(std::string& strValue);

protected:

   std::istream* mInputStream;

private:
   CalStreamSource(); //Can't use this
};

#endif
