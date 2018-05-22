//****************************************************************************//
// datasource.h                                                              //
// Copyright (C) 2001-2003 Bruno 'Beosil' Heidelberger                       //
//****************************************************************************//
// This library is free software; you can redistribute it and/or modify it    //
// under the terms of the GNU Lesser General Public License as published by   //
// the Free Software Foundation; either version 2.1 of the License, or (at    //
// your option) any later version.                                            //
//****************************************************************************//

#ifndef CAL_DATASOURCE_H
#define CAL_DATASOURCE_H

//****************************************************************************//
// Includes                                                                   //
//****************************************************************************//

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <string>

#include "cal3d/global.h"

/**
 * CalDataSource abstract interface class.
 *
 * This is an abstract class designed to represent a source of Cal3d data,
 * whether it is an ifstream, istream, or even a memory buffer. Inheriting
 * classes must implement the 'read' functions below.
 */

class CAL3D_API CalDataSource
{
public:

   virtual bool ok() const = 0;
   virtual void setError() const = 0;
   virtual bool readBytes(void* pBuffer, int length) = 0;
   virtual bool readFloat(float& value) = 0;
   virtual bool readInteger(int& value) = 0;
   virtual bool readString(std::string& strValue) = 0;
   virtual ~CalDataSource() {};
   
};

#endif
