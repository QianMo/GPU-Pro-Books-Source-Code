//****************************************************************************//
// error.h                                                                    //
// Copyright (C) 2001, 2002 Bruno 'Beosil' Heidelberger                       //
//****************************************************************************//
// This library is free software; you can redistribute it and/or modify it    //
// under the terms of the GNU Lesser General Public License as published by   //
// the Free Software Foundation; either version 2.1 of the License, or (at    //
// your option) any later version.                                            //
//****************************************************************************//

#ifndef CAL_ERROR_H
#define CAL_ERROR_H

#include "cal3d/global.h"


namespace CalError
{
  enum Code
  {
    OK = 0,
    INTERNAL,
    INVALID_HANDLE,
    MEMORY_ALLOCATION_FAILED,
    FILE_NOT_FOUND,
    INVALID_FILE_FORMAT,
    FILE_PARSER_FAILED,
    INDEX_BUILD_FAILED,
    NO_PARSER_DOCUMENT,
    INVALID_ANIMATION_DURATION,
    BONE_NOT_FOUND,
    INVALID_ATTRIBUTE_VALUE,
    INVALID_KEYFRAME_COUNT,
    INVALID_ANIMATION_TYPE,
    FILE_CREATION_FAILED,
    FILE_WRITING_FAILED,
    INCOMPATIBLE_FILE_VERSION,
    NO_MESH_IN_MODEL,
    BAD_DATA_SOURCE,
    NULL_BUFFER,
    INVALID_MIXER_TYPE,
    MAX_ERROR_CODE
  };

  CAL3D_API Code getLastErrorCode();
  CAL3D_API const std::string& getLastErrorFile();
  CAL3D_API int getLastErrorLine();
  CAL3D_API const std::string& getLastErrorText();
  CAL3D_API void printLastError();
  CAL3D_API void setLastError(Code code, const std::string& strFile, int line, const std::string& strText = "");

  CAL3D_API std::string getErrorDescription(Code code);

  inline std::string getLastErrorDescription() {
      return getErrorDescription(getLastErrorCode());
  }
}

#endif
