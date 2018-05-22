#include "ModelLoader.h"
#include "MemoryLeakTracker.h"
#include "..\Libs\cyHair\cyHairFile.h"

CoreResult LoadHairModel(Core *core, std::wstring &filename, Model *model)
{
	cyHairFile hairfile;

	// Load the hair model
	int result = hairfile.LoadFromFile(filename.c_str());

    // Check for errors
    switch(result) 
	{
        case CY_HAIR_FILE_ERROR_CANT_OPEN_FILE:
			CoreLog::Information("Error: Cannot open hair file!\n");
            return CORE_FILENOTFOUND;
        case CY_HAIR_FILE_ERROR_CANT_READ_HEADER:
            CoreLog::Information("Error: Cannot read hair file header!\n");
            return CORE_MISC_ERROR;
        case CY_HAIR_FILE_ERROR_WRONG_SIGNATURE:
            CoreLog::Information("Error: File has wrong signature!\n");
            return CORE_MISC_ERROR;
        case CY_HAIR_FILE_ERROR_READING_SEGMENTS:
            CoreLog::Information("Error: Cannot read hair segments!\n");
            return CORE_MISC_ERROR;
        case CY_HAIR_FILE_ERROR_READING_POINTS:
            CoreLog::Information("Error: Cannot read hair points!\n");
            return CORE_MISC_ERROR;
        case CY_HAIR_FILE_ERROR_READING_COLORS:
            CoreLog::Information("Error: Cannot read hair colors!\n");
            return CORE_MISC_ERROR;
        case CY_HAIR_FILE_ERROR_READING_THICKNESS:
            CoreLog::Information("Error: Cannot read hair thickness!\n");
            return CORE_MISC_ERROR;
        case CY_HAIR_FILE_ERROR_READING_TRANSPARENCY:
            CoreLog::Information("Error: Cannot read hair transparency!\n");
            return CORE_MISC_ERROR;
    };

    int hairCount = hairfile.GetHeader().hair_count;
    int pointCount = hairfile.GetHeader().point_count;
	float *dirs = new float[pointCount * 3];
	CoreColor hairColor;
	hairColor.r = hairfile.GetHeader().d_color[0];
	hairColor.g = hairfile.GetHeader().d_color[1];
	hairColor.b = hairfile.GetHeader().d_color[2];
	hairColor.a = hairfile.GetHeader().d_transparency;
  
    // Compute directions
    if(!hairfile.FillDirectionArray(dirs))
        CoreLog::Information("Error: Cannot compute hair directions!\n");

    // Draw arrays
    int pointIndex = 0;
    const unsigned short *segments = hairfile.GetSegmentsArray();
	float *points = hairfile.GetPointsArray();
	
	CoreVector3 *vertexData = new CoreVector3[hairfile.GetHeader().point_count * 4];

    if(segments)
	{
        // If segments array exists
        for(int hairIndex = 0; hairIndex < hairCount; hairIndex++) 
		{
			for(int point = pointIndex; point < pointIndex + segments[hairIndex]; point++)
			{
				vertexData[point * 4 + 0].x = points[point * 3 + 0];
				vertexData[point * 4 + 0].z = points[point * 3 + 1];
				vertexData[point * 4 + 0].y = points[point * 3 + 2];

				vertexData[point * 4 + 1].x = dirs[point * 3 + 0];
				vertexData[point * 4 + 1].z = dirs[point * 3 + 1];
				vertexData[point * 4 + 1].y = dirs[point * 3 + 2];

				vertexData[point * 4 + 2].x = points[(point + 1) * 3 + 0];
				vertexData[point * 4 + 2].z = points[(point + 1) * 3 + 1];
				vertexData[point * 4 + 2].y = points[(point + 1) * 3 + 2];

				vertexData[point * 4 + 3].x = dirs[(point + 1) * 3 + 0];
				vertexData[point * 4 + 3].z = dirs[(point + 1) * 3 + 1];
				vertexData[point * 4 + 3].y = dirs[(point + 1) * 3 + 2];
			}
            pointIndex += segments[hairIndex] + 1;
        }
	}
    else
	{
        // If segments array does not exist, use default segment count
        int dsegs = hairfile.GetHeader().d_segments;
        for(int hairIndex = 0; hairIndex < hairCount; hairIndex++)
		{
			for(int point = pointIndex; point < pointIndex + dsegs; point++)
			{
				vertexData[point * 4 + 0].x = points[point * 3 + 0];
				vertexData[point * 4 + 0].z = points[point * 3 + 1];
				vertexData[point * 4 + 0].y = points[point * 3 + 2];

				vertexData[point * 4 + 1].x = dirs[point * 3 + 0];
				vertexData[point * 4 + 1].z = dirs[point * 3 + 1];
				vertexData[point * 4 + 1].y = dirs[point * 3 + 2];

				vertexData[point * 4 + 2].x = points[(point + 1) * 3 + 0];
				vertexData[point * 4 + 2].z = points[(point + 1) * 3 + 1];
				vertexData[point * 4 + 2].y = points[(point + 1) * 3 + 2];

				vertexData[point * 4 + 3].x = dirs[(point + 1) * 3 + 0];
				vertexData[point * 4 + 3].z = dirs[(point + 1) * 3 + 1];
				vertexData[point * 4 + 3].y = dirs[(point + 1) * 3 + 2];
			}

            pointIndex += dsegs + 1;
        }
    }

	D3D11_INPUT_ELEMENT_DESC inputLayout[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
	
	CoreResult cr = model->Init(core, filename, NULL, 0, DXGI_FORMAT_UNKNOWN, vertexData, sizeof(CoreVector3) * 2, hairfile.GetHeader().point_count * 2, inputLayout, 2, D3D11_PRIMITIVE_TOPOLOGY_LINELIST, hairColor, NULL);
	
	delete vertexData;
	delete dirs;

	return cr;
}