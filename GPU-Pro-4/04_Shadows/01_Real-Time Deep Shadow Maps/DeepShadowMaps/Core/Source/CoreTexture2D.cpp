#include "Core.h"


CoreTexture2D::CoreTexture2D()
{
	width = 0;
	height = 0;
	texture = NULL;
	sRGB = false;
}

// Load a texture from a stream
CoreResult CoreTexture2D::init(Core* core, std::istream *in[], UINT textureCount, UINT mipLevels,
							   UINT cpuAccessFlags, UINT miscFlags, D3D11_USAGE usage, UINT bindFlags,
							   UINT sampleCount, UINT sampleQuality, bool sRGB)
{
	BYTE* data = NULL;
	BYTE** dataArray = new BYTE* [textureCount];

	ZeroMemory(dataArray, sizeof(BYTE*) * textureCount);

	this->core = core;
	this->mipLevels = mipLevels;
	this->sampleCount = sampleCount;
	this->sampleQuality = sampleQuality;
	this->cpuAccessFlags = cpuAccessFlags;
	this->usage = usage;
	this->bindFlags = bindFlags;
	this->textureCount = textureCount;
	this->miscFlags = miscFlags;
	this->sRGB = sRGB;

	UINT successfullyLoaded = 0;
	for(UINT ui = 0 ; ui < textureCount ; ui++)
	{
		if(!in[ui]->good())
		{
			CoreLog::Information(L"in[%d] is not good, skipping!", ui);
			continue;
		}

		if(loadJpg(*in[ui], &data) != CORE_OK)
			if(loadPng(*in[ui], &data) != CORE_OK)
				if(loadBitmap(*in[ui], &data) != CORE_OK)
					if(loadTga(*in[ui], &data) != CORE_OK)
					{
						CoreLog::Information(L"Couldn't load your image (supported types are: *.jpg *.png *.bmp *.tga)");
						continue;
					}
		successfullyLoaded++;
		
		dataArray[ui] = data;
	}
	
	if(successfullyLoaded == 0)	// None of the textures could be loaded successfully
	{
		delete dataArray;
		return CORE_MISC_ERROR;
	}

	CoreResult result = createAndFillTexture(dataArray);
	for(UINT ui = 0 ; ui < textureCount ; ui++)	// Clean up
		if(dataArray[ui] != NULL) 
			delete dataArray[ui];
	delete dataArray;

	return result;
}

CoreResult CoreTexture2D::init(Core* core, const std::vector <std::istream *> &in, UINT mipLevels,
		  	    UINT cpuAccessFlags, UINT miscFlags, D3D11_USAGE usage, UINT bindFlags,
			    UINT sampleCount, UINT sampleQuality,  bool sRGB)
{
	UINT textureCount = (UINT)in.size();
	BYTE* data = NULL;
	BYTE** dataArray;
	dataArray = new BYTE* [textureCount];

	ZeroMemory(dataArray, sizeof(BYTE*) * textureCount);

	this->core = core;
	this->mipLevels = mipLevels;
	this->sampleCount = sampleCount;
	this->sampleQuality = sampleQuality;
	this->cpuAccessFlags = cpuAccessFlags;
	this->usage = usage;
	this->bindFlags = bindFlags;
	this->textureCount = textureCount;
	this->miscFlags = miscFlags;
	this->sRGB = sRGB;

	UINT successfullyLoaded = 0;
	for(UINT ui = 0 ; ui < textureCount ; ui++)
	{
		if(!in[ui]->good())
		{
			CoreLog::Information(L"in[%d] is not good, skipping!", ui);
			continue;
		}

		if(loadJpg(*in[ui], &data) != CORE_OK)
			if(loadPng(*in[ui], &data) != CORE_OK)
				if(loadBitmap(*in[ui], &data) != CORE_OK)
					if(loadTga(*in[ui], &data) != CORE_OK)
					{
						CoreLog::Information(L"Couldn't load your image (supported types are: *.jpg *.png *.bmp *.tga)");
						continue;
					}
		successfullyLoaded++;
		
		dataArray[ui] = data;
	}
	
	if(successfullyLoaded == 0)	// None of the textures could be loaded successfully
	{
		delete dataArray;
		return CORE_MISC_ERROR;
	}

	CoreResult result = createAndFillTexture(dataArray);
	for(UINT ui = 0 ; ui < textureCount ; ui++)	// Clean up
		if(dataArray[ui] != NULL) 
			delete dataArray[ui];
	delete dataArray;

	return result;
}


// Create a texture from memory
CoreResult CoreTexture2D::init(Core* core, BYTE** data, UINT width, UINT height, UINT textureCount, UINT mipLevels,
							   DXGI_FORMAT format, UINT cpuAccessFlags, UINT miscFlags, D3D11_USAGE usage, UINT bindFlags,
							   UINT sampleCount, UINT sampleQuality)
{
	BYTE** dataArray = NULL;

	this->core = core;
	this->mipLevels = mipLevels;
	this->sampleCount = sampleCount;
	this->sampleQuality = sampleQuality;
	this->cpuAccessFlags = cpuAccessFlags;
	this->usage = usage;
	this->bindFlags = bindFlags;
	this->textureCount = textureCount;
	this->format = format;
	this->width = width;
	this->height = height;
	this->miscFlags = miscFlags;
	
	if(data)
	{
		dataArray = new BYTE* [textureCount];
		ZeroMemory(dataArray, sizeof(BYTE*) * textureCount);
		for(UINT ui = 0 ; ui < textureCount ; ui++)
		{
			if(!data[ui])
			{
				CoreLog::Information(L"Data[%d] == NULL , skipping!", ui);
				continue;
			}
			

			dataArray[ui] = data[ui];
		}
	}
	CoreResult res;
	res = createAndFillTexture(dataArray);
	
	if(data)
		delete dataArray;

	return res;
}


// Directly use an ID3D11Texture2D object
CoreResult CoreTexture2D::init(Core* core, ID3D11Texture2D* texture)
{
	if(texture == NULL)
		return CORE_INVALIDARGS;

	this->core = core;
	
	D3D11_TEXTURE2D_DESC textureDesc;
	
	texture->GetDesc(&textureDesc);

	textureCount = textureDesc.ArraySize;
	cpuAccessFlags = textureDesc.CPUAccessFlags;
	height = textureDesc.Height;
	width = textureDesc.Width;
	mipLevels = textureDesc.MipLevels;
	bindFlags = textureDesc.BindFlags;
	sampleQuality = textureDesc.SampleDesc.Count;
	sampleQuality = textureDesc.SampleDesc.Quality;
	
	texture->AddRef();
	this->texture = texture;
	return CORE_OK;
}

// Loads a Bitmap from a stream
CoreResult CoreTexture2D::loadBitmap(std::istream& in, BYTE** data)
{
	BITMAPFILEHEADER bmpFileHead;
	BITMAPINFOHEADER bmpInfoHead;
	BYTE *tempData;

	std::istream::pos_type pos = in.tellg();
	in.read((char *)&bmpFileHead, sizeof(BITMAPFILEHEADER));
	if(bmpFileHead.bfType != 19778)
	{
		in.clear();
		in.seekg(pos);	// return to original pos
		return CORE_WRONG_FORMAT;
	}

	in.read((char *)&bmpInfoHead, sizeof(BITMAPINFOHEADER));

	if (bmpInfoHead.biCompression)
	{
		CoreLog::Information(L"Compressed Bitmaps aren't supported!");
		in.clear();
		in.seekg(pos);
		return CORE_MISC_ERROR;
	}

	if (bmpInfoHead.biBitCount != 32 && bmpInfoHead.biBitCount != 24)
	{
		CoreLog::Information(L"Only 32bit and 24bit Bitmaps are supported!");
		in.clear();
		in.seekg(pos);
		return CORE_NODATA;
	}

	UINT bpp = bmpInfoHead.biBitCount / 8;
	if(width == 0)
		width = bmpInfoHead.biWidth;
	else
		if(width != bmpInfoHead.biWidth)
		{
			CoreLog::Information(L"Width of Image does not match the width of the first image, skipping");
			in.clear();
			in.seekg(pos);
			return CORE_WRONG_FORMAT;
		}
	if(height == 0)
		height = bmpInfoHead.biHeight;
	else
		if(height!= bmpInfoHead.biHeight)
		{
			CoreLog::Information(L"Height of Image does not match the height of the first image, skipping");
			in.clear();
			in.seekg(pos);
			return CORE_WRONG_FORMAT;
		}
	if (this->sRGB)
		format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
	else
		format = DXGI_FORMAT_R8G8B8A8_UNORM;


	tempData = new BYTE[width * height * 4];
	BYTE *tempData2 = new BYTE[width * height * bpp];

	if(!tempData)
	{
		CoreLog::Information(L"Couldn't load your Texture: OutOfMem");
		in.clear();
		in.seekg(pos);
		return CORE_OUTOFMEM;
	}

	if(!tempData2)
	{
		CoreLog::Information(L"Couldn't load your Texture: OutOfMem");
		in.clear();
		in.seekg(pos);
		delete tempData;
		return CORE_OUTOFMEM;
	}

	in.read((char *)tempData2, width * height * bpp);

	
	for(UINT i = 0 ; i < height; i++)
	{
		BYTE *tmpSrc = tempData2 + width * bpp * (height - i - 1);					// Image is inverted
		BYTE *tmpDst = tempData + width * 4 * i;
		for(UINT j = 0 ; j < width ; j++)
		{
			tmpDst[2] = tmpSrc[0];
			tmpDst[1] = tmpSrc[1];
			tmpDst[0] = tmpSrc[2];
			if(bpp == 3)		// Add alpha = 255 if none provided
				tmpDst[3] = 255;
			else
				tmpDst[3] = tmpSrc[3];
			
			tmpSrc += bpp;
			tmpDst += 4;	
		}
	}
	delete tempData2;

	*data = tempData;
	return CORE_OK;
}


// Loads a TGA from a stream
CoreResult CoreTexture2D::loadTga(std::istream& in, BYTE** data)
{
	BYTE *tempData;
	std::istream::pos_type pos = in.tellg();

	TGAHeader tgaHeader;
	
	in.read((char *)&tgaHeader, sizeof(TGAHeader));

	if(tgaHeader.imagetype != 2)	// No better check available
	{
		in.clear();
		in.seekg(pos);
		return CORE_WRONG_FORMAT;
	}

	if ((tgaHeader.imgspec.bpp != 24 && tgaHeader.imgspec.bpp != 32))
	{
		CoreLog::Information(L"The Format of your TGA is unsupported");
		in.clear();
		in.seekg(pos);
		return CORE_WRONG_FORMAT;	
	}

	if(width == 0)
		width = tgaHeader.imgspec.w;
	else
		if(width != tgaHeader.imgspec.w)
		{
			CoreLog::Information(L"Width of Image does not match the width of the first image, skipping");
			in.clear();
			in.seekg(pos);
			return CORE_WRONG_FORMAT;
		}
	if(height == 0)
		height = tgaHeader.imgspec.h;
	else
		if(height!= tgaHeader.imgspec.h)
		{
			CoreLog::Information(L"Height of Image does not match the height of the first image, skipping");
			in.clear();
			in.seekg(pos);
			return CORE_WRONG_FORMAT;
		}
	
	UINT bpp = tgaHeader.imgspec.bpp / 8;
	if (sRGB)
		format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
	else
		format = DXGI_FORMAT_R8G8B8A8_UNORM;

	tempData = new BYTE[width * height * 4];
	BYTE *tempData2 = new BYTE[width * height * bpp];

	if(!tempData)
	{
		CoreLog::Information(L"Couldn't load your Texture: OutOfMem");
		in.clear();
		in.seekg(pos);
		return CORE_OUTOFMEM;
	
	}

	if(!tempData2)
	{
		CoreLog::Information(L"Couldn't load your Texture: OutOfMem");
		in.clear();
		in.seekg(pos);
		delete tempData;
		return CORE_OUTOFMEM;
	}

	in.read((char *)tempData2, width * height * bpp);

	
	for(UINT i = 0 ; i < height; i++)
	{
		BYTE *tmpSrc = tempData2 + width * bpp * (height - i - 1);					// Image is inverted
		BYTE *tmpDst = tempData + width * 4 * i;
		for(UINT j = 0 ; j < width ; j++)
		{
			tmpDst[2] = tmpSrc[0];
			tmpDst[1] = tmpSrc[1];
			tmpDst[0] = tmpSrc[2];
			if(bpp == 3)		// Add alpha = 255 if none provided
				tmpDst[3] = 255;
			else
				tmpDst[3] = tmpSrc[3];
			
			tmpSrc += bpp;
			tmpDst += 4;	
		}
	}
	delete tempData2;

	*data = tempData;

	return CORE_OK;
}

// ------------------------------------------ jpeglib stuff -------------------------------
boolean StreamFillInputBuffer(j_decompress_ptr cinfo)
{
	StreamSourceManager *sourceManager = (StreamSourceManager *)cinfo->src;

	sourceManager->In->read((char *)sourceManager->Buffer, JPEG_INPUT_BUF_SIZE);
	cinfo->src->bytes_in_buffer = (unsigned int)sourceManager->In->gcount();
	
	sourceManager->pub.next_input_byte = (const JOCTET *)sourceManager->Buffer;
	return true;
}
void StreamSkipInputData (j_decompress_ptr cinfo, long lByteCount)
{
	if(lByteCount < (long)cinfo->src->bytes_in_buffer)
	{
		cinfo->src->next_input_byte += lByteCount;
		cinfo->src->bytes_in_buffer -= lByteCount;
	}
	else
	{
		((StreamSourceManager *)cinfo->src)->In->seekg(lByteCount - cinfo->src->bytes_in_buffer, std::ios::cur);
		cinfo->src->bytes_in_buffer = 0;
	}
}


void StreamInitTermSource (j_decompress_ptr cinfo){}

// For logging to CoreLog, do not exit(1) the app
void LogMessage (j_common_ptr cinfo)
{
	LogErrorManager *err = (LogErrorManager *) cinfo->err;

	// If we got 53, this is not a jpg file, so don't gen a msg, since we want to try other formats
	if(err->Pub.msg_code != 53)
	{
		char buffer[JMSG_LENGTH_MAX];

		// Create the message
		(*cinfo->err->format_message) (cinfo, buffer);
		
		// Convert to Unicode
		int Len = (int)strlen(buffer) + 1;
		WCHAR* unicodeBuffer = new WCHAR[Len];
		
		MultiByteToWideChar(CP_ACP, WC_DEFAULTCHAR, buffer, Len, unicodeBuffer, Len); 
		CoreLog::Information(std::wstring(unicodeBuffer));
		delete unicodeBuffer;
	}
	
	// Return control to the setjmp point 
	longjmp(err->JumpBuffer, 1);
}

// Ignore warning messages
void LogMessageDiscard (j_common_ptr cinfo)
{
	
}


// ------------------------------------------ jpeglib stuff end ---------------------------
// Loads a JPG from an Stream
CoreResult CoreTexture2D::loadJpg(std::istream &in, BYTE** data)
{
	jpeg_decompress_struct *cinfo = new jpeg_decompress_struct;
	LogErrorManager jerr;
	JSAMPARRAY buffer;								
	DWORD rowStride;
	BYTE *tempData;
	std::istream::pos_type pos = in.tellg();
	
	cinfo->err = jpeg_std_error((jpeg_error_mgr *)&jerr);
	jerr.Pub.error_exit = LogMessage;
	jerr.Pub.output_message	= LogMessageDiscard;

	if (setjmp(jerr.JumpBuffer)) 
	{
		// If we get here, the JPEG code has signaled an error.
		// We need to clean up the JPEG object and return.

		// If we got 53, this is not a jpg file, so try others
		CoreResult result = CORE_MISC_ERROR;
		in.clear();
		in.seekg(pos);
		if(cinfo->err->msg_code == 53)
			result = CORE_WRONG_FORMAT;
		
		delete ((StreamSourceManager *)cinfo->src)->Buffer;
		delete cinfo->src;
		jpeg_destroy_decompress(cinfo);
		delete cinfo;
		
		return result;
	}

	jpeg_create_decompress(cinfo);

	// Stream handling
	cinfo->src = (jpeg_source_mgr *)new StreamSourceManager;
	cinfo->src->init_source = StreamInitTermSource;
	cinfo->src->fill_input_buffer = StreamFillInputBuffer;
	cinfo->src->skip_input_data = StreamSkipInputData;
	cinfo->src->resync_to_restart = jpeg_resync_to_restart;
	cinfo->src->term_source = StreamInitTermSource;
	cinfo->src->bytes_in_buffer = 0;			// forces fill_input_buffer on first read 
	cinfo->src->next_input_byte = NULL;			// until buffer loaded
	((StreamSourceManager *)cinfo->src)->In = &in;
	((StreamSourceManager *)cinfo->src)->Buffer = (BYTE *)::operator new(JPEG_INPUT_BUF_SIZE);

	if(!((StreamSourceManager *)cinfo->src)->Buffer)
	{
		in.clear();
		in.seekg(pos);
		CoreLog::Information(L"Couldn't load your Texture: OutOfMem");
		delete cinfo->src;
		jpeg_abort((j_common_ptr)cinfo);
		delete cinfo;
		return CORE_OUTOFMEM;
	}	

	jpeg_read_header(cinfo, true);

	jpeg_start_decompress(cinfo);

	UINT bpp = cinfo->output_components;

	if(bpp != 3)
	{
		in.clear();
		in.seekg(pos);
		CoreLog::Information(L"Only 24-bit jpgs are supported.");
		delete ((StreamSourceManager *)cinfo->src)->Buffer;
		delete cinfo->src;
		jpeg_abort((j_common_ptr)cinfo);
		delete cinfo;
		return CORE_WRONG_FORMAT;
	}
	
	if (sRGB)
		format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
	else
		format = DXGI_FORMAT_R8G8B8A8_UNORM;

	if(width == 0)
		width = cinfo->output_width;
	else
		if(width != cinfo->output_width)
		{
			in.clear();
			in.seekg(pos);
			CoreLog::Information(L"Width of Image does not match the width of the first image, skipping");
			delete ((StreamSourceManager *)cinfo->src)->Buffer;
			delete cinfo->src;
			jpeg_abort((j_common_ptr)cinfo);
			delete cinfo;
			return CORE_WRONG_FORMAT;
		}
	if(height == 0)
		height = cinfo->output_height;
	else
		if(height!= cinfo->output_height)
		{
			in.clear();
			in.seekg(pos);
			CoreLog::Information(L"Height of Image does not match the height of the first image, skipping");
			delete ((StreamSourceManager *)cinfo->src)->Buffer;
			delete cinfo->src;
			jpeg_abort((j_common_ptr)cinfo);
			delete cinfo;
			return CORE_WRONG_FORMAT;
		}


	tempData = (BYTE *)::operator new(cinfo->output_height * cinfo->output_width * 4);

	if(!tempData)
	{
		in.clear();
		in.seekg(pos);
		CoreLog::Information(L"Couldn't load your Texture: OutOfMem");
		delete ((StreamSourceManager *)cinfo->src)->Buffer;
		delete cinfo->src;
		jpeg_abort((j_common_ptr)cinfo);
		delete cinfo;
		return CORE_OUTOFMEM;
	}

	rowStride = cinfo->output_width * cinfo->output_components;
	UINT newRowStride = cinfo->output_width * 4;

	buffer = (*cinfo->mem->alloc_sarray)									// Buffer is automatically deleted when finished 
		((j_common_ptr) cinfo, JPOOL_IMAGE, rowStride, 1);

	while (cinfo->output_scanline < cinfo->output_height)
	{
		jpeg_read_scanlines(cinfo, buffer, 1);	

		for(DWORD dw = 0 ; dw < width ; dw ++)		 
		{
			(tempData + newRowStride * (cinfo->output_scanline - 1) + dw * 4)[0] = (buffer[0] + dw * bpp)[0];
			(tempData + newRowStride * (cinfo->output_scanline - 1) + dw * 4)[1] = (buffer[0] + dw * bpp)[1];
			(tempData + newRowStride * (cinfo->output_scanline - 1) + dw * 4)[2] = (buffer[0] + dw * bpp)[2];
			(tempData + newRowStride * (cinfo->output_scanline - 1) + dw * 4)[3] = 255;		// Add alpha
		}
	}
	

	delete ((StreamSourceManager *)cinfo->src)->Buffer;
	delete cinfo->src;

	jpeg_destroy_decompress(cinfo);
    delete cinfo;

	*data = tempData;
	
	return CORE_OK;
}


// ------------------------------------------ libpng stuff --------------------------------
void ErrorFunction(png_structp pngPtr, png_const_charp message)
{
	// Convert to Unicode
	int Len = (int)strlen(message) + 1;
	WCHAR* unicodeBuffer = new WCHAR[Len];
	
	MultiByteToWideChar(CP_ACP, WC_DEFAULTCHAR, message, Len, unicodeBuffer, Len); 
	CoreLog::Information(std::wstring(unicodeBuffer));
	delete unicodeBuffer;
	// jump back
	longjmp(pngPtr->jmpbuf, 1);
}

void ReadFunction(png_structp pngPtr, png_bytep buffer, png_size_t size)
{
	((std::istream *)pngPtr->io_ptr)->read((char *)buffer, size);
	
	if (((std::istream *)pngPtr->io_ptr)->gcount() != size)
		png_error(pngPtr, "Read Error");
}
// ------------------------------------------ libpng stuff end ----------------------------
// Loads a PNG from a stream
CoreResult CoreTexture2D::loadPng(std::istream& in, BYTE** data)
{
	png_structp pngPtr;
	png_infop infoPtr;

	BYTE sig[8];
	std::istream::pos_type pos = in.tellg();

	in.read((char *)sig, 8);
    if (!png_check_sig((png_bytep) sig, 8))
	{
		in.clear();
		in.seekg(pos);
		return CORE_WRONG_FORMAT;   // Bad signature
	}
	
	// Create a png struct with a custom error function
	pngPtr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, ErrorFunction, NULL);
	if(!pngPtr)
	{
		in.clear();
		in.seekg(pos);
		return CORE_OUTOFMEM;
	}

	// Create an info struct
	infoPtr = png_create_info_struct(pngPtr);
    if (!infoPtr) 
	{
        png_destroy_read_struct(&pngPtr, NULL, NULL);
		in.clear();
		in.seekg(pos);
        return CORE_OUTOFMEM;
    }

	if (setjmp(png_jmpbuf(pngPtr))) 
	{
        // If we get here something really bad happened
		in.clear();
		in.seekg(pos);
		png_destroy_read_struct(&pngPtr, &infoPtr, NULL);
        return CORE_MISC_ERROR;
    }

	png_set_read_fn(pngPtr, (void *)&in, ReadFunction);
	
	png_set_sig_bytes(pngPtr, 8);  // We already read the 8 signature bytes

	png_read_info(pngPtr, infoPtr);  // Read all PNG info
	

	// Now read the png
	if (infoPtr->color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_expand(pngPtr);	// Expand to RGB
							
    if (infoPtr->color_type == PNG_COLOR_TYPE_GRAY && infoPtr->bit_depth < 8)
        png_set_expand(pngPtr); // Expand to RGB

    if (png_get_valid(pngPtr, infoPtr, PNG_INFO_tRNS))
        png_set_expand(pngPtr); // Expand to RGB

	if (infoPtr->color_type == PNG_COLOR_TYPE_GRAY || infoPtr->color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(pngPtr); // Expand to RGB

	if(infoPtr->channels < 4)		// Add alpha to the image
		png_set_add_alpha(pngPtr, 255, PNG_FILLER_AFTER);


	if(infoPtr->bit_depth < 16)
	{
		if (sRGB)
			format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
		else
			format = DXGI_FORMAT_R8G8B8A8_UNORM;	// others will be expanded
	}
	else
		format = DXGI_FORMAT_R16G16B16A16_UNORM;
	
    

	// All transforms are done, now update the infoPtr
    png_read_update_info(pngPtr, infoPtr);

	if(width == 0)
		width = infoPtr->width;
	else
		if(width != infoPtr->width)
		{
			CoreLog::Information(L"Width of Image does not match the width of the first image, skipping");
			in.clear();
			in.seekg(pos);
			png_destroy_read_struct(&pngPtr, &infoPtr, NULL);
			return CORE_WRONG_FORMAT;
		}

	if(height == 0)
		height = infoPtr->height;
	else
		if(height!= infoPtr->height)
		{
			CoreLog::Information(L"Height of Image does not match the height of the first image, skipping");
			in.clear();
			in.seekg(pos);
			png_destroy_read_struct(&pngPtr, &infoPtr, NULL);
			return CORE_WRONG_FORMAT;
		}

	BYTE *tempData;
    if ((tempData = new BYTE[width * height * infoPtr->bit_depth / 2]) == NULL) 
	{
        png_destroy_read_struct(&pngPtr, &infoPtr, NULL);
		in.clear();
		in.seekg(pos);
        return CORE_OUTOFMEM;
    }

	png_bytepp rowPointers;
    if ((rowPointers = new png_bytep[height]) == NULL) 
	{
        png_destroy_read_struct(&pngPtr, &infoPtr, NULL);
        delete tempData;
        in.clear();
		in.seekg(pos);
        return CORE_OUTOFMEM;
    }


    // Set the row pointers to point at the correct offsets
    for(UINT ui = 0;  ui < height;  ui++)
        rowPointers[ui] = tempData + ui * width * infoPtr->bit_depth / 2;

    // Read the whole image
    png_read_image(pngPtr, rowPointers);

	// Done. We don't need those row pointers any more
    delete rowPointers;

    png_read_end(pngPtr, infoPtr);
	png_destroy_read_struct(&pngPtr, &infoPtr, NULL);
	*data = tempData;
	return CORE_OK;
}

// As the name says...
CoreResult CoreTexture2D::createAndFillTexture(BYTE** data)
{
	if(core)
	{
		D3D11_TEXTURE2D_DESC texDesc;
		ZeroMemory(&texDesc, sizeof(D3D11_TEXTURE2D_DESC));
		texDesc.Height = height;
		texDesc.Width = width;
		texDesc.MipLevels = mipLevels;
		texDesc.Format = format;
		texDesc.SampleDesc.Count = sampleCount;
		texDesc.SampleDesc.Quality = sampleQuality;
		texDesc.ArraySize = textureCount;
		texDesc.Usage = usage;
		texDesc.BindFlags = bindFlags;
		texDesc.CPUAccessFlags = cpuAccessFlags;
		texDesc.MiscFlags = miscFlags;

        D3D11_SUBRESOURCE_DATA *subResData = NULL;

		if(data)
		{
			subResData = new D3D11_SUBRESOURCE_DATA[textureCount];

			for(UINT ui = 0 ; ui < textureCount ; ui++)
			{
				subResData[ui].pSysMem = data[ui];
				subResData[ui].SysMemSlicePitch = 0;
				subResData[ui].SysMemPitch = GetNumberOfBytesFromDXGIFormt(format) * width;
			}
		}
		
		HRESULT result;

		if (mipLevels != 0 || !(miscFlags & D3D11_RESOURCE_MISC_GENERATE_MIPS))
		{
			result = core->GetDevice()->CreateTexture2D(&texDesc, subResData, &texture);
		}
		else
		{
			//Create texture with space for mipmaps and load data to mipmap level 0
			HRESULT result = core->GetDevice()->CreateTexture2D(&texDesc, NULL, &texture);
			texture->GetDesc(&texDesc);
			mipLevels = texDesc.MipLevels;
			if(FAILED(result))
			{
				delete subResData;
				CoreLog::Information(L"Couldn't create D3D Texture, HRESULT = %x!", result);
				return CORE_MISC_ERROR;
			}
			
			if(data)
			{
				for (UINT ui = 0; ui < textureCount; ui++)
				{
					UINT index = D3D11CalcSubresource(0, ui, mipLevels);
					core->GetImmediateDeviceContext()->UpdateSubresource(texture, index, NULL, data[ui], GetNumberOfBytesFromDXGIFormt(format) * width, 0);
				}
			}
		}

		
		delete subResData;

		return CORE_OK;
	}
	else
		texture = NULL;
	

	return CORE_OK;
}

// CleanUp
void CoreTexture2D::finalRelease()
{
	SAFE_RELEASE(texture);
}


// Retrieves the RenderTargetView from the texture
CoreResult CoreTexture2D::CreateRenderTargetView(D3D11_RENDER_TARGET_VIEW_DESC* rtvDesc, ID3D11RenderTargetView** rtv)
{
	HRESULT result = core->GetDevice()->CreateRenderTargetView(texture, rtvDesc, rtv);
	if(FAILED(result))
	{
		CoreLog::Information(L"Could not create RenderTargetView, HRESULT = %x", result);
		return CORE_MISC_ERROR;
	}
	return CORE_OK;
}

// Retrieves the DepthStencilView from the texture
CoreResult CoreTexture2D::CreateDepthStencilView(D3D11_DEPTH_STENCIL_VIEW_DESC* dsvDesc, ID3D11DepthStencilView** dsv)
{
	HRESULT result = core->GetDevice()->CreateDepthStencilView(texture, dsvDesc, dsv);
	if(FAILED(result))
	{
		CoreLog::Information(L"Could not create DepthStencilView, HRESULT = %x", result);
		return CORE_MISC_ERROR;
	}
	return CORE_OK;
}


// Creates a ShaderResourceView with this texture as resource
CoreResult CoreTexture2D::CreateShaderResourceView(D3D11_SHADER_RESOURCE_VIEW_DESC* srvDesc, ID3D11ShaderResourceView** srv)
{
	HRESULT result = core->GetDevice()->CreateShaderResourceView(texture, srvDesc, srv);
	if(FAILED(result))
	{
		CoreLog::Information(L"Could not create Shader Resource View, HRESULT = %x", result);
		return CORE_MISC_ERROR;
	}
	return CORE_OK;
}