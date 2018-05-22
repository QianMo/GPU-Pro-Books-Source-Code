// ================================================================================ //
// Copyright (c) 2011 Arturo Garcia, Francisco Avila, Sergio Murguia and Leo Reyes	//
//																					//
// Permission is hereby granted, free of charge, to any person obtaining a copy of	//
// this software and associated documentation files (the "Software"), to deal in	//
// the Software without restriction, including without limitation the rights to		//
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies	//
// of the Software, and to permit persons to whom the Software is furnished to do	//
// so, subject to the following conditions:											//
//																					//
// The above copyright notice and this permission notice shall be included in all	//
// copies or substantial portions of the Software.									//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR		//
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,			//
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE		//
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER			//
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,	//
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE	//
// SOFTWARE.																		//
// ================================================================================ //

#include <ArgumentsParser.h>
ArgumentsParser::ArgumentsParser() :
	/* These are the default values for the Ray Tracer */
	m_iNumThreads(1), 
	m_sAccelerationStruct("bvh"), 
	m_bIsReflective(true), 
	m_bIsMultiplicative(true),
	m_iNumReflections(0), 
	m_fSpeed(0.1), 
	m_iTextureWidth(1024), 
	m_iTextureHeight(1024), 
	m_iScreenMultiplier(1),
	m_sModel("teapot.txt"), 
	m_iMaxPrimsNode(1), 
	m_sBVHSplit("sah"), 
	m_pError(NULL),
	m_iGroupSizeX(0),
	m_iGroupSizeY(0), 
	m_iGroupSizeZ(0), 
	m_iLBVHDepth(21), 
	m_iIterations(1),
	Glib::OptionGroup("ray_tracer_group","Intel Ray Tracer","For help type: raytracer --help")
{
	
	//Glib::OptionGroup group = Glib::OptionGroup("ray_tracer_group","Intel Ray Tracer","For help type: raytracer --help");
	Glib::init();					// Initialize the Glib library asap
	//Glib::OptionContext context;	// The OptionContext class defines which options are accepted by the commandline option parser
	//optionContext.add_group(*this);		// Add arguments parsed on the either at the command line or the archive

	/* Option for the Number of threads */
	Glib::OptionEntry entry1;
	entry1.set_short_name('t');
	entry1.set_long_name("num-threads");
	entry1.set_arg_description("N");
	entry1.set_description("Use N threads when running the ray tracer");
	entry1.set_flags(Glib::OptionEntry::FLAG_OPTIONAL_ARG | Glib::OptionEntry::FLAG_IN_MAIN);
	add_entry(entry1, m_iNumThreads);
	/* Option for the Acceleration Structure */
	Glib::OptionEntry entry2;
	entry2.set_short_name('s');
	entry2.set_long_name("cpu_structure");
	entry2.set_arg_description("NAME");
	entry2.set_description("Use NAME as the acceleration structure");
	entry2.set_flags(Glib::OptionEntry::FLAG_OPTIONAL_ARG | Glib::OptionEntry::FLAG_IN_MAIN);
	add_entry(entry2, m_sAccelerationStruct);
	/* Option for the reflective property */
	Glib::OptionEntry entry3;
	entry3.set_short_name('r');
	entry3.set_long_name("reflective");
	entry3.set_arg_description("N");
	entry3.set_description("Use 1 (true) or 0 (false) if is reflective");
	entry3.set_flags(Glib::OptionEntry::FLAG_OPTIONAL_ARG | Glib::OptionEntry::FLAG_IN_MAIN);
	add_entry(entry3, m_bIsReflective);
	/* Option for the Multiplicative Vertex property */
	Glib::OptionEntry entry4;
	entry4.set_short_name('m');
	entry4.set_long_name("multiplicative-vertex");
	entry4.set_arg_description("N");
	entry4.set_description("Use 1 (true) or 0 (false) if is mutliplicative");
	entry4.set_flags(Glib::OptionEntry::FLAG_OPTIONAL_ARG | Glib::OptionEntry::FLAG_IN_MAIN);
	add_entry(entry4, m_bIsMultiplicative);
	/* Option for the number of reflections */
	Glib::OptionEntry entry5;
	entry5.set_short_name('n');
	entry5.set_long_name("num-reflections");
	entry5.set_arg_description("N");
	entry5.set_description("Use N reflections");
	entry5.set_flags(Glib::OptionEntry::FLAG_OPTIONAL_ARG | Glib::OptionEntry::FLAG_IN_MAIN);
	add_entry(entry5, m_iNumReflections);
	/* Option for the Texture Width */
	Glib::OptionEntry entry6;
	entry6.set_short_name('w');
	entry6.set_long_name("texture-width");
	entry6.set_arg_description("N");
	entry6.set_description("Use N as the width of the texture");
	entry6.set_flags(Glib::OptionEntry::FLAG_OPTIONAL_ARG | Glib::OptionEntry::FLAG_IN_MAIN);
	add_entry(entry6, m_iTextureWidth);
	/* Option for the Texture Height */
	Glib::OptionEntry entry7;
	entry7.set_short_name('h');
	entry7.set_long_name("texture-height");
	entry7.set_arg_description("N");
	entry7.set_description("Use N as the height of the texture");
	entry7.set_flags(Glib::OptionEntry::FLAG_OPTIONAL_ARG | Glib::OptionEntry::FLAG_IN_MAIN);
	add_entry(entry7, m_iTextureHeight);
	/* Option for the Screen multiplier */
	Glib::OptionEntry entry8;
	entry8.set_short_name('c');
	entry8.set_long_name("screen-multiplier");
	entry8.set_arg_description("N");
	entry8.set_description("Use N as the screen multiplier");
	entry8.set_flags(Glib::OptionEntry::FLAG_OPTIONAL_ARG | Glib::OptionEntry::FLAG_IN_MAIN);
	add_entry(entry8, m_iScreenMultiplier);
	/* Option for the model file */
	Glib::OptionEntry entry9;
	entry9.set_short_name('f');
	entry9.set_long_name("model-file");
	entry9.set_arg_description("PATH");
	entry9.set_description("Use PATH to get the model");
	entry9.set_flags(Glib::OptionEntry::FLAG_OPTIONAL_ARG | Glib::OptionEntry::FLAG_IN_MAIN);
	add_entry_filename(entry9, std::string(m_sModel.raw()));

	Glib::OptionEntry entry10;
	entry10.set_short_name('i');
	entry10.set_long_name("iterations");
	entry10.set_arg_description("ITERATIONS");
	entry10.set_description("Use ITERATIONS when generating the image");
	entry10.set_flags(Glib::OptionEntry::FLAG_OPTIONAL_ARG | Glib::OptionEntry::FLAG_IN_MAIN);
	add_entry(entry10, m_iIterations);
}


ArgumentsParser::~ArgumentsParser()
{
  if(m_pKeyFile)
	g_key_file_free(m_pKeyFile);
}

int ArgumentsParser::ParseData()
{
	try
	{
#ifdef WINDOWS
		int argc_copy = 1, argc = 1;
		LPWSTR * arglist;
		arglist = CommandLineToArgvW(GetCommandLineW(), &argc);
		argc_copy = argc;
		char **argv = new char*[argc];
		if( NULL == arglist )
		{
			printf("CommandLineToArgvW failed\n");
			return 0;
		 }
		else
		{
			std::string tmp;
			for(int i=0; i < argc; i++)
			{
				cvtLPW2stdstring(tmp,arglist[i]);
				argv[i] = new char[tmp.size()+1];
				size_t j;
				for(j=0; j<tmp.size(); j++)
				{
					argv[i][j] = tmp [j];
				}
				argv[i][j] = '\0';
			}
			if (argc>1) printf("These are the arguments: ");
			for(int i=0; i<argc; i++)
				printf("%s\n",argv[i]);
		}
#endif
		// Get parsed values
		//context.parse(argc,argv);	// If we want to use command line arguments comment/uncomment this line.
		LoadConfigurationFromFile("conf.ini");	//If we want to use the .ini file comment//uncomment out this line,

#ifdef WINDOWS
		for(int i=0; i<argc_copy; i++)
			delete [] argv[i];
		delete [] argv;
#endif
	}
	catch(const Glib::Error& ex)
	{
		std::cerr << "Exception: " << ex.what() << std::endl;
	}
	
	return 1;
}

int	ArgumentsParser::LoadConfigurationFromFile(const char* sFile)
{
	// Create a new GKeyFile object
	m_pKeyFile = g_key_file_new();

	// Load the GKeyFilefrom raytracer.ini or return an error code
	if(!g_key_file_load_from_file(m_pKeyFile, sFile, G_KEY_FILE_KEEP_COMMENTS,&m_pError))
	{
		g_error(m_pError->message);
		g_key_file_free(m_pKeyFile);
		return -1;
	}
	else
	{
		/* This is where we get all of your values from the file */
		m_iNumThreads =			g_key_file_get_integer(m_pKeyFile,"ray_tracer_cpu","num_threads",&m_pError);;
		
		m_bIsReflective =		g_key_file_get_boolean(m_pKeyFile, "reflection", "is_reflective",&m_pError );
		m_bIsMultiplicative =	g_key_file_get_boolean(m_pKeyFile, "reflection", "is_multiplicative_vertex",&m_pError );
		m_iNumReflections =		g_key_file_get_integer(m_pKeyFile,"reflection","num_reflex",&m_pError);

		m_iMaxPrimsNode =		g_key_file_get_integer(m_pKeyFile, "bvh","bvh_max_prims_node",&m_pError);
		m_sBVHSplit =			g_key_file_get_string(m_pKeyFile, "bvh", "bvh_split_algorithm",&m_pError);

		m_fSpeed =				static_cast<float>(g_key_file_get_double(m_pKeyFile, "camera", "speed" ,&m_pError));
		
		m_sModel =				g_key_file_get_string(m_pKeyFile,"scene", "model_file",&m_pError);
		m_sAccelerationStruct=	g_key_file_get_string(m_pKeyFile,"scene", "acceleration_structure",&m_pError);

		m_iTextureWidth =		g_key_file_get_integer(m_pKeyFile,"options","texture_width",&m_pError);
		m_iTextureHeight =		g_key_file_get_integer(m_pKeyFile,"options","texture_height",&m_pError);
		m_iScreenMultiplier =	g_key_file_get_integer(m_pKeyFile,"options","screen_multiplier",&m_pError);

		m_iGroupSizeX =			g_key_file_get_integer(m_pKeyFile,"ray_tracer_cs","cs_group_size_x",&m_pError);
		m_iGroupSizeY =			g_key_file_get_integer(m_pKeyFile,"ray_tracer_cs","cs_group_size_y",&m_pError);
		m_iGroupSizeZ =			g_key_file_get_integer(m_pKeyFile,"ray_tracer_cs","cs_group_size_z",&m_pError);

		m_iLBVHDepth=			g_key_file_get_integer(m_pKeyFile,"lbvh","lbvh_depth",&m_pError);
		
		m_iIterations =			g_key_file_get_integer(m_pKeyFile,"global","iterations",&m_pError);
		
		ShowConfiguration(0);
	}

	return 0;
}

void ArgumentsParser::ShowConfiguration(unsigned int uiNumProcesses)
{
	std::stringstream tmp ( std::stringstream::in | std::stringstream::out ); 
	tmp << "\n---------------------------------------"
		<< "\nINI FILE"
		<< "\n---------------------------------------"
		<< "\nNumber of processes:\t"<< uiNumProcesses
		<< "\nNumber of threads:\t"<< m_iNumThreads
		<< "\nAcceleration struc.:\t"<< m_sAccelerationStruct
		<< "\nIs reflective:\t\t" << m_bIsReflective
		<< "\nIs Multiplicative:\t"<< m_bIsMultiplicative
		<< "\n# of Reflections:\t" << m_iNumReflections
		<< "\nMax Prims Node:\t\t" << m_iMaxPrimsNode
		<< "\nSplit Algorithm:\t" << m_sBVHSplit
		<< "\nSpeed:\t\t\t" << m_fSpeed
		<< "\nModel:\t\t\t" << m_sModel
		<< "\nTexture width:\t\t" << m_iTextureWidth
		<< "\nTexture height:\t\t" << m_iTextureHeight
		<< "\nScreen multiplier:\t" << m_iScreenMultiplier
		<< "\nIterations:\t\t" << m_iIterations
		<< "\n---------------------------------------";
	printf("%s\n",tmp.str().c_str());
}

bool ArgumentsParser::on_pre_parse(Glib::OptionContext& context, Glib::OptionGroup& group)
{
	return Glib::OptionGroup::on_pre_parse(context, group);
}

bool ArgumentsParser::on_post_parse(Glib::OptionContext& context, Glib::OptionGroup& group)
{
	return Glib::OptionGroup::on_post_parse(context, group);
}

void ArgumentsParser::on_error(Glib::OptionContext& context, Glib::OptionGroup& group)
{
	std::cerr << "There was an error parsing the arguments." <<std::endl;
	Glib::OptionGroup::on_error(context, group);
}