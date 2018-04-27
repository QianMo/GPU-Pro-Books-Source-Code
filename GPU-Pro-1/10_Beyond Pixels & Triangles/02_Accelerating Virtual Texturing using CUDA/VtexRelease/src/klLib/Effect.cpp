/**
 *
 *  This software module was originally developed for research purposes,
 *  by Multimedia Lab at Ghent University (Belgium).
 *  Its performance may not be optimized for specific applications.
 *
 *  Those intending to use this software module in hardware or software products
 *  are advized that its use may infringe existing patents. The developers of 
 *  this software module, their companies, Ghent Universtity, nor Multimedia Lab 
 *  have any liability for use of this software module or modifications thereof.
 *
 *  Ghent University and Multimedia Lab (Belgium) retain full right to modify and
 *  use the code for their own purpose, assign or donate the code to a third
 *  party, and to inhibit third parties from using the code for their products. 
 *
 *  This copyright notice must be included in all copies or derivative works.
 *
 *  For information on its use, applications and associated permission for use,
 *  please contact Prof. Rik Van de Walle (rik.vandewalle@ugent.be). 
 *
 *  Detailed information on the activities of
 *  Ghent University Multimedia Lab can be found at
 *  http://multimedialab.elis.ugent.be/.
 *
 *  Copyright (c) Ghent University 2004-2009.
 *
 **/

#include "shared.h"
#include "Effect.h"
#include "FileSystem.h"
#include "Console.h"
#include "cgfx_dump.h"

#include <windows.h>

CGcontext klCgContext = NULL;

//
// Support code for loading and compiling effects files
//

void EffectNameToFile(const char *effect, char *fileName) {
    strcpy(fileName,"./base/shaders/");
    fileName += strlen(fileName);
    while ( *effect ) {
        if ( *effect == '.' ) {
            *fileName++ = '/';
        } else {
            *fileName++ = *effect;
        }
        effect++;
    }
    *fileName++ = '.';
    *fileName++ = 'c';
    *fileName++ = 'g';
    *fileName++ = 0;
}

class PreprocessBuffer {
    char buffer[0xFFFF];
    int ofs;
    int lineNo;

    std::string  errors;

    // Put text to the buffer to send to the compiler
    void putString(const char *string) {
        while ( *string ) {
            buffer[ofs++] = *string++;
        }
    }
    
    // Process #include directives on the given file
    bool processFile(const char *fileName) {
        char lineDirectiveBuff[256];
        lineNo = 1;
        char *programText = fileSystem.openText(fileName);
        if ( programText == NULL ) {
            std::string  err = std::string ("Source file '") + fileName + "' not found.";
            errors += err;
            return false;
        }

        sprintf(lineDirectiveBuff,"\n#line 1 \"%s\"\n", fileName);
        putString(lineDirectiveBuff);

        //Copy to output till we find #include
        char *current = programText;
        while ( true ) {
            char *include = strstr(current, "#include" );
            if ( include ) {
                // Copy till include
                while ( current<include ) {
                    char c = *current++;
                    if ( c == '\n' ) {
                        lineNo++;
                    }
                    buffer[ofs++] = c;
                }

                // skip #include
                include += 8; 

                // Skip whitespace beween #include and opening "
                while ( *include == ' ' || *include == '\t' ) include++;

                // Check opening '"'
                if ( *include != '"' ) {
                    std::string  err = "Expected string literal after #include.";
                    errors += err;
                    return false;
                }
                include++;

                // Parse filename
                char includeName[256];
                int includeNameLen = 0;
                while ( *include != '"' ) {
                    includeName[includeNameLen++] = *include;
                    include++;
                }
                includeName[includeNameLen++] = 0;
                include++;

                // Emit a line directive for the include
                sprintf(lineDirectiveBuff,"\n#line 1 \"%s\"\n", includeName);
                putString(lineDirectiveBuff);

                // Emit the include file itself
                char *includeText = fileSystem.openText(includeName);
                if ( includeText == NULL ) {
                    std::string  err = std::string ("Included file '") + includeName + "' not found.";
                    errors += err;
                    return false;
                }
                putString(includeText);
                delete [] includeText;

                // Emit a line directive to continue the including file
                sprintf(lineDirectiveBuff,"\n#line %i \"%s\"\n", lineNo, fileName);
                putString(lineDirectiveBuff);

                // Continue
                current = include;
            } else {
                // Copy till end of string
                while ( *current ) {
                    buffer[ofs++] = *current++;
                }         
                break;
            }

        }

        delete [] programText;
        return true;
    }

    static void OnCgError(CGcontext ctx, CGerror err, void *thisp ) {
        ((PreprocessBuffer*)thisp)->onCgError(ctx,err);
    }

    void onCgError(CGcontext ctx, CGerror err) {	    
	    if ( err == CG_COMPILER_ERROR ) {
		    const char *listing = cgGetLastListing(ctx);
		    if ( listing != NULL ) {
                errors += "\n";
                errors += listing;
            } else {
                errors += "\n";
                errors += "Empty listing";
            }
        } else {
            errors += "\n";
            errors += cgGetErrorString(err);
        }
    }

public:

    CGeffect compileFile(const char *fileName) {
        CGeffect handle;
        ofs = 0; //Empty the buffer
        errors.clear();

        // Store old CG error handler
        void *oldAppData;
        CGerrorHandlerFunc oldFunc = cgGetErrorHandler(&oldAppData);
        cgSetErrorHandler(OnCgError,this);

        // Fill buffer with text to compile
        processFile("./base/shaders/system/std.cg");
        processFile(fileName);
        buffer[ofs++] = 0; // null terminator

        // Compile it!
        handle = cgCreateEffect(klCgContext, buffer, NULL);

        // Restore old handler
        cgSetErrorHandler(oldFunc,oldAppData);

        return handle;
    }

    const char *getErrors(void) {
        if (errors.empty()) {
            return NULL;
        } else {
            return errors.c_str();
        }
    }

} preprocessBuffer;

//
// The real effect implementation
//

klEffect::klEffect( const char *effectName) {
    loadResources(effectName);
}

void klEffect::loadResources( const char *effectName) {
    char fileName[256];
    EffectNameToFile(effectName,fileName);

    // Reload till it compiles or the user clicks no...
    while (true) {
        handle = preprocessBuffer.compileFile(fileName);

        // Deal with compiler errors
        const char *err = preprocessBuffer.getErrors();
        if ( err ) {
            std::string errorMsg = std::string ("Error(s) compiling '")+fileName+"':\n"+err+"\nRetry compilation?\n";
            klLog(errorMsg.c_str());
            if ( MessageBox(NULL, errorMsg.c_str(), "Effect Loader", MB_YESNO | MB_ICONERROR ) != IDYES ) {
                break;
            }
        } else {
            break;
        }
    }

    cgSetEffectName(handle, effectName);

    // Find what technique we need to use...
    tech = cgGetFirstTechnique(handle);
    while (tech && cgValidateTechnique(tech) == CG_FALSE) {
        const char *errString = cgGetLastListing(klCgContext);
        if ( errString != NULL ) {
            klError("*** Cg Compiler ***\n%s", errString);
        }
        klLog("Technique %s did not validate.  Skipping.\n", cgGetTechniqueName(tech));
        tech = cgGetNextTechnique(tech);
    }

    parmModelViewProjection = getBuildInSemanticParameter("MODELVIEWPROJECTION");
    parmModelToWorld        = getBuildInSemanticParameter("MODELTOWORLD");
    parmWorldToModel        = getBuildInSemanticParameter("WORLDTOMODEL");
    parmInvViewProjection   = getBuildInSemanticParameter("INVVIEWPROJECTION");
    parmInvProjection       = getBuildInSemanticParameter("INVPROJECTION");

    parmModelCameraOrigin   = getBuildInSemanticParameter("MODELCAMERAORIGIN");
    parmModelLightOrigin    = getBuildInSemanticParameter("MODELLIGHTORIGIN"); 

    parmTime                = getBuildInSemanticParameter("TIME"); 

    parmUserParms           = getBuildInSemanticParameter("USERPARAMS");

    if ( parmInvProjection ) {
        cgConnectParameter(effectManager.parmInvProjection, parmInvProjection);
    }

    if ( parmInvViewProjection ) {
        cgConnectParameter(effectManager.parmInvViewProjection, parmInvViewProjection);
    }

    if ( parmTime ) {
        cgConnectParameter(effectManager.parmTime, parmTime);
    }

}

void klEffect::freeResources(void) {
    cgDestroyEffect(handle);
    handle = NULL;
    currentInstance = 0;
    parmModelViewProjection = NULL;
    parmModelToWorld = NULL;
    parmWorldToModel = NULL;
    parmInvViewProjection = NULL;
    parmInvProjection = NULL;

    parmModelCameraOrigin = NULL;
    parmModelLightOrigin = NULL;
    instanceDependentPrograms.clear();
}


void klEffect::reload(const char *effectName) {
    freeResources();
    loadResources(effectName);
}

void klEffect::dumpCompiledPrograms(void) {
    std::vector<CGprogram> dump;

    // Find the programs
    CGparameter parameter = cgGetFirstEffectParameter(handle);
    while(parameter) {
        int numConnected = cgGetNumConnectedToParameters(parameter);
        for (int i=0; i<numConnected; i++) {
            CGparameter programParam = cgGetConnectedToParameter(parameter, i);
            CGprogram prog = cgGetParameterProgram(programParam);
            if ( prog == NULL ) continue;

            // Already in list?
            bool inList = false;
            for ( size_t j=0; j<dump.size(); j++ ) {
                if ( dump[j] == prog ) {
                    inList = true;
                    break;
                }
            }

            // Add if not
            if ( !inList ) {
                dump.push_back(prog);
            }
        }

        parameter = cgGetNextParameter(parameter);
    }

    // And dump them!
    for ( size_t i=0; i<dump.size(); i++ ) {
        const char *src = cgGetProgramString(dump[i], CG_COMPILED_PROGRAM);
        if ( src ) {
            char fname[256];
            sprintf(fname,"./base/dump/%s.%i.txt",cgGetEffectName(handle),i);
            FILE *f = fopen(fname,"w");
            if (f) {
                fprintf(f,src);
                fclose(f);
            }
        }
    }
}


CGparameter klEffect::getBuildInSemanticParameter(const char *name) {
    CGparameter parameter = cgGetEffectParameterBySemantic(handle,name);
    
    if ( parameter == 0 ) return NULL;
    if ( !cgIsParameterUsed(parameter, handle) ) return NULL;

    // Find the programs using this parameter so we can synchronise them when
    // the instance changes during rendering
    int numConnected = cgGetNumConnectedToParameters(parameter);
    for (int i=0; i<numConnected; i++) {
        CGparameter programParam = cgGetConnectedToParameter(parameter, i);
        if (cgIsParameterReferenced(programParam)) {
            CGprogram prog = cgGetParameterProgram(programParam);
            if ( prog == NULL ) continue;
            
            // Already in list?
            bool inList = false;
            for ( size_t j=0; j<instanceDependentPrograms.size(); j++ ) {
                if ( instanceDependentPrograms[j] == prog ) {
                    inList = true;
                    break;
                }
            }

            // Add if not
            if ( !inList ) {
                instanceDependentPrograms.push_back(prog);
            }
        }
    }

    return parameter;
}

void klEffect::setup(void) {
//    
    CGpass pass = cgGetFirstPass(tech);
    cgSetPassState(pass);  
}

void klEffect::setupInstance(klInstanceParameters &instance) {
    if ( currentInstance == instance.instanceId ) {
        return;
    }

    // Set cg effects state, this just caches it and sets the 'dirty' bit
    if ( parmModelViewProjection ) {
        cgGLSetMatrixParameterfc(parmModelViewProjection, instance.modelViewProjection.toPtr());
    }

    if ( parmModelToWorld ) {
        cgGLSetMatrixParameterfc(parmModelToWorld, instance.modelToWorld.toPtr());
    }

    if ( parmWorldToModel ) {
        cgGLSetMatrixParameterfc(parmWorldToModel, instance.worldToModel.toPtr());
    }

    if ( parmModelCameraOrigin ) {
        cgGLSetParameter3fv(parmModelCameraOrigin, instance.modelSpaceCamera.toPtr());
    }

    if ( parmModelLightOrigin ) {
        cgGLSetParameter3fv(parmModelLightOrigin, instance.modelSpaceLight.toPtr());
    }

    if ( parmUserParms ) {
        cgSetParameterValuefc(parmUserParms, klInstanceParameters::NUM_PARAMETERS, instance.userParams);
    }

    // Now syncronise the dirty parameters with opengl
    for ( size_t i=0; i<instanceDependentPrograms.size(); i++ ) {
        cgUpdateProgramParameters(instanceDependentPrograms[i]);
    }

    currentInstance = instance.instanceId;

}

void klEffect::reset(void) {
    CGpass pass = cgGetFirstPass(tech);
    cgResetPassState(pass);  
}


void Con_DumpCg(const char *) {
    dumpCgContext(klCgContext,true);  
}

void Con_ReloadEffects(const char *args) {
    effectManager.reload();
}

void Con_ReloadTextures(const char *args) {
    textureManager.reload();
}

klEffectManager effectManager;

void klEffectManager::init(void) {
    if ( klCgContext == NULL ) {
        klCgContext = cgCreateContext();
        cgSetParameterSettingMode(klCgContext, CG_DEFERRED_PARAMETER_SETTING);        
        cgGLRegisterStates(klCgContext);
        cgGLSetManageTextureParameters(klCgContext, CG_FALSE);

        //Global parameters
        parmInvProjection     = cgCreateParameter(klCgContext, CG_FLOAT4x4);
        parmInvViewProjection = cgCreateParameter(klCgContext, CG_FLOAT4x4);
        parmTime              = cgCreateParameter(klCgContext, CG_FLOAT);

        console.registerCommand("dumpcg",Con_DumpCg);
        console.registerCommand("reloadeffects",Con_ReloadEffects);
        console.registerCommand("reloadtextures",Con_ReloadTextures);
    }
}

klEffect *klEffectManager::getInstance(const char *name) {
    klEffect *effect = new klEffect(name);
    effect->dumpCompiledPrograms();
    return effect;   
}

void klEffectManager::setGlobalParameter(GlobalParamKind pk, const float *val) {
    switch (pk) {
        case PK_INVPROJECTION:
            cgSetMatrixParameterfc(parmInvProjection, val);
            break;
        case PK_INVVIEWPROJECTION:
            cgSetMatrixParameterfc(parmInvViewProjection, val);
            break;
        case PK_TIME:
            //cgSetParameterfc(parmTime, val);
            cgSetParameter1f(parmTime,*val);
            break;
    }
}