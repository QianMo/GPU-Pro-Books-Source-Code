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
#include <Cg/cg.h>

static void dumpCgStateAssignment(CGstateassignment sa, CGbool unabridged)
{
  CGstate state = cgGetStateAssignmentState(sa);
  CGtype type = cgGetStateType(state);
  int numDependents = cgGetNumDependentStateAssignmentParameters(sa);
  CGprogram program;
  const float *fvalues;
  const int *ivalues;
  const CGbool *bvalues;
  int numValues;
  int i;

  klLog("        StateAssignment 0x%p:\n", sa);
  klLog("          State Name: %s\n",
    cgGetStateName(state));
  klLog("          State Type: %s\n",
    cgGetTypeString(type));
  switch (type) {
  case CG_PROGRAM_TYPE:
    program = cgGetProgramStateAssignmentValue(sa);
    klLog("          State Assignment Value: 0x%p\n", program);

    if ( !cgIsProgramCompiled(program ) ) {
        cgCompileProgram(program);
    }

    klLog("Source: %s\n",cgGetProgramString(program,CG_PROGRAM_SOURCE));
    klLog("Compiled: %s\n",cgGetProgramString(program,CG_COMPILED_PROGRAM));
    klLog("Entry: %s\n",cgGetProgramString(program,CG_PROGRAM_ENTRY));
    break;
  case CG_FLOAT:
  case CG_FLOAT1:
  case CG_FLOAT2:
  case CG_FLOAT3:
  case CG_FLOAT4:
    fvalues = cgGetFloatStateAssignmentValues(sa, &numValues);
    klLog("          State Assignment Value:");
    for (i=0; i<numValues; i++) {
      klLog(" %f", fvalues[i]);
    }
    klLog("\n");
    break;
  case CG_INT:
  case CG_INT1:
  case CG_INT2:
  case CG_INT3:
  case CG_INT4:
    ivalues = cgGetIntStateAssignmentValues(sa, &numValues);
    klLog("          State Assignment Value:");
    for (i=0; i<numValues; i++) {
      klLog(" %d (0x%x)", ivalues[i], ivalues[i]);
    }
    klLog("\n");
    break;
  case CG_BOOL:
  case CG_BOOL1:
  case CG_BOOL2:
  case CG_BOOL3:
  case CG_BOOL4:
    bvalues = cgGetBoolStateAssignmentValues(sa, &numValues);
    klLog("          State Assignment Value:");
    for (i=0; i<numValues; i++) {
      klLog(" %s", bvalues[i] ? "TRUE" : "false");
    }
    klLog("\n");
    break;
  case CG_STRING:
    klLog("          State Assignment Value: %s\n", cgGetStringStateAssignmentValue(sa));
    break;
  case CG_TEXTURE:
    klLog("          State Assignment Value: 0x%p\n", cgGetTextureStateAssignmentValue(sa));
    break;
  case CG_SAMPLER1D:
  case CG_SAMPLER2D:
  case CG_SAMPLER3D:
  case CG_SAMPLERCUBE:
  case CG_SAMPLERRECT:
    klLog("          State Assignment Value: 0x%p\n", cgGetSamplerStateAssignmentValue(sa));
    break;
  default:
    klLog("UNEXPECTED CASE: 0x%x (%d)\n", type, type);
    break;
  }
  for (i=0; i<numDependents; i++) {
    CGparameter parameter = cgGetDependentStateAssignmentParameter(sa, i);

    klLog("            Dependent on Parameter %s (0x%p)\n",
      cgGetParameterName(parameter), parameter);
  }
}

static void dumpCgSamplerStateAssignment(CGstateassignment sa, CGbool unabridged)
{
  CGstate state = cgGetSamplerStateAssignmentState(sa);
  CGtype type = cgGetStateType(state);
  int numDependents = cgGetNumDependentStateAssignmentParameters(sa);
  CGprogram program;
  const float *fvalues;
  const int *ivalues;
  const CGbool *bvalues;
  int numValues;
  int i;

  klLog("            SamplerStateAssignment 0x%p:\n", sa);
  klLog("              State Name: %s\n",
    cgGetStateName(state));
  klLog("              State Type: %s\n",
    cgGetTypeString(type));
  switch (type) {
  case CG_PROGRAM_TYPE:
    program = cgGetProgramStateAssignmentValue(sa);
    klLog("              State Assignment Value: 0x%p\n", program);
    break;
  case CG_FLOAT:
  case CG_FLOAT1:
  case CG_FLOAT2:
  case CG_FLOAT3:
  case CG_FLOAT4:
    fvalues = cgGetFloatStateAssignmentValues(sa, &numValues);
    klLog("              State Assignment Value:");
    for (i=0; i<numValues; i++) {
      klLog(" %f", fvalues[i]);
    }
    klLog("\n");
    break;
  case CG_INT:
  case CG_INT1:
  case CG_INT2:
  case CG_INT3:
  case CG_INT4:
    ivalues = cgGetIntStateAssignmentValues(sa, &numValues);
    klLog("              State Assignment Value:");
    for (i=0; i<numValues; i++) {
      klLog(" %d (0x%x)", ivalues[i], ivalues[i]);
    }
    klLog("\n");
    break;
  case CG_BOOL:
  case CG_BOOL1:
  case CG_BOOL2:
  case CG_BOOL3:
  case CG_BOOL4:
    bvalues = cgGetBoolStateAssignmentValues(sa, &numValues);
    klLog("              State Assignment Value:");
    for (i=0; i<numValues; i++) {
      klLog(" %s", bvalues[i] ? "TRUE" : "false");
    }
    klLog("\n");
    break;
  case CG_STRING:
    klLog("              State Assignment Value: %s\n", cgGetStringStateAssignmentValue(sa));
    break;
  case CG_TEXTURE:
    klLog("              State Assignment Value: 0x%p\n", cgGetTextureStateAssignmentValue(sa));
    break;
  case CG_SAMPLER1D:
  case CG_SAMPLER2D:
  case CG_SAMPLER3D:
  case CG_SAMPLERCUBE:
  case CG_SAMPLERRECT:
    klLog("              State Assignment Value: 0x%p\n", cgGetSamplerStateAssignmentValue(sa));
    break;
  default:
    klLog("UNEXPECTED CASE: 0x%x (%d)\n", type, type);
    break;
  }
  for (i=0; i<numDependents; i++) {
    CGparameter parameter = cgGetDependentStateAssignmentParameter(sa, i);

    klLog("                Dependent on Parameter %s (0x%p)\n",
      cgGetParameterName(parameter), parameter);
  }
}

static const char *parameterClassString(CGparameterclass pc)
{
  switch (pc) {
  case CG_PARAMETERCLASS_UNKNOWN:
    return "unknown";
  case CG_PARAMETERCLASS_SCALAR:
    return "scalar";
  case CG_PARAMETERCLASS_VECTOR:
    return "vector";
  case CG_PARAMETERCLASS_MATRIX:
    return "matrix";
  case CG_PARAMETERCLASS_STRUCT:
    return "struct";
  case CG_PARAMETERCLASS_ARRAY:
    return "array";
  case CG_PARAMETERCLASS_OBJECT:
    return "object";
  default:
    return "invalid";
  }
}

static void dumpCgParameterInfo(CGparameter parameter, int unabridged)
{
  const char *semantic = cgGetParameterSemantic(parameter);
  CGparameterclass paramclass = cgGetParameterClass(parameter);

  klLog("              Class: %s\n", parameterClassString(paramclass));
  klLog("              Type: %s\n", cgGetTypeString(cgGetParameterType(parameter)));
  klLog("              Variability: %s\n", cgGetEnumString(cgGetParameterVariability(parameter)));
  if (semantic && strlen(semantic)) {
    klLog("              Semantic: %s\n", semantic);
  }
  klLog("              Resource: %s:%ld (base %s)\n",
    cgGetResourceString(cgGetParameterResource(parameter)),
    cgGetParameterResourceIndex(parameter),
    cgGetResourceString(cgGetParameterBaseResource(parameter)));
  klLog("              Direction: %s\n", cgGetEnumString(cgGetParameterDirection(parameter)));
}

static CGbool interestingCgProgramParameter(CGparameter parameter)
{
  return cgIsParameterReferenced(parameter);
}

static void dumpCgProgramParameter(CGparameter parameter, CGbool unabridged)
{
  if (cgIsParameterReferenced(parameter)) {
    klLog("            Parameter %s (0x%p):\n",
      cgGetParameterName(parameter), parameter);
    dumpCgParameterInfo(parameter, unabridged);
  } else {
    if (unabridged) {
      klLog("            __unreferenced Parameter %s (0x%p):\n",
        cgGetParameterName(parameter), parameter);
      dumpCgParameterInfo(parameter, unabridged);
    }
  }
}

static CGbool interestingCgConnectedParameter(CGparameter parameter)
{
  if (cgIsParameterReferenced(parameter)) {
    CGprogram program = cgGetParameterProgram(parameter);
    CGbool compiled = cgIsProgramCompiled(program);

    return compiled;
  }
  return CG_FALSE;
}

static void dumpCgConnectedParameter(CGparameter parameter, CGbool unabridged)
{
  CGprogram program = cgGetParameterProgram(parameter);

  if (cgIsParameterReferenced(parameter)) {
    CGbool compiled = cgIsProgramCompiled(program);
    if (unabridged || compiled) {
      klLog("            Connected to Parameter %s (0x%p) of\n",
        cgGetParameterName(parameter), parameter);
      if (compiled) {
        klLog("              Program 0x%p\n",
          program);
      } else {
        klLog("              __uncompiled Program 0x%p\n",
          program);
      }
    }
  } else {
    if (unabridged) {
      klLog("            __unreferenced Connected to Parameter %s (0x%p) of\n",
        cgGetParameterName(parameter), parameter);
      if (cgIsProgramCompiled(program)) {
        klLog("              __uncompiled Program 0x%p\n",
          program);
      } else {
        klLog("              Program 0x%p\n",
          program);
      }
    }
  }
}

static CGbool interestingCgEffectParameter(CGparameter parameter)
{
  int numConnected = cgGetNumConnectedToParameters(parameter);
  int i;

  for (i=0; i<numConnected; i++) {
    CGbool interesting = interestingCgConnectedParameter(cgGetConnectedToParameter(parameter, i));

    if (interesting) {
      return CG_TRUE;
    }
  }
  return CG_FALSE;
}

static void dumpCgEffectParameterInfo(CGparameter parameter, int numConnected, CGbool unabridged)
{
  const char *semantic = cgGetParameterSemantic(parameter);
  int i;

  if (semantic && strlen(semantic)) {
    klLog("            Semantic: %s\n", semantic);
  }
  if (cgGetParameterClass(parameter) == CG_PARAMETERCLASS_SAMPLER) {
    CGstateassignment sa;

    for (sa = cgGetFirstSamplerStateAssignment(parameter); sa; sa = cgGetNextStateAssignment(sa)) {
      dumpCgSamplerStateAssignment(sa, unabridged);
    }
  }
  for (i=0; i<numConnected; i++) {
    dumpCgConnectedParameter(cgGetConnectedToParameter(parameter, i), unabridged);
  }
}

static void dumpCgEffectParameter(CGparameter parameter, CGbool unabridged)
{
  if (unabridged || interestingCgEffectParameter(parameter)) {
    int numConnected = cgGetNumConnectedToParameters(parameter);

    if (numConnected) {
      const char *name = cgGetParameterName(parameter);


      klLog("          Parameter %s (0x%p) for effect:\n", name, parameter);
      dumpCgEffectParameterInfo(parameter, numConnected, unabridged);
    } else {
      if (unabridged) {
        const char *name = cgGetParameterName(parameter);

        klLog("          __unconnected Parameter %s (0x%p) for effect:\n",
          name, parameter);
        dumpCgEffectParameterInfo(parameter, 0, unabridged);
      }
    }
  }
}

static void dumpCgEffectParameters(CGeffect effect, CGbool unabridged)
{
  CGparameter parameter;

  for (parameter = cgGetFirstEffectParameter(effect);
    parameter;
    parameter = cgGetNextParameter(parameter)) {

      dumpCgEffectParameter(parameter, unabridged);
    }
}

static CGbool interestingCgProgramParameters(CGprogram program, CGenum nameSpace)
{
  CGparameter parameter;

  for (parameter = cgGetFirstParameter(program, nameSpace);
    parameter;
    parameter = cgGetNextParameter(parameter)) {
      if (interestingCgProgramParameter(parameter)) {
        return CG_TRUE;
      }
    }
    return CG_FALSE;
}

static void dumpCgProgramParameters(CGprogram program, 
                                    CGenum nameSpace, 
                                    char const *nameSpaceName, 
                                    CGbool unabridged)
{
  CGparameter parameter;

  klLog("  %s:\n", nameSpaceName);
  for (parameter = cgGetFirstParameter(program, nameSpace);
    parameter;
    parameter = cgGetNextParameter(parameter)) {
      dumpCgProgramParameter(parameter, unabridged);
    }
}

static void dumpCgProgram(CGprogram program, CGbool unabridged)
{
    klLog("  Profile: %s\n", cgGetProfileString(cgGetProgramProfile(program)));
    if (unabridged || interestingCgProgramParameters(program, CG_GLOBAL)) {
        dumpCgProgramParameters(program, CG_GLOBAL, "Global Parameters", unabridged);
    }
    if (unabridged || interestingCgProgramParameters(program, CG_PROGRAM)) {
        dumpCgProgramParameters(program, CG_PROGRAM, "Program Parameters", unabridged);
    }
}

static void dumpCgPrograms(CGcontext context, CGbool unabridged)
{
  CGprogram program;
  int programNumber = 0;

  for (program = cgGetFirstProgram(context);
       program;
       program = cgGetNextProgram(program), programNumber++) {
    if (cgIsProgramCompiled(program)) {
      klLog("Program 0x%p (%d):\n", program, programNumber);
      dumpCgProgram(program, unabridged);
    } else {
      if (unabridged) {
        klLog("__uncompiled Program 0x%p (%d):\n", program, programNumber);
        dumpCgProgram(program, unabridged);
      }
    }
  }
}

static void dumpCgStateAssignments(CGpass pass, CGbool unabridged)
{
  CGstateassignment sa;

  for (sa = cgGetFirstStateAssignment(pass);
       sa;
       sa = cgGetNextStateAssignment(sa)) {

      dumpCgStateAssignment(sa, unabridged);
    }
}

static void dumpCgPasses(CGtechnique technique, CGbool unabridged)
{
  CGpass pass;
  int passNumber = 0;

  for (pass = cgGetFirstPass(technique);
    pass;
    pass = cgGetNextPass(pass), passNumber++) {
    klLog("      Pass %s (0x%p, %d):\n",
      cgGetPassName(pass), pass, passNumber);
    dumpCgStateAssignments(pass, unabridged);
  }
}

static void dumpCgTechniques(CGeffect effect, CGbool unabridged)
{
  CGtechnique technique;
  int techniqueNumber = 0;

  for (technique = cgGetFirstTechnique(effect);
       technique;
       technique = cgGetNextTechnique(technique), techniqueNumber++) {
    if (cgIsTechniqueValidated(technique)) {
      klLog("    Technique %s (0x%p, %d):\n", cgGetTechniqueName(technique), technique, techniqueNumber);
      dumpCgPasses(technique, unabridged);
    } else {
      if (unabridged) {
        klLog("    __unvalidated Technique %s (%d):\n", cgGetTechniqueName(technique), techniqueNumber);
        dumpCgPasses(technique, unabridged);
      }
    }
  }
}

static void dumpCgEffects(CGcontext context, CGbool unabridged)
{
  CGeffect effect;
  int effectNumber = 0;

  for (effect = cgGetFirstEffect(context);
       effect;
       effect = cgGetNextEffect(effect), effectNumber++) {
    klLog("Effect 0x%p (%d):\n", effect, effectNumber);
    klLog("  Effect Parameters:\n");
    dumpCgEffectParameters(effect, unabridged);
    klLog("  Techniques:\n");
    dumpCgTechniques(effect, unabridged);
  }
}

void dumpCgContext(CGcontext context, CGbool unabridged)
{
  dumpCgPrograms(context, unabridged);
  dumpCgEffects(context, unabridged);
}
