--------------------------------------------------------------
- Demo program for Rule-based Geometry Synthesis in Real-time
- ShaderX 8 article.
--------------------------------------------------------------

-------------
How it works?
-------------
The demo program ("RuleBasedGeometry.exe") uses L-system 
descriptions given in XML files to generate C++ and HLSL code 
which are used to synthesize geometry fully on the GPU in 
real-time, by the given rules.

Note:
The demo program was tested with ATI radeon 4850 Pro and
NVIDIA 8800 GTX GPUs, DirectX SDK 2008 november and 2009 march.


---------------------------------------
Selecting the grammar description file
---------------------------------------
The name of the input XML grammar description should be given
in "INPUT_NAME.cfg". This input file is loaded when the program
starts. 

Input grammars can be loaded at run-time by right clicking 
anywhere in the program and selecting the "Open" menu. 

Since CPU type is also generated from the Module type given in
the grammar description, when the Module type changes (e.g.
attributes are added, removed, or renamed), the CPU code need
to be recompiled (see below).

Any modification in a grammar description except the change of
the module attributes affects only the generated shader code
(and the values of C++ variables, but not their type). Thus,
any grammar description that has the same attribute type as
the currently loaded grammar can be safely loaded in run-time,
without recompiling the CPU (C++) code and restarting the
application.


---------------------------------------
Recompilation of the project (CPU code)
---------------------------------------
When building the Visual Studio project, a CPU code generator
("CPPCodeGeneration.exe") program is used to generate the 
Module types from the grammar file given in "INPUT_NAME.cfg".
This step is added to the build rules, thus, by rebuilding 
the Visual Studio project, the C++ Module types are 
auto-generated and recompiled.


------------------------
Grammar descriptor files
------------------------
The description of an L-system is given in an XML file.
One file should contain only one L-system description, the
others are neglected.

Every descriptor XML file should follow the format described
in the followings. Please note that the demo program does
only some basic syntax checks. Thus, loading input files 
that do not follow the described format may cause program
crashes. 


-------------------------
Grammar descriptor format
-------------------------
For examples of valid grammar descriptor files please check
the XML files given in the "grammars" directory. The syntax
of this format is described in the followings.

The input format contains many pre-defined types, operators
rule selection methods, etc. Feel free to extend these with 
new ones, by modifying the XMLGrammarLoader class for the 
parsing, and CodeGeneration, CPUCodeGeneration, or
DXCPPCodeGeneration classes for code generation.

The input format is case-sensitive.

- Defining a new L-system:
  The description of an L-system is given in a <grammar> 
  tag. Only the first one is considered at loading.
  
- Symbols of the grammar:
  Symbols are given in <symbols> tag. Every specific 
  symbol is defined in a <symbol> tag. Every symbol must
  have a name. Symbols can have additional parameters:
  - Instancing type:
    Instancing type requires a given mesh name in "mesh"
    attribute. This name will be concatenated to the 
    mesh library defined in the grammar's <properties>
    tag.
    Instancing type also requires a technique name.
    The name of the shader file ("technique" attribute), 
    that contains the instancing code must be given 
    (the ".fx" file extension is added automatically).
    Every instancing shader file must contain a HLSL 
    technique called "instance", which will be used for
    instancing.
  - Rule selection:
    Symbols can have rule selection methods to select
    from multiple successors. It is given in the 
    "rule_selection" attribute. It must have one of the
    following values:
    - "first": the first rule is always chosen.
      The rules are ordered as they are specified in the
      grammar descriptor.
    - "condition": every rule can have a condition, 
      that is, an expression of the module attributes.
      when it evaluates to true, the rule is chosen.
      when the condition is satisfied for multiple rules,
      the first one is chosen. String "module" represents
      the current predecessor module.
    - "random": every rule can have a probability, rules
      are chosen randomly (stochastic selection).
    - "lod": every rule can have a mininum and maximum
      size in pixels. If the predecessor satisfies this,
      the rule is chosen.
      
- Rules of the grammar:
  Rules are given in the <rules> tag. Specific rules are
  given in <rule> tags. Every rule can have the following
  attributes:
  - "condition": the condition used in rule selection.
    Use it in conjunction with "condition" rule selection.
    The value of this attribute is a logical expression
    of the module attributes. Attributes are given as
    "module.attribute_name". E.g. size <= 2 will become
    "module.size <= 2". Please note that characters "<"
    and "&" are escape sequences in XML. Use &gt; and
    &amp; instead ( http://en.wikipedia.org/wiki/XML ).
    The given condition is simply copied to the shader
    code, so invalid expressions will generate an error
    in the HLSL shader compiler.
  - "probability": the probability of the rule. 
    Use it in conjunction with "random" rule selection.
    Must be between 0 and 1.
  - "lod_min_pixels", "lod_max_pixels": the minimum and
    maximum allowed size in pixels. 
    Use it in conjunction with "lod" rule selection.
  -------------------
  - Rule predecessor:
  Every rule must have a predecessor given in a 
  <predecessor> tag. The symbol is specified in the 
  "symbol" attribute.  
  -----------------
  - Rule successor:
  Rules can have one successor (at least 1 rule must have
  at least one successor module). Successor modules are
  given in <successor> tags. The symbol is given by the
  "symbol" attribute.
  ----------------------
  - Successor operators:
  Successors can have operators that change module 
  attributes. Examples for predefined operators can
  be found in the examples grammars.
  ----------------------
  - Operator parameters:
  Operators can have the following two child tags:
    - <animation>: the given operator parameter becomes 
      time-dependent. currently, only one animation type
      is implemented: the parameter oscillates linearly 
      between the given minimum and maximum values by 
      the given period. 
    - <random>: noise is added to the given operator 
      parameter

- Module type: the module type (ordered set of module
  attributes) is given in the module_attributes tag.
  There are predefined attributes that are used by the
  predefined rule selection methods and operators.
  Note that changing the module type results in a change
  in the C++ Module type too, thus, the CPU codes need 
  to be recompiled (see details above).
  
- Grammar axiom:
  The axiom of the grammar (from which the generation 
  starts) is given in the <axiom> tag. It can have 
  a "file" attribute that specifies an XML file that
  contains the grammar axiom. Alternatively, the axiom
  modules with their attribute values can be given here
  using <module> tags and their <attribute> child tags.
  
- Properties:
  Some additional properties in <properties> tag. 
  Can have the following child tags:
  - <symbol_prefix>: every symbol name will get this 
    prefix in the generated code. Useful to avoid name 
    conflicts with shader keywords.
  - <mesh_library>: the directory containing the mesh 
    files.
  - <generation_depth>: initial the depth of 
    the generation step. Note that the actual depth can 
    be changed in run-time by pressing the NUMPAD +/- 
    keys.
    

-----------------    
Example grammars:
-----------------
Example grammars can be found in the "grammars" 
directory. Note that the module type of all of the
examples is combined to one module type (e.g. attribute 
"isDoor" is only used by the labyrinth example, but 
included to every examples). 
The reason for this is that using different module 
types would need the recompilation of the C++ codes. 
By using the same module type in all of the examples, 
any of them can be loaded in run-time. 

- Sierpinski pyramids:
  "sierpinski_pyramids_simple.xml" is the most basic 
  example. It contains the description for the fractal
  called Sierpinski-pyramid. 
  "sierpinski_pyramids_colored.xml" extends the previous
  example with random coloring by using the "add_noise"
  function for the color of the uppermost child pyramid.
  
- Fern:
  "fern.xml" describes a very simple fractal fern.
  It shows 3 elements of the grammar description:
  - axiom loading from file by setting the "file" 
    attribute
  - animation: the angle parameter of the rotation
    operator is animated. Although the motion is quite
    far from reality :) it show that by evaluating the
    L-system in real-time, time-dependent rules can
    be written.
  - random: a small noise is added to the parameters 
    of the rotation operator. Note that this is 
    different than using the "add_noise" operator,
    since "add_noise" directly modifies the attribute's
    value directly, which is not appropriate for 
    attributes like orientation. Orientation is stored 
    as a quaternion, adding a random noise vector to a 
    quaternion makes no sense.
    
- Labyrinth:
  "labyrinth.xml" describes a simple, fully traversable 
  labyrinth. It demonstrates the connection between
  programs and rules:
  - variable assignement is made with the "set" 
    operator. any expression of the module attributes
    and built-in functions can be written here. the
    successor node is referred as "output".
    (the string is copied to the generated code, so
    when the value is invalid, the HLSL shader compiler
    may generate an error)
  - sequence: writing successor modules and operators
  - branching: rules with condition. 
  - loops: a condition of a recusrive rule contains a 
    variable that changes in the successor. an example
    is the splitting of rooms until a specific size
    is reached, the size is halved in each iteration.
    