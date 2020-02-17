#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import io
import os
import sys

import numpy as np  # type: ignore

from onnx import defs, FunctionProto, helper, OperatorStatus
from onnx.defs import OpSchema, ONNX_DOMAIN, ONNX_ML_DOMAIN
from onnx.backend.test.case import collect_snippets
from onnx.backend.sample.ops import collect_sample_implementations
from typing import Any, Text, Sequence, Dict, List, Type, Set, Tuple


#controls on ONNF code gen
#specify attr default value 
special_attr_defaults = dict([
#        ("AveragePool "+"kernel_shape", ('ints', '{}')),
#        ("MaxPool "+"kernel_shape", ('ints', '{}')),
#        ("Cast "+"to", ('int', '0')),
#        ("Concat "+"axis", ('int', '0')),
#        ("Conv "+"group", ('int', '1')),
#        ("Unsqueeze "+"axes", ('ints', '{}')),
#        ("RNN "+"activation_alpha", ('floats', '{}')),
#        ("RNN "+"activation_beta", ('floats', '{}')),
        ])

#specify the function name in src/builder/frontend_dialect_transformer.cpp
#the reason for Conv and MaPool is to handled optional arguments
special_op_handler = dict([
        ("Conv", "ImportNodeConv"),
        ("MaxPool", "ImportNodeMaxPool"),
        ("Gemm", "ImportNodeGemm"),
        ("Pad", "ImportNodePad"),
        #("Transpose", "ImportNodeTranspose")
        ])

#add an Op in this list if ShapeInterference is defined for this Op
ShapeInferenceList=['Exp', 'Tanh', 'Sinh', 'Cosh', 'Sigmoid', 'Relu',
                   'Add', 'Mul', 'Div', 'Sub', 'And', 'Or', 'Xor',
                   'Sum', 'Max', 'Min', 'MatMul', 'Gemm', 'LeakyRelu',
                   'Elu', 'Selu', 'HardSigmoid', 'Reshape', 'Reciprocal',
                   'Identity', 'Cos', 'Log', 'Transpose', 'Softmax',
                   'ReduceMax', 'ReduceMin', 'ReduceProd', 'ReduceSum',
                   'Softplus', 'Softsign', 'Sqrt', 'Unsqueeze', 'Sign']

CanonicalList=['Add', 'Identity', 'ReduceL1', 'ReduceL2', 'ReduceLogSum',
               'ReduceLogSumExp', 'ReduceSumSquare']

#add an Op in this list if the Op needs result type deduction which is required
#when writing declarative rewriting rules.
custom_builder_ops_list = ['Abs', 'Mul', 'Exp']

manual_code_in_op_def = dict([
      ('DummyExample', '  let extraClassDeclaration = [{ \n'+
                    '    static StringRef getPermAttrName() { return "perm"; }\n'+
                    '    }];\n')
      ])


SNIPPETS = collect_snippets()
SAMPLE_IMPLEMENTATIONS = collect_sample_implementations()
ONNX_ML = not bool(os.getenv('ONNX_ML') == '0')

ONNX_ML = False
print("ONNX_ML", ONNX_ML)


if ONNX_ML:
    ext = '-ml.md'
else:
    ext = '.md'


def display_number(v):  # type: (int) -> Text
    if defs.OpSchema.is_infinite(v):
        return '&#8734;'
    return Text(v)


def should_render_domain(domain):  # type: (Text) -> bool
    if domain == ONNX_ML_DOMAIN and not ONNX_ML:
        return False
    elif ONNX_ML and domain != ONNX_ML_DOMAIN:
        return False
    return True


def format_name_with_domain(domain, schema_name):  # type: (Text, Text) -> Text
    if domain:
        return '{}.{}'.format(domain, schema_name)
    else:
        return schema_name


def display_attr_type(v):  # type: (OpSchema.AttrType) -> Text
    assert isinstance(v, OpSchema.AttrType)
    s = Text(v)
    s = s[s.rfind('.') + 1:].lower()
    if s[-1] == 's':
        s = 'list of ' + s
    return s


def display_domain(domain):  # type: (Text) -> Text
    if domain:
        return "the '{}' operator set".format(domain)
    else:
        return "the default ONNX operator set"


def display_domain_short(domain):  # type: (Text) -> Text
    if domain:
        return domain
    else:
        return 'ai.onnx (default)'


def display_version_link(name, version):  # type: (Text, int) -> Text
    changelog_md = 'Changelog' + ext
    name_with_ver = '{}-{}'.format(name, version)
    return '<a href="{}#{}">{}</a>'.format(changelog_md, name_with_ver, name_with_ver)

def get_unique_output_name(schema, name):
    for input in schema.inputs :
        if input.name == name :
            return 'out_'+name
    return name

def display_schema(schema, versions):  # type: (OpSchema, Sequence[OpSchema]) -> Text
    s = ''

    # doc
    if schema.doc:
        s += '\n'
        s += '\n'.join('  ' + line
                       for line in schema.doc.lstrip().splitlines())
        s += '\n'

    # since version
    s += '\n#### Version\n'
    if schema.support_level == OpSchema.SupportType.EXPERIMENTAL:
        s += '\nNo versioning maintained for experimental ops.'
    else:
        s += '\nThis version of the operator has been ' + ('deprecated' if schema.deprecated else 'available') + ' since version {}'.format(schema.since_version)
        s += ' of {}.\n'.format(display_domain(schema.domain))
        if len(versions) > 1:
            # TODO: link to the Changelog.md
            s += '\nOther versions of this operator: {}\n'.format(
                ', '.join(display_version_link(format_name_with_domain(v.domain, v.name),
                                               v.since_version) for v in versions[:-1]))

    # If this schema is deprecated, don't display any of the following sections
    if schema.deprecated:
        return s

    # attributes
    if schema.attributes:
        s += '\n#### Attributes\n\n'
        s += '<dl>\n'
        for _, attr in sorted(schema.attributes.items()):
            # option holds either required or default value
            opt = ''
            if attr.required:
                opt = 'required'
            elif attr.default_value.name:
                default_value = helper.get_attribute_value(attr.default_value)

                def format_value(value):  # type: (Any) -> Text
                    if isinstance(value, float):
                        formatted = str(np.round(value, 5))
                        # use default formatting, unless too long.
                        if (len(formatted) > 10):
                            formatted = str("({:e})".format(value))
                        return formatted
                    elif isinstance(value, (bytes, bytearray)) and sys.version_info[0] == 3:
                        return str(value.decode('utf-8'))
                    return str(value)

                if isinstance(default_value, list):
                    default_value = [format_value(val) for val in default_value]
                else:
                    default_value = format_value(default_value)
                opt = 'default is {}'.format(default_value)

            s += '<dt><tt>{}</tt> : {}{}</dt>\n'.format(
                attr.name,
                display_attr_type(attr.type),
                ' ({})'.format(opt) if opt else '')
            s += '<dd>{}</dd>\n'.format(attr.description)
        s += '</dl>\n'

    # inputs
    s += '\n#### Inputs'
    if schema.min_input != schema.max_input:
        s += ' ({} - {})'.format(display_number(schema.min_input),
                                 display_number(schema.max_input))
    s += '\n\n'
    if schema.inputs:
        s += '<dl>\n'
        for input in schema.inputs:
            option_str = ""
            if OpSchema.FormalParameterOption.Optional == input.option:
                option_str = " (optional)"
            elif OpSchema.FormalParameterOption.Variadic == input.option:
                if input.isHomogeneous:
                    option_str = " (variadic)"
                else:
                    option_str = " (variadic, heterogeneous)"
            s += '<dt><tt>{}</tt>{} : {}</dt>\n'.format(input.name, option_str, input.typeStr)
            s += '<dd>{}</dd>\n'.format(input.description)
        s += '</dl>\n'

    # outputs
    s += '\n#### Outputs'
    if schema.min_output != schema.max_output:
        s += ' ({} - {})'.format(display_number(schema.min_output),
                                 display_number(schema.max_output))
    s += '\n\n'

    if schema.outputs:
        s += '<dl>\n'
        for output in schema.outputs:
            option_str = ""
            if OpSchema.FormalParameterOption.Optional == output.option:
                option_str = " (optional)"
            elif OpSchema.FormalParameterOption.Variadic == output.option:
                if output.isHomogeneous:
                    option_str = " (variadic)"
                else:
                    option_str = " (variadic, heterogeneous)"
            s += '<dt><tt>{}</tt>{} : {}</dt>\n'.format(get_unique_output_name(schema, output.name), option_str, output.typeStr)
            s += '<dd>{}</dd>\n'.format(output.description)
        s += '</dl>\n'

    # type constraints
    s += '\n#### Type Constraints'
    s += '\n\n'
    if schema.type_constraints:
        s += '<dl>\n'
        for type_constraint in schema.type_constraints:
            allowedTypes = type_constraint.allowed_type_strs
            if (len(allowedTypes) > 0):
                allowedTypeStr = allowedTypes[0]
            for allowedType in allowedTypes[1:]:
                allowedTypeStr += ', ' + allowedType
            s += '<dt><tt>{}</tt> : {}</dt>\n'.format(
                type_constraint.type_param_str, allowedTypeStr)
            s += '<dd>{}</dd>\n'.format(type_constraint.description)
        s += '</dl>\n'

    # Function Body
    if schema.has_function:  # type: ignore
        s += '\n#### Function\n'
        s += '\nThe Function can be represented as a function.\n'

    return s


def support_level_str(level):  # type: (OpSchema.SupportType) -> Text
    return \
        "<sub>experimental</sub> " if level == OpSchema.SupportType.EXPERIMENTAL else ""

def convert_type(tstr) :
    tfrom = np.array(['bool', 'int8', 'int16', 'int32', 'int64',
            'unkown', 'float16', 'float', 'double'])
    tto =np.array(['I1', 'I8', 'I16', 'I32', 'I64',
         'BF16', 'F16', 'F32', 'F64'])
    index = -1
    for i in range(len(tfrom)) :
        if tfrom[i] in tstr :
            index = i
            break
    if index == -1 :
        print("error", tstr)
        return ''
    else :
        return tto[i]

def  collect_types(schema, input) :
    allowedTypeStr=''
    #first step just ignore the type constraints
    return allowedTypeStr
    if input.typeStr :
        tstr = input.typeStr
    else :
        return allwedTypeStr
    if schema.type_constraints:
        for type_constraint in schema.type_constraints:
            if type_constraint.type_param_str != tstr :
                continue
            allowedTypes = type_constraint.allowed_type_strs
            allowedTypeStr=''
            if (len(allowedTypes) > 0):
                t = convert_type(allowedTypes[0])
                if t == '' :
                    return ''
                allowedTypeStr += t
            for allowedType in allowedTypes[1:]:
                t = convert_type(allowedType)
                if t == '' :
                    return ''
                if  not t in allowedTypeStr :
                    allowedTypeStr += ', '+t

            return allowedTypeStr

    return allowedTypeStr

def gen_schema(schema) :
    line_indent = '  '

    #s = 'def ONNX'+schema.name+str(schema.since_version)+'Op:ONNX_Op<"'+schema.name+'", \n'
    s = 'def ONNX'+schema.name+'Op:ONNX_Op<"'+schema.name+'", \n'
    s += line_indent+'  [NoSideEffect'
    if schema.name in ShapeInferenceList :
        s+= ', DeclareOpInterfaceMethods<ShapeInferenceOpInterface>'
    s += ']> {'

    if schema.name in CanonicalList:
        s += '\n'+line_indent+'let hasCanonicalizer = 1;'

    #summary
    s += '\n'+line_indent
    s += 'let summary = "ONNX '+schema.name+' operation";'

    #description
    s += '\n'+line_indent
    s += 'let description = [{'
    if schema.doc:
        """
        s += '\n'.join(line_indent + line
                   for line in schema.doc.lstrip().splitlines())
        """
        for line in schema.doc.lstrip().splitlines():
            line = line.replace('}]', '\}\]')
            s += '\n'+line_indent+'  '+'"'+line+'"'
    else :
        s += '\n'+line_indent*2 +'no doc for this op from onnx'
    s += '\n'+line_indent+'}];'

    #input
    s+= '\n'+line_indent+'let arguments = (ins '
    isfirst = True
    # add operands
    operand_ins = get_operand_ins(schema)
    for operand_type_name in operand_ins:
        if not isfirst:
            s+= ',\n           '
        else:
            isfirst = False
        s+=operand_type_name[0]+':$'+operand_type_name[1]

    # add attributes
    attr_ins = get_attr_ins(schema)
    for attr_type_name in attr_ins:
        if not isfirst:
            s += ',\n           '
        else :
            isfirst = False
        s += attr_type_name[0]+':$'+attr_type_name[1]
    s+= ');'

    #output
    s+= '\n'+line_indent+'let results = (outs '
    if schema.outputs:
        for output in schema.outputs:
            if output != schema.outputs[0] :
                s+= ',\n           '
            #need to interpret output.typeStr
            etypes=collect_types(schema, output)
            if etypes == '':
                s+= 'AnyTypeOf<[AnyMemRef, AnyTensor]>'
            else:
                s+= 'TensorOf<['+etypes+']>'
            s += ':$'+get_unique_output_name(schema, output.name)
    s+= ');\n'

    #s+= 'let hasCanonicalizer = 1;'

    # add custom builders
    # use element type of the first operand to construct an UnrankedTensorType for the output.
    if schema.name in custom_builder_ops_list:
        s += line_indent+'let builders = [\n'
        # custom builders with operands only.
        # E.g. OpBuilder<"Builder *builder, OperationState &state, Value X, Value, Y", [{}]>
        s += line_indent*2+'OpBuilder<"Builder *builder, OperationState &state,'
        isfirst = True
        first_operand = ""
        for _, arg_name in operand_ins:
            if not isfirst:
                s += ', '
            else:
                isfirst = False
                first_operand = arg_name
            s += 'Value '+arg_name
        s += '", [{\n'

        first_operand = operand_ins[0][1]
        s += line_indent*3+'auto elementType = '+first_operand+'.getType().cast<TensorType>().getElementType();\n'
        s += line_indent*3+'build(builder, state, UnrankedTensorType::get(elementType), '
        isfirst = True
        for _, arg_name in operand_ins:
            if not isfirst:
                s += ', '
            else:
                isfirst = False
            s += arg_name
        s += ');\n'

        s += line_indent*2+'}]>,\n'

        # custom builders with all operands and attributes having aggregate parameters.
        # E.g. OpBuilder<"Builder *builder, OperationState &state, ValueRange operands, ArrayRef<NamedAttribute> attributes", [{}]>'
        s += line_indent*2
        s += 'OpBuilder<"Builder *builder, OperationState &state, ValueRange operands, ArrayRef<NamedAttribute> attributes", [{\n'
        s += line_indent*3+'auto elementType = operands[0].getType().cast<TensorType>().getElementType();\n'
        s += line_indent*3+'std::vector<mlir::Type> outputTypes;\n'
        s += line_indent*3+'outputTypes.emplace_back(UnrankedTensorType::get(elementType));\n'
        s += line_indent*3+'build(builder, state, outputTypes, operands, attributes);\n'

        s += line_indent*2+'}]>'
        s += '\n'+line_indent+'];\n'

    #add special code
    if schema.name in manual_code_in_op_def :
        s += manual_code_in_op_def[schema.name]

    s += '}\n\n'

    return s

"""
special cases:
* Split: attr split default value: sizeof(output1) namely 1
* Conv: attr dilations default value is {num_dim of first input - 2, 1}
* Conv: attr kernel_shape type is ints
* Transpose: attr perm default value is {} empty int list
"""

def gen_code(schema,fefile) :

    handle_variadic = False

    line_indent = '  '
    fefile.write('    '+'}else if (OpName == "'+schema.name+'") {\n')
    op_type_str='mlir::ONNX'+schema.name+'Op'
    if schema.name in special_op_handler :
        fefile.write('       '+special_op_handler[schema.name]+'(node, '
          +str(len(schema.inputs))
          +', ' +str(len(schema.outputs)))
    elif len(schema.outputs) > 1 :
        fefile.write('       '+'ImportNodeMultipleOuts<'+op_type_str+'>(node, '
          +str(len(schema.inputs))
          +', ' +str(len(schema.outputs)))
    else :
        fefile.write('       '+'ImportNodeOneOut<'+op_type_str+'>(node, '
          +str(len(schema.inputs))
          +', ' +str(len(schema.outputs)))

    variadicIn = 'false'
    variadicOut = 'false'
    for input in schema.inputs:
        if OpSchema.FormalParameterOption.Variadic == input.option:
            if input.isHomogeneous:
                variadicIn = 'true'
                handle_variadic = True
    for output in schema.outputs:
        if OpSchema.FormalParameterOption.Variadic == output.option:
            if output.isHomogeneous:
                variadicOut = 'true'
    if not handle_variadic:
        fefile.write(');\n')
    else:
        fefile.write(', '+variadicIn+', '+variadicOut+');\n')

def get_operand_ins(schema):
    operand_type_and_name_list = []  # [(optype, opname)]
    if schema.inputs:
        for input in schema.inputs:
            optype = ""

            etypes=collect_types(schema, input)

            if OpSchema.FormalParameterOption.Optional == input.option:
                #TODO : handle optional
                print("warning: optional input for"+schema.name+' '+input.name)
            elif OpSchema.FormalParameterOption.Variadic == input.option:
                if input.isHomogeneous:
                    optype += 'Variadic<'
                else:
                    #TODO handle(variadic, heterogeneous) "
                    print("warning: (variadic, heterogeneous) for"+schema.name+' '+input.name)
            if etypes == '':
                optype += 'AnyTypeOf<[AnyMemRef, AnyTensor]>'
            else:
                optype += 'TensorOf<['+etypes+']>'

            if OpSchema.FormalParameterOption.Optional == input.option:
                #TODO : handle optional
                t=''
            elif OpSchema.FormalParameterOption.Variadic == input.option:
                if input.isHomogeneous:
                    optype += '>'
                else:
                    #TODO handle(variadic, heterogeneous) "
                    t=''
            operand_type_and_name_list.append((optype, input.name))
    return operand_type_and_name_list

def get_attr_ins(schema) :
    
    def get_attr_type_basic(attr_type) :
        if attr_type == 'int' :
            mytype = 'I64Attr'
        elif attr_type == 'float' :
            mytype = 'F32Attr'
        elif attr_type == 'ints' :
            mytype = 'I64ArrayAttr'
        elif attr_type == 'floats' :
            mytype = 'F32ArrayAttr'
        elif attr_type == "string" :
            mytype = 'StrAttr'
        elif attr_type == "strings" :
            mytype = 'StrArrayAttr'
        else :
            mytype ='AnyAttr'
        #TODO: tensor and sparse tensor
        return mytype

    def get_attr_type_optional(attr_type) :
        mytype = 'OptionalAttr<'
        mytype += get_attr_type_basic(attr_type)
        mytype += '>'
        return mytype

    def get_attr_type_with_default(attr_type, attr_default) :
        mytype = 'DefaultValuedAttr<'
        mytype += get_attr_type_basic(attr_type)
        mytype += ', "'+attr_default+'">'
        return mytype

    attr_type_and_name_list = []  # :: [(attrtype, attrname)]
    attr_line = ''
    if schema.attributes:
        for _, attr in sorted(schema.attributes.items()):
            #attr_line = line_indent+line_indent+line_indent+line_indent
            found = False
            attr_type = ""
            if schema.name+' '+attr.name in special_attr_defaults:
                (attr_type_str, attr_default_str) = special_attr_defaults[schema.name+' '+attr.name]
                attr_type = get_attr_type_with_default(attr_type_str, attr_default_str)
                found = True
            elif attr.required:
                s = Text(attr.type)
                attr_type_str  = s[s.rfind('.') + 1:].lower()
                attr_type = get_attr_type_basic(attr_type_str)
                found = True

            # option holds either required or default value
            elif attr.default_value.name:
                s = Text(attr.type)
                attr_type_str  = s[s.rfind('.') + 1:].lower()

                default_value = helper.get_attribute_value(attr.default_value)
                def format_value(value):  # type: (Any) -> Text
                    if isinstance(value, float):
                        formatted = str(np.round(value, 5))
                        # use default formatting, unless too long.
                        if (len(formatted) > 10):
                            formatted = str("({:e})".format(value))
                        return formatted
                    elif isinstance(value, (bytes, bytearray)) and sys.version_info[0] == 3:
                        return str(value.decode('utf-8'))
                    return str(value)

                if isinstance(default_value, list):
                    default_value = [format_value(val) for val in default_value]
                    attr_option_str = '{}'.format(default_value)
                    attr_option_str = attr_option_str.replace('[', '{', 1)
                    attr_option_str = attr_option_str.replace(']', '}', 1)
                    if attr_type_str == 'strings' :
                        attr_option_str = attr_option_str.replace("'", '\\"')
                    else :
                        attr_option_str = attr_option_str.replace("'", '')
                else:
                    default_value = format_value(default_value)
                    attr_option_str = default_value
                attr_type = get_attr_type_with_default(attr_type_str, attr_option_str)
                found = True
            else:
                s = Text(attr.type)
                attr_type_str  = s[s.rfind('.') + 1:].lower()
                attr_type = get_attr_type_optional(attr_type_str)
            if found:
                attr_type_and_name_list.append((attr_type, attr.name))
    return attr_type_and_name_list

def main(args):  # type: (Type[Args]) -> None
    with io.open(args.changelog, 'w', newline='') as fout:
        fout.write('## Operator Changelog\n')
        fout.write(
            "*This file is automatically generated from the\n"
            "            [def files](/onnx/defs) via [this script](/onnx/defs/gen_doc.py).\n"
            "            Do not modify directly and instead edit operator definitions.*\n")

        # domain -> version -> [schema]
        dv_index = defaultdict(lambda: defaultdict(list))  # type: Dict[Text, Dict[int, List[OpSchema]]]
        for schema in defs.get_all_schemas_with_history():
            dv_index[schema.domain][schema.since_version].append(schema)

        fout.write('\n')

        for domain, versionmap in sorted(dv_index.items()):
            if not should_render_domain(domain):
                continue

            s = '# {}\n'.format(display_domain_short(domain))

            for version, unsorted_schemas in sorted(versionmap.items()):
                s += '## Version {} of {}\n'.format(version, display_domain(domain))
                for schema in sorted(unsorted_schemas, key=lambda s: s.name):
                    name_with_ver = '{}-{}'.format(format_name_with_domain(domain, schema.name),
                                                   schema.since_version)
                    s += ('### <a name="{}"></a>**{}**' + (' (deprecated)' if schema.deprecated else '') + '</a>\n').format(name_with_ver, name_with_ver)
                    s += display_schema(schema, [schema])
                    s += '\n'

            fout.write(s)

    with io.open(args.output, 'w', newline='', encoding="utf-8") as fout:
        fout.write('## Operator Schemas\n')
        fout.write(
            "*This file is automatically generated from the\n"
            "            [def files](/onnx/defs) via [this script](/onnx/defs/gen_doc.py).\n"
            "            Do not modify directly and instead edit operator definitions.*\n")

        # domain -> support level -> name -> [schema]
        index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # type: Dict[Text, Dict[int, Dict[Text, List[OpSchema]]]]
        for schema in defs.get_all_schemas_with_history():
            #print("check point 0", schema.name, schema.domain, schema.support_level)
            #gen_schema(schema)
            index[schema.domain][int(schema.support_level)][schema.name].append(schema)

        fout.write('\n')

        # Preprocess the Operator Schemas
        # [(domain, [(support_level, [(schema name, current schema, all versions schemas)])])]
        operator_schemas = list()  # type: List[Tuple[Text, List[Tuple[int, List[Tuple[Text, OpSchema, List[OpSchema]]]]]]]
        exsting_ops = set()  # type: Set[Text]
        for domain, _supportmap in sorted(index.items()):
            if not should_render_domain(domain):
                continue

            processed_supportmap = list()
            for _support, _namemap in sorted(_supportmap.items()):
                processed_namemap = list()
                for n, unsorted_versions in sorted(_namemap.items()):
                    versions = sorted(unsorted_versions, key=lambda s: s.since_version)
                    schema = versions[-1]
                    #print("check point 2", schema)
                    if schema.name in exsting_ops:
                        continue
                    exsting_ops.add(schema.name)
                    processed_namemap.append((n, schema, versions))
                processed_supportmap.append((_support, processed_namemap))
            operator_schemas.append((domain, processed_supportmap))

        # Table of contents
        for domain, supportmap in operator_schemas:
            s = '* {}\n'.format(display_domain_short(domain))
            fout.write(s)
            function_ops = list()
            for _, namemap in supportmap:
                for n, schema, versions in namemap:
                    if schema.has_function:  # type: ignore
                        function_ops.append((n, schema, versions))
                        continue
                    s = '  * {}<a href="#{}">{}</a>\n'.format(
                        support_level_str(schema.support_level),
                        format_name_with_domain(domain, n),
                        format_name_with_domain(domain, n))
                    fout.write(s)
            if len(function_ops):
                fout.write('\n')
                fout.write('  **Operators with function registered:**\n')
                for n, schema, versions in function_ops:
                    s = '  * {}<a href="#{}">{}</a>\n'.format(
                        support_level_str(schema.support_level),
                        format_name_with_domain(domain, n),
                        format_name_with_domain(domain, n))
                    fout.write(s)

        fout.write('\n')
        tdfile= io.open(args.tdfile, 'w', newline='') 
        tdfile.write('//********************************************************\n'+
                     '//   Warning: Do not modify this file directly\n'+
                     '//   This file is automatically generated via script\n'+
                     '//   Details can be found in doc/readonnxdefs.md\n'+
                     '//********************************************************\n\n'
               )
        fefile=io.open('op_build_table.inc', 'w', newline='')
        firstfunc = True

        fefile.write('//********************************************************\n'+
                     '//   Warning: Do not modify this file directly\n'+
                     '//   This file is automatically generated via script\n'+
                     '//   Details can be found in doc/readonnxdefs.md\n'+
                     '//********************************************************\n\n'
               )
        fefile.write('    '+'if (OpName == "DUMMY") {\n')
        for domain, supportmap in operator_schemas:
            s = '## {}\n'.format(display_domain_short(domain))
            fout.write(s)

            for _, namemap in supportmap:
                for op_type, schema, versions in namemap:
                    # op_type
                    #print("check point 1", schema.name, len(schema.inputs), len(schema.outputs))
                    gen_code(schema, fefile)

                    r = gen_schema(schema)
                    tdfile.write(r)
                    s = ('### {}<a name="{}"></a><a name="{}">**{}**' + (' (deprecated)' if schema.deprecated else '') + '</a>\n').format(
                        support_level_str(schema.support_level),
                        format_name_with_domain(domain, op_type),
                        format_name_with_domain(domain, op_type.lower()),
                        format_name_with_domain(domain, op_type))
                    
                    s += display_schema(schema, versions)

                    s += '\n\n'

                    if op_type in SNIPPETS:
                        s += '#### Examples\n\n'
                        for summary, code in sorted(SNIPPETS[op_type]):
                            s += '<details>\n'
                            s += '<summary>{}</summary>\n\n'.format(summary)
                            s += '```python\n{}\n```\n\n'.format(code)
                            s += '</details>\n'
                            s += '\n\n'
                    if op_type.lower() in SAMPLE_IMPLEMENTATIONS:
                        s += '#### Sample Implementation\n\n'
                        s += '<details>\n'
                        s += '<summary>{}</summary>\n\n'.format(op_type)
                        s += '```python\n{}\n```\n\n'.format(SAMPLE_IMPLEMENTATIONS[op_type.lower()])
                        s += '</details>\n'
                        s += '\n\n'

                    fout.write(s)
        fefile.write('    }')
        fefile.close()


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    docs_dir = os.path.join(base_dir, 'docs')
    print(docs_dir)

    class Args(object):
        output = os.path.join(docs_dir, 'Operators' + ext)
        changelog = os.path.join(docs_dir, 'Changelog' + ext)
        tdfile = os.path.join(base_dir, 'onnxop.inc')
    print(Args)
    main(Args)
