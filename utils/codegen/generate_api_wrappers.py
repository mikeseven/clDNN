#!/usr/bin/env python2

# INTEL CONFIDENTIAL
# Copyright 2016 Intel Corporation
#
# The source code contained or described herein and all documents related to the source code ("Material") are owned by
# Intel Corporation or its suppliers or licensors. Title to the Material remains with Intel Corporation or its
# suppliers and licensors. The Material contains trade secrets and proprietary and confidential information of Intel
# or its suppliers and licensors. The Material is protected by worldwide copyright and trade secret laws and treaty
# provisions. No part of the Material may be used, copied, reproduced, modified, published, uploaded, posted,
# transmitted, distributed, or disclosed in any way without Intel's prior express written permission.
#
# No license under any patent, copyright, trade secret or other intellectual property right is granted to
# or conferred upon you by disclosure or delivery of the Materials, either expressly, by implication, inducement,
# estoppel or otherwise. Any license under such intellectual property rights must be express and approved by Intel
# in writing.
#
#
# For details about script please contact following people:
#  * [Version: 1.0] Walkowiak, Marcin <marcin.walkowiak@intel.com>

import argparse
import os

import re


# Pattern that filters file names that are headers.
headerFileNamePattern = re.compile('^[a-zA-Z0-9_]+\\.[hH]$')
# Marker that marks API function/data member (at its beginning).
apiMemberMarker = 'CLDNN_API'
# Macros that should
apiMacroMemberMatchers = [
    (re.compile('^\\s*CLDNN_DECLARE_PRIMITIVE_TYPE_ID\\s*\\(\\s*([a-zA-Z0-9_]+)\\s*\\)\\s*;', re.MULTILINE),
     'cldnn_primitive_type_id cldnn_\\1_type_id(cldnn_status* status)')
]
# C language and project reserved keywords (that cannot be used as function/parameter name).
reservedKeywords = [
    'auto', 'else', 'long', 'switch', 'break', 'enum', 'register', 'typedef', 'case', 'extern', 'return', 'union',
    'char', 'float', 'short', 'unsigned', 'const', 'for', 'signed', 'void', 'continue', 'goto', 'sizeof', 'volatile',
    'default', 'if', 'static', 'while', 'do', 'int', 'struct', '_Packed', 'double'
]
# C language and project reserved keyword patterns (that cannot be used as function/parameter name).
reservedKeywordPatterns = [
    re.compile('^__[a-z0-9_]+__$', re.IGNORECASE)
]


apiMemberMatcher = re.compile('^\\s*' + re.escape(apiMemberMarker) + '\\s+([^;]+);', re.MULTILINE)
typeIdentifierSplitter = re.compile('^(.*?)([a-zA-Z_][a-zA-Z0-9_]*)$')


def stripCommentsAndPreprocessor(content):
    """ Strips out comments and preprocessor constructs from text written in C language (or compatible).

    :param content: Text with code written in C language (or compatible).
    :type content: str
    :return: Content of C language code with comments and preprocessor constructs stripped out.
    :rtype: str
    """
    # FSA states:
    # 0 - normal context, start state
    # 1 - string context
    # 2 - string context, after \ character (character escape)
    # 3 - normal context, after / character (possible comment)
    # 4 - multi-line comment context
    # 5 - multi-line comment context, after * character (possible end of comment)
    # 6 - single-line comment context
    # 7 - single-line comment context, after \ character (escape)
    # 8 - preprocessor definition/instruction context
    # 9 - preprocessor definition/instruction context, after \ character (escape)

    state = 0
    strippedOutputArray = []
    for c in content:
        # normal context, start state
        if state == 0:
            if c == '"':
                state = 1   # string
                strippedOutputArray.append(c)
            elif c == '/':
                state = 3   # possible comment (no out)
            elif c == '#':
                state = 8   # preprocessor (no out)
            else:
                strippedOutputArray.append(c)
        # string context
        elif state == 1:
            if c == '\\':
                state = 2   # escape sequence
                strippedOutputArray.append(c)
            elif c == '"':
                state = 0
                strippedOutputArray.append(c)
            else:
                strippedOutputArray.append(c)
        # string context, after \ character (character escape)
        elif state == 2:
            state = 1   # do not leave string context on any character
            strippedOutputArray.append(c)
        # normal context, after / character (possible comment)
        elif state == 3:
            if c == '*':
                state = 4   # multi-line comment (no out)
            elif c == '/':
                state = 6   # single-line comment (no out)
            else:
                state = 0   # not comment (flush previous token)
                strippedOutputArray.append('/')
                strippedOutputArray.append(c)
        # multi-line comment context
        elif state == 4:
            if c == '*':
                state = 5   # possible end of comment (no out)
        # multi-line comment context, after * character (possible end of comment)
        elif state == 5:
            if c == '/':
                state = 0   # end of comment (no out)
            elif c == '*':
                pass   # not end of comment, but check next token for possible end of comment (no out)
            else:
                state = 4   # not end of comment (no out)
        # single-line comment context
        elif state == 6:
            if c == '\n':
                state = 0   # end of comment (append new line)
                strippedOutputArray.append('\n')
            elif c == '\\':
                state = 7   # escape in comment (can escape new line character) (no out)
        # single-line comment context, after \ character (escape)
        elif state == 7:
            state = 6   # do not leave comment on any character (no out)
        # preprocessor definition/instruction context
        elif state == 8:
            if c == '\n':
                state = 0   # end of preprocessor construct (no out)
            elif c == '\\':
                state = 9   # escape in preprocessor construct (no out)
        # preprocessor definition/instruction context, after \ character (escape)
        elif state == 9:
            state = 8   # do not leave preprocessor construct on any character (no out)

    return ''.join(strippedOutputArray)


def isReservedName(name):
    """ Determines whether specified name is reserved in C language or project.

    :param name: Name to check.
    :type name: str
    :return: True, if name is reserved; otherwise, False.
    :rtype: bool
    """
    if name.strip() in reservedKeywords:
        return True
    for keywordPattern in reservedKeywordPatterns:
        if keywordPattern.match(name.strip()):
            return True
    return False


automaticSplitVarIndex = 0


def splitTypeAndIdentifier(decl):
    match = typeIdentifierSplitter.match(decl.strip())
    if match and not isReservedName(match.group(2)):
        return match.group(1).strip(), match.group(2).strip()
    else:
        global automaticSplitVarIndex
        automaticSplitVarIndex += 1
        return decl.strip(), 'arg{0:05d}'.format(automaticSplitVarIndex)


def parseApiMemberDeclarator(apiDecl):
    parenLevel = 0
    state = 0

    name = ''
    returnType = ''
    isFunction = False
    paramDecls = []   # Collection of extracted parameter declarations
    attrs = ''

    # Reversed array where tokens are collected:
    nameRArray = []         # API member name
    returnTypeRArray = []   # Return type declaration
    paramRArray = []        # Parameter declarator

    cLoc = len(apiDecl)
    cAttributeSplitLoc = cLoc
    while cLoc > 0:
        cLoc -= 1
        c = apiDecl[cLoc]

        if parenLevel == 0:
            # API member declarator context, start state
            if state == 0:
                if c == ')':
                    state = 1   # possible function declarator
                    isFunction = True
                    attrs = apiDecl[cLoc + 1:]
            # function parameter declaration
            elif state == 1:
                if c == ')':   # nesting of parentheses (stop normal parsing, only collect tokens)
                    parenLevel += 1
                    paramRArray.append(c)
                elif c == '(':
                    state = 2   # end of parameters declaration (move to function name, store parameter if needed)
                    if len(paramRArray) > 0:
                        paramDecls.append(''.join(paramRArray[::-1]).strip())
                        paramRArray = []
                elif c == ',':   # start of next parameter declaration
                    paramDecls.append(''.join(paramRArray[::-1]).strip())
                    paramRArray = []
                else:
                    paramRArray.append(c)
            # function name (optional whitespace)
            elif state == 2:
                if not c.isspace():
                    cLoc += 1
                    state = 3   # ignore whitespace until non-whitespace character is encountered (re-parse token)
            # function name
            elif state == 3:
                if c.isalnum() or c == '_':
                    nameRArray.append(c)
                else:
                    name = ''.join(nameRArray[::-1]).strip()
                    nameRArray = []

                    cLoc += 1   # re-parse unmatched token
                    if isReservedName(name):
                        cAttributeSplitLoc = cLoc

                        name = ''
                        returnType = ''
                        isFunction = False
                        paramDecls = []
                        attrs = apiDecl[cLoc:]

                        state = 0   # if parsed function declaration has reserved name, it need to be treated as attribute
                    else:
                        state = 4   # name is not reserved - treat next tokens as return type
            # return type declarator
            elif state == 4:
                returnTypeRArray.append(c)
        else:
            # Nesting of parentheses - collect tokens only.
            if c == ')':
                parenLevel += 1
            elif c == '(':
                parenLevel -= 1
            paramRArray.append(c)

    if isFunction:
        if len(nameRArray) > 0:
            name = ''.join(nameRArray[::-1]).strip()
        if len(returnTypeRArray) > 0:
            returnType = ''.join(returnTypeRArray[::-1]).strip()
        if len(paramRArray) > 0:
            paramDecls.append(''.join(paramRArray[::-1]).strip())
    else:
        returnType, name = splitTypeAndIdentifier(apiDecl[:cAttributeSplitLoc])

    paramDeclInfos = []
    for decl in reversed(paramDecls):
        paramType, paramName = splitTypeAndIdentifier(decl)
        paramDeclInfos.append({'name': paramName, 'type': paramType})

    return {
        'name': name,
        'isFunction': isFunction,
        'returnType': returnType,
        'params': paramDeclInfos,
        'attrs': attrs
    }


# print parseApiMemberDeclarator('int const   __attribute__((pure)) ')
# print parseApiMemberDeclarator('int foo1   __attribute__((pure)) ')
# print parseApiMemberDeclarator('int foo1')
# print parseApiMemberDeclarator('void a(int, const a*bb)')
# print parseApiMemberDeclarator('int foo __attribute__((static))')
# print parseApiMemberDeclarator('int foo()__attribute__((static))')
# print parseApiMemberDeclarator('int foo()__attribute__((static)) __attribute__((data(1,2,3))) do() NN')
# print parseApiMemberDeclarator('int foo (int a, int b)__attribute__((static)) __attribute__((data(1,2,3))) do() NN')
# print parseApiMemberDeclarator('DD(int,a)* foo(int a, const D(1,I())* b)__attribute__((static)) __attribute__((data(1,2,3))) do() NN')


def parseHeaderFile(headerFilePath):
    """ Opens, reads and parses header file and extracts information about API functions inside.

    :param headerFilePath: Path to header file that will be parsed.
    :return: List of API function declarations. Each declaration contains dictionary describing function name,
             parameter types and return type.
    :rtype: list
    """
    apiMembersInfo = []

    headerFile = file(headerFilePath)
    headerContent = headerFile.read()
    strippedContent = stripCommentsAndPreprocessor(headerContent)
    matchedFunctionDecls = apiMemberMatcher.findall(strippedContent)
    for decl in matchedFunctionDecls:
        apiMembersInfo.append(parseApiMemberDeclarator(decl))

    for matcher, replace in apiMacroMemberMatchers:
        matchedMacroDecls = matcher.finditer(strippedContent)
        for decl in matchedMacroDecls:
            apiMembersInfo.append(parseApiMemberDeclarator(decl.expand(replace)))

    return apiMembersInfo


def main(parsedOptions):
    """ Main script function.

    The script generates header file with wrappers for all API functions from headers contained in specific directory.

    :param parsedOptions: Arguments parsed by argparse.ArgumentParser class.
    :return: Exit code for script.
    :rtype: int
    """
    scanDirectory = parsedOptions.dir if parsedOptions.dir is not None and parsedOptions.dir != '' else os.curdir

    apiMembersInfo = []

    for scanDir, scanSubdirectories, scanFileNames in os.walk(scanDirectory):
        for scanFileName in scanFileNames:
            if headerFileNamePattern.match(scanFileName):
                apiMembersInfo.extend(parseHeaderFile(os.path.join(scanDir, scanFileName)))

    for apiMemberInfo in apiMembersInfo:
        if apiMemberInfo['isFunction']:
            print '{0} (*{1}_fptr)({2}){5} = NULL;\n{0} {1}({4}){5} {{\n    assert({1}_fptr != NULL);\n    return {1}_fptr({3});\n}}\n'.format(
                apiMemberInfo['returnType'],
                apiMemberInfo['name'],
                ', '.join([x['type'] for x in apiMemberInfo['params']]),
                ', '.join([x['name'] for x in apiMemberInfo['params']]),
                ', '.join([x['type'] + ' ' + x['name'] for x in apiMemberInfo['params']]),
                (' ' + apiMemberInfo['attrs']) if len(apiMemberInfo['attrs']) > 0 else '')

    print 'int cldnn_load_symbols(lib_handle_t handle) {'
    for apiMemberInfo in apiMembersInfo:
        if apiMemberInfo['isFunction']:
            print '    {1}_fptr = ({0} (*)({2}){5}) load_symbol(handle, "{1}");\n    if ({1}_fptr == NULL) {{\n        return -1;\n    }}\n'.format(
                apiMemberInfo['returnType'],
                apiMemberInfo['name'],
                ', '.join([x['type'] for x in apiMemberInfo['params']]),
                ', '.join([x['name'] for x in apiMemberInfo['params']]),
                ', '.join([x['type'] + ' ' + x['name'] for x in apiMemberInfo['params']]),
                (' ' + apiMemberInfo['attrs']) if len(apiMemberInfo['attrs']) > 0 else '')
    print '    return 0;\n}'


if __name__ == "__main__":
    optParser = argparse.ArgumentParser(description = 'Generates wrappers for all API functions contained in headers' +
                                                      'of specific directory.')

    optParser.add_argument('dir', metavar = '<dir>', type = str, default = None,
                           help = 'Directory to scan for header files. Default/None specified:' +
                                  ' current working directory.')
    optParser.add_argument('--version', action = 'version', version = '%(prog)s 1.0')

    options = optParser.parse_args()

    exitCode = main(options)
    optParser.exit(exitCode)
