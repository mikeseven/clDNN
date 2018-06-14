#!/usr/bin/env python2.7

import argparse
import array
import collections
import fnmatch
import functools
import glob as my_glob
import itertools
import json
import logging
import os
import re
import struct
import textwrap
#import typing


# Module logger.
_logger = logging.getLogger(__name__)  # type: logging.Logger

# ----------------------------------------------------------------------------------------------------------------------

_collect_filters_opt_name_re_template = '{name}_re'  # type: str


# noinspection PyShadowingNames
def collect_filters(parsed_args, filter_opt_name, wildcards_only=False):
    # type: (argparse.Namespace, str, bool) -> typing.List[typing.Callable[[typing.AnyStr], bool]]
    """
    Collects specified filter option values and prepares list of filter functions that can perform needed filtering.

    :param parsed_args:     Result form `argparse.ArgumentParser.parse_args()` in which filter options are located.
    :param filter_opt_name: Name of filter option in `argparse.ArgumentParser`. The name points to wildcards filter
                            option.
                            The name for regular expression filter option is created based on this name with additional
                            `_re` suffix.
    :param wildcards_only:  Indicates that only wildcards filter should be collected.
    :return:                Collection of filter functions in the form `(name: str) -> bool` that will match provided
                            file name or path and return `True` if name satisfies filter, `False` otherwise.
    """
    if not isinstance(parsed_args, argparse.Namespace):
        return []

    parsed_args_dict = vars(parsed_args)
    matchers = []

    if filter_opt_name in parsed_args_dict:
        ws_filters = set(itertools.chain.from_iterable(parsed_args_dict[filter_opt_name]))
        for ws_filter in ws_filters:
            matchers.append(lambda name:
                            fnmatch.fnmatch(name, ws_filter) or
                            fnmatch.fnmatch(name, os.path.join('*', ws_filter)))

    if not wildcards_only:
        filter_opt_name_re = _collect_filters_opt_name_re_template.format(name=filter_opt_name)
        if filter_opt_name_re in parsed_args_dict:
            re_filters = set(itertools.chain.from_iterable(parsed_args_dict[filter_opt_name_re]))
            for re_filter in re_filters:
                re_matcher = re.compile(re_filter, re.M | re.S | re.U)
                matchers.append(lambda name: re_matcher.match(name) is not None)

    return matchers

# ----------------------------------------------------------------------------------------------------------------------


# noinspection PyShadowingNames
def collect_files(parsed_args, file_opt_name, file_filter_opt_name, wildcards_only=False):
    # type: (argparse.Namespace, str, str) -> typing.List[str]
    """
    Collect files based on file search specifiers and inclusion filters.

    :param parsed_args:          Result form `argparse.ArgumentParser.parse_args()` in which file selection and filter
                                 options are located.
    :param file_opt_name:
    :param file_filter_opt_name: Name of filter option in `argparse.ArgumentParser`. The name points to wildcards filter
                                 option. This filter will be applied to list of found files (their paths).
                                 The name for regular expression filter option is created based on this name with
                                 additional `_re` suffix.
    :param wildcards_only:       Indicates that only wildcards filter should be applied.
    :return:                     List of collected files (absolute paths to them).
    """
    if not isinstance(parsed_args, argparse.Namespace):
        return []

    my_glob._ishidden = lambda x: False  # Trying to apply WA for poor check for hidden files in GLOB.
    parsed_args_dict = vars(parsed_args)
    found_files = set()

    if file_opt_name in parsed_args_dict:
        file_glob_patterns = parsed_args_dict[file_opt_name]

        for file_glob_pattern in file_glob_patterns:
            glob_found_files = my_glob.glob(file_glob_pattern)

            for glob_found_file in glob_found_files:
                if os.path.isdir(glob_found_file):
                    found_files.update({os.path.abspath(os.path.join(walk_root, found_file))
                                        for walk_root, walk_dirs, walk_files in os.walk(glob_found_file)
                                        for found_file in walk_files})
                elif os.path.isfile(glob_found_file):
                    found_files.add(os.path.abspath(glob_found_file))

    filtered_files = list(found_files)
    file_filters = collect_filters(parsed_args, file_filter_opt_name, wildcards_only)

    if len(file_filters) > 0:
        filtered_files = []

        for found_file in found_files:
            for file_filter in file_filters:
                if file_filter(found_file):
                    filtered_files.append(found_file)
                    break

    filtered_files.sort()
    return filtered_files


# ----------------------------------------------------------------------------------------------------------------------

# noinspection PyShadowingNames
class NndFile(object):
    """
    Represents .nnd file format.
    """
    # Format of NND file header (basic + extended).
    #                  MMMDVDEL
    #                  GGGTECSY
    __header_format = '<3scBBBB'
    # Size of NND File header (basic + extended).
    __header_size   = struct.calcsize(__header_format)

    # Format of NND file header dimension size specifier.
    __dim_size_format = '<Q'
    # Size of NND file header dimension size specifier.
    __dim_size_size   = struct.calcsize(__dim_size_format)

    # Check version 3 constrains.
    assert __header_size   == 8
    assert __dim_size_size == 8

    # ------------------------------------------------------------------------------------------------------------------
    # DATA TYPE enumeration.
    # ------------------------------------------------------------------------------------------------------------------
    DT_FP32   = 'F'  # type: str # Data type: Single-precision float (32-bit).
    DT_FP16   = 'H'  # type: str # Data type: Half-precision float (16-bit).
    DT_INT16  = 's'  # type: str # Data type: Integer (signed, 16-bit)
    DT_UINT16 = 'S'  # type: str # Data type: Integer (unsigned, 16-bit)
    DT_INT8   = 'b'  # type: str # Data type: Integer (signed, 8-bit)
    DT_UINT8  = 'B'  # type: str # Data type: Integer (unsigned, 8-bit)

    @staticmethod
    def get_supported_data_types():
        # type: () -> typing.List[str]
        """
        Gets enum values of supported data types for `NndFile`.

        :return: List of enum values (data types).
        """
        return list({NndFile.DT_FP32, NndFile.DT_FP16,
                     NndFile.DT_INT16, NndFile.DT_UINT16,
                     NndFile.DT_INT8, NndFile.DT_UINT8})

    # ------------------------------------------------------------------------------------------------------------------
    # LAYOUT enumeration.
    # ------------------------------------------------------------------------------------------------------------------
    LAYOUT_F    = 'O'          # type: str
    LAYOUT_O    = LAYOUT_F     # type: str
    LAYOUT_I    = LAYOUT_F     # type: str
    LAYOUT_X    = LAYOUT_F     # type: str
    LAYOUT_BF   = 'OI'         # type: str
    LAYOUT_BX   = LAYOUT_BF    # type: str
    LAYOUT_OI   = LAYOUT_BF    # type: str
    LAYOUT_FB   = 'IO'         # type: str
    LAYOUT_XB   = LAYOUT_FB    # type: str
    LAYOUT_IO   = LAYOUT_FB    # type: str
    LAYOUT_BFYX = 'OIYX'       # type: str
    LAYOUT_OIYX = LAYOUT_BFYX  # type: str
    LAYOUT_YXFB = 'YXIO'       # type: str
    LAYOUT_YXIO = LAYOUT_YXFB  # type: str

    @staticmethod
    def get_supported_layouts():
        # type: () -> typing.List[str]
        """
        Gets enum values of supported layouts for `NndFile`.

        :return: List of enum values (layouts).
        """
        return list({NndFile.LAYOUT_F, NndFile.LAYOUT_O, NndFile.LAYOUT_I, NndFile.LAYOUT_X,
                     NndFile.LAYOUT_BFYX, NndFile.LAYOUT_OIYX, NndFile.LAYOUT_YXFB, NndFile.LAYOUT_YXIO,
                     NndFile.LAYOUT_BF, NndFile.LAYOUT_BX, NndFile.LAYOUT_OI,
                     NndFile.LAYOUT_FB, NndFile.LAYOUT_XB, NndFile.LAYOUT_IO})

    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self):
        """
        Creates empty `NndFile`.
        """
        self.__file_path = ''
        self.__data_type = NndFile.DT_FP32
        self.__version   = 3
        self.__sizes     = [0, 0, 0, 0]
        self.__layout    = NndFile.LAYOUT_BFYX
        self.__data      = array.array('f')

    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def __parse_hdr_data_type(hdr_data_type):
        # type: (typing.AnyStr) -> typing.Tuple[str, int, str]
        """
        Parses data type code from NND file header and returns information about it.

        :param hdr_data_type: Type-code of data type from NND header (value extracted from header).
        :return:              Tuple with following values:
                               * enum value identifying data type in NndFile (one of `DT_`-prefixed constants
                                 in `NndFile`).
                               * size of element of data type in bytes.
                               * type-code that should be used in `array.array` to correctly store the data.
        :raise RuntimeError: Specified type-code of data type cannot be parsed. Data type is not supported.
        """
        hdr_data_type_map = {
            'F': (NndFile.DT_FP32,   4, 'f'),
            'H': (NndFile.DT_FP16,   2, 'H'),
            's': (NndFile.DT_INT16,  2, 'h'),
            'S': (NndFile.DT_UINT16, 2, 'H'),
            'b': (NndFile.DT_INT8,   1, 'b'),
            'B': (NndFile.DT_UINT8,  1, 'B'),
        }
        if hdr_data_type in hdr_data_type_map:
            return hdr_data_type_map[hdr_data_type]
        raise RuntimeError('Unsupported data type encountered in NND file.')

    @staticmethod
    def __encode_hdr_data_type(data_type):
        # type: (str) -> typing.Tuple[str, int]
        """
        Encodes data type (one of `DT_`-prefixed constants from `NndFile`) into NND header type-code.

        :param data_type: Enum value (`DT_`-prefixed constants from `NndFile`) that specifies data type of `NndFile`.
        :return:          Tuple with following values:
                           * NND header type-code for specified data type.
                           * size of element of data type in bytes.
        """
        data_type_map = {
            NndFile.DT_FP32:   ('F', 4),
            NndFile.DT_FP16:   ('H', 2),
            NndFile.DT_INT16:  ('s', 2),
            NndFile.DT_UINT16: ('S', 2),
            NndFile.DT_INT8:   ('b', 1),
            NndFile.DT_UINT8:  ('B', 1),
        }
        return data_type_map[data_type]

    # ------------------------------------------------------------------------------------------------------------------

    # Old format of layout type-code is shifted by this value if data type is FP16 instead of FP32.
    __hdr_layout_fp16_base = 19

    @staticmethod
    def __parse_hdr_layout(hdr_layout):
        # type: (int) -> typing.Tuple[str, int]
        """
        Parses layout format from NND file header and returns information about it.

        :param hdr_layout: Type-code of layout format from NND header (value extracted from header).
        :return:           Tuple with following values:
                            * enum value identifying layout in NndFile (one of `LAYOUT_`-prefixed constants
                              in `NndFile`).
                            * maximum number of dimensions supported by selected layout.
        :raise RuntimeError: Specified type-code of layout cannot be parsed. Layout is not supported.
        """
        # Support for old layout format with encoded data type (not needed since data type is passed as well).
        hdr_layout_rem = hdr_layout % NndFile.__hdr_layout_fp16_base

        hdr_layout_map = {
            0:  (NndFile.LAYOUT_F,    1),
            1:  (NndFile.LAYOUT_FB,   2),
            2:  (NndFile.LAYOUT_BF,   2),
            4:  (NndFile.LAYOUT_YXFB, 4),
            8:  (NndFile.LAYOUT_BFYX, 4),
            11: (NndFile.LAYOUT_YXFB, 4),
            19: (NndFile.LAYOUT_F,    1),
            20: (NndFile.LAYOUT_FB,   2),
            21: (NndFile.LAYOUT_BF,   2),
            23: (NndFile.LAYOUT_YXFB, 4),
            27: (NndFile.LAYOUT_BFYX, 4),
            30: (NndFile.LAYOUT_YXFB, 4),
        }
        if hdr_layout_rem in hdr_layout_map:
            return hdr_layout_map[hdr_layout_rem]
        raise RuntimeError('Unsupported layout type encountered in NND file.')

    @staticmethod
    def __encode_hdr_layout(layout, data_type=None, emit_old_format=False):
        # type: (str, str, bool) -> int
        """
        Encodes layout (one of `LAYOUT_`-prefixed constants from `NndFile`) into NND header type-code.

        :param layout:          Enum value (`LAYOUT_`-prefixed constants from `NndFile`) that specifies
                                layout of `NndFile`.
        :param data_type:       Enum value (`DT_`-prefixed constants from `NndFile`) that specifies
                                data type of `NndFile`. If not specified, the layout is emitted in new format and
                                `emit_old_format` flag is ignored.
        :param emit_old_format: Indicates that old format of layout type-code should be emitted if possible.
                                The old format incorporates data type information inside. It is defined only for
                                `NndFile.DT_FP32` and `NndFile.DT_FP16`. For rest, the flag will be ignored.
        :return:                Type-code with selected layout for NND file header.
        """
        layout_map = {
            NndFile.LAYOUT_F:     0,
            NndFile.LAYOUT_FB:    1,
            NndFile.LAYOUT_BF:    2,
            NndFile.LAYOUT_BFYX:  8,
            NndFile.LAYOUT_YXFB: 11,
        }
        return layout_map[layout] + \
            (NndFile.__hdr_layout_fp16_base if emit_old_format and data_type == NndFile.DT_FP16 else 0)

    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def __needs_quant_factors(data_type):
        # type: (str) -> bool
        """
        Indicates that specified data type needs additional file with quantization factors to be correctly converted
        into different data type in NND file.

        :param data_type: Enum value (`DT_`-prefixed constants from `NndFile`) that specifies data type of `NndFile`.
        :return: `True` if data type of NND requires additional factors; otherwise, `False`.
        """
        calib_map = {
            NndFile.DT_FP32:   False,
            NndFile.DT_FP16:   False,
            NndFile.DT_INT16:  True,
            NndFile.DT_UINT16: True,
            NndFile.DT_INT8:   True,
            NndFile.DT_UINT8:  True,
        }
        return calib_map[data_type]

    def needs_quant_factors(self):
        # type: () -> bool
        """
        Indicates that current NND file needs additional file with quantization factors to be correctly converted
        to NND file which uses different type of data element.

        :return: `True` if NND requires additional factors file; otherwise, `False`.
        """
        return NndFile.__needs_quant_factors(self.__data_type)

    # ------------------------------------------------------------------------------------------------------------------

    def __get_output_features_count(self):
        # type: () -> int
        """
        Gets information about number of output features available in current NND file.

        :return: Number of output features, or `0` if format does not support output features.
        """
        if self.__layout == NndFile.LAYOUT_O:
            return self.__sizes[-1]
        elif self.__layout == NndFile.LAYOUT_OI:
            return self.__sizes[-2]
        elif self.__layout == NndFile.LAYOUT_IO:
            return self.__sizes[-1]
        elif self.__layout == NndFile.LAYOUT_OIYX:
            return self.__sizes[-4]
        elif self.__layout == NndFile.LAYOUT_YXIO:
            return self.__sizes[-1]
        else:
            return 0

    def __set_output_features_count(self, val):
        # type: (int) -> typing.NoReturn
        """
        Sets number of output features available in current NND file.

        :param val: Number of output features. Discarded if format does not support output features.
        """
        if self.__layout == NndFile.LAYOUT_O:
            self.__sizes[-1] = int(val)
        elif self.__layout == NndFile.LAYOUT_OI:
            self.__sizes[-2] = int(val)
        elif self.__layout == NndFile.LAYOUT_IO:
            self.__sizes[-1] = int(val)
        elif self.__layout == NndFile.LAYOUT_OIYX:
            self.__sizes[-4] = int(val)
        elif self.__layout == NndFile.LAYOUT_YXIO:
            self.__sizes[-1] = int(val)

    __output_features_count = property(__get_output_features_count, __set_output_features_count)  # type: int

    def __get_output_feature_iterator(self):
        # type: () -> typing.Generator[typing.Tuple[int, int, int]]
        """
        Creates iterator / enumerator that enumerates over all output feature offsets in `NndFile` data.

        :return: Generator function that generates tuples with following values:
                  * output feature offset (in elements) in `NndFile` data.
                  * size / step of output feature range starting from returned offset (all elements in that range
                    are from the same output feature).
                  * index (`0`-based) of output feature. For some layouts, single output feature set might not be
                    contiguous.
        :raise ValueError: Cannot iterate over feature offsets. The NND file layout does not contain output features.
        """
        output_feature_step = 0
        iterations_count    = 0
        features_count      = self.__output_features_count
        if self.__layout == NndFile.LAYOUT_O:
            output_feature_step = 1
            iterations_count    = functools.reduce(lambda x, y: x * y, self.__sizes, 1)
        elif self.__layout == NndFile.LAYOUT_OI:
            output_feature_step = functools.reduce(lambda x, y: x * y, self.__sizes[-1:], 1)
            iterations_count    = functools.reduce(lambda x, y: x * y, self.__sizes[:-1], 1)
        if self.__layout == NndFile.LAYOUT_IO:
            output_feature_step = 1
            iterations_count    = functools.reduce(lambda x, y: x * y, self.__sizes, 1)
        elif self.__layout == NndFile.LAYOUT_OIYX:
            output_feature_step = functools.reduce(lambda x, y: x * y, self.__sizes[-3:], 1)
            iterations_count    = functools.reduce(lambda x, y: x * y, self.__sizes[:-3], 1)
        elif self.__layout == NndFile.LAYOUT_YXIO:
            output_feature_step = 1
            iterations_count    = functools.reduce(lambda x, y: x * y, self.__sizes, 1)

        if iterations_count <= 0:
            return

        if output_feature_step <= 0 or features_count <= 0:
            raise ValueError('Layout type is not supported for this function (no output features in layout {0}): "{1}".'
                             .format(self.__layout, self.__file_path))

        output_feature_offset = 0
        for output_feature_idx in xrange(iterations_count):
            yield output_feature_offset, output_feature_step, output_feature_idx % features_count
            output_feature_offset += output_feature_step

    def __get_input_features_count(self):
        # type: () -> int
        """
        Gets information about number of input features available in current NND file.

        :return: Number of input features, or `0` if format does not support input features.
        """
        if self.__layout == NndFile.LAYOUT_I:
            return self.__sizes[-1]
        elif self.__layout == NndFile.LAYOUT_OI:
            return self.__sizes[-1]
        elif self.__layout == NndFile.LAYOUT_IO:
            return self.__sizes[-2]
        elif self.__layout == NndFile.LAYOUT_OIYX:
            return self.__sizes[-3]
        elif self.__layout == NndFile.LAYOUT_YXIO:
            return self.__sizes[-2]
        else:
            return 0

    def __set_input_features_count(self, val):
        # type: (int) -> typing.NoReturn
        """
        Sets number of input features available in current NND file.

        :param val: New number of input features. Discarded if format does not support input features.
        """
        if self.__layout == NndFile.LAYOUT_I:
            self.__sizes[-1] = int(val)
        elif self.__layout == NndFile.LAYOUT_OI:
            self.__sizes[-1] = int(val)
        elif self.__layout == NndFile.LAYOUT_IO:
            self.__sizes[-2] = int(val)
        elif self.__layout == NndFile.LAYOUT_OIYX:
            self.__sizes[-3] = int(val)
        elif self.__layout == NndFile.LAYOUT_YXIO:
            self.__sizes[-2] = int(val)

    __input_features_count = property(__get_input_features_count, __set_input_features_count)  # type: int

    def __get_input_feature_iterator(self):
        # type: () -> typing.Generator[typing.Tuple[int, int, int]]
        """
        Creates iterator / enumerator that enumerates over all input feature offsets in `NndFile` data.

        :return: Generator function that generates tuples with following values:
                  * input feature offset (in elements) in `NndFile` data.
                  * size / step of input feature range starting from returned offset (all elements in that range
                    are from the same input feature).
                  * index (`0`-based) of input feature. For some layouts, single input feature set might not be
                    contiguous.
        :raise ValueError: Cannot iterate over feature offsets. The NND file layout does not contain input features.
        """
        input_feature_step = 0
        iterations_count   = 0
        features_count     = self.__input_features_count
        if self.__layout == NndFile.LAYOUT_I:
            input_feature_step = 1
            iterations_count   = functools.reduce(lambda x, y: x * y, self.__sizes, 1)
        elif self.__layout == NndFile.LAYOUT_OI:
            input_feature_step = 1
            iterations_count   = functools.reduce(lambda x, y: x * y, self.__sizes, 1)
        elif self.__layout == NndFile.LAYOUT_IO:
            input_feature_step = functools.reduce(lambda x, y: x * y, self.__sizes[-1:], 1)
            iterations_count   = functools.reduce(lambda x, y: x * y, self.__sizes[:-1], 1)
        elif self.__layout == NndFile.LAYOUT_OIYX:
            input_feature_step = functools.reduce(lambda x, y: x * y, self.__sizes[-2:], 1)
            iterations_count   = functools.reduce(lambda x, y: x * y, self.__sizes[:-2], 1)
        elif self.__layout == NndFile.LAYOUT_YXIO:
            input_feature_step = functools.reduce(lambda x, y: x * y, self.__sizes[-1:], 1)
            iterations_count   = functools.reduce(lambda x, y: x * y, self.__sizes[:-1], 1)

        if iterations_count <= 0:
            return

        if input_feature_step <= 0 or features_count <= 0:
            raise ValueError('Layout type is not supported for this function (no input features in layout {0}): "{1}".'
                             .format(self.__layout, self.__file_path))

        input_feature_offset = 0
        for input_feature_idx in xrange(iterations_count):
            yield input_feature_offset, input_feature_step, input_feature_idx % features_count
            input_feature_offset += input_feature_step

    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def from_file(file_path):
        # type: (typing.AnyStr) -> NndFile
        """
        Creates `NndFile` instance form .nnd file and return newly created object.

        :param file_path: Path (relative or absolute) to .nnd file.
        :return:          Instance of `NndFile` with content of specified file.
        :raise RuntimeError: File cannot be accessed.
        :raise RuntimeError: File contains malformed or invalid header.
        :raise RuntimeError: File version is not supported by script.
        :raise RuntimeError: File uses unsupported data type.
        :raise RuntimeError: File uses unsupported layout.
        :raise RuntimeError: File is too short to contain required data.
        """
        with open(file_path, mode=u'rb') as nnd_file:
            nnd_content = nnd_file.read()

            if len(nnd_content) < NndFile.__header_size:
                raise RuntimeError('NND file is too short ({0} B) to contain valid header: "{1}".'
                                   .format(len(nnd_content), file_path))

            nnd_hdr_magic, nnd_hdr_data_type, nnd_hdr_version, nnd_hdr_dims_count, nnd_hdr_elem_size, nnd_hdr_layout = \
                struct.unpack(NndFile.__header_format, nnd_content[:NndFile.__header_size])

            if nnd_hdr_magic != 'nnd':
                raise RuntimeError('NND file contains invalid header (magic: "{0}", expected: "nnd"): "{1}".'
                                   .format(nnd_hdr_magic, file_path))
            if nnd_hdr_version != 3:
                raise RuntimeError('The script does not support encountered version of NND file'
                                   '(version: {0}, expected: 3). Only version 3 is currently supported: "{1}".'
                                   .format(nnd_hdr_version, file_path))
            parsed_elem_type, parsed_elem_size, array_storage_type = NndFile.__parse_hdr_data_type(nnd_hdr_data_type)
            if parsed_elem_size != nnd_hdr_elem_size:
                raise RuntimeError('The encountered size of data element ({0}B) does not match calculated size '
                                   'of data type (elem size: {0}B, expected: {1}B as for "{2}" data type): "{3}".'
                                   .format(nnd_hdr_elem_size, parsed_elem_size, parsed_elem_type, file_path))

            nnd_hdr_dims_count = max(nnd_hdr_dims_count, 0)
            if len(nnd_content) < NndFile.__header_size + NndFile.__dim_size_size * nnd_hdr_dims_count:
                raise RuntimeError('NND file is too short ({0}B) to contain valid header: "{1}".'
                                   .format(len(nnd_content), file_path))
            parsed_layout, parsed_max_dims_count = NndFile.__parse_hdr_layout(nnd_hdr_layout)
            if nnd_hdr_dims_count > parsed_max_dims_count:
                raise RuntimeError('The encountered number of dimensions ({0}) is greater than supported by '
                                   'specified layout (dim count: {0}, max. supported: {1} for layout format {2}): '
                                   '"{3}".'
                                   .format(nnd_hdr_dims_count, parsed_max_dims_count, nnd_hdr_layout, file_path))

            parsed_sizes = [1] * parsed_max_dims_count
            nnd_hdr_offset = NndFile.__header_size
            for dim_idx in xrange(nnd_hdr_dims_count):
                parsed_size, = struct.unpack(NndFile.__dim_size_format,
                                             nnd_content[nnd_hdr_offset:nnd_hdr_offset + NndFile.__dim_size_size])
                parsed_sizes[dim_idx] = parsed_size
                nnd_hdr_offset += NndFile.__dim_size_size

            parsed_sizes.reverse()
            parsed_data_size = functools.reduce(lambda x, y: x * y, parsed_sizes) * parsed_elem_size
            if len(nnd_content) < nnd_hdr_offset + parsed_data_size:
                raise RuntimeError('NND file is too short ({0}B) to contain required data (expected header size: {1}B, '
                                   'expected data size: {2}B): "{3}".'
                                   .format(len(nnd_content), nnd_hdr_offset, parsed_data_size, file_path))

            parsed_file = NndFile()
            parsed_file.__file_path = os.path.abspath(file_path)
            parsed_file.__data_type = parsed_elem_type
            parsed_file.__version   = nnd_hdr_version
            parsed_file.__sizes     = parsed_sizes
            parsed_file.__layout    = parsed_layout
            parsed_file.__data      = array.array(array_storage_type)
            parsed_file.__data.fromstring(nnd_content[nnd_hdr_offset:nnd_hdr_offset + parsed_data_size])
            return parsed_file

    def to_file(self, file_path=None, use_old_layout_format=False):
        # type: (typing.AnyStr, bool) -> typing.NoReturn
        """
        Writes content of current instance of `NndFile` to .nnd file.

        :param file_path:             Path (relative or absolute) to .nnd file. If not specified, the write is done to
                                      the file which was used to load data.
        :param use_old_layout_format: Indicates that old format of layout should be written into .nnd file.
        """
        used_file_path = file_path if file_path is not None else self.__file_path

        nnd_hdr_data_type, nnd_hdr_elem_size = NndFile.__encode_hdr_data_type(self.__data_type)
        nnd_hdr_layout = NndFile.__encode_hdr_layout(self.__layout, self.__data_type, use_old_layout_format)
        nnd_hdr_data = ('nnd', nnd_hdr_data_type, self.__version, len(self.__sizes), nnd_hdr_elem_size, nnd_hdr_layout)
        nnd_hdr_bytes = struct.pack(NndFile.__header_format, *nnd_hdr_data)

        nnd_hdr_sizes = list(self.__sizes)
        nnd_hdr_sizes.reverse()
        for nnd_hdr_size in nnd_hdr_sizes:
            nnd_hdr_bytes += struct.pack(NndFile.__dim_size_format, nnd_hdr_size)

        with open(used_file_path, 'wb') as nnd_file:
            nnd_file.write(nnd_hdr_bytes)
            nnd_file.write(self.__data.tostring())

    @staticmethod
    def from_iterable(proposed_file_path, data_type, data):
        # type: (typing.AnyStr, str, typing.Iterable[typing.Union[int, float, bool]]) -> NndFile
        """
        Creates NND file from iterable.

        :param proposed_file_path: Suggested file path.
        :param data_type:          Data type of new NND file.
        :param data:               Data representation (for FP16 values should be represented as 16-bit integers with
                                   the same binary representation as FP16).
        :return:                   One-dimensional NND file with (converted) content of iterable.
        :raise ValueError          Proposed file path is invalid.
        :raise NotImplementedError Specified data type is not currently supported in NND.
        """
        if proposed_file_path is None or proposed_file_path.strip() == '':
            raise ValueError('Suggested file path for NND file may not be empty.')
        if data_type not in NndFile.get_supported_data_types():
            raise NotImplementedError('NND file does not support specified data type: "{0}".'
                                      .format(data_type))

        nnd_storage_type = NndFile.__parse_hdr_data_type(NndFile.__encode_hdr_data_type(data_type)[0])[2]
        nnd_data = array.array(nnd_storage_type, data)

        nnd_file = NndFile()
        nnd_file.__file_path = proposed_file_path
        nnd_file.__data_type = data_type
        nnd_file.__sizes     = [len(nnd_data)]
        nnd_file.__layout    = NndFile.LAYOUT_F
        nnd_file.__data      = nnd_data
        return nnd_file

    def to_nnd(self, proposed_file_path, copy=True):
        # type: (typing.AnyStr, bool) -> NndFile
        """
        Creates NND file from current file.

        :param proposed_file_path: Suggested file path.
        :param copy:               Indicates whether object should be copied. If `False`, the current object is
                                   modified and returned.
        :return:                   New NND file with the same content (different file name) or current file (depends
                                   on `copy` argument).
        :raise ValueError Proposed file path is invalid.
        """
        if proposed_file_path is None or proposed_file_path.strip() == '':
            raise ValueError('Suggested file path for NND file may not be empty.')

        if not copy:
            self.__file_path = proposed_file_path
            return self

        nnd_file = NndFile()
        nnd_file.__file_path = proposed_file_path
        nnd_file.__data_type = self.__data_type
        nnd_file.__version   = self.__version
        nnd_file.__sizes     = list(self.__sizes)
        nnd_file.__layout    = self.__layout
        nnd_file.__data      = array.array(self.__data.typecode, self.__data)
        return nnd_file

    # ------------------------------------------------------------------------------------------------------------------

    def decalibrate(self, calib_nnd_file):
        # type: (NndFile) -> typing.NoReturn
        """
        Decalibrates current NND file with calibration factors from other `NndFile`.

        :param calib_nnd_file: Instance of `NndFile` with calibration factors used for decalibration.
        :raise RuntimeError        No input features in current NND file.
        :raise ValueError          Invalid calibration factors file (no file, no output features in file).
        :raise ValueError          Mismatched number of features between current file and calibration factors.
        :raise NotImplementedError Layout of data type not supported in current NND file or calibration factors file.
        """
        if self.__input_features_count <= 0:
            raise RuntimeError('Current NND file does not have input features to decalibrate: "{0}".'
                               .format(self.__file_path))
        if calib_nnd_file is None:
            raise ValueError('NND file with calibration factors is invalid (no file specified).')
        if calib_nnd_file.__output_features_count <= 0:
            raise ValueError('NND file with calibration factors is invalid (has no output features): "{0}.'
                             .format(calib_nnd_file.__file_path))
        if self.__input_features_count != calib_nnd_file.__output_features_count:
            raise ValueError('NND file cannot be decalibrated using specified calibration factors. Mismatch '
                             'in features counts ({0} vs. {1}): "{2}", "{3}"'
                             .format(self.__input_features_count, calib_nnd_file.__output_features_count,
                                     self.__file_path, calib_nnd_file.__file_path))
        if calib_nnd_file.__layout != NndFile.LAYOUT_O:
            raise NotImplementedError('Layout of calibration factors for decalibration is not supported '
                                      'yet ("{0}"; only supports "{1}"): "{2}".'
                                      .format(calib_nnd_file.__layout, NndFile.LAYOUT_O, calib_nnd_file.__file_path))
        if calib_nnd_file.__data_type != NndFile.DT_FP32:
            raise NotImplementedError('Data type of calibration factors for decalibration is not supported '
                                      'yet ("{0}"; only supports "{1}"): "{2}".'
                                      .format(calib_nnd_file.__data_type, NndFile.DT_FP32, calib_nnd_file.__file_path))

        calib_factors = calib_nnd_file.__data

        if self.__data_type != NndFile.DT_FP32:
            # TODO: Convert to FP32 and next convert back.
            # TODO: Implement calibrate() when this will work fully.
            raise NotImplementedError('Data type of current NND file for decalibration is not supported '
                                      'yet ("{0}"; only supports "{1}"): "{2}".'
                                      .format(self.__data_type, NndFile.DT_FP32, self.__file_path))
        else:
            norm_calib_data = self.__data

        for if_off, if_step, if_idx in self.__get_input_feature_iterator():
            calib_factor = calib_factors[if_idx]
            if calib_factor > 0:
                norm_calib_data[if_off:if_off + if_step] = array.array(
                    norm_calib_data.typecode,
                    [val / calib_factor for val in norm_calib_data[if_off:if_off + if_step]])
            else:
                norm_calib_data[if_off:if_off + if_step] = array.array(norm_calib_data.typecode, [0] * if_step)

        if self.__data_type != NndFile.DT_FP32:
            # TODO: Convert back.
            pass

    def split_on_output_features(self, *split_nnd_file_paths):
        # type: (*typing.AnyStr) -> typing.List[NndFile]
        """
        Splits current NND file on output features. Split size is equal to number of file paths specified.

        :param split_nnd_file_paths: Names of new .nnd files to write split data to.
        """
        if self.__output_features_count <= 0:
            raise RuntimeError('Current NND file does not have output features to split: "{0}".'
                               .format(self.__file_path))

        split_size = len(split_nnd_file_paths)
        if split_size <= 0:
            return []
        if split_size == 1:
            return [self.to_nnd(split_nnd_file_paths[0])]

        if self.__output_features_count % split_size != 0:
            raise ValueError('Selected split size ({0}) does not divide evenly number of output features ({1}) '
                             'in current NND file: "{2}".'
                             .format(split_size, self.__output_features_count, self.__file_path))

        new_features_count = self.__output_features_count / split_size

        nnd_files = []
        for split_file_path in split_nnd_file_paths:
            nnd_file = NndFile()
            nnd_file.__file_path = split_file_path
            nnd_file.__data_type = self.__data_type
            nnd_file.__version = self.__version
            nnd_file.__sizes = list(self.__sizes)
            nnd_file.__layout = self.__layout
            nnd_file.__data = array.array(self.__data.typecode)

            nnd_file.__output_features_count = new_features_count
            nnd_files.append(nnd_file)

        for of_off, of_step, of_idx in self.__get_output_feature_iterator():
            nnd_files[of_idx / new_features_count].__data.extend(self.__data[of_off:of_off + of_step])

        return nnd_files

    @staticmethod
    def __abs_fp16(x):
        # type: (int) -> int
        """
        Calculates absolute value for FP16 representation (in `short`s).

        :param x: Input FP16 `short` representation.
        :return:  Output FP16 `short` representation which is `abs(x)`.
        """
        return x & 0x7FFF

    def __max_abs_per_output_feature(self):
        # type: () -> array.array
        """
        For each output feature, it calculates `max({abs(x0), abs(x1), ...})` from all elements
        belonging to this output feature. After calculation results are returned in `array.array` of the same storage
        type as `NndFile` data is stored.

        :return: Array of maximum of absolute values, calculated from all elements in output feature, for each output
                 feature. The array contains exactly `__output_features_count` elements.
        """
        if self.__output_features_count <= 0:
            return array.array(self.__data.typecode)

        abs_func = NndFile.__abs_fp16 if self.__data_type == NndFile.DT_FP16 else abs
        of_max_abs = array.array(self.__data.typecode, [0] * self.__output_features_count)

        for of_off, of_step, of_idx in self.__get_output_feature_iterator():
            of_max_abs[of_idx] = max(of_max_abs[of_idx], max(map(abs_func, self.__data[of_off:of_off + of_step])))
        return of_max_abs

    def __max_abs_per_input_feature(self):
        # type: () -> array.array
        """
        For each input feature, it calculates `max({abs(x0), abs(x1), ...})` from all elements
        belonging to this input feature. After calculation results are returned in `array.array` of the same storage
        type as `NndFile` data is stored.

        :return: Array of maximum of absolute values, calculated from all elements in input feature, for each input
                 feature. The array contains exactly `__input_features_count` elements.
        """
        if self.__input_features_count <= 0:
            return array.array(self.__data.typecode)

        abs_func = NndFile.__abs_fp16 if self.__data_type == NndFile.DT_FP16 else abs
        if_max_abs = array.array(self.__data.typecode, [0] * self.__input_features_count)

        for if_off, if_step, if_idx in self.__get_input_feature_iterator():
            if_max_abs[if_idx] = max(if_max_abs[if_idx], max(map(abs_func, self.__data[if_off:if_off + if_step])))
        return if_max_abs

    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def __gen_quant_factors(max_abs_vals, qunat_data_type, inverse=False):
        # type: (array.array, str) -> array.array
        """
        Calculates quantization factors based on `max(abs())` values and data type that will be calibrated.

        :param max_abs_vals:    Array of `max(abs())` returned (one per output feature).
                                For details please look into e.g. `NndFile.__max_abs_per_output_feature()`.
        :param qunat_data_type: One of `NndFile.DT_`-prefixed enum values. It allows to estimate needed output range.
        :param inverse:         Indicates that inverse of quantization factors should be returned.
        :return:                Array of corresponding quantization factors (per output feature).
        """
        range_factor = CalibCalculator.calib_range_factors[qunat_data_type]

        if inverse:
            return array.array('f', [range_factor / (max_abs_val if max_abs_val > 0.0 else 1.0)
                                     for max_abs_val in max_abs_vals])
        else:
            return array.array('f', [(max_abs_val if max_abs_val > 0.0 else 1.0) / range_factor
                                     for max_abs_val in max_abs_vals])

    @staticmethod
    def __gen_convert_sat(data_type):
        # type: (str) ->  typing.Callable[[typing.Union[int, float]], typing.Union[int, float]]
        """
        Function generator that returns saturated conversion functions for numbers.
        :param data_type: Data type to which values will be converted. One of `NndFile.DT_`-prefixed enum values.
        :return:          Function `(Any
        :raise NotImplementedError FP16 conversion is not yet supported.
        """
        if data_type == NndFile.DT_FP32:
            return lambda x: float(x)
        elif data_type == NndFile.DT_FP16:
            raise NotImplementedError('Saturated conversion to FP16 ({0}) is not supported yet.'
                                      .format(data_type))
        elif data_type == NndFile.DT_INT16:
            return lambda x: int(round(min(max(-32768, x), 32767)))
        elif data_type == NndFile.DT_UINT16:
            return lambda x: int(round(min(max(0, x), 65536)))
        elif data_type == NndFile.DT_INT8:
            return lambda x: int(round(min(max(-128, x), 127)))
        elif data_type == NndFile.DT_UINT8:
            return lambda x: int(round(min(max(0, x), 255)))

    # TODO: Comments.
    def quantize(self, new_data_type=None, nnd_qf_file=None, out_nnd_file_path=None, out_nnd_qf_file_path=None):
        # type: (str, typing.Optional[NndFile], typing.Optional[typing.AnyStr], typing.Optional[typing.AnyStr]) -> typing.Tuple[NndFile, typing.Optional[NndFile]]
        """

        :param new_data_type:
        :param nnd_qf_file:
        :param out_nnd_file_path:
        :param out_nnd_qf_file_path:
        :return:
        """
        ret_nnd_qf_file = out_nnd_qf_file_path is not None

        new_data_type        = new_data_type if new_data_type is not None else self.__data_type
        out_nnd_file_path    = out_nnd_file_path if out_nnd_file_path is not None else self.__file_path
        out_nnd_qf_file_path = out_nnd_qf_file_path if out_nnd_qf_file_path is not None else '__temp__'

        if self.needs_quant_factors() and nnd_qf_file is None:
            raise RuntimeError('Cannot convert NND values to new data type. The NND requires additional calibration '
                               'file to perform correct conversion: "{0}".'
                               .format(self.__file_path))
        if self.__output_features_count <= 0:
            raise RuntimeError('Current NND file does not have output features to quantize: "{0}".'
                               .format(self.__file_path))

        if new_data_type not in self.get_supported_data_types():
            raise ValueError('Target data type for quantization is not supported ("{0}"): "{1}".'
                             .format(new_data_type, self.__file_path))

        if self.__data_type != NndFile.DT_FP32:
            # TODO: Convert to FP32 and quantize.
            raise NotImplementedError('Data type of current NND file for quantization is not supported '
                                      'yet ("{0}"; only supports "{1}"): "{2}".'
                                      .format(self.__data_type, NndFile.DT_FP32, self.__file_path))

        new_nnd_storage_type = NndFile.__parse_hdr_data_type(NndFile.__encode_hdr_data_type(new_data_type)[0])[2]

        convert_sat = NndFile.__gen_convert_sat(new_data_type)

        max_abs = self.__max_abs_per_output_feature()
        qf_data = NndFile.__gen_quant_factors(max_abs, new_data_type)
        iqf_data = NndFile.__gen_quant_factors(max_abs, new_data_type, True)

        out_nnd_file = self.to_nnd(out_nnd_file_path)
        new_data = array.array(new_nnd_storage_type, [0] * len(self.__data))

        for of_off, of_step, of_idx in self.__get_output_feature_iterator():
            quant_factor = iqf_data[of_idx]
            new_data[of_off:of_off + of_step] = array.array(
                new_nnd_storage_type,
                [convert_sat(val * quant_factor) for val in self.__data[of_off:of_off + of_step]])

        out_nnd_file.__data_type = new_data_type
        out_nnd_file.__data      = new_data

        out_nnd_qf_file = None
        if ret_nnd_qf_file:
            out_nnd_qf_file = NndFile.from_iterable(out_nnd_qf_file_path, self.__data_type, qf_data)

        return out_nnd_file, out_nnd_qf_file


# ----------------------------------------------------------------------------------------------------------------------


# noinspection PyShadowingNames
class CalibCalculator(object):
    """
    Calibration calculator class.
    """
    __dump_file_value_matcher = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?',
                                           re.M | re.S)  # type: typing.Pattern[str]
    # Pattern groups: 1 - primitive name, 2 - batch index, 3 - feature index
    __dump_file_name_matcher = re.compile(r'(.*)_gpu_b([0-9]+)_f([0-9]+)\.txt$',
                                          re.M | re.S)  # type: typing.Pattern[str]

    # Pattern groups: 1 - primitive or (primitive weight) name, 2 - opt. group index,
    #                 3,4,5,6 - type indicator: weights (3), biases (4), quantization factors (5) or means (6).
    __weights_file_name_matcher = re.compile(
        r'(.*?)(?:_g([0-9]+))?_(?:(w)(?:eights?)?|(b)(?:ias(?:es)?)?|w?(q)f|(m)eans?)\.nnd$', re.M | re.S)
    __weights_to_qf_renamer      = re.compile('_w(?:eights?)?\.nnd$')
    __weights_to_qf_renamer_repl = '_qf.nnd'

    __def_file_name_template       = '{prim_name}_{attrib}{type}.nnd'               # type: str
    __def_group_file_name_template = '{prim_name}_g{group_idx}_{attrib}{type}.nnd'  # type: str
    __def_frontier_name_template   = 'frontier{frontier_idx}'                       # type: str

    __allowed_decalib_modes = ['-', '+', '*']  # type: typing.List[str]

    calib_range_factors = {
        NndFile.DT_FP32:   1.0,      # Floating point do not need to rescale to get good granularity.
        NndFile.DT_FP16:   1.0,
        NndFile.DT_INT16:  65535.4,  # Assuming we have rounding we should be in range.
        NndFile.DT_UINT16: 32767.4,
        NndFile.DT_INT8:   127.49,
        NndFile.DT_UINT8:  255.49,
    }  # type: typing.Dict[str, float]

    __WT_W  = 0  # type: int # File is weights file.
    __WT_B  = 1  # type: int # File is biases file.
    __WT_QF = 2  # type: int # File is (weights) quantization factors file.
    __WT_M  = 3  # type: int # File is (weights) mean values file.
    __WT_CF = 4  # type: int # File is calibration factors file.

    __file_type_quals = ['weights', 'bias', 'qf', 'mean', 'cf']  # type: typing.List[typing.AnyStr]

    CCO_RET_CALIB_FILES      = 0x00000001  # type: int # Return information about calibration factors files.
    CCO_RET_FRONTIER_FILES   = 0x00000002  # type: int # Return information about frontier calibration factors files.
    CCO_RET_WEIGHT_FILES     = 0x00000004  # type: int # Return information about calibrated/quantized weight files.
    CCO_SAVE_CALIB_FILES     = 0x00001000  # type: int # Save calibration factors files to output directory.
    CCO_SAVE_FRONTIER_FILES  = 0x00002000  # type: int # Save frontier calibration factors files to output directory.
    CCO_SAVE_WEIGHT_FILES    = 0x00004000  # type: int # Save calibrated/quantized weight files to output directory.
    CCO_USE_OLD_NND_LAYOUT   = 0x01000000  # type: int # Use old layout format when saving .nnd files.
    CCO_OMIT_WEIGHTS_DECALIB = 0x02000000  # type: int # Suppress weights decalibration.
    CCO_OMIT_WEIGHTS_QUANT   = 0x04000000  # type: int # Suppress weights quantization.
    CCO_ADD_RAW_CALIIB_FILES = 0x08000000  # type: int # Adds raw (not decalibrated) calibration files to calib files.
    CCO_UNIFY_WG_NAMES       = 0x10000000  # type: int # Indicates that weights should be renamed in output to common format.

    def __init__(self, dump_dir_path, output_dir_path=None, dump_dir_incl_subdirs=False,
                 calib_data_type=NndFile.DT_INT8, weights_dir_path=None, weights_dir_incl_subdirs=False,
                 file_name_template=None, group_file_name_template=None, frontier_name_template=None):
        # type: (typing.AnyStr, typing.AnyStr, bool, str, typing.AnyStr, bool, typing.AnyStr, typing.AnyStr, typing.AnyStr) -> None
        """
        Creates new instance of calibration calculator.

        :param dump_dir_path:            Path (relative or absolute) to directory where dump files from `examples`
                                         application reside.
                                         The path must exist and point to directory.
        :param output_dir_path:          Path (relative or absolute) to directory where output .nnd calibration files
                                         should be written.
                                         If not specified, current working directory will be used.
        :param dump_dir_incl_subdirs     Indicates that subdirectories of dump directory should be also taken into
                                         consideration when searching for dump files (recursive mode).
        :param calib_data_type           Data type to which range calibration factors will calibrate outputs.
                                         Calibration factors are always FP32, but they can calibrate to different ranges
                                         depending on this parameter.
                                         Data type must be one of `NndFile.DT_`-prefixed enum values.
        :param weights_dir_path:         Optional path (relative or absolute) to directory where weights/biases/QF
                                         files from `examples` application reside.
                                         If specified (not `None`), the path must exist and point to directory.
        :param weights_dir_incl_subdirs  Indicates that subdirectories of weights directory should be also taken into
                                         consideration when searching for weights/biases/QF files (recursive mode).
        :param file_name_template:       Template for .nnd file name (single feature group). It support placeholders:
                                          * {prim_name} - name of primitive.
                                          * {attrib}    - additional attributes / file types (ending with underscore
                                                           if specified).
                                          * {type}      - main type of file. It is replaced by:
                                                           ** 'cf'      (calibration factors)
                                                           ** 'qf'      (quantization factors)
                                                           ** 'weights' (weights)
                                                           ** 'bias'    (biases)
                                                           ** 'mean'    (means)
        :param group_file_name_template: Template for .nnd file name (multiple feature groups). It support placeholders:
                                          * {prim_name} - name of primitive.
                                          * {group_idx} - index (`0`-based) of feature group.
                                          * {attrib}    - additional attributes / file types (ending with underscore
                                                          if specified).
                                          * {type}      - main type of file. It is replaced by:
                                                           ** 'cf'      (calibration factors)
                                                           ** 'qf'      (quantization factors)
                                                           ** 'weights' (weights)
                                                           ** 'bias'    (biases)
                                                           ** 'mean'    (means)
        :param frontier_name_template:   Template for primitive frontier name (name which will be used for group of
                                         primitives that use the same calibration factors). It supports placeholders:
                                          * {frontier_idx} - unique index that can differentiate frontiers.
        :raise RuntimeError Path to dump directory does not exist or it is inaccessible.
        :raise RuntimeError Path to dump directory does not point to directory.
        :raise RuntimeError Path to weights directory does not exist or it is inaccessible.
        :raise RuntimeError Path to weights directory does not point to directory.
        :raise RuntimeError Path for output files exists, but points to something else than directory.
        :raise NotImplementedError Specified data type for calibration is not yet supported.
        """
        if not os.path.exists(dump_dir_path):
            raise RuntimeError('The specified path to dump files from `examples` application cannot be found or '
                               'it is inaccessible: "{0}".'
                               .format(dump_dir_path))
        if not os.path.isdir(dump_dir_path):
            raise RuntimeError('The specified path to dump files from `examples` application does not point to '
                               'directory: "{0}".'
                               .format(dump_dir_path))
        if output_dir_path is not None and os.path.exists(output_dir_path) and not os.path.isdir(output_dir_path):
            raise RuntimeError('The specified output path for calibration files exists, but does not '
                               'point to directory: {0}.'
                               .format(output_dir_path))
        if weights_dir_path is not None and not os.path.exists(weights_dir_path):
            raise RuntimeError('The specified path to weights files from `examples` application cannot be found or '
                               'it is inaccessible: "{0}".'
                               .format(weights_dir_path))
        if weights_dir_path is not None and not os.path.isdir(weights_dir_path):
            raise RuntimeError('The specified path to weights files from `examples` application does not point to '
                               'directory: "{0}".'
                               .format(weights_dir_path))
        if calib_data_type not in NndFile.get_supported_data_types():
            raise NotImplementedError('Specified calibration data type is not supported yet: "{0}".'
                                      .format(calib_data_type))

        self.__dump_dir_path            = dump_dir_path
        self.__dump_dir_incl_subdirs    = dump_dir_incl_subdirs
        self.__output_dir_path          = output_dir_path if output_dir_path is not None else os.getcwd()
        self.__weights_dir_path         = weights_dir_path
        self.__weights_dir_incl_subdirs = weights_dir_incl_subdirs

        self.__calib_data_type          = calib_data_type

        self.__file_name_template       = file_name_template if file_name_template is not None else \
            CalibCalculator.__def_file_name_template
        self.__group_file_name_template = group_file_name_template if group_file_name_template is not None else \
            CalibCalculator.__def_group_file_name_template
        self.__frontier_name_template   = frontier_name_template if frontier_name_template is not None else \
            CalibCalculator.__def_frontier_name_template

    @staticmethod
    def __parse_calib_opts_from_json(calib_opts):
        # type: (typing.AnyStr) -> typing.Dict[typing.AnyStr, typing.Tuple[typing.List[typing.List[typing.Tuple[typing.AnyStr, typing.Optional[int]]]], int, str, typing.AnyStr, str]]
        """
        Parses JSON containing calibration options into internal representation.

        JSON representation is in following form (comments are for clarification and not allowed in JSON):
        ```
        {
            "<prim1>": [],
            "<prim2>": [],
            "<prim3>": ["<prim1>", "<prim2>"],
            "<prim4>": {"deps": ["<prim3>"], "groups": 2, "decalib_mode": true},
            "<prim5>": {"deps": [{"<prim4>": 0}, {"<prim4>": 1}]}
        }
        ```
        Lines:
         * 1, 2: indicates that calibration factors should be calculated for prim1 and prim2.
         * 3: indicates that calibration factors needs to be calculated for prim3. Also since prim1 and prim2 are
              dependencies the prim1 and prim2 will have common calibration factors calculated. Also common factors from
              prim1, prim2 will be used to decalibrate weights of prim3 (if it has weights).
         * 4: indicates that prim3 is dominator of prim4, weights of prim4 will not be decalibrated, instead
              decalibration will be applied to calibration factors of prim4. The result factors are treated as
              split into 2 groups.
         * 5: dependency on two feature groups of prim4.

        :param calib_opts: String containing calibration options in JSON format.
        :return: Calibration options (normalized). The options are stored in dictionary with the following elements:
                  * key   - primitive name (normalized, lowercase).
                  * value - tuple containing:
                     ** list of frontiers for primitive. Each frontier is a list of dependencies that are
                        dominance frontier of current primitive (used to determine common factors for entire dominance
                        frontier of primitive, decalibrate calibration factors or decalibrate weights before
                        quantization). Each element is a tuple containing dependency
                        primitive name (normalized, lowercase) and feature group (can be `None` if treated as
                        concatenated entity).
                     ** number of feature groups for primitive (split size).
                     ** value indicating mode in which factors calculated for current primitive should be
                        decalibrated by factors from dominance frontier(s) of primitive. Allowed values:
                        '-' (no decalibration), '+' (decalibrate by (first) frontier),
                        '*' (decalibrate by all frontiers).
                     ** weights file(s) root name.
        :raise RuntimeError: Some options in JSON are malformed.
        """
        _logger.info('Parsing calibration options (JSON format)...')

        calib_opts_json = calib_opts.strip()  # type: typing.AnyStr
        if not calib_opts_json.startswith('{'):
            calib_opts_json = '{' + calib_opts_json
        if calib_opts_json.endswith(','):
            calib_opts_json = calib_opts_json[:-1] + '}'
        elif not calib_opts_json.endswith('}'):
            calib_opts_json += '}'

        calib_opts = json.loads(calib_opts_json)
        # Normalization.
        norm_calib_opts = {}
        prim_names = set()
        for prim_name, prim_opts in calib_opts.iteritems():
            norm_prim_name         = prim_name.lower()
            norm_weights_root_name = norm_prim_name
            norm_dump_mode         = ''
            if norm_prim_name in prim_names:
                _logger.warning('Duplicated entry in calibration options JSON for primitive: "{0}". '
                                'It will be omitted.'
                                .format(norm_prim_name))
                continue
            prim_names.add(norm_prim_name)

            prim_groups_count = 1
            prim_decalib_mode = '-'
            prim_deps         = []
            if isinstance(prim_opts, dict):
                if 'deps' in prim_opts:
                    prim_opt_deps = prim_opts['deps']
                    if isinstance(prim_opt_deps, list):
                        prim_deps = prim_opt_deps
                    elif isinstance(prim_opt_deps, (str, unicode)):
                        prim_deps = [prim_opt_deps]
                    else:
                        raise RuntimeError('Primitive dependencies list is malformed (expected list or string): "{0}"'
                                           .format(prim_opt_deps))

                if 'groups' in prim_opts:
                    prim_groups_count = max(int(prim_opts['groups']), 1)
                elif 'split' in prim_opts:
                    prim_groups_count = max(int(prim_opts['split']), 1)

                prim_opt_decalib = None
                if 'decalibrate_mode' in prim_opts:
                    prim_opt_decalib = prim_opts['decalibrate_mode']
                elif 'decalib_mode' in prim_opts:
                    prim_opt_decalib = prim_opts['decalib_mode']

                if prim_opt_decalib is not None:
                    if isinstance(prim_opt_decalib, (str, unicode)):
                        prim_decalib_mode = str(prim_opt_decalib)
                    elif prim_opt_decalib:
                        prim_decalib_mode = '+'

                if 'weights' in prim_opts:
                    norm_weights_root_name = prim_opts['weights']

                if 'dump_mode' in prim_opts:
                    norm_dump_mode = str(prim_opts['dump_mode'])
            elif isinstance(prim_opts, list):
                prim_deps = prim_opts
            elif isinstance(prim_opts, (str, unicode)):
                prim_deps = [prim_opts]
            else:
                raise RuntimeError('Primitive options is malformed (expected list, object or string): "{0}"'
                                   .format(prim_opts))

            norm_prim_all_deps = []
            norm_prim_deps     = []
            for prim_dep in prim_deps:
                if isinstance(prim_dep, list):
                    frontier_deps = prim_dep
                    if len(frontier_deps) <= 0:
                        continue

                    if len(norm_prim_deps) > 0:
                        norm_prim_all_deps.append(norm_prim_deps)
                        norm_prim_deps = []

                    for frontier_dep in frontier_deps:
                        if isinstance(frontier_dep, dict):
                            for prim_dep_name, prim_dep_group_idx in frontier_dep.iteritems():
                                norm_prim_deps.append((prim_dep_name.lower(), max(int(prim_dep_group_idx), 0)))
                        elif isinstance(frontier_dep, (str, unicode)):
                            norm_prim_deps.append((frontier_dep.lower(), None))
                        else:
                            raise RuntimeError('Primitive dependency group specification is malformed '
                                               '(expected object or string): "{0}"'
                                               .format(frontier_dep))
                elif isinstance(prim_dep, dict):
                    for prim_dep_name, prim_dep_group_idx in prim_dep.iteritems():
                        norm_prim_deps.append((prim_dep_name.lower(), max(int(prim_dep_group_idx), 0)))
                elif isinstance(prim_dep, (str, unicode)):
                    norm_prim_deps.append((prim_dep.lower(), None))
                else:
                    raise RuntimeError('Primitive dependency specification is malformed (expected list, '
                                       'object or string): "{0}"'
                                       .format(prim_dep))

            if len(norm_prim_deps) > 0:
                norm_prim_all_deps.append(norm_prim_deps)

            norm_calib_opts[norm_prim_name] = (norm_prim_all_deps, prim_groups_count, prim_decalib_mode, norm_weights_root_name, norm_dump_mode)

        return norm_calib_opts

    @staticmethod
    def __validate_calib_opts(norm_calib_opts):
        # type: (typing.Dict[typing.AnyStr, typing.Tuple[typing.List[typing.List[typing.Tuple[typing.AnyStr, typing.Optional[int]]]], int, str, typing.AnyStr, str]]) -> typing.NoReturn
        """
        Validates calibration options (internal completeness and lack of cycles). Raises `RuntimeError` if options
        are malformed.

        :param norm_calib_opts: Normalized calibration options (e.g. as returned from
                                `CalibCalculator.__parse_calib_opts_from_json()`).
        :raise RuntimeError: Calibration options contain depencency cycle(s).
        :raise RuntimeError: Primitive was specified as dependency without proper definition in calibration options.
        :raise RuntimeError: Unknown decalibration mode selected in primitive.
        """
        # Simple BFS with coloring of already scanned nodes for cycle detection.
        for prim_name, prim_opts in norm_calib_opts.iteritems():
            scan_queue = collections.deque({dep[0]
                                            for frontier_deps in prim_opts[0]
                                            for dep in frontier_deps})

            scanned_prims = set()
            while len(scan_queue) > 0:
                scanned_prim = scan_queue.popleft()

                if prim_name == scanned_prim:
                    raise RuntimeError('Calibration options contain cycle on primitive dependencies. '
                                       'Cycle was closed by "{0}" primitive.'
                                       .format(scanned_prim))
                if scanned_prim not in norm_calib_opts:
                    raise RuntimeError('Primitive specified as dependency does not have definition in calibration '
                                       'options: "{0}" primitive.'
                                       .format(scanned_prim))

                scanned_prims.add(scanned_prim)
                scan_queue.extend({dep[0]
                                   for frontier_deps in norm_calib_opts[scanned_prim][0]
                                   for dep in frontier_deps
                                   if dep[0] not in scanned_prims})

            if prim_opts[2] not in CalibCalculator.__allowed_decalib_modes:
                raise RuntimeError('Unknown decalibration mode ("{0}") selected: "{1}" primitive.'
                                   .format(prim_opts[2], prim_name))

    @staticmethod
    def __canonize_prim_dep(norm_calib_opts, dep):
        # type: (typing.Dict[typing.AnyStr, typing.Tuple[typing.List[typing.List[typing.Tuple[typing.AnyStr, typing.Optional[int]]]], int, str, typing.AnyStr, str]], typing.Tuple[typing.AnyStr, typing.Optional[int]]) -> typing.Tuple[typing.AnyStr, int, typing.Optional[int]]
        """
        Canonize form of dependency and extend it with additional information (number of feature groups).

        :param norm_calib_opts: Normalized calibration options (e.g. as returned from
                                `CalibCalculator.__parse_calib_opts_from_json()`).
                                Please note, that calibration options should be validated (this method does not
                                perform validation and assumes that data is valid - please use
                                `CalibCalculator.__validate_calib_opts()` to ensure that parameter is valid).
        :param dep:             Dependency (from norm_calib_opts) to cannonize.
        :return:                Dependency in unified form.
        """
        if dep[0] not in norm_calib_opts:
            return dep[0], 1, None  # Assuming that unknown dependency has no split.
        dep_groups_count = norm_calib_opts[dep[0]][1]
        if dep_groups_count <= 1:
            return dep[0], 1, None  # Canonize dependency which has no split.
        return dep[0], dep_groups_count, None if dep[1] is None else dep[1] % dep_groups_count

    @staticmethod
    def __calculate_dominance_frontiers(norm_calib_opts):
        # type: (typing.Dict[typing.AnyStr, typing.Tuple[typing.List[typing.List[typing.Tuple[typing.AnyStr, typing.Optional[int]]]], int, str, typing.AnyStr, str]]) -> typing.List[typing.Set[typing.Tuple[typing.AnyStr, int, typing.Optional[int]]]]
        """
        Calculates dominance frontier from normalized calibration options.

        :param norm_calib_opts: Normalized calibration options (e.g. as returned from
                                `CalibCalculator.__parse_calib_opts_from_json()`).
                                Please note, that calibration options should be validated (this method does not
                                perform validation and assumes that data is valid - please use
                                `CalibCalculator.__validate_calib_opts()` to ensure that parameter is valid).
        :return:                Frontiers information. The collection is dictionary with:
                                 * key - frontier name
                                 * value - set of tuples that describes which primitives are grouped into same frontier.
                                           They contain following elements:
                                            * primitive name
                                            * number of feature groups in primitive (split size)
                                            * index (`0`-based) of feature group to use (`None` if all should be taken)
        :raise NotImplementedError Mixed mode of dependency using (as single concatenated entity and as multiple feature
                                   groups) is not yet supported.
        """
        _logger.info('Calculating primitive dominance frontiers from calibration options...')

        # Checking dependencies for information how to handle split primitives (either as one concatenated dependency
        # or as multiple dependencies (each representing single feature group).
        all_deps = {CalibCalculator.__canonize_prim_dep(norm_calib_opts, dep)
                    for _, prim_opts in norm_calib_opts.iteritems()
                    for frontier_deps in prim_opts[0]
                    for dep in frontier_deps}
        # true - indicates that primitive should be treated as single concatenated entity.
        dep_concat_handlers = {dep_name: None for dep_name in {dep[0] for dep in all_deps}}  # type: typing.Dict[typing.AnyStr, typing.Optional[bool]]
        for dep in all_deps:
            if dep_concat_handlers[dep[0]] is None:
                dep_concat_handlers[dep[0]] = dep[2] is None
            elif dep_concat_handlers[dep[0]] != dep[2] is None:
                raise NotImplementedError('Using split primitive both as single concatenated dependency and '
                                          'as multiple feature-group-split dependencies at the same time is '
                                          'not yet supported: "{0}" primitive dependency.'
                                          .format(dep[0]))

        # Creating dominance frontier clusters based on simple chaining approach.
        dep_map = {(prim_name, max(prim_opts[1], 1), group_idx, frontier_deps_idx):
                   {CalibCalculator.__canonize_prim_dep(norm_calib_opts, dep) for dep in frontier_deps}
                   for prim_name, prim_opts in norm_calib_opts.iteritems()
                   for frontier_deps_idx, frontier_deps in enumerate(prim_opts[0] if len(prim_opts[0]) > 0 else [[]])
                   for group_idx in ([None] if prim_name not in dep_concat_handlers or dep_concat_handlers[prim_name]
                                     else xrange(prim_opts[1]))}  # type: typing.Dict[typing.Tuple[typing.AnyStr, int, typing.Optional[int], int], typing.Set[typing.Tuple[typing.AnyStr, int, typing.Optional[int]]]]

        rev_dep_map = {prim_name: set() for prim_name in {prim_full_name[:3] for prim_full_name in dep_map}}
        for prim_name, prim_deps in dep_map.iteritems():
            for prim_dep in prim_deps:
                rev_dep_map[prim_dep].add(prim_name)

        frontiers    = []
        scanned_deps = set()
        for dep_name, next_prim_names in rev_dep_map.iteritems():
            if dep_name in scanned_deps:
                continue
            scanned_deps.add(dep_name)

            frontier      = {dep_name}
            frontier_deps = {chained_dep_name
                             for next_prim_name in next_prim_names
                             for chained_dep_name in dep_map[next_prim_name]
                             if chained_dep_name not in scanned_deps}
            while len(frontier_deps) > 0:
                dep_name = frontier_deps.pop()
                scanned_deps.add(dep_name)

                frontier.add(dep_name)
                frontier_deps.update([chained_dep_name
                                      for next_prim_name in rev_dep_map[dep_name]
                                      for chained_dep_name in dep_map[next_prim_name]
                                      if chained_dep_name not in scanned_deps])

            frontiers.append(frontier)

        _logger.info('    Found {0} frontiers:'.format(len(frontiers)))
        for frontier_idx, deps in enumerate(frontiers):
            _logger.info('    {0:>4}.    {{{1}}}'
                         .format(frontier_idx + 1, ', '.join(['"{0}" (group: {1})'
                                                             .format(dep[0], dep[2] if dep[2] is not None else 'all')
                                                              for dep in deps])))

        return frontiers

    def __gen_nnd_name(self, prim_name, groups_count=1, group_idx=None, attrib=None, name_type=None):
        # type: (typing.AnyStr, int, int, typing.Optional[typing.AnyStr], int) -> typing.AnyStr
        """
        Generated name of .nnd file connected to primitive (or it feature group).

        :param prim_name:    Name of primitive.
        :param groups_count: Number of feature groups in primitive (split size).
        :param group_idx:    Index (`0`-based) of feature group. Ignored, if `groups_count` is one (or less).
        :param attrib:       Attribute string that allows to generate different name depending on attributes.
        :param name_type:    Type of .nnd file (one of `__WT_`-prefixed enum values). Default: `__WT_CF`.
        :return:             Proposed name of .nnd file.
        """
        attrib_str = attrib + '_' if attrib is not None else ''
        type_str = CalibCalculator.__file_type_quals[name_type if name_type is not None else CalibCalculator.__WT_CF]
        if groups_count > 1 and group_idx is not None:
            return self.__group_file_name_template.format(prim_name=prim_name, group_idx=group_idx % groups_count + 1,
                                                          attrib=attrib_str, type=type_str)
        else:
            return self.__file_name_template.format(prim_name=prim_name, attrib=attrib_str, type=type_str)

    @staticmethod
    def __max_abs_from_dump_file(file_path):
        # type: (typing.AnyStr) -> float
        """
        Calculates max of absolute values of all elements dumped in specified file.

        :param file_path: Path (absolute or relative) to file which contains data dumped in format of clDNN `examples`
                          application.
        :return:          Value which is maximum of absolute values of all elements from file.
        """
        with open(file_path, u'rb') as dump_file:
            dump_contents = dump_file.read()
            dump_abs_vals = map(lambda val: abs(float(val)),
                                CalibCalculator.__dump_file_value_matcher.findall(dump_contents))
            return max(dump_abs_vals) if len(dump_abs_vals) > 0 else 0.0

    @staticmethod
    def __abs_vals_from_dump_file(file_path):
        # type: (typing.AnyStr) -> array.array
        """
        Calculates absolute values of all elements dumped in specified file and return vector of them (in the same
        order as in the file).

        :param file_path: Path (absolute or relative) to file which contains data dumped in format of clDNN `examples`
                          application.
        :return:          Absolute values from file.
        """
        with open(file_path, u'rb') as dump_file:
            dump_contents = dump_file.read()
            dump_abs_vals = map(lambda val: abs(float(val)),
                                CalibCalculator.__dump_file_value_matcher.findall(dump_contents))
            return array.array('f', dump_abs_vals)

    @staticmethod
    def __extract_props_from_dump_file_name((dump_file_name, dump_file_dir)):
        # type: (typing.Tuple[typing.AnyStr, typing.AnyStr]) -> typing.Tuple[typing.AnyStr, int, int, typing.AnyStr, typing.AnyStr]
        name_parts = CalibCalculator.__dump_file_name_matcher.match(dump_file_name)
        return name_parts.group(1), int(name_parts.group(2)), int(name_parts.group(3)), dump_file_name, dump_file_dir

    @staticmethod
    def __list_dump_files(dump_dir_path, recursive=False):
        # type: (typing.AnyStr, bool) -> typing.List[typing.Tuple[typing.AnyStr, int, int, typing.AnyStr, typing.AnyStr]]
        """
        List all candidate dump files.

        :param dump_dir_path: Directory root where dump files reside.
        :param recursive:     Indicates that subdirectories should be also scanned recursively for candidate files.
        :return:              List of tuples. Each tuple contain:
                               * primitive name.
                               * index of batch (`0`-based).
                               * index of (output) feature (`0`-based); index is flatten among all feature groups.
                               * name of the dump file.
                               * relative path to subdirectory
        """
        _logger.info('Searching for all dump files. This may take a while: "{0}"...'
                     .format(dump_dir_path))

        if recursive:
            candidate_files = [(walk_file, os.path.relpath(walk_root, dump_dir_path))
                               for walk_root, _, walk_files in os.walk(dump_dir_path)
                               for walk_file in walk_files
                               if CalibCalculator.__dump_file_name_matcher.match(walk_file)]
        else:
            candidate_files = [(list_file, '.') for list_file in os.listdir(dump_dir_path)
                               if os.path.isfile(os.path.join(dump_dir_path, list_file)) and
                               CalibCalculator.__dump_file_name_matcher.match(list_file)]

        _logger.info('    Found {0} candidate dump files.'.format(len(candidate_files)))
        _logger.info('Extracting information from all dump files. This may take a while: "{0}"...'
                     .format(dump_dir_path))
        return map(CalibCalculator.__extract_props_from_dump_file_name, candidate_files)

    @staticmethod
    def __extract_props_from_weights_file_name((weights_file_name, weights_file_dir)):
        # type: (typing.Tuple[typing.AnyStr, typing.AnyStr]) -> typing.Tuple[typing.AnyStr, typing.Optional[int], int, typing.AnyStr, typing.AnyStr]
        name_parts = CalibCalculator.__weights_file_name_matcher.match(weights_file_name)
        group_idx = int(name_parts.group(2)) if name_parts.group(2) is not None and name_parts.group(2) != '' else None
        group_idx = group_idx - 1 if group_idx is not None and group_idx > 0 else None
        if name_parts.group(3) is not None and name_parts.group(3)  != '':
            file_type = CalibCalculator.__WT_W
        elif name_parts.group(4) is not None and name_parts.group(4) != '':
            file_type = CalibCalculator.__WT_B
        elif name_parts.group(5) is not None and name_parts.group(5) != '':
            file_type = CalibCalculator.__WT_QF
        elif name_parts.group(6) is not None and name_parts.group(6) != '':
            file_type = CalibCalculator.__WT_M
        else:
            assert False

        return name_parts.group(1), group_idx, file_type, weights_file_name, weights_file_dir

    @staticmethod
    def __list_weights_files(weights_dir_path=None, recursive=False):
        # type: (typing.AnyStr, bool) -> typing.List[typing.Tuple[typing.AnyStr, typing.Optional[int], int, typing.AnyStr, typing.AnyStr]]
        """
        List all candidate weights/biases/QF files.

        :param weights_dir_path: Directory root where files for weights, biases and weight quantization factors reside.
                                 Can be `None` which means that weights will not be searched.
        :param recursive:        Indicates that subdirectories should be also scanned recursively for candidate files.
        :return:                 List of tuples. Each tuple contain:
                                  * primitive name.
                                  * index of feature group/split (`0`-based). The value is optional (can be `None`).
                                  * file type (one of `__WT_`-prefixed enum values).
                                  * name of the weights/biases/QF file.
                                  * relative path to subdirectory (relative to `weights_dir_path`).
        """
        if weights_dir_path is None:
            _logger.info('Searching for all weights files is disabled.')
            return []

        _logger.info('Searching for all weights files. This may take a while: "{0}"...'
                     .format(weights_dir_path))

        if recursive:
            candidate_files = [(walk_file, os.path.relpath(walk_root, weights_dir_path))
                               for walk_root, _, walk_files in os.walk(weights_dir_path)
                               for walk_file in walk_files
                               if CalibCalculator.__weights_file_name_matcher.match(walk_file)]
        else:
            candidate_files = [(list_file, '.') for list_file in os.listdir(weights_dir_path)
                               if os.path.isfile(os.path.join(weights_dir_path, list_file)) and
                               CalibCalculator.__weights_file_name_matcher.match(list_file)]

        _logger.info('    Found {0} candidate weights files.'.format(len(candidate_files)))
        _logger.info('Extracting information from all weights files. This may take a while: "{0}"...'
                     .format(weights_dir_path))
        return map(CalibCalculator.__extract_props_from_weights_file_name, candidate_files)

    @staticmethod
    def __get_dump_files_for_primitive(dump_dir_path, dump_file_infos, prim_name, groups_count=1, group_idx=None,
                                       dump_mode='normal'):
        # type: (typing.AnyStr, typing.List[typing.Tuple[typing.AnyStr, int, int, typing.AnyStr, typing.AnyStr]], typing.AnyStr, int, typing.Optional[int], str) -> typing.Tuple[typing.AnyStr, int, int, int, typing.List[typing.Tuple[int, int, int, typing.Union[typing.AnyStr, float], typing.Optional[typing.AnyStr]]]]
        """
        Get all dump files (with information) connected to specified primitive or specified feature group (split) of
        primitive.

        :param dump_dir_path:   Directory root where dump files reside.
        :param dump_file_infos: Information about all considered dump files. The information can be returned using
                                `CalibCalculator.__list_dump_files()`.
        :param prim_name:       Name of primitive.
        :param groups_count:    Number of feature groups (split size) in primitive.
        :param group_idx:       Index (`0`-based) of feature group to select. Specify `None`, if all groups should be
                                selected (split primitive output is treated as single concatenated entity).
                                Ignored when there is only one group.
        :param dump_mode:       Mode in which primitive was dumped. One of:
                                 * 'normal'        - fall-back mode in which each output feature is dumped to different
                                                     file.
                                 * 'expand_single' - mode (for fully-connected layers) where all output features are
                                                     dumped to single file.
        :return:                Information about primitive dump files. It is tuple containing:
                                 * name of primitive.
                                 * calculated batch size.
                                 * calculated number of output features.
                                 * number of feature groups (split size).
                                 * list of tuples describing each dump file for selected primitive:
                                    ** batch index (`0`-based).
                                    ** feature index (`0`-based).
                                    ** group index (`0`-based).
                                    ** name of dump file or, if 'expand_single' mode is used - value from file.
                                    ** directory path of dump file (relative to dump directory path). `None` if
                                       `dump_mode` is equal 'expand_single'.
        :raise RuntimeError: Calculated cumulative number of features is not dividable by count of feature groups.
        """
        if dump_mode not in ['expand_single', 'normal']:
            dump_mode = 'normal'

        _logger.info('Gathering dump information about primitive: "{0}" primitive (group: {1}, dump mode: "{2}")...'
                     .format(prim_name, group_idx if group_idx is not None else 'all', dump_mode))

        prim_file_infos = [file_info for file_info in dump_file_infos if file_info[0].lower() == prim_name.lower()]
        if len(prim_file_infos) <= 0:
            return prim_name, 0, 0, 0, []

        batch_idxs   = {prim_file_info[1] for prim_file_info in prim_file_infos}
        feature_idxs = {prim_file_info[2] for prim_file_info in prim_file_infos}
        batch_size     = max(batch_idxs) + 1
        features_count = max(feature_idxs) + 1

        if dump_mode == 'expand_single':
            if features_count > 1:
                _logger.warning('Calculated feature count based on names of dump files ({0}) indicates that "normal" '
                                'dump mode should be used: "{1}" primitive.'
                                .format(features_count, prim_name))

            exp_prim_file_infos = []
            exp_features_count  = None
            log_features_count_file_name = None
            for exp_prim_name, batch_idx, feature_idx, file_name, file_dir in prim_file_infos:
                exp_dump_file_path = os.path.join(dump_dir_path, file_dir, file_name)
                abs_vals           = CalibCalculator.__abs_vals_from_dump_file(exp_dump_file_path)
                if len(abs_vals) <= 0:
                    raise RuntimeError('Expanded dump file has no valid values / factors inside: "{0}" ("{1}") file, '
                                       '"{2}" primitive.'
                                       .format(file_name, exp_dump_file_path, prim_name))

                if exp_features_count is None:
                    exp_features_count           = len(abs_vals)
                    log_features_count_file_name = file_name
                elif exp_features_count != len(abs_vals):
                    raise RuntimeError('Mismatch in calculated feature count in expanded files ({0} vs. {1} '
                                       'for files: "{2}", "{3}"): "{4}" primitive.'
                                       .format(exp_features_count, len(abs_vals),
                                               log_features_count_file_name, file_name, prim_name))

                exp_prim_file_infos_part = [(exp_prim_name, batch_idx, abs_val_idx, abs_val, None)
                                            for abs_val_idx, abs_val in enumerate(abs_vals)]
                exp_prim_file_infos.extend(exp_prim_file_infos_part)

            assert exp_features_count is not None
            prim_file_infos = exp_prim_file_infos
            feature_idxs    = set(xrange(exp_features_count))
            features_count  = exp_features_count

        if features_count % groups_count != 0:
            raise RuntimeError('Cumulative output feature count calculated from dump files ({0}) '
                               'cannot be split (is not dividable) into specified number of groups ({1}): '
                               '"{2}" primitive.'
                               .format(features_count, groups_count, prim_name))

        features_count /= groups_count

        if len(feature_idxs) < features_count * groups_count:
            missing_feature_idxs = list(set(xrange(features_count)) - feature_idxs)
            _logger.warning('Dump files for primitive do not provide contiguous coverage for all output features '
                            '(distinct feature indices: {0}, features count: {1}, groups count: {2}, cumulative '
                            'features count: {3}). It appears that some dump files are missing ({4}): "{5}" primitive.'
                            .format(len(feature_idxs), features_count, groups_count, features_count * groups_count,
                                    missing_feature_idxs, prim_name))
        if len(batch_idxs) < batch_size:
            missing_batch_idxs = list(set(xrange(batch_size)) - batch_idxs)
            _logger.warning('Dump files for primitive do not provide contiguous coverage for entire batch '
                            '(distinct batch indices: {0}, batch size: {1}). It appears that some dump files '
                            'are missing ({2}): "{3}" primitive.'
                            .format(len(batch_idxs), batch_size, missing_batch_idxs, prim_name))

        # Treat as single concat entity if group index has special value.
        if group_idx is None:
            features_count *= groups_count
            groups_count    = 1

        file_infos = [(batch_idx, feature_idx % features_count, feature_idx / features_count, file_name, file_dir)
                      for _, batch_idx, feature_idx, file_name, file_dir in prim_file_infos]
        if groups_count > 1 and group_idx is not None:
            file_infos = [file_info for file_info in file_infos if file_info[2] == group_idx]
        return prim_name, batch_size, features_count, groups_count, file_infos

    @staticmethod
    def __get_dump_files_for_frontier(dump_dir_path, dump_file_infos, norm_calib_opts, frontier_idx, frontier):
        # type: (typing.AnyStr, typing.List[typing.Tuple[typing.AnyStr, int, int, typing.AnyStr, typing.AnyStr]], typing.Dict[typing.AnyStr, typing.Tuple[typing.List[typing.List[typing.Tuple[typing.AnyStr, typing.Optional[int]]]], int, str, typing.AnyStr, str]], int, typing.Set[typing.Tuple[typing.AnyStr, int, typing.Optional[int]]]) -> typing.Tuple[int, typing.Set[typing.Tuple[typing.AnyStr, int, typing.Optional[int]]], int, int, typing.List[typing.Tuple[int, int, int, typing.AnyStr, typing.AnyStr]]]
        """
        Gathers information about all dump files needed to calculate calibration of specified primitive
        dominance frontier.

        :param dump_dir_path:   Directory root where dump files reside.
        :param dump_file_infos: Information about all considered dump files. The information can be returned using
                                `CalibCalculator.__list_dump_files()`.
        :param norm_calib_opts: Normalized calibration options (e.g. as returned from
                                `CalibCalculator.__parse_calib_opts_from_json()`).
                                Please note, that calibration options should be validated (this method does not
                                perform validation and assumes that data is valid - please use
                                `CalibCalculator.__validate_calib_opts()` to ensure that parameter is valid).
        :param frontier_idx:    Index of frontier (`0`-based) for which dump files are gathered.
        :param frontier:        Collection of calculated dominance frontiers (e.g. as values of dictionary returned from
                                `CalibCalculator.__calculate_dominance_frontiers()`)
        :return:                Information about primitive dump files. It is tuple containing:
                                 * index (`0`-based) of frontier.
                                 * calculated batch size.
                                 * calculated number of output features.
                                 * list of tuples describing each dump file for selected primitive:
                                    ** batch index (`0`-based).
                                    ** feature index (`0`-based).
                                    ** group index (`0`-based).
                                    ** name of dump file.
                                    ** directory path of dump file (relative to dump directory path).
        :raise RuntimeError: Primitives from the same primitive frontier have different number of features. Cannot
                             calculate unified calibration factors.
        """
        _logger.info('Gathering dump information about primitives in dominance frontier: frontier # {0}...'
                     .format(frontier_idx + 1))

        batch_size      = 0
        feature_count   = 0
        prim_file_infos = []
        prev_prim_file_info = None
        for prim_name, groups_count, group_idx in frontier:
            prim_dump_mode = norm_calib_opts[prim_name][4]
            prim_file_info = CalibCalculator.__get_dump_files_for_primitive(dump_dir_path, dump_file_infos, prim_name,
                                                                            groups_count, group_idx, prim_dump_mode)
            if prev_prim_file_info is not None:
                if prev_prim_file_info[2] != prim_file_info[2]:
                    raise RuntimeError('Primitives from the same frontier have different number of features '
                                       '({0} in "{1}" primitive vs. {2} in "{3}" primitive). Cannot calculate '
                                       'unified calibration factors: "{1}", "{3}" primitives.'
                                       .format(prev_prim_file_info[2], prev_prim_file_info[0],
                                               prim_file_info[2], prim_file_info[0]))
                if prev_prim_file_info[1] != prim_file_info[1]:
                    _logger.warning('Primitives from the same frontier have different batch size '
                                    '({0} in "{1}" primitive vs. {2} in "{3}" primitive). Assuming batch size is '
                                    'maximum of the two.'
                                    .format(prev_prim_file_info[1], prev_prim_file_info[0],
                                            prim_file_info[1], prim_file_info[0]))
                    batch_size = max(batch_size, prim_file_info[1])
            else:
                batch_size    = prim_file_info[1]
                feature_count = prim_file_info[2]

            prim_file_infos.extend(prim_file_info[4])

            prev_prim_file_info = prim_file_info

        return frontier_idx, frontier, batch_size, feature_count, prim_file_infos

    @staticmethod
    def __get_weights_files_for_primitive(weights_file_infos, norm_calib_opts, prim_name,
                                          groups_count=1, group_idx=None):
        # type: (typing.List[typing.Tuple[typing.AnyStr, typing.Optional[int], int, typing.AnyStr, typing.AnyStr]], typing.Dict[typing.AnyStr, typing.Tuple[typing.List[typing.List[typing.Tuple[typing.AnyStr, typing.Optional[int]]]], int, str, typing.AnyStr, str]], typing.AnyStr, int, typing.Optional[int]) -> typing.Tuple[typing.AnyStr, typing.AnyStr, int, typing.List[typing.Tuple[typing.List[typing.Tuple[typing.AnyStr, typing.AnyStr]], typing.List[typing.Tuple[typing.AnyStr, typing.AnyStr]], typing.List[typing.Tuple[typing.AnyStr, typing.AnyStr]], typing.List[typing.Tuple[typing.AnyStr, typing.AnyStr]]]]]
        """
        Get all dump files (with information) connected to specified primitive or specified feature group (split) of
        primitive.

        :param weights_file_infos: Information about all considered dump files. The information can be returned using
                                   `CalibCalculator.__list_weights_files()`.
        :param norm_calib_opts:    Normalized calibration options (e.g. as returned from
                                   `CalibCalculator.__parse_calib_opts_from_json()`).
                                   Please note, that calibration options should be validated (this method does not
                                   perform validation and assumes that data is valid - please use
                                   `CalibCalculator.__validate_calib_opts()` to ensure that parameter is valid).
        :param prim_name:          Name of primitive.
        :param groups_count:       Number of feature groups (split size) in primitive (for validation).
        :param group_idx:          Index (`0`-based) of feature group to select. Specify `None`, if all groups should be
                                   selected.
                                   Ignored when there is only one group.
        :return:                   Information about primitive dump files. It is tuple containing:
                                    * name of primitive.
                                    * root name of weights (usually the same as name of primitive).
                                    * calculated number of feature groups (split size).
                                    * list of tuples describing each weights file for selected primitive:
                                       ** group index (`0`-based).
                                       ** list of tuples/pairs (file name and directory path of weights file (relative
                                          to weights directory path)) describing weights files.
                                       ** list of tuples/pairs (file name and directory path of biases file (relative
                                          to weights directory path)) describing biases files.
                                       ** list of tuples/pairs (file name and directory path of QF file (relative
                                          to weights directory path)) describing QF files.
                                       ** list of tuples/pairs (file name and directory path of mean file (relative
                                          to weights directory path)) describing mean files.
        :raise RuntimeError: Calculated cumulative number of features is not dividable by count of feature groups.
        """
        _logger.info('Gathering weights information about primitive: "{0}" primitive (group: {1})...'
                     .format(prim_name, group_idx if group_idx is not None else 'all'))

        file_root_name = norm_calib_opts[prim_name][3]

        prim_file_infos = [file_info
                           for file_info in weights_file_infos
                           if file_info[0].lower() == file_root_name.lower()]
        if len(prim_file_infos) <= 0:
            return prim_name, file_root_name, 0, []

        group_idxs = {prim_file_info[1] for prim_file_info in prim_file_infos if prim_file_info[1] is not None}
        group_idxs_count  = len(group_idxs)
        # Threat `None` as distinct index (equivalent 0) only if set of indices would be empty otherwise.
        calc_groups_count = max(group_idxs) + 1 if group_idxs_count > 0 else 1
        group_idxs_count  = max(group_idxs_count, 1)

        if group_idxs_count < calc_groups_count:
            missing_group_idxs = [idx + 1 for idx in set(xrange(calc_groups_count)) - group_idxs]
            _logger.warning('Weights files for primitive do not provide contiguous coverage for entire split '
                            '(distinct group indices: {0}, split size: {1}). It appears that some weights files '
                            'are missing ({2}): "{3}" primitive, "{4}" weight name root.'
                            .format(group_idxs_count, calc_groups_count, missing_group_idxs, prim_name, file_root_name))
        if calc_groups_count != groups_count:
            _logger.warning('Calculated number of feature groups (split size) is different than declared in primitive '
                            'configuration (calculated split size: {0}, config. split size: {1}). Excessive '
                            'files will be omitted: "{2}" primitive, "{3}" weight name root.'
                            .format(calc_groups_count, groups_count, prim_name, file_root_name))
            calc_groups_count = min(calc_groups_count, groups_count)

        file_infos = [([], [], [], []) for _ in xrange(calc_groups_count)]  # type: typing.List[typing.Tuple[typing.List[typing.Tuple[typing.AnyStr, typing.AnyStr]], typing.List[typing.Tuple[typing.AnyStr, typing.AnyStr]], typing.List[typing.Tuple[typing.AnyStr, typing.AnyStr]], typing.List[typing.Tuple[typing.AnyStr, typing.AnyStr]]]]
        for _, file_group_idx, file_type, file_name, file_dir in prim_file_infos:
            assert 0 <= file_type < 4
            if file_group_idx is not None and file_group_idx < calc_groups_count:
                file_infos[file_group_idx][file_type].append((file_name, file_dir))
            elif file_group_idx is None and calc_groups_count == 1:
                file_infos[0][file_type].append((file_name, file_dir))

        if groups_count > 1 and group_idx is not None:
            file_infos = file_infos[group_idx:group_idx + 1] if group_idx < calc_groups_count else [([], [], [], [])]
        return prim_name, file_root_name, calc_groups_count, file_infos

    @staticmethod
    def __max_abs_for_frontier(dump_dir_path, dump_file_infos, norm_calib_opts, frontier_idx, frontier):
        # type: (typing.AnyStr, typing.List[typing.Tuple[typing.AnyStr, int, int, typing.AnyStr, typing.AnyStr]], typing.Dict[typing.AnyStr, typing.Tuple[typing.List[typing.List[typing.Tuple[typing.AnyStr, typing.Optional[int]]]], int, str, typing.AnyStr, str]], int, typing.Set[typing.Tuple[typing.AnyStr, int, typing.Optional[int]]]) -> array.array
        """
        Calculates max of absolute values of all elements of all batch sets for all primitives in dominance frontier.
        The maximum is calculated separately per each output feature and one element is returned per output feature.

        :param dump_dir_path:   Directory root where dump files reside.
        :param dump_file_infos: Information about all considered dump files. The information can be returned using
                                `CalibCalculator.__list_dump_files()`.
        :param norm_calib_opts: Normalized calibration options (e.g. as returned from
                                `CalibCalculator.__parse_calib_opts_from_json()`).
                                Please note, that calibration options should be validated (this method does not
                                perform validation and assumes that data is valid - please use
                                `CalibCalculator.__validate_calib_opts()` to ensure that parameter is valid).
        :param frontier_idx:    Index (`0`-based) of frontier for which `max(abs())` is calculated.
        :param frontier:        Collection of calculated dominance frontiers (e.g. as values of dictionary returned from
                                `CalibCalculator.__calculate_dominance_frontiers()`)
        :return:                Array of floats containing maximums of absolute values per output feature.
        """
        _logger.info('Calculating maximums of absolutes for dominance frontier: frontier # {0}...'
                     .format(frontier_idx + 1))

        sel_dump_files = CalibCalculator.__get_dump_files_for_frontier(dump_dir_path, dump_file_infos, norm_calib_opts,
                                                                       frontier_idx, frontier)

        _logger.info('    Processing {0} dump file(s) or expanded dump value(s)...'
                     .format(len(sel_dump_files[4])))

        max_abs_vals = array.array('f', [0.0] * sel_dump_files[3])
        for sel_dump_file in sel_dump_files[4]:
            dump_feature_idx = sel_dump_file[1]
            if sel_dump_file[4] is None and isinstance(sel_dump_file[3], (float, int)):
                max_abs_val = float(sel_dump_file[3])
            else:
                dump_file_path = os.path.join(dump_dir_path, sel_dump_file[4], sel_dump_file[3])
                max_abs_val = CalibCalculator.__max_abs_from_dump_file(dump_file_path)
            max_abs_vals[dump_feature_idx] = max(max_abs_vals[dump_feature_idx], max_abs_val)

        return max_abs_vals

    @staticmethod
    def __gen_calib_factors(max_abs_vals, calib_data_type):
        # type: (array.array, str) -> array.array
        """
        Calculates calibration factors based on `max(abs())` values and data type that will be calibrated.

        :param max_abs_vals:    Array of `max(abs())` returned (one per output feature).
                                For details please look into `CalibCalculator.__max_abs_for_frontier()`.
        :param calib_data_type: One of `NndFile.DT_`-prefixed enum values. It allows to estimate needed output range.
        :return:                Array of corresponding calibration factors (per output feature).
        """
        range_factor = CalibCalculator.calib_range_factors[calib_data_type]

        return array.array('f', [(range_factor / max_abs_val if max_abs_val > 0.0 else 0.0)
                                 for max_abs_val in max_abs_vals])

    # TODO: Type + comments.
    # TODO: Split into multiple functions.
    def __calculate_calib_factors(self, norm_calib_opts, out_dir_path, calib_data_type, calc_opts):
        # type: (typing.Dict[typing.AnyStr, typing.Tuple[typing.List[typing.List[typing.Tuple[typing.AnyStr, typing.Optional[int]]]], int, str, typing.AnyStr, str]], typing.AnyStr, str, int) -> typing.Tuple[typing.Optional[typing.Dict],typing.Optional[typing.List], typing.Optional[typing.List],typing.Optional[typing.List]]

        CalibCalculator.__validate_calib_opts(norm_calib_opts)
        frontiers = CalibCalculator.__calculate_dominance_frontiers(norm_calib_opts)
        if len(frontiers) <= 0:
            return ({},
                    [] if (calc_opts & CalibCalculator.CCO_RET_CALIB_FILES) != 0 else None,
                    [] if (calc_opts & CalibCalculator.CCO_RET_WEIGHT_FILES) != 0 else None,
                    [] if (calc_opts & CalibCalculator.CCO_RET_FRONTIER_FILES) != 0 else None)

        # Listing files (dumps and weights).
        dump_file_infos = CalibCalculator.__list_dump_files(self.__dump_dir_path, self.__dump_dir_incl_subdirs)
        if len(dump_file_infos) <= 0:
            raise RuntimeError('Dump directory does not contain any `examples` dump files: "{0}" '
                               '(recursive search: {1}).'
                               .format(self.__dump_dir_path, 'yes' if self.__dump_dir_incl_subdirs else 'no'))

        weights_file_infos = CalibCalculator.__list_weights_files(self.__weights_dir_path,
                                                                  self.__weights_dir_incl_subdirs)

        use_old_nnd_layout = (calc_opts & CalibCalculator.CCO_USE_OLD_NND_LAYOUT) != 0

        # Calculating frontier calibration factors.
        frontier_calib_factor_sets = [
            CalibCalculator.__gen_calib_factors(
                CalibCalculator.__max_abs_for_frontier(self.__dump_dir_path, dump_file_infos, norm_calib_opts,
                                                       frontier_idx, frontier),
                calib_data_type)
            for frontier_idx, frontier in enumerate(frontiers)
        ]

        frontier_nnd_files = []  # type: typing.List[typing.Tuple[str, NndFile]]
        for frontier_idx, calib_factor_set in enumerate(frontier_calib_factor_sets):
            frontier_nnd_file_name = self.__gen_nnd_name(
                self.__frontier_name_template.format(frontier_idx=frontier_idx + 1))
            frontier_nnd_file_path = os.path.join(out_dir_path, frontier_nnd_file_name)

            frontier_nnd_file = NndFile.from_iterable(frontier_nnd_file_path, NndFile.DT_FP32, calib_factor_set)

            frontier_nnd_files.append((frontier_nnd_file_name, frontier_nnd_file))

        # Connecting primitive dependencies to appropriate frontier calibration factors (for later decalibration
        # either for weights or for other calibration factors).
        # noinspection PyTypeChecker
        prims_frontier_idxs_map = {frontier_prim: ((frontier_idx,) + frontier_nnd_files[frontier_idx])
                                   for frontier_idx, frontier in enumerate(frontiers)
                                   for frontier_prim in frontier}  # type: typing.Dict[typing.Tuple[typing.AnyStr, int, typing.Optional[int]], typing.Tuple[int, typing.AnyStr, NndFile]]
        prims_effective_deps = {prim_name: [[prims_frontier_idxs_map[
                                                 CalibCalculator.__canonize_prim_dep(norm_calib_opts, dep)]
                                             for dep in dep_group]
                                            for dep_group in prim_opts[0]]
                                for prim_name, prim_opts in norm_calib_opts.iteritems()}  # type: typing.Dict[typing.AnyStr, typing.List[typing.List[typing.Tuple[int, typing.AnyStr, NndFile]]]]

        # Calculating post-actions needed to process frontier mappings into calibration files.
        calib_nnd_files   = []
        weights_nnd_files = []
        for frontier_idx, frontier in enumerate(frontiers):
            for prim_name, prim_groups_count, prim_group_idx in frontier:
                prim_decalib_mode = norm_calib_opts[prim_name][2]
                prim_deps         = prims_effective_deps[prim_name]

                prim_weights = CalibCalculator.__get_weights_files_for_primitive(weights_file_infos, norm_calib_opts,
                                                                                 prim_name, prim_groups_count,
                                                                                 prim_group_idx)

                # Primitive needs split if is used as single concat entity, but has multiple groups. In this case
                # calculated frontier calibration factors needs to be divided into multiple files.
                prim_needs_split   = prim_groups_count > 1 and prim_group_idx is None
                # Primitive requires decalibration of current calibration factors by calibration factors
                # from its dependencies (this decalibration disables decalibration on weights for current
                # primitive / primitive split group).
                prim_needs_decalib = prim_decalib_mode != '-'
                # Quick check whether primitive has weights
                prim_has_deps      = len(prim_deps) > 0 and any([len(prim_dep_group) > 0
                                                                 for prim_dep_group in prim_deps])
                prim_has_weights   = prim_weights[2] > 0 and len(prim_weights[3]) > 0 \
                                     and any([len(prim_group_weights[CalibCalculator.__WT_W]) > 0
                                              for prim_group_weights in prim_weights[3]])
                prim_needs_decalib_weights = not prim_needs_decalib and prim_has_weights \
                                             and (calc_opts & CalibCalculator.CCO_OMIT_WEIGHTS_DECALIB) == 0

                if prim_needs_decalib and not prim_has_deps:
                    _logger.warning('Primitive needs decalibration, but dependencies for primitive are not specified. '
                                    'The decalibration for primitive will be omitted: "{0}" primitive.'
                                    .format(prim_name))
                    prim_needs_decalib = False
                if prim_needs_decalib_weights and not prim_has_deps:
                    _logger.warning('Primitive needs decalibration on weights, but dependencies for primitive '
                                    'are not specified. The decalibration on weights for primitive will be omitted: '
                                    '"{0}" primitive.'
                                    .format(prim_name))
                    prim_needs_decalib_weights = False

                # Support decalibration on factors.
                base_nnd_file_name, base_nnd_file = frontier_nnd_files[frontier_idx]
                tmp_nnd_file = base_nnd_file.to_nnd('__tmp__.nnd')
                if prim_needs_decalib:
                    if prim_decalib_mode == '+':
                        distinct_decalib_idxs = {prim_dep[0]
                                                 for prim_dep_group in prim_deps
                                                 for prim_dep in prim_dep_group}
                        if len(distinct_decalib_idxs) > 1:
                            raise RuntimeError('Primitive cannot be decalibrated in the "+" mode. It has {0} sets of '
                                               'independent calibration factors (calibration frontiers) on which it '
                                               'depends: "{1}" primitive.'
                                               .format(len(distinct_decalib_idxs), prim_name))
                        assert len(distinct_decalib_idxs) > 0
                        decalib_nnd_file_name, decalib_nnd_file = frontier_nnd_files[distinct_decalib_idxs.pop()]
                        _logger.info('    Decalibrating calibration factors for "{0}" primitive with "{1}"...'
                                     .format(prim_name, decalib_nnd_file_name))
                        tmp_nnd_file.decalibrate(decalib_nnd_file)
                    elif prim_decalib_mode == '*':
                        all_decalib_idxs = [prim_dep[0]
                                            for prim_dep_group in prim_deps
                                            for prim_dep in prim_dep_group]
                        assert len(all_decalib_idxs) > 0
                        for decalib_idx in all_decalib_idxs:
                            decalib_nnd_file_name, decalib_nnd_file = frontier_nnd_files[decalib_idx][1]
                            _logger.info('    Decalibrating calibration factors for "{0}" primitive with "{1}"...'
                                         .format(prim_name, decalib_nnd_file_name))
                            tmp_nnd_file.decalibrate(decalib_nnd_file)

                # Support splitting of factors.
                if prim_needs_split:
                    prim_cf_file_names = [self.__gen_nnd_name(prim_name, prim_groups_count, split_group_idx)
                                          for split_group_idx in xrange(prim_groups_count)]
                    prim_cf_file_paths = [os.path.join(out_dir_path, cf_file_name)
                                          for cf_file_name in prim_cf_file_names]
                    prim_cf_files = tmp_nnd_file.split_on_output_features(*prim_cf_file_paths)
                    calib_nnd_files.extend(zip(prim_cf_file_names, prim_cf_files, [frontier_idx] * prim_groups_count))
                else:
                    prim_cf_file_name = self.__gen_nnd_name(prim_name, prim_groups_count, prim_group_idx)
                    prim_cf_file_path = os.path.join(out_dir_path, prim_cf_file_name)
                    prim_cf_file = tmp_nnd_file.to_nnd(prim_cf_file_path, copy=False)
                    calib_nnd_files.append((prim_cf_file_name, prim_cf_file, frontier_idx))

                # Support dumping of factors before decalibration.
                if prim_needs_decalib and (calc_opts & CalibCalculator.CCO_ADD_RAW_CALIIB_FILES) != 0:
                    if prim_needs_split:
                        prim_cf_file_names = [self.__gen_nnd_name(prim_name, prim_groups_count, split_group_idx,
                                                                  'nondecalib')
                                              for split_group_idx in xrange(prim_groups_count)]
                        prim_cf_file_paths = [os.path.join(out_dir_path, cf_file_name)
                                              for cf_file_name in prim_cf_file_names]
                        prim_cf_files = base_nnd_file.split_on_output_features(*prim_cf_file_paths)
                        calib_nnd_files.extend(zip(prim_cf_file_names, prim_cf_files,
                                                   [frontier_idx] * prim_groups_count))
                    else:
                        prim_cf_file_name = self.__gen_nnd_name(prim_name, prim_groups_count, prim_group_idx,
                                                                'nondecalib')
                        prim_cf_file_path = os.path.join(out_dir_path, prim_cf_file_name)
                        prim_cf_file = base_nnd_file.to_nnd(prim_cf_file_path)
                        calib_nnd_files.append((prim_cf_file_name, prim_cf_file, frontier_idx))

                # ------------------------------------------------------------------------------------------------------
                # WEIGHTS HANDLING:
                # ------------------------------------------------------------------------------------------------------
                if not prim_has_weights:
                    continue

                _logger.info('Loading weights of "{0}" primitive...'.format(prim_name))
                prim_wg_files = [[(wg_file_name,
                                   NndFile.from_file(os.path.join(self.__weights_dir_path, wg_dir, wg_file_name)))
                                  for wg_file_name, wg_dir in prim_group_weights[CalibCalculator.__WT_W]]
                                 for prim_group_weights in prim_weights[3]]

                if prim_needs_decalib_weights:
                    # For weights we only use '+'-like decalibration mode.
                    distinct_decalib_idxs = {prim_dep[0]
                                             for prim_dep_group in prim_deps
                                             for prim_dep in prim_dep_group}
                    if len(distinct_decalib_idxs) > 1:
                        raise RuntimeError('Primitive weight cannot be decalibrated. It has {0} sets of '
                                           'independent calibration factors (calibration frontiers) on which it '
                                           'depends: "{1}" primitive.'
                                           .format(len(distinct_decalib_idxs), prim_name))
                    assert len(distinct_decalib_idxs) > 0
                    decalib_nnd_file_name, decalib_nnd_file = frontier_nnd_files[distinct_decalib_idxs.pop()]

                    # Need to split calibration file when current primitive is splitted, so each group will get
                    # proper number of output calibration factors.
                    if prim_needs_split:
                        prim_cf_file_names = ['{0}.split_g{1}'.format(decalib_nnd_file_name, split_group_idx + 1)
                                              for split_group_idx in xrange(prim_groups_count)]
                        prim_cf_files      = decalib_nnd_file.split_on_output_features(*prim_cf_file_names)
                        log_group_idxs     = range(1, prim_groups_count + 1)
                    else:
                        prim_cf_file_names = [decalib_nnd_file_name]
                        prim_cf_files      = [decalib_nnd_file]
                        log_group_idxs     = [prim_group_idx + 1 if prim_group_idx is not None else 'all']

                    for log_group_idx, prim_wg_file_list, prim_cf_file_name, prim_cf_file in \
                            zip(log_group_idxs, prim_wg_files, prim_cf_file_names, prim_cf_files):
                        if len(prim_wg_file_list) < 1:
                            _logger.warning('Missing one of the weights of primitive. The decalibration for this '
                                            'weight will be omitted: "{0}" primitive (group: {1}).'
                                            .format(prim_name, log_group_idx))
                            continue
                        if len(prim_wg_file_list) > 1:
                            _logger.warning('More than one weight found for primitive or primitive split group. '
                                            'The decalibration will be applied to following weight files: '
                                            '"{0}" primitive (group: {1}).'
                                            .format(prim_name, log_group_idx))
                            for prim_wg_file_name, _ in prim_wg_file_list:
                                _logger.warning('     -- "{0}"'.format(prim_wg_file_name))

                        for prim_wg_file_name, prim_wg_file in prim_wg_file_list:
                            _logger.info('Decalibrating weights file "{0}" with "{1}" calibration factors...'
                                         .format(prim_wg_file_name, prim_cf_file_name))
                            prim_wg_file.decalibrate(prim_cf_file)

                # Renaming weights .nnd files to proper (preparing data for quantization).
                if prim_needs_split:
                    norm_group_idxs = range(prim_groups_count)
                    log_group_idxs  = range(1, prim_groups_count + 1)
                else:
                    norm_group_idxs = [prim_group_idx]
                    log_group_idxs  = [prim_group_idx + 1 if prim_group_idx is not None else 'all']

                # Quantization of weights.
                if (calc_opts & CalibCalculator.CCO_OMIT_WEIGHTS_QUANT) == 0:
                    for norm_group_idx, log_group_idx, prim_wg_file_list in \
                            zip(norm_group_idxs, log_group_idxs, prim_wg_files):
                        if (calc_opts & CalibCalculator.CCO_UNIFY_WG_NAMES) != 0:
                            for file_idx, (_, prim_wg_file) in enumerate(prim_wg_file_list):
                                dup_rename_str = str(file_idx) if file_idx > 0 else None
                                prim_wg_file_name = self.__gen_nnd_name(prim_name, prim_groups_count, norm_group_idx,
                                                                        dup_rename_str, CalibCalculator.__WT_W)
                                prim_wg_file_path = os.path.join(self.__output_dir_path, prim_wg_file_name)

                                prim_qf_file_name = self.__gen_nnd_name(prim_name, prim_groups_count, norm_group_idx,
                                                                        dup_rename_str, CalibCalculator.__WT_QF)
                                prim_qf_file_path = os.path.join(self.__output_dir_path, prim_qf_file_name)

                                _logger.info('    Performing weights quantization for "{0}" primitive into "{1}" '
                                             'and "{2}" files...'
                                             .format(prim_name, prim_wg_file_name, prim_qf_file_name))
                                quant_wg_file, quant_qf_file = prim_wg_file.quantize(self.__calib_data_type, None,
                                                                                     prim_wg_file_path,
                                                                                     prim_qf_file_path)

                                weights_nnd_files.append((prim_wg_file_name, quant_wg_file))
                                weights_nnd_files.append((prim_qf_file_name, quant_qf_file))
                        else:
                            for prim_wg_file_name, prim_wg_file in prim_wg_file_list:
                                prim_wg_file_path = os.path.join(self.__output_dir_path, prim_wg_file_name)

                                prim_qf_file_name = self.__weights_to_qf_renamer.sub(self.__weights_to_qf_renamer_repl,
                                                                                     prim_wg_file_name)
                                prim_qf_file_path = os.path.join(self.__output_dir_path, prim_qf_file_name)

                                _logger.info('    Performing weights quantization for "{0}" primitive into "{1}" '
                                             'and "{2}" files...'
                                             .format(prim_name, prim_wg_file_name, prim_qf_file_name))
                                quant_wg_file, quant_qf_file = prim_wg_file.quantize(self.__calib_data_type, None,
                                                                                     prim_wg_file_path,
                                                                                     prim_qf_file_path)

                                weights_nnd_files.append((prim_wg_file_name, quant_wg_file))
                                weights_nnd_files.append((prim_qf_file_name, quant_qf_file))
                # Rename:
                else:
                    for norm_group_idx, log_group_idx, prim_wg_file_list in \
                            zip(norm_group_idxs, log_group_idxs, prim_wg_files):
                        if (calc_opts & CalibCalculator.CCO_UNIFY_WG_NAMES) != 0:
                            for file_idx, (_, prim_wg_file) in enumerate(prim_wg_file_list):
                                dup_rename_str = str(file_idx) if file_idx > 0 else None
                                prim_wg_file_name = self.__gen_nnd_name(prim_name, prim_groups_count, norm_group_idx,
                                                                        dup_rename_str, CalibCalculator.__WT_W)
                                prim_wg_file_path = os.path.join(self.__output_dir_path, prim_wg_file_name)
                                renamed_nnd_file = prim_wg_file.to_nnd(prim_wg_file_path, copy=False)

                                weights_nnd_files.append((prim_wg_file_name, renamed_nnd_file))
                        else:
                            for prim_wg_file_name, prim_wg_file in prim_wg_file_list:
                                prim_wg_file_path = os.path.join(self.__output_dir_path, prim_wg_file_name)
                                renamed_nnd_file = prim_wg_file.to_nnd(prim_wg_file_path, copy=False)

                                weights_nnd_files.append((prim_wg_file_name, renamed_nnd_file))

        # Saving files.
        if (calc_opts & CalibCalculator.CCO_SAVE_CALIB_FILES) != 0:
            _logger.info('    Saving calibration factors .nnd files...')
            for nnd_file_name, nnd_file, base_idx in calib_nnd_files:
                _logger.info('     -- "{0}" based on frontier # {1}'.format(nnd_file_name, base_idx + 1))
                nnd_file.to_file(use_old_layout_format=use_old_nnd_layout)

        if (calc_opts & CalibCalculator.CCO_SAVE_WEIGHT_FILES) != 0:
            _logger.info('    Saving weights .nnd files...')
            for nnd_file_name, nnd_file in weights_nnd_files:
                _logger.info('     -- "{0}"'.format(nnd_file_name))
                nnd_file.to_file(use_old_layout_format=use_old_nnd_layout)

        if (calc_opts & CalibCalculator.CCO_SAVE_FRONTIER_FILES) != 0:
            _logger.info('    Saving frontier .nnd files...')
            for nnd_file_name, nnd_file in frontier_nnd_files:
                _logger.info('     -- "{0}"'.format(nnd_file_name))
                nnd_file.to_file(use_old_layout_format=use_old_nnd_layout)

        # Preparing return value.
        ret_val = (prims_effective_deps,
                   calib_nnd_files if (calc_opts & CalibCalculator.CCO_RET_CALIB_FILES) != 0 else None,
                   weights_nnd_files if (calc_opts & CalibCalculator.CCO_RET_WEIGHT_FILES) != 0 else None,
                   frontier_nnd_files if (calc_opts & CalibCalculator.CCO_RET_FRONTIER_FILES) != 0 else None)
        return ret_val

    def calculate_calibration_factors(self, calib_opts_file_path, output_dir_path=None, calib_data_type=None,
                                      calc_opts=0):
        if not os.path.exists(calib_opts_file_path) or not os.path.isfile(calib_opts_file_path):
            raise RuntimeError('The specified path to calibration options/settings does not point to existing file '
                               'or it is inaccessible: "{0}".'
                               .format(calib_opts_file_path))

        used_out_dir_path = output_dir_path if output_dir_path is not None else self.__output_dir_path
        if os.path.exists(used_out_dir_path) and not os.path.isdir(used_out_dir_path):
            raise RuntimeError('The specified output path for calibration files exists, but does not '
                               'point to directory: {0}.'
                               .format(used_out_dir_path))

        used_calib_data_type = calib_data_type if calib_data_type is not None else self.__calib_data_type
        if used_calib_data_type not in NndFile.get_supported_data_types():
            raise NotImplementedError('Specified calibration data type is not supported yet: "{0}".'
                                      .format(used_calib_data_type))

        if not os.path.exists(used_out_dir_path):
            os.makedirs(used_out_dir_path)
        with open(calib_opts_file_path, 'r') as calib_opts_file:
            calib_opts_json = calib_opts_file.read()
            calib_opts_file.close()

            norm_calib_opts = CalibCalculator.__parse_calib_opts_from_json(calib_opts_json)
            return self.__calculate_calib_factors(norm_calib_opts, used_out_dir_path, used_calib_data_type, calc_opts)

    def calculate_calibration_factors_from_json(self, calib_opts_json, output_dir_path=None, calib_data_type=None,
                                                calc_opts=0):
        used_out_dir_path = output_dir_path if output_dir_path is not None else self.__output_dir_path
        if os.path.exists(used_out_dir_path) and not os.path.isdir(used_out_dir_path):
            raise RuntimeError('The specified output path for calibration files exists, but does not '
                               'point to directory: {0}.'
                               .format(used_out_dir_path))

        used_calib_data_type = calib_data_type if calib_data_type is not None else self.__calib_data_type
        if used_calib_data_type not in NndFile.get_supported_data_types():
            raise NotImplementedError('Specified calibration data type is not supported yet: "{0}".'
                                      .format(used_calib_data_type))

        if not os.path.exists(used_out_dir_path):
            os.makedirs(used_out_dir_path)

        norm_calib_opts = CalibCalculator.__parse_calib_opts_from_json(calib_opts_json)
        return self.__calculate_calib_factors(norm_calib_opts, used_out_dir_path, used_calib_data_type, calc_opts)


# ----------------------------------------------------------------------------------------------------------------------
# Command-line Parser
# ----------------------------------------------------------------------------------------------------------------------

_script_version = '%(prog)s    1.0.0 (RC1)'  # type: str

_data_type_map = {
    'i8':     NndFile.DT_INT8,
    'u8':     NndFile.DT_UINT8,
    'i16':    NndFile.DT_INT16,
    'u16':    NndFile.DT_UINT16,
    'int8':   NndFile.DT_INT8,
    'uint8':  NndFile.DT_UINT8,
    'int16':  NndFile.DT_INT16,
    'uint16': NndFile.DT_UINT16,
}  # type: typing.Dict[str, str]


def create_cmd_parser():
    # type: () -> typing.Tuple[argparse.ArgumentParser, typing.Dict[str, str]]
    """
    Creates command parser and returns it with "about" descriptions dictionary.

    :return: Command parser and descriptions.
    """
    opts_file_desc = textwrap.dedent('''\
    -----------------------------------------------------------------------------------------------
                                CALIBRATION / QUANTIZATION OPTIONS FILE
    -----------------------------------------------------------------------------------------------
    
    The file should contain JSON (initial brace and hanging comma allowed).
    
    Content of sample <quant-opts-file> (this one is for AlexNet):

        "input":        [],
        "conv1":        ["input"],
        "conv2_group2": {"deps": "conv1",        "split": 2, "weights": "conv2"},
        "conv3":        "conv2_group2",
        "conv4_group2": {"deps": "conv3",        "split": 2, "weights": "conv4"},
        "conv5_group2": {"deps": "conv4_group2", "split": 2, "weights": "conv5"},
        "fc6":          {"deps": "conv5_group2", "dump_mode": "expand_single"},
        "fc7":          {"deps": "fc6",          "dump_mode": "expand_single"},
        "fc8":          {"deps": "fc7",          "dump_mode": "expand_single"},

    As you can see "deps" describes primitive dependencies (for decalibration of weights).
    It can be specified directly in following forms:

        "<prim-1>": [],
        "<prim-2>": {},
        "<prim-3>": "<prim-dep>",
        "<prim-4>": ["<prim-dep>"],
        "<prim-5>": [["<prim-dep>"]],
        "<prim-6>": ["<prim-dep-1>", "<prim-dep-2>"],
        "<prim-7>": [["<prim-dep-1>"], ["<prim-dep-2>"]],

    In cases above primitives:
        #1, #2     - have no dependencies.
        #3, #4, $5 - is dependant on one primitive.
        #6         - is dependant on two primitives (with one dependency frontier - they
                     will generate / use one unified calibration factors set).
        #7         - is dependant on two primitives (in two frontiers - each has own
                     calibration set; should be used with proper decalibration mode).

    You can also use the same syntax "deps" if you add other options for primitive.

        {"deps": [["<prim-dep-1>"], ["<prim-dep-2>"]], ...},

    Other options for primitive (specified in JSON in the same way as "deps"):
        "decalibrate_mode", "decalib_mode" - Currently can be: "-" (default), "+", "*".
                                             Use mostly for eltwise-like layers.
                                             * "-" means no decalibration of calibration
                                                   factors (instead weights are
                                                   decalibrated) by calibration from
                                                   dependency frontiers.
                                             * "+" use first dependency frontier
                                                   to decalibrate (one frontier should
                                                   be in dependencies).
                                             * "*" apply decalibration from each
                                                   frontier to calibration factors.
                                             "+", "*" disable weights decalibration
                                             and decalibrate directly calibration
                                             factors. Also returned calibration
                                             factors for primitive cannot be used
                                             for for manual decalibration. Use
                                             script options "--add-raw-cf" to get
                                             non-decalibrated factors.
        "dump_mode"                        - Currently can be: "normal" (default),
                                             "expand_single".
                                             * "normal" mode process dump files as files
                                               describing single feature each.
                                             * "expand_single" expands each file and
                                               treats every value from file as single
                                               feature (there should be only one file per
                                               batched item). Used mainly with
                                               fully-connected primitives as they dumped
                                               this way.
        "split"                            - Number of feature groups the primitive is
                                             split into (split size).
                                             Default: 1 (no split)
        "weights"                          - Name (root part of .nnd file name).
                                             Default: <the same as primitive name>

    -----------------------------------------------------------------------------------------------
    ''')

    # Generic arguments:
    gen_parser = argparse.ArgumentParser(add_help=False)
    # gen_parser.add_argument('input',
    #                         nargs='*',
    #                         help='Path to input file or directory (either literal or wildcards). If directory is '
    #                              'specified,all files in the directory will be added as an input. If you want to '
    #                              'restrict files to specific names, please use -f and -fre options.')
    # gen_parser.add_argument('-f', '-filter', '--filter',
    #                         dest='include_filter',
    #                         metavar='pattern',
    #                         action='append',
    #                         default=[],
    #                         nargs='+',
    #                         help='Wildcard pattern that filters input file names. Files that do not match any '
    #                              'of filters (specified by -f or -fre) are omitted.')
    # gen_parser.add_argument('-fre', '-filter_re', '--filter_re',
    #                         dest='include_filter_re',
    #                         metavar='pattern',
    #                         action='append',
    #                         default=[],
    #                         nargs='+',
    #                         help='Regular expression pattern that filters input file names. Files that do not match '
    #                              'any of filters (specified by -f or -fre) are omitted. Matching is done in '
    #                              'case-sensitive manner (re.M and re.S modifiers are also applied).')
    gen_parser.add_argument('-v', '--verbose',
                            dest='verbose_level',
                            action='store_const',
                            const=2,
                            default=1,
                            help='Presents verbose output from script.')
    gen_parser.add_argument('-vv', '--very-verbose',
                            dest='verbose_level',
                            action='store_const',
                            const=3,
                            default=1,
                            help='Presents verbose output from script.')
    gen_parser.add_argument('-q', '--quiet',
                            dest='verbose_level',
                            action='store_const',
                            const=0,
                            default=1,
                            help='Suppress any output (including warnings). Only errors are presented.')
    gen_parser.add_argument('-V', '--version',
                            action='version',
                            version=_script_version)
    gen_parser.add_argument('-?', '-h', '--help',
                            action='help',
                            help='Shows this help message for "%(prog)s".')

    # Command "quantize" arguments:
    cmd_quant_parser = argparse.ArgumentParser(add_help=False)
    cmd_quant_parser.add_argument('opts_file',
                                  metavar='<quant-opts-file>',
                                  help='Path to file with calibration and quantization options. The file contains '
                                       'information in JSON format and describes primitives to calibrate / quantize, '
                                       'their dependencies, how to decalibrate them and how to access weights or dump '
                                       'files.')
    cmd_quant_parser.add_argument('dump_dir',
                                  metavar='<ex-dump-dir>',
                                  help='Path to dump directory containing all .txt files dumped for specified network '
                                       'using "examples" application (option "--dump_hidden_layers").')
    cmd_quant_parser.add_argument('weights_dir',
                                  metavar='<weights-dir>',
                                  help='Path to weights directory containing all .nnd files needed to quantize. '
                                       'Currently only weights that use FP32 (float) as data type are supported.')
    cmd_quant_parser.add_argument('-t', '--type', '--data-type',
                                  dest='data_type',
                                  type=str.lower,
                                  metavar='<data-type>',
                                  default='int8',
                                  choices=_data_type_map.keys(),
                                  help='Data type to which weights will be quantized. Must be one of: {0}. '
                                       'Default: %(default)s.'.format(', '.join(sorted(_data_type_map.keys()))))
    cmd_quant_parser.add_argument('-o', '--output-dir',
                                  dest='output_dir',
                                  metavar='<output-dir>',
                                  default='.',
                                  help='Path to output directory for quantized weights, calibration factors '
                                       'and quantization factors. If it does not exist, it will be created. '
                                       'Defaults to: "%(default)s" (current working directory).')
    cmd_quant_parser.add_argument('-ids', '--incl-dump-sub-dirs',
                                  dest='recurse_dump',
                                  action='store_true',
                                  help='Additionally scans <ex-dump-dir> sub-directories for dump files (recursively).')
    cmd_quant_parser.add_argument('-iws', '--incl-weights-sub-dirs',
                                  dest='recurse_wg',
                                  action='store_true',
                                  help='Additionally scans <weights-dir> sub-directories for weights (recursively).')
    cmd_quant_parser.add_argument('-ar', '--add-raw-cf',
                                  dest='add_cf_raw',
                                  action='store_true',
                                  help='Adds additionally raw (non-decalibrated) calibration factors files to '
                                       'output directory. Files will have "nondecalib" attribute in file name. '
                                       'The additional files are returned for primitives which CF were '
                                       'decalibrated.')
    cmd_quant_parser.add_argument('-af', '--add-frontier-cf',
                                  dest='add_cf_frontier',
                                  action='store_true',
                                  help='Additionally scans <weights-dir> sub-directories for weights (recursively).')
    cmd_quant_parser.add_argument('-ulo', '--use-old-nnd-layout',
                                  dest='use_old_layout',
                                  action='store_true',
                                  help='If possible, uses old layout format (with encoded data type) when saving '
                                       '.nnd files.')
    cmd_quant_parser.add_argument('-n', '-nn', '--norm-names',
                                  dest='norm_names',
                                  action='store_true',
                                  help='Normalizes names of quantized weights and factors (of .nnd files). '
                                       'Names will be based on primitive names.')
    cmd_quant_parser.add_argument('-dwc', '--disable-weights-decalibration',
                                  dest='disable_wg_decalib',
                                  action='store_true',
                                  help='Disables decalibration phase for weights. The phase decalibrates weights with '
                                       'calibration factors form dependency primitive.')
    cmd_quant_parser.add_argument('-dwq', '--disable-weights-quantization',
                                  dest='disable_wg_quant',
                                  action='store_true',
                                  help='Disables quantization phase for weights. Weights will be using original data '
                                       'type.')

    # Parsing arguments:
    cmd_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[gen_parser],
        description=textwrap.dedent('''\
            Utility tool for examples application.
    
            The tool provides functionality to dump, convert and manipulate .nnd files.
            '''),
        add_help=False)
    subcmds = cmd_parser.add_subparsers(dest='subcmd_name',
                                        title='Available sub-commands')

    # - d/dump (dumping content of .nnd files):
    # subcmds.add_parser('d',
    #                    parents=[cmd_dump_parser, gen_parser],
    #                    add_help=False,
    #                    help='Dumps .nnd file in human-readable format.')
    # subcmds.add_parser('dump',
    #                    parents=[cmd_dump_parser, gen_parser],
    #                    add_help=False,
    #                    help='Dumps .nnd file in human-readable format.')

    # - q/quant/quantize (weights quantization):
    subcmds.add_parser('q',
                       parents=[cmd_quant_parser, gen_parser],
                       description='Calibrates and quantizes .nnd files into one of integral data formats.',
                       add_help=False,
                       help='Calibrates and quantizes .nnd files into one of integral data formats.')
    subcmds.add_parser('quant',
                       parents=[cmd_quant_parser, gen_parser],
                       description='Calibrates and quantizes .nnd files into one of integral data formats.',
                       add_help=False,
                       help='Calibrates and quantizes .nnd files into one of integral data formats.')
    subcmds.add_parser('quantize',
                       parents=[cmd_quant_parser, gen_parser],
                       description='Calibrates and quantizes .nnd files into one of integral data formats.',
                       add_help=False,
                       help='Calibrates and quantizes .nnd files into one of integral data formats.')

    # - about-opts-file (information about options file):
    subcmds.add_parser('about-opts-file',
                       parents=[gen_parser],
                       formatter_class=argparse.RawDescriptionHelpFormatter,
                       description=opts_file_desc,
                       add_help=False,
                       help='Information about options file.')

    return cmd_parser, {'about-opts-file': opts_file_desc}


def prepare_main():
    # type: () -> typing.NoReturn
    """
    Handles generic arguments and prepares for "main" function.

    :return: Parsed arguments.
    """
    cmd_parser, about_descriptions = create_cmd_parser()
    parsed_args = cmd_parser.parse_args()

    if 'subcmd_name' in parsed_args and parsed_args.subcmd_name.startswith('about-'):
        print about_descriptions[parsed_args.subcmd_name]
        exit(0)

    log_level = logging.WARNING
    if 'verbose_level' in parsed_args:
        if parsed_args.verbose_level >= 3:
            log_level = logging.DEBUG
        elif parsed_args.verbose_level == 2:
            log_level = logging.INFO
        elif parsed_args.verbose_level == 0:
            log_level = logging.ERROR

    logging.basicConfig(level=log_level)

    exit_code = 0
    try:
        exit_code = main(parsed_args)
    except Exception as ex:
        if log_level <= logging.INFO:
            _logger.exception(ex)
        else:
            _logger.error(ex)
        exit(5)
    exit(exit_code)


def main(parsed_args):
    # type: (argparse.Namespace) -> int
    """
    "main" function. Performs tasks for .nnd tool.

    :param parsed_args: Parsed arguments.
    :return:            Exit code.
    """
    if 'subcmd_name' not in parsed_args:
        return 0

    if parsed_args.subcmd_name in ['q', 'quant', 'quantize']:
        return main_quantize(parsed_args)

    return 0


def main_quantize(parsed_args):
    # type: (argparse.Namespace) -> int
    """
    "quantize" handler function. Performs quantization tasks for .nnd tool.

    :param parsed_args: Parsed arguments.
    :return:            Exit code.
    """

    _logger.info('Starting quantization...')
    _logger.info('    Using options file (data type: {0})     : "{1}"'
                 .format(parsed_args.data_type, parsed_args.opts_file))
    _logger.info('    Using dump directory    (recursive: {0}): "{1}"'
                 .format('yes' if parsed_args.recurse_dump else ' no', parsed_args.dump_dir))
    _logger.info('    Using weights directory (recursive: {0}): "{1}"'
                 .format('yes' if parsed_args.recurse_wg else ' no', parsed_args.weights_dir))
    _logger.info('    Using output directory                  : "{0}"'
                 .format(parsed_args.output_dir))

    calibrator = CalibCalculator(
        dump_dir_path=parsed_args.dump_dir,
        output_dir_path=parsed_args.output_dir,
        dump_dir_incl_subdirs=parsed_args.recurse_dump,
        calib_data_type=_data_type_map[parsed_args.data_type],
        weights_dir_path=parsed_args.weights_dir,
        weights_dir_incl_subdirs=parsed_args.recurse_wg
    )

    # Namespace(add_cf_frontier=False, add_cf_raw=False, data_type='int8', disable_wg_decalib=False,
    # disable_wg_quant=False, dump_dir='b', norm_names=False, opts_file='a', output_dir='.', recurse_dump=False,
    # recurse_wg=False, subcmd_name='q', use_old_layout=False, verbose_level=2, weights_dir='c')
    calc_options  = CalibCalculator.CCO_SAVE_CALIB_FILES
    calc_options |= CalibCalculator.CCO_SAVE_WEIGHT_FILES
    if parsed_args.add_cf_frontier:
        calc_options |= CalibCalculator.CCO_SAVE_FRONTIER_FILES
    if parsed_args.add_cf_raw:
        calc_options |= CalibCalculator.CCO_ADD_RAW_CALIIB_FILES
    if parsed_args.use_old_layout:
        calc_options |= CalibCalculator.CCO_USE_OLD_NND_LAYOUT
    if parsed_args.norm_names:
        calc_options |= CalibCalculator.CCO_UNIFY_WG_NAMES
    if parsed_args.disable_wg_decalib:
        calc_options |= CalibCalculator.CCO_OMIT_WEIGHTS_DECALIB
    if parsed_args.disable_wg_quant:
        calc_options |= CalibCalculator.CCO_OMIT_WEIGHTS_QUANT

    calibrator.calculate_calibration_factors(
        calib_opts_file_path=parsed_args.opts_file,
        calc_opts=calc_options
    )

    return 0


if __name__ == '__main__':
    prepare_main()

    # ------------------------------------------------------------------------------------------------------------------
    # Tests:
    # print repr(parsed_args)
    # print repr(collect_files(parsed_args, 'input', 'include_filter'))
    # nnd_file = NndFile.from_file(r"C:\Users\mwalkowi\Desktop\alexnet\resnet50\fc1000_weights.nnd")
    # nnd_file.convert_data_type()
    # nnd_file.to_file('test_fc1000_weights.nnd')
    #
    # nnd_file2 = NndFile.from_file(r"C:\Users\mwalkowi\Desktop\alexnet\resnet50\fc1000_bias.nnd")
    # nnd_file2.convert_data_type()
    # nnd_file2.to_file('test_fc1000_bias.nnd')

    # a = CalibCalculator.__list_dump_files(r"C:\Users\mwalkowi\Desktop\alexnet\cldnn_dumps\1528448067")
    # print len(a)
    # print len(b)
    # a.sort()
    # b.sort()
    # print a
    # print b

    # a = CalibCalculator.__list_dump_files(r"C:\Users\mwalkowi\Desktop\alexnet\cldnn_dumps\1528448067")
    # print CalibCalculator.get_dump_files_for_primitive(a, 'conv2_group2', 2)

    # test = """
    # {
    #     "<prim1>": ["<prim2>"],
    #     "<prim2>": ["<prim01>"],
    #     "<prim3>": ["<prim1>", "<prim2>"],
    #     "<prim4>": {"deps": ["<prim3>"], "groups": 2, "decalib_mode": true},
    #     "<prim5>": {"deps": [{"<prim4>": 0}, {"<prim4>": 1}]},
    #     "<prim01>": []
    # }
    # """
    # test = """
    # {
    #     "A": ["C", "D"],
    #     "B": ["D", "E"],
    #     "C": "F",
    #     "D": {"deps": ["F", "H"], "decalibrate_mode": "-"},
    #     "E": {"deps": "G"},
    #     "F": ["H"],
    #     "G": [[]],
    #     "H": [],
    #     "I": [["E", {"J": 0}], [{"J": 3}]],
    #     "J": {"deps": [], "split": 4},
    #     "K": [{"J": 1}, {"J": 3}]
    # }
    # """
    # test = """
    # "conv1":        {"split": 3},
    # "conv2_group2": {"split": 8},
    # "conv3":        [{"conv2_group2": 1}, {"conv1": 1}],
    # """
    # norm_opts = CalibCalculator.__parse_calib_opts_from_json(test)
    # print norm_opts
    # CalibCalculator.__validate_calib_opts(norm_opts)
    # fronts = CalibCalculator('').__calculate_dominance_frontiers(norm_opts)
    # path = r'C:\Users\mwalkowi\Desktop\alexnet\cldnn_dumps\1528448067'
    # files = CalibCalculator('').__list_dump_files(path)
    # print CalibCalculator('').__max_abs_for_frontier(path, files, 6, fronts[6])
