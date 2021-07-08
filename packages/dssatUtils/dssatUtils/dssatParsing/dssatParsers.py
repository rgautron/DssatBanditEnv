"""
DSSAT .SOIL/.CLI input file parser
Inspired from https://scipy.github.io/old-wiki/pages/Cookbook/Reading_Custom_Text_Files_with_Pyparsing.html
and https://github.com/pyparsing/wikispaces_archive/blob/master/discussion/all_wiki_discussion_toc_2012.md
"""
from pyparsing import *
import numpy as np
import ast

ParserElement.setDefaultWhitespaceChars(' \t')


class DssatIntegrationParser():
    """
    Utility class to read DSSAT specific internal input_files
    """

    def __init__(self, soil_file_path, weather_file_path):
        self.soil_file_path = soil_file_path
        self.weather_file_path = weather_file_path
        self.inner_block = None
        self.outer_block = None

    def get_grammar(self, weather_or_soil):
        """
        Define the grammar to read DSSAT text input_files
        @param weather_or_soil: if weather or soil is considered
        @type weather_or_soil: string in ['weather','soil']
        @return: nothing
        @rtype: None
        """

        def get_col_nb(header):
            feature_values << Word(printables) * len(header) + EOL

        EOL = LineEnd().suppress()

        feature_values = Forward()

        self.outer_block_begin = Literal('*')
        if weather_or_soil == 'weather':
            outer_block_id = Combine(Word(alphanums) + Optional(OneOrMore(' ' + Word(alphanums))))
        else:
            outer_block_id = Combine(Word(alphanums) + Optional(OneOrMore('_' + Word(alphanums)))) + \
                             Suppress(restOfLine)

        inner_block_begin = Literal('@')
        inner_block_id = OneOrMore(Word(alphanums)) + EOL
        inner_block_prefix = Group(feature_values)

        if weather_or_soil == 'weather':  # optional if
            self.inner_block = \
                Suppress(SkipTo(inner_block_begin, include=True)) + \
                Group(Group(inner_block_id.setParseAction(get_col_nb)) + Group(ZeroOrMore(inner_block_prefix)))
        else:
            self.inner_block = \
                Optional(Suppress(SkipTo(Literal('@SCOM'), include=False))) + \
                Suppress(SkipTo(inner_block_begin, include=True)) + \
                Group(Group(inner_block_id.setParseAction(get_col_nb)) + Group(ZeroOrMore(inner_block_prefix)))

        self.inner_block.ignore(('!' + restOfLine))

        self.outer_block = \
            Suppress(self.outer_block_begin) + \
            outer_block_id + \
            Group(ZeroOrMore(self.inner_block))

        self.outer_block.ignore(('!' + restOfLine))

    def isOuterBegin(self, line):
        """
        Utility function for parsing to detect an outer block
        @param line: the text line to be evaluated
        @type line: list of strings
        @return: True/False
        @rtype: bool
        """
        bList = self.outer_block_begin.searchString(line).asList()
        if len(bList) > 0:
            return True
        return False

    def parse(self, text):
        """
        Utility function to parse the whole file
        @param text: the text to be parsed
        @type text: string
        @return: list of parsed element
        @rtype: list of strings
        """
        strList = []
        blockStr = ''
        inBlock = False
        for line in text.splitlines():
            if self.isOuterBegin(line):  # Start new outer block
                if len(blockStr) > 0 and inBlock:  # Close out previous block
                    strList.append(blockStr)
                    blockStr = ''
                else:
                    inBlock = True
            if inBlock:
                blockStr += line + '\n'

        if inBlock and len(blockStr) > 0:  # Close out final block
            strList.append(blockStr)

        pList = []
        for blockStr in strList:
            bList = self.outer_block.searchString(blockStr).asList()
            pList.append(bList[0])
        return pList

    def digest_parse(self, pList):
        """
        Convert parsing results into a dictionary
        @param pList: parsed element
        @type pList: list of strings
        @return: dictionary of results
        @rtype: dictionary
        """
        dic_res = {}
        for section in pList[1:]:
            section_name = section[0]
            dic_res[section_name] = {}
            for table in section[1]:
                header = table[0]
                values = np.array(table[1]).T
                for feature, value in zip(header, values):
                    dic_res[section_name][feature] = np.array(list(cast_list_of_strings(value)))  # casts values
        return dic_res

    def get_dic_from_file(self, weather_or_soil):
        """
        Load input_files to parse from their path
        @param weather_or_soil: if weather or soil is considered
        @type weather_or_soil: string in ["weather","soil"]
        @return: dictionary result of the parsing
        @rtype: dictionary
        """
        if weather_or_soil == 'weather':
            path = self.weather_file_path
        elif weather_or_soil == 'soil':
            path = self.soil_file_path
        else:
            raise ValueError('weather_or_soil takes values in ["weather","soil"]')

        with open(path, mode='r', encoding="utf8", errors='ignore') as f_:
            data = f_.read()
        self.get_grammar(weather_or_soil)
        pList = self.parse(data)
        dic_res = self.digest_parse(pList)
        return dic_res


class DssatExpFileParser(DssatIntegrationParser):
    """
    Utility class to read DSSAT specific internal input_files
    """

    def __init__(self):
        self.inner_block = None
        self.outer_block = None

    def get_grammar(self):
        """
        Define the grammar to read DSSAT text input_files
        @param weather_or_soil: if weather or soil is considered
        @type weather_or_soil: string in ['weather','soil']
        @return: nothing
        @rtype: None
        """

        def get_col_nb(header):
            feature_values << Word(printables) * len(header) + EOL

        EOL = LineEnd().suppress()

        feature_values = Forward()

        self.outer_block_begin = Literal('*')
        outer_block_id = Combine(Word(alphanums) + Optional(OneOrMore(' ' + Word(alphanums))))

        inner_block_begin = Literal('@')
        inner_block_id = OneOrMore(Word(alphanums + '%' + '#')) + EOL
        inner_block_prefix = Group(feature_values)

        self.inner_block = \
            Suppress(SkipTo(inner_block_begin, include=True)) + \
            Group(Group(inner_block_id.setParseAction(get_col_nb)) + Group(ZeroOrMore(inner_block_prefix)))

        self.inner_block.ignore(('!' + restOfLine))

        self.outer_block = \
            Suppress(self.outer_block_begin) + \
            outer_block_id + \
            Group(ZeroOrMore(self.inner_block))

        self.outer_block.ignore(('!' + restOfLine))

    def digest_parse(self, pList):
        """
        Convert parsing results into a dictionary
        @param pList: parsed element
        @type pList: list of strings
        @return: dictionary of results
        @rtype: dictionary
        """
        dic_res = {}
        for section in pList:
            section_name = section[0]
            dic_res[section_name] = {}
            for table in section[1]:
                header = table[0]
                values = np.array(table[1]).T
                for feature, value in zip(header, values):
                    if feature in dic_res[section_name]:
                        feature += '_2'
                    dic_res[section_name][feature] = np.array(list(cast_list_of_strings(value)))  # casts values
        return dic_res

    def get_dic_from_file(self, path):
        """
        Load input_files to parse from their path
        @param weather_or_soil: if weather or soil is considered
        @type weather_or_soil: string in ["weather","soil"]
        @return: dictionary result of the parsing
        @rtype: dictionary
        """
        with open(path, mode='r', encoding="utf8", errors='ignore') as f_:
            data = f_.read()
        self.get_grammar()
        pList = self.parse(data)
        dic_res = self.digest_parse(pList)
        return dic_res


def cast_list_of_strings(L):
    """
    Casts a list of strings into their relative types, error tolerant
    from https://stackoverflow.com/questions/2859674/converting-python-list-of-strings-to-their-type
    @param L: the strings
    @type L: list
    @return: casted strings
    @rtype: list
    """

    def tryeval(val):
        try:
            val = ast.literal_eval(val)
        except:
            pass
        return val

    return [tryeval(x) for x in L]


if __name__ == '__main__':
    pass