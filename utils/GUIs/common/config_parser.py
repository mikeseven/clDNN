import os
import ConfigParser

class ConfigParserWrapper(object):
    def __init__(self, config_file_name=None):
        self.config_file_name = \
            config_file_name if config_file_name else os.path.join(os.getcwd(), 'conf.ini')


    def read(self, section):
        res = {}
        try:
            if os.path.exists(self.config_file_name):
                parser = ConfigParser.SafeConfigParser()
                parser.optionxform = str
                parser.read(self.config_file_name)
                for k,v in parser.items(section):
                    res[k] = v
        except:
            pass
        return res

    def write(self, section, params):
        try:
            parser = ConfigParser.SafeConfigParser()
            parser.optionxform = str
            if os.path.exists(self.config_file_name):
                parser.read(self.config_file_name)
            if parser.has_section(section):
                parser.remove_section(section)
            parser.add_section(section)
            for k, v in params.items():
                parser.set(section, k, v)

            with open(self.config_file_name, "w") as f:
                parser.write(f)
        except:
            pass


