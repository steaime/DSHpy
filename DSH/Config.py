import configparser
import numpy as np
import pkg_resources
pkg_installed = {pkg.key for pkg in pkg_resources.working_set}
if 'json' in pkg_installed:
    import json
    use_json = True
else:
    import ast
    use_json = False

class Config():
    """Class that develops on configparser with customized methods"""
    
    def __init__(self, ConfigFile=None, defaultConfigFiles=[]):
        """Initializes the configfile class
        
        Parameters
        ----------
        ConfigFile : filename of main configuration file
        defaultConfigFiles : list of filenames with default configurations
                        parameters not present in the main configuration file
                        will be loaded from the default configuration files
        """
        self.config = configparser.ConfigParser(allow_no_value=True)
        for conf_f in defaultConfigFiles:
            self.config.read(conf_f)
        if (ConfigFile is not None):
            self.config.read(ConfigFile)

    def Import(self, dict_config, section_name='default'):
        """Imports all dictionary entries into a given section
        
        Parameters
        ----------
        dict_config : dictionary to be imported
        section_name : section name (string)
                    if None, keys in dict_config will be interpreted as section names
                    entries in dict_config must be dictionaries
        """
        if (section_name is None):
            for key in dict_config.keys():
                if (type(dict_config[key]) is dict):
                    if (not self.config.has_section(key)):
                        self.config.add_section(key)
                    for subkey in dict_config[key].keys():
                        self.config.set(key, subkey, str(dict_config[key][subkey]))
        else:
            if (not self.config.has_section(section_name)):
                self.config.add_section(section_name)
            for key in dict_config.keys():
                self.config.set(section_name, key, str(dict_config[key]))
    
    def Export(self, configFile):
        """Export all entries in configuration file
        
        Parameters
        ----------
        configFile : full path of configuration file
        """
        cfgfile = open(configFile,'w')
        self.config.write(cfgfile)
        cfgfile.close()

    def Get(self, sect, key, default=None, cast_type=None, silent=True):
        """Gets configuration entry
        
        Parameters
        ----------
        sect : section name (string)
        key : key name (string)
        default : variable to be returned if section/key is not present
        cast_type : if not None, cast variable to type
                    default type is string
        silent : output default without printing any warning message
        
        Returns
        -------
        variable. Can be dictionary or list.
        """
        if (self.config.has_option(sect, key)):
            res = self.config[sect][key]
            if (str(res)[0] in ['[','(', '{']):
                if use_json:
                    res = json.loads(res)
                else:
                    res = ast.literal_eval(res)
            if (type(res) in [list,tuple]):
                for i in range(len(res)):
                    if (type(res[i]) in [list,tuple]):
                        if (cast_type is not None):
                            for j in range(len(res[i])):
                                res[i][j] = cast_type(res[i][j])
                    else:
                        if (cast_type is not None):
                            res[i] = cast_type(res[i])
                        
                return res
            elif (cast_type is bool):
                return self.config.getboolean(sect, key)
            elif (cast_type is int):
                return self.config.getint(sect, key)
            elif (cast_type is float):
                if (res == 'nan'):
                    return np.nan
                else:
                    return self.config.getfloat(sect, key)
            else:
                if (cast_type is None):
                    return res
                else:
                    return cast_type(res)
        else:
            if not silent:
                print('"' + key + '" not found in section "' + sect + '": default value ' + str(default) + ' returned.')
            return default
    
    def GetSections(self):
        return self.config.sections()
    
    def Set(self, sect, key, value):
        self.config.set(sect, key, str(value))
        
    def ToDict(self, section=None):
        """Export to dictionary
        
        Parameters
        ----------
        section : section to export as a dictionary
                if None, export all sections
        """
        if (section is None):
            return self.config._sections
        else:
            return self.config._sections[section]