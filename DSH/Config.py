import os
import configparser
import collections
import logging
import numpy as np

from DSH import SharedFunctions as sf
    
def Duplicate(other_config):
    """ Creates a duplicate of a Config object
    """
    new_config = Config()
    new_config.Import(other_config.ToDict(section=None), section_name=None)
    return new_config

def ExportDict(dict_to_export, out_filename, section_name=None):
    conf = Config()
    conf.Import(dict_to_export, section_name=section_name)
    conf.Export(out_filename)

def LoadMetadata(MetaData, SectionName=None, DefaultFiles=[]):
    
    """Reads metadata file
    it also reads the default configuration file
    in case of duplicates, information from MetaDataFile is used
    
    Parameters
    ----------
    MetaData : dict or filename
    SectionName : if MetaData is a dictionnary, eventually load subsection of configuration parameters
    DefaultFiles : list of full path containing default configuration parameters
    
    Returns
    -------
    outConfig : Config object containing desired metadata
    """
    if (type(MetaData) in [dict, collections.OrderedDict]):
        logging.debug('Appending input dictionary to section ' + str(SectionName))
        outConfig = Config(None, defaultConfigFiles=DefaultFiles)
        outConfig.Import(MetaData, section_name=SectionName)
    else:
        outConfig = Config(MetaData, defaultConfigFiles=DefaultFiles)
        logging.debug('Loading config file ' + str(MetaData) + ' (' + str(outConfig.CountSections()) + 
                      ' sections, ' + str(outConfig.CountKeys()) + ' keys)')
    
    return outConfig

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
            if (os.path.isfile(conf_f)):
                self.config.read(conf_f)
            else:
                raise IOError('Configuration file ' + str(conf_f) + ' not found')
        if (ConfigFile is not None):
            if (os.path.isfile(ConfigFile)):
                self.config.read(ConfigFile)
            else:
                raise IOError('Configuration file ' + str(ConfigFile) + ' not found')

    def __repr__(self):
        return '<Config class: %s sections, %s keys>' % (self.CountSections(), self.CountKeys())
    
    def __str__(self):
        str_res  = '\n|---------------|'
        str_res += '\n| Config class: |'
        str_res += '\n|---------------+---------------'
        str_res += '\n| Section count : ' + str(self.CountSections())
        str_res += '\n| Total keys    : ' + str(self.CountKeys())
        for sect_name in self.GetSections():
            sect_keys = self.config._sections[sect_name].keys()
            str_res += '\n| ' + str(sect_name).ljust(14) + ': <' + str(len(sect_keys)) + ' keys>'
            for cur_key in sect_keys:
                str_res += '\n| ' + ' '.ljust(16) + str(cur_key) + ' = ' + self.config._sections[sect_name][cur_key]
        str_res += '\n|---------------+---------------'
        return str_res
    
    def Import(self, dict_config, section_name='default'):
        """Imports all dictionary entries into a given section
        
        Parameters
        ----------
        dict_config : dictionary to be imported
        section_name : section name (string). 
                            keys in dict_config will become options of section section_name
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
            return sf.StrParse(self.config[sect][key], cast_type)
        else:
            if not silent:
                print('"' + key + '" not found in section "' + sect + '": default value ' + str(default) + ' returned.')
            return default
    
    def HasOption(self, sect, key):
        return self.config.has_option(sect, key)
    
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
            return dict(self.config._sections)
        else:
            if (self.config.has_section(section)):
                return dict(self.config._sections[section])
            else:
                print('WARNING: section not found in current configuration. Available sections are: ' + str(self.GetSections()))
                return {}
    
    def CountSections(self):
        return len(self.config.sections())
    
    def CountKeys(self):
        num_keys = 0
        for cur_sect in self.config.sections():
            num_keys += len(self.config._sections[cur_sect].keys())
        return num_keys
