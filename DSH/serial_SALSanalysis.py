import sys
import os
import shutil
import logging
from DSH import SALS as LS

logger = logging.getLogger()
#fhandler = logging.FileHandler(filename='mylog.log', mode='a')
fhandler = logging.FileHandler(filename='serial_SALS.log', mode='w')
formatter = logging.Formatter('%(asctime)s - %(name)s | %(levelname)s : %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)

ConfigFile = None
strerr = 'no error'
if (len(sys.argv) > 1):
    tmp_config = str(sys.argv[1]).strip()
    if os.path.isfile(tmp_config):
        ConfigFile = tmp_config
    else:
        strerr = 'Configuration file read from command line (' + str(tmp_config) + ') not found'
else:
    strerr = 'Configuration file needs to be specified in the command line'
        
if ConfigFile is not None:
    logging.info('Loading analysis parameters from file ' + str(ConfigFile))
    SALS_analyzer = LS.LoadFromConfig(ConfigFile, outputSubfolder=None)
    logging.info('Analysis ended')
else:
    logging.error(strerr)
    print(strerr)

shutil.copyfile('serial_SALS.log', os.path.join(SALS_analyzer.LastSaveFolder, 'SALS_analysis.log'))