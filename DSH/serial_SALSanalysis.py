import sys
import os
import shutil
import logging
from DSH import SALS as LS

logger = logging.getLogger()
#fhandler = logging.FileHandler(filename='mylog.log', mode='a')
log_fname = 'serial_SALS.log'
for i in range(10000):
    if os.path.isfile(log_fname):
        log_fname = 'serial_SALS_' + str(i).zfill(4) + '.log'
    else:
        break
fhandler = logging.FileHandler(filename=log_fname, mode='w')
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

if "--debug" in sys.argv:
    debugMode = True
    logging.info('Debug mode activated from command line')
else:
    debugMode = False
        
if ConfigFile is not None:
    logging.info('Loading analysis parameters from file ' + str(ConfigFile))
    SALS_analyzer = LS.LoadFromConfig(ConfigFile, outputSubfolder=None, debugMode=debugMode)
    logging.info('Analysis ended')
else:
    logging.error(strerr)
    print(strerr)

shutil.copyfile(log_fname, os.path.join(SALS_analyzer.LastSaveFolder, log_fname))