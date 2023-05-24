from glob import glob
import shutil

ABC_PATH = 'E:\\ABC_Dataset\\abc_0000_obj_v00'

folders = glob(ABC_PATH + '\\*')

for folder in folders:
    files = glob(folder + '\\*.obj')
    
    for file in files:
        shutil.move(file, ABC_PATH) 


