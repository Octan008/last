import glob
import os
import shutil

folders = glob.glob('/src/last/#log_trash/*')
# folders = glob.glob('/src/last/#refreshlog/*')
for folder in folders:
    files = glob.glob(folder + '/*/**/*.png', recursive=True)
    if len(files) == 0:
        pass
        # print(folder, len(files))
        shutil.rmtree(folder)
    else:
        pass
        # print(folder, len(files))