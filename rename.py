import shutil
import os
import sys


if len(sys.argv) != 3:
    print('Not the right number of arguments')
    print('Example usage: python3 renamefiles.py parent_dir_name file_extension')
    exit(-1)


for root, dirs, files in os.walk(sys.argv[1]):
    for file in files:
        if file.endswith(sys.argv[2]):
            # RENAME FILES WITH PARENT_DIR_NAME + FILE_NAME.sys.argv[2]
            spl = root.split("/"); newname = spl[-1]+'_'+file; sup = ("/").join(spl[:-1])
            shutil.move(root+"/"+file, sup+"/"+newname)


            # REMOVE PARENT DIR            
            # checking whether the folder is empty or not
            if len(os.listdir(root)) == 0:
                # removing the file using the os.remove() method
                os.rmdir(root)


