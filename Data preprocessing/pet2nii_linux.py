import os
import errno

'''
This is for to make .bash file to run in linux
upet2mnc --> mnc2nii
after run write.bash file in linux
you gonna get original_nii,float_nii,double_nii,byte_nii,int_nii,signed_nii,unsigned_nii files
'''

print('current loc : ', os.getcwd())



home_dir = '/home/bmssa'
os.system('cd %s' %home_dir)
os.system('ls %s' %home_dir)

# check_dir = '/home/bmssa/what'
# os.system('mkdir %s' %check_dir)

dir_Dataset = os.path.join(home_dir, 'Dataset')
dir_Rat = os.path.join(dir_Dataset, 'Rat')
dir_Rat_KHI = os.path.join(dir_Rat, 'RatData_KimHyungIhl')
folders1 = os.listdir(dir_Rat_KHI)


def mkdir_func(dir_name):
    '''

    :param dir_name: directory name
    :return: x
    '''
    try:
        os.mkdir(dir_name)
        print('folder {} created '.format(dir_name))
    except FileExistsError:
        print(dir_name, 'directory already exists')
    except FileNotFoundError:
        print(dir_name, 'no such file or directory')

def get_files_by_file_size(dirname, reverse=False):
    #dirname= dir_folder1
    """ Return list of file paths in directory sorted by file size """

    # Get list of files
    filepaths = []
    for basename in os.listdir(dirname):
        filename = os.path.join(dirname, basename)
        if os.path.isfile(filename):
            filepaths.append(filename)

    # Re-populate list with filename, size tuples
    for i in range(len(filepaths)):
        filepaths[i] = (filepaths[i], os.path.getsize(filepaths[i]))

    # Sort list by file size
    # If reverse=True sort from largest to smallest
    # If reverse=False sort from smallest to largest
    filepaths.sort(key=lambda filename: filename[1], reverse=reverse)

    # Re-populate list with just filenames
    for i in range(len(filepaths)):
        filepaths[i] = filepaths[i][0]

    return filepaths

new_root = 'Rat_upet2nii'
dir_new1_name = os.path.join(dir_Rat, new_root)
mkdir_func(dir_new1_name)


# file_array = get_files_by_file_size(dir_folder1, reverse=False)
# len(file_array)
# for i in file_array:
#     dir_file1 = os.path.join(dir_folder1, i)
#     print(i, os.stat(dir_file1).st_size)

dir_write = os.path.join(dir_Rat, 'write1.bash')
f = open(dir_write, 'w')

for folder in folders1:
    dir_folder1 = os.path.join(dir_Rat_KHI, folder)
    # print(dir_folder1) #'/home/bmssa/Dataset/Rat/RatData_KimHyungIhl/140811_NV_Opto_140708-3'
    # get_files_by_file_size(dir_folder1, reverse=False)
    if os.path.isdir(dir_folder1)==True:
        print(folder) #140811_NV_Opto_140708-3
        folders2 = os.listdir(dir_folder1)

        #make new directory
        dir_new_folder1 = os.path.join(dir_new1_name, folder)
        mkdir_func(dir_new_folder1)

        for file1 in folders2:
            dir_file1 = os.path.join(dir_folder1, file1)
            # print(':', os.stat(dir_file1).st_size)
            if os.path.isdir(dir_file1) ==True:
                files2 = os.listdir(dir_file1)
                # print('   --folder', file1,':', os.stat(dir_file1).st_size)


                for file2 in files2:
                    dir_file2 = os.path.join(dir_file1, file2)
                    # print('      ----', file2, ':', os.stat(dir_file1).st_size)
                    if file2[-4:]=='.img':
                        # print('   == img fold :', file2)
                        file2_img = file2
                        dir_file2_img = dir_file2
                        dic_new_mnc = os.path.join(dir_new_folder1, file2[:6] + ".mnc")
                        f.write('upet2mnc {} {}\n'.format(dir_file2, dic_new_mnc))
                        f.write('mnc2nii {} {}\n'.format(dic_new_mnc, dic_new_mnc[:-4]+'_original.nii'))
                        f.write('mnc2nii -float {} {}\n'.format(dic_new_mnc, dic_new_mnc[:-4]+'_float.nii'))
                        f.write('mnc2nii -double {} {}\n'.format(dic_new_mnc, dic_new_mnc[:-4]+'_double.nii'))
                        f.write('mnc2nii -byte {} {}\n'.format(dic_new_mnc, dic_new_mnc[:-4]+'_byte.nii'))
                        f.write('mnc2nii -int {} {}\n'.format(dic_new_mnc, dic_new_mnc[:-4]+'_int.nii'))
                        f.write('mnc2nii -signed {} {}\n'.format(dic_new_mnc, dic_new_mnc[:-4]+'_signed.nii'))
                        f.write('mnc2nii -unsigned {} {}\n'.format(dic_new_mnc, dic_new_mnc[:-4]+'_unsigned.nii'))
                        f.write('\n')

            elif os.path.isfile(dir_file1)==True:
                # print('   --file  ', file1)
                if file1[-4:] == '.img':
                    print('   == img file :', file1)
                    file1_img = file1
                    dir_file2_img = dir_file1
                    if 'static' in file1:
                        print('   ==== static  : ', file1)
                        dic_new_mnc = os.path.join(dir_new_folder1, 'static.mnc')
                    elif 'Dynamic' in file1:
                        print('   ====Dynamic : ', file1)
                        dic_new_mnc = os.path.join(dir_new_folder1, 'static.mnc')
                    else:
                        print('   ====nothing : ', file1)
                        dic_new_mnc = os.path.join(dir_new_folder1, 'nothing.mnc')
                    f.write('upet2mnc {} {}\n'.format(dir_file1, dic_new_mnc))
                    f.write('mnc2nii {} {}\n'.format(dic_new_mnc, dic_new_mnc[:-4] + '_original.nii'))
                    f.write('mnc2nii -float {} {}\n'.format(dic_new_mnc, dic_new_mnc[:-4] + '_float.nii'))
                    f.write('mnc2nii -double {} {}\n'.format(dic_new_mnc, dic_new_mnc[:-4] + '_double.nii'))
                    f.write('mnc2nii -byte {} {}\n'.format(dic_new_mnc, dic_new_mnc[:-4] + '_byte.nii'))
                    f.write('mnc2nii -int {} {}\n'.format(dic_new_mnc, dic_new_mnc[:-4] + '_int.nii'))
                    f.write('mnc2nii -signed {} {}\n'.format(dic_new_mnc, dic_new_mnc[:-4] + '_signed.nii'))
                    f.write('mnc2nii -unsigned {} {}\n'.format(dic_new_mnc, dic_new_mnc[:-4] + '_unsigned.nii'))
                    f.write('\n')
        print()
        f.write('\n')
print()
f.write('\n')
f.close()

# dir_write = os.path.join(dir_Rat, 'write.bash')
# f = open(dir_write, 'w')
# f.write('This\n')


# os.system("~/.bashrc");
# os.system("source /opt/minc/1.9.18/minc-toolkit-config.sh")
#
# os.system("sudo source /opt/minc/1.9 p.18/minc-toolkit-config.sh")
#
# import subprocess as sub
# sub.Popen('/opt/minc/1.9.18/minc-toolkit-config.sh', encoding='utf-8')
#
# subprocess.call(["/opt/minc/1.9.18/minc-toolkit-config.sh"])
#
# subprocess.Popen('opt/minc/1.9.18/minc-toolkit-config.sh', shell=True)
# # subprocess.run('upet2mnc #!%s #!%s' %(dir_file2_img ,dic_new_mnc), shell=True, check=True)
# subprocess.run('upet2mnc %s %s' %(dir_file2_img ,dic_new_mnc), shell=True, check=True)
#
#
# os.system('source /opt/minc/1.9.18/minc-toolkit-config.sh')
# os.system('upet2mnc %s %s' %(dir_file2_img ,dic_new_mnc))
#
# upet2mnc 140811_NV_Opto_140708-1_Atn_1500s_static_em_v1.pet.img test.mnc
# os.system("upet2mnc '/home/bmssa/Dataset/Rat/RatData_KimHyungIhl/140811_StV_Stim_140709-3/140811_StV_Stim_140709-3_Atn_1500s_static_em_v1.pet.img' '/home/bmssa/Dataset/Rat/Rat_upet2nii/140811_StV_Stim_140709-3/static.mnc'")
#
# upet2mnc /home/bmssa/Dataset/Rat/RatData_KimHyungIhl/140811_StV_Stim_140709-3/140811_StV_Stim_140709-3_Atn_1500s_static_em_v1.pet.img /home/bmssa/Dataset/Rat/Rat_upet2nii/140811_StV_Stim_140709-3/static.mnc': 'upet2mnc /home/bmssa/Dataset/Rat/RatData_KimHyungIhl/140811_StV_Stim_140709-3/140811_StV_Stim_140709-3_Atn_1500s_static_em_v1.pet.img /home/bmssa/Dataset/Rat/Rat_upet2nii/140811_StV_Stim_140709-3/static.mnc
# dir_file2_img
# dic_new_mnc
#
# print('upet2mnc %s %s' %(dir_file2_img ,dic_new_mnc))
# print('upet2mnc {} {}'.format(dir_file2_img ,dic_new_mnc))
# os.system('upet2mnc {} {}'.format(dir_file2_img ,dic_new_mnc))
# #weird
#
# weird_dir = '/home/bmssa/Dataset/Rat/RatData_KimHyungIhl/140812_NV_Opto_140708-9Atn'
# weird_files = os.listdir(weird_dir)
#
# '/home/bmssa/Dataset/Rat/Rat_upet2nii/140811_StV_Stim_140709-3/static.mnc'
# '/home/bmssa/Dataset/Rat/RatData_KimHyungIhl/140811_StV_Stim_140709-3/140811_StV_Stim_140709-3_Atn_1500s_static_em_v1.pet.img'
#
# os.system("upet2mnc '/home/bmssa/Dataset/Rat/RatData_KimHyungIhl/140811_StV_Stim_140709-3/140811_StV_Stim_140709-3_Atn_1500s_static_em_v1.pet.img' '/home/bmssa/Dataset/Rat/Rat_upet2nii/140811_StV_Stim_140709-3/static.mnc'")
# a='ls'
# b=dir_file2_img
# os.system('%s %s'%(a, dir_Rat_KHI))
git push --set-upstream origin master

git remote set-url origin
git push --set-upstream origin master
