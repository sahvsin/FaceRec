import os


def change_permissions(path, mode):
    '''
    Changes file access permissions of a single file using CHMOD

    Inputs:
    path -- path of the file whose permission you wish to change
    mode -- accepts up to 4 digits (in octal, base 8) determining permission/access to the file
            1st digit: special flag for "setuid", "setgid", "sticky" options
            2nd digit: change user (file owner) permissions
            3rd digit: change group (user's group) permissions
            4th digit: other (non-user/non-group) permissions
            0: None (no read, write, execution permission)
            1: execute only
            2: write only
            3: write and execute only
            4: read only
            5: read and execute
            6: read and write
            7: read, write, and execute
    '''

    os.chmod(path, mode)


def recursive_change_permissions(path, mode):
    '''
    Change file access permissions of an entire directory, its subdirectories, and files
    '''

    for root, dirs, files in os.walk(path, topdown=False):
        for directory in [os.path.join(root, d) for d in dirs]:
            os.chmod(directory, mode)
            print(directory)
    for a_file in [os.path.join(root, f) for f in files]:
        os.chmod(a_file, mode)
        print(a_file)



#path = input("What directory/file?\n")
#mode = int(input("What permission? (octal, start with 0o)\n"), 8)

#change_permissions(path, mode)
#recursive_change_permissions(path, mode)








