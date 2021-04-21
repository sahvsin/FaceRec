import numpy as np
from img_utils import encode_img




def verify_image(path, new_h, new_w, identity, database, channel_order, frmodel, thresh=0.7):

    '''
    Computes the Euclidean distance (L2 norm) between the two image encodings to verify the current user
    
    Inputs:
    path -- path of the file whose permission you wish to change
    new_h -- the image's resized/new height
    new_w -- the image's resized/new width
    identity -- username input by the person to verify
    database -- hashmap mapping each authenticated user's name to the encoding of their respective image
    channel_order -- order of color channels to process the image (RGB vs BGR)
    frmodel -- the face recognition model
    thresh -- Distance/Norm threshold, the smaller the threshold the stricter the system

    Output:
    Boolean determining whether/not user is verified to be among the list of authenticated users
    '''

    #forward propagate the image across the model/network to get its encoding
    encoding = encode_img(path, new_h, new_w, channel_order, frmodel)

    #compute the distance/norm between the processed image and the referential one (stored in database)
    distance = np.linalg.norm(encoding-database[identity], axis=1)
    #print(distance)

    #compare the distance with the threshold (smaller = accept, bigger = deny)
    if distance < thresh:
        print("Accepted!  Hello, " + identity + "!")
        return True
    else:
        print("Denied! You are not " + identity + "!")
        return False
