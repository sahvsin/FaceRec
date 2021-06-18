# Linux Face Verification
A Face Verification system built in TensorFlow + Keras and OpenCV using chmod to control access to files (only in Linux).
<br><br>

![](./GIFs/LAC.gif)
<p align="center">
  Initially the file (Test.py) is locked ("X" in file manager on the right) and then unlocks after verifying the user against the image of the authorized user, a red Bubly can. Then the system rejects an invalid user (green Bubly) and re-locks the file ("X" returns). 
</p>

<br><br>
First have the user type their name and the system will first check if that name is among those in the database of verified users. If not, the system restricts access to the specified file(s) using the Python [OS](https://docs.python.org/3/library/os.html) module to invoke the Unix [chmod](https://www.computerhope.com/unix/uchmod.htm) command.

Otherwise, the system takes a live snapshot with a connected camera using [OpenCV](https://opencv.org/). If it can't detect a camera, the system exits.

After taking a snapshot, it builds Google's [FaceNet](https://arxiv.org/abs/1503.03832) model in [Keras](https://keras.io/) using Adam Optimization and a custom triplet loss function built in [TensorFlow](https://www.tensorflow.org/). Then the system loads pre-trained parameters found online onto the model to expedite its training.

With the model trained, the system builds a database of 128-point vector encodings by processing each verified user's image through the FaceNet model.

Finally, the system verifies the user by passing their taken image through the FaceNet model and computes the Euclidean distance (L2 norm) between its encoding and the associated encoding in the database. If the distance is below a threshold, the user is verified and the files become unlocked via chmod. Otherwise they get locked out. 

<br><br>
## Credits

Please make sure to check out the CREDITS.txt file.  I couldn't have done this project without resources/papers provided by those credited.

--Credit to Schroff et al. for their research and paper on FaceNet, the model I use
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;link to paper - https://arxiv.org/abs/1503.03832

--Credit to the CMU OpenFace team for providing open-source pre-trained weights for FaceNet
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Weights are derived from training on CASIA-WebFace and FaceScrub datasets (total of 500k images)
