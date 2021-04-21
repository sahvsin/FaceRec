import cv2



def take_snapshot():
    
    '''
    capture video stream from (defualt) connected camera and save a snapshot of a frame of the stream
    '''

    cap = cv2.VideoCapture(0)
    filename = 'images/camera/camera_0.jpg'
    print("CONTROLS:\n")
    print("PRESS \"p\" key to PAUSE the recording\n")
    print("PRESS \"c\" key to CAPTURE an image\n")
    print("PRESS \"q\" key to QUIT the video stream/feed\n")

    while(True):

        #capture frame-by-frame
        ret, frame = cap.read()
        
        #key detection
        key = cv2.waitKey(1) & 0xFF

        if not ret:
            break

        cv2.imshow('webcam', frame)

        #pause the capture (blocking code)
        if key == ord('p'):
            while(True):

                #key detection
                key2 = cv2.waitKey(1) or 0xFF
                
                #show the same frame indefinetely until p is pressed again to resume capture
                cv2.imshow('webcam', frame)

                if key2 == ord('p'):
                    break
                #save the image to file if c is pressed
                elif key2 == ord('c'):
                    cv2.imwrite(filename, frame)

        
        #end the capture after q is pressed
        if key == ord('q'):
            break
        #save the image to file if c is pressed
        elif key == ord('c'):
            cv2.imwrite(filename, frame)

    #when all is done (break from webcam capturing), release the capture
    cap.release()
    cv2.destroyAllWindows()
