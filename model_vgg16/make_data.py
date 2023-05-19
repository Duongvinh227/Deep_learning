import cv2
import os

while True:
    class_name = input("Please enter the product name (instant writing, no accents)/or press q to exit: ").strip()
    if class_name=="q":
        break
    else:
        frame_count = 0
        record = False
        cat_folder = os.path.join("train_data", class_name)
        print("Destination folder ", cat_folder)
        print("Press the R key to sample/stop sampling. Press Q to exit")

        import shutil
        if os.path.exists(cat_folder):
            shutil.rmtree(cat_folder)
        os.mkdir(cat_folder)

        cap = cv2.VideoCapture(1)
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, dsize=(640,640), fx=0.5, fy=0.5)
                frame = cv2.flip(frame,1)
                cv2.imshow('frame', frame)
            if record:
                # Write file
                dest_file = 'img' + str(frame_count) + ".png"
                dest_file = os.path.join(cat_folder, dest_file)
                cv2.imwrite(dest_file, frame)
                frame_count += 1
                print(frame_count)
                if frame_count==200:
                    print("Got all 200 photos!")
                    break
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                record = not record
        cap.release()
        cv2.destroyAllWindows()

