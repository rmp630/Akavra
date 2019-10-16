from keras.models import model_from_json
from keras.optimizers import SGD
import numpy as np
from time import sleep
model = model_from_json(open('./models/Face_model_architecture.json').read())
#model.load_weights('_model_weights.h5')
model.load_weights('./models/Face_model_weights.h5')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

def extract_face_features(gray, detected_face, offset_coefficients):
        (x, y, w, h) = detected_face
        #print x , y, w ,h
        horizontal_offset = np.int(np.floor(offset_coefficients[0] * w))
        vertical_offset = np.int(np.floor(offset_coefficients[1] * h))
	

        extracted_face = gray[y+vertical_offset:y+h, 
                          x+horizontal_offset:x-horizontal_offset+w]
        #print extracted_face.shape
        new_extracted_face = zoom(extracted_face, (48. / extracted_face.shape[0], 
                                               48. / extracted_face.shape[1]))
        new_extracted_face = new_extracted_face.astype(np.float32)
        new_extracted_face /= float(new_extracted_face.max())
        return new_extracted_face
from scipy.ndimage import zoom
def detect_face(frame):
        cascPath = "./models/haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascPath)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(48, 48) #,
            #    flags=HAAR_FEATURE_MAX
            )
        return gray, detected_faces

import cv2
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
emotiondict = [0,0,0,0,0,0,0]

hdict = [""]

while True:
    # Capture frame-by-frame
    #    sleep(0.8)

    ret, frame = video_capture.read()

    # detect faces
    gray, detected_faces = detect_face(frame)
    
    face_index = 0
    
    # predict output
    for face in detected_faces:
        (x, y, w, h) = face
        if w > 48:
            # draw rectangle around face 
            cv2.rectangle(frame, (x, y), (x+w, y+h), (250, 250, 0), 2)
            
            # extract features
            extracted_face = extract_face_features(gray, face, (0.075, 0.05)) #(0.075, 0.05)

            # predict 
            prediction_result = model.predict_classes(extracted_face.reshape(1,48,48,1))

            # draw extracted face in the top right corner
            frame[face_index * 48: (face_index + 1) * 48, -49:-1, :] = cv2.cvtColor(extracted_face * 255, cv2.COLOR_GRAY2RGB)

            # annotate main image with a label
            if prediction_result == 3:
                emotiondict[3]+=1
                cv2.putText(frame, "Happy",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, 50, 4)
            elif prediction_result == 0:
                emotiondict[0]+=1
                cv2.putText(frame, "Angry",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, 50, 4)
            elif prediction_result == 1:
                emotiondict[1]+=1
                cv2.putText(frame, "Disgust",(x,y), cv2.FONT_HERSHEY_SIMPLEX,1, 50, 4)
            elif prediction_result == 2:
                emotiondict[2]+=1
                cv2.putText(frame, "Fear",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, 50, 4)
            elif prediction_result == 4:
                emotiondict[4]+=1
                cv2.putText(frame, "Sad",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, 50, 4)
            elif prediction_result == 5:
                emotiondict[5]+=1
                cv2.putText(frame, "Surprise",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, 50, 4)

        else :
                emotiondict[6]+=1
                cv2.putText(frame, "Neutral",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, 50, 4)
                face_index += 1
                
    
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()


happymovie = ['Avengers: Endgame','The Sound of Music','Good Will Hunting','Forrest Gump','Despicable Me']


angrymovie = ['Armageddon','The Lion King','El Camino','Moana','Sharknado']
sadmovie = ['The Perks of Being a Wallflower','The Princess Bride','The Breakfast Club','Back to the Future','Frozen']


fearmovie = ['The Conjuring','The Exorcist','Sinister','Psycho','Annabelle']

disgustmovie = ['The Pursuit of Happyness','The Karate Kid','Django Unchained','Inglourious Basterds','Saving Private Ryan']


surprisemovie = ['Silence of the Lambs','Shutter Island','Searching','Mission Impossible: Ghost Protocol','Now You See Me']


neutralmovie= ['Avengers: Endgame','The Conjuring','Silence of the Lambs','The Pursuit of Happyness','Frozen']

resindex = emotiondict.index(max(emotiondict))

if resindex == 3:
                print('Movies to watch when you are HAPPY:')
                print(*happymovie, sep = "\n") 
                # cv2.putText(frame, "Happy!!",(x,y), cv2.FONT_ITALIC, 2, 155, 10)
elif resindex == 0:
                print('Movies to watch when you are ANGRY:')
                print(*angrymovie, sep = "\n") 
                # cv2.putText(frame, "Angry",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
elif resindex == 1:
                print('Movies to watch when you are DISGUSTED:')
                print(*disgustmovie, sep = "\n") 
                # cv2.putText(frame, "Disgust",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
elif resindex == 2:
                print('Movies to watch when you are FRIGHTENED xD:')
                print(*fearmovie, sep = "\n") 
                # cv2.putText(frame, "Fear",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
elif resindex == 4:
                print('Movies to watch when you are SAD:')
                print(*sadmovie, sep = "\n") 
                # cv2.putText(frame, "Sad",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
elif resindex == 5:
                print('Movies to watch when you are SUPRPRISED:')
                print(*surprisemovie, sep = "\n") 
                # cv2.putText(frame, "Surprise",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)

else :
                print('Movies to watch when you are Kind Of NEUTRAL:')
                print(*neutralmovie, sep = "\n") 
