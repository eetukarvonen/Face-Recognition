import face_recognition
from PIL import Image, ImageDraw

obamaPic = face_recognition.load_image_file('./img/known/Barack Obama.jpg')
obabaEnc = face_recognition.face_encodings(obamaPic)[0]

niinistoPic = face_recognition.load_image_file('./img/known/Sauli Niinisto.jpg')
niinistoEnc = face_recognition.face_encodings(niinistoPic)[0]

haukioPic = face_recognition.load_image_file('./img/known/Jenni Haukio.jpg')
haukioEnc = face_recognition.face_encodings(haukioPic)[0]

# Array of encodings and names
knownFaceEncodings = [obabaEnc, niinistoEnc, haukioEnc]
knownFaceNames = ["Barack Obama", "Sauli Niinsto", "Jenni Haukio"]

# Load test image
testImage = face_recognition.load_image_file('./img/groups/haukio-obama-niinisto.jpg')

# Find faces in test image
faceLocations = face_recognition.face_locations(testImage)
faceEnc = face_recognition.face_encodings(testImage, faceLocations)

# Convert to PIL format
pilImage = Image.fromarray(testImage)

# Create ImageDraw instance
draw = ImageDraw.Draw(pilImage)

# Loop through faces in test image
for (top, right, bottom, left), faceEncoding in zip(faceLocations, faceEnc):
    matches = face_recognition.compare_faces(knownFaceEncodings, faceEncoding)

    name = "Unknown Person"

    if True in matches:
        firstMatchIndex = matches.index(True)
        name = knownFaceNames[firstMatchIndex]

    # Draw face box
    draw.rectangle(((left, top), (right, bottom)), outline=(0,0,0))

    # Draw label
    textWidth, textHeight = draw.textsize(name)
    draw.rectangle(((left, bottom - textHeight - 10), (right, bottom)), fill=(0,0,0), outline=(0,0,0))
    draw.text((left + 6, bottom - textHeight - 5), name, fill=(255,255,255,255))

del draw

pilImage.show()

pilImage.save('recognized.jpg')