import easyocr

reader = easyocr.Reader(['fr'])
result = reader.readtext("C:/Users/user/Downloads/Identite_ visuelle ASSOCIATION ASBH.png")
for detection in result:
    print(detection[1])
