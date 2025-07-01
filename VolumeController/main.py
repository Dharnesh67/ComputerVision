import cv2 
import time 
import numpy as np
import HandTrackingModule as htm
import math
################################
WCam=cv2.VideoCapture(0)
WCam.set(3, 1640)  # Width
WCam.set(4, 1480)  # Height
################################
prevtime = 0


# We are using Pycaw for volume control, but it is not included in this code snippet.
# You can install it using pip:
# pip install pycaw

#make an instance of the HandTrackingModule
# this will allow us to use the methods defined in the HandTrackingModule
Detector=htm.HandDetector(
    detectionCon=0.9,  # Detection confidence threshold
    maxHands=2,        # Maximum number of hands to detect
    
)

from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

print(f"- Muted: {bool(volume.GetMute())}")
print(f"- Volume level: {volume.GetMasterVolumeLevel()} dB")
volRange = volume.GetVolumeRange()
print(f"- Volume range: {volRange[0]} dB - {volRange[1]} dB")
minVol = volRange[0]
maxVol = volRange[1]

# volume.SetMasterVolumeLevel(-5.0, None)  # Set initial volume level

# - Muted: False
# - Volume level: -3.27508544921875 dB
# - Volume range: -63.5 dB - 0.0 dB

def LengthOfLine(x1, y1, x2, y2):
    """Calculate the length of a line segment given its endpoints."""
    return int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

vol=0
VolumeBar=400
volPer=0
smoothening=5

while True:
    success, img=WCam.read()
    if not success:
        continue
        
    # Flip image for mirror effect
    img = cv2.flip(img, 1)
    
    # send image in the HandTrackingModule to find hands
    Detector.findHands(img)
    LMSlist=Detector.findPosition(img,draw=False)
    # https://www.google.com/search?q=hand+landmarks+mediapipe&rlz=1C1ONGR_enIN1054IN1054&oq=hand+landmarks+&gs_lcrp=EgZjaHJvbWUqBwgBEAAYgAQyBggAEEUYOTIHCAEQABiABDIMCAIQABgUGIcCGIAEMgcIAxAAGIAEMgcIBBAAGIAEMggIBRAAGBYYHjIICAYQABgWGB4yCAgHEAAYFhgeMgoICBAAGAoYFhgeMggICRAAGBYYHtIBCDM3ODdqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8#vhid=JntrWWUSgmurWM&vssid=_Gb1faM_IFrzgseMPhZni4Qs_70
    if len(LMSlist)>0:
        # print(LMSlist[4],LMSlist[8])  # Print the coordinates of the index finger tip and thumb tip
        x1,y1=LMSlist[4][1],LMSlist[4][2]
        x2,y2=LMSlist[8][1],LMSlist[8][2]
        # midpoint between the two fi`ngers
        cx,cy=(x1+x2)//2,(y1+y2)//2
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        length=math.hypot(x2-x1, y2-y1)
        
        # Constrain length to valid range
        length = np.clip(length, 50, 300)
        
        # Convert to volume
        vol=np.interp(length, [50, 300], [minVol, maxVol])
        VolumeBar= np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])
        
        # Smoothing
        volPer = smoothening * round(volPer/smoothening)
        
        # Change color when fingers are close
        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
        
        volume.SetMasterVolumeLevel(vol, None)
        
    # Volume Bar
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(VolumeBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    
    Ctime=time.time()
    fps=1/(Ctime-prevtime)
    prevtime = Ctime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    
    cv2.imshow("Volume Controller", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break