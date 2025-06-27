ctime= time.time()
    fps=1/(ctime-prevtime)
    prevtime=ctime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)