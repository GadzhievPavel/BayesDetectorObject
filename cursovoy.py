import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2

def init_(img):

    return

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file',
    help='Path to video file (if not using camera)')
parser.add_argument('-c', '--color', type=str, default='gray',
    help='Color space: "gray" (default) or "rgb"')
parser.add_argument('-b', '--bins', type=int, default=16,
    help='Number of bins per channel (default 16)')
parser.add_argument('-w', '--width', type=int, default=0,
    help='Resize video to specified width in pixels (maintains aspect)')
args = vars(parser.parse_args())

if not args.get('file', False):
    capture = cv2.VideoCapture(0)
else:
    capture = cv2.VideoCapture(args['file'])

color = args['color']
bins = args['bins']
resizeWidth = args['width']

# Initialize plot.
fig_init, initx = plt.subplots()
fig, ax = plt.subplots()
if color == 'rgb':
    ax.set_title('Histogram (RGB)')
    initx.set_title('Init histigram(RGB)')
else:
    ax.set_title('Histogram (grayscale)')
    initx.set_title('Init histigram(RGB)')
ax.set_xlabel('Bin')
ax.set_ylabel('Frequency')
initx.set_xlabel('Bin')
initx.set_ylabel('Frequency')

# Initialize plot line object(s). Turn on interactive plotting and show plot.
lw = 1
alpha = 0.5
if color == 'rgb':
    lineR, = ax.plot(np.arange(bins), np.zeros((bins,)), c='r', lw=lw, alpha=alpha)
    lineG, = ax.plot(np.arange(bins), np.zeros((bins,)), c='g', lw=lw, alpha=alpha)
    lineB, = ax.plot(np.arange(bins), np.zeros((bins,)), c='b', lw=lw, alpha=alpha)

    init_line_R = fig_init.plot(np.arange(bins), np.zeros((bins,)), c='r', lw=lw, alpha=alpha)
    init_line_G = fig_init.plot(np.arange(bins), np.zeros((bins,)), c='g', lw=lw, alpha=alpha)
    init_line_B = fig_init.plot(np.arange(bins), np.zeros((bins,)), c='b', lw=lw, alpha=alpha)

else:
    lineGray, = ax.plot(np.arange(bins), np.zeros((bins,1)), c='k', lw=lw)
ax.set_xlim(0, bins-1)
ax.set_ylim(0, 1)
plt.ion()
plt.show()

# Grab, process, and display video frames. Update plot line object(s).
while True:
    (grabbed, frame) = capture.read()

    if not grabbed:
        break

    # Resize frame to width, if specified.
    if resizeWidth > 0:
        (height, width) = frame.shape[:2]
        resizeHeight = int(float(resizeWidth / width) * height)
        frame = cv2.resize(frame, (resizeWidth, resizeHeight),
            interpolation=cv2.INTER_AREA)
                # Normalize histograms based on number of pixels per frame.
    numPixels = np.prod(frame.shape[:2])
    if color == 'rgb':
        (b, g, r) = cv2.split(frame)
        histogramR = cv2.calcHist([r], [0], None, [bins], [0, 255])
        histogramG = cv2.calcHist([g], [0], None, [bins], [0, 255])
        histogramB = cv2.calcHist([b], [0], None, [bins], [0, 255])
        lineR.set_ydata(histogramR)
        lineG.set_ydata(histogramG)
        lineB.set_ydata(histogramB)
    else:
        #print(numPixels)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        histogram = cv2.calcHist([frame], [0], None, [bins], [0, 255])/numPixels
        print(histogram.shape)
        lineGray.set_ydata(histogram)
    cv2.imshow('Image', frame)    
    fig.canvas.draw()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()