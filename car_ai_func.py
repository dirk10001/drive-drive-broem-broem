
import cv2
import numpy as np
import math

#create a bit mask to filter out unused region of the image
def create_mask(width, height):
        polygons = np.array([
                                [(0, height), (width, height),
                                (round(0.95*width), round(0.5 * height)),
                                (round(0.05*width), round(0.5 * height)) ]
                            ])
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, polygons, 255)
        return mask

#create a mask
mask = create_mask(640, 480)

#method that is used in car_ai and learning-model
def preProcess(image):
    return skewAndCrop(image)

#crop the image and skew so that unrelevant parts of the image are not used.
#crop the top oof the images
#skew make a trapezium and transform it to a rectangl distoring the images
#to see the results use cv2.imwrite
def skewAndCrop(image):
    #remove top
    crop_img = image[138:480, 0:640]
    rows, cols, ch = crop_img.shape
    #define trapezium
    pts1 = np.float32(
        [[cols*.25, 0],
         [cols*.75, 0],
         [0, rows],
         [cols,    rows]]
    )
    #define target transformation
    pts2 = np.float32(
        [[0, 0],
         [cols,     0],
         [0,        rows],
         [cols,     rows]]
    )
    #create a transformation matrix
    M = cv2.getPerspectiveTransform(pts1,pts2)
    #transform the cropped image
    dst = cv2.warpPerspective(crop_img, M, (cols, rows))
    return dst

#detect edges in the image using the Canny algorithm
def edgeDetect(image):
    global mask
    #transform to grey
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #blur for better results
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #edge detect
    canny = cv2.Canny(blur, 40, 200)
    #apply mask to remove unrelevant image data
    segment = cv2.bitwise_and(canny, mask)
    return segment

#use this method to apply edge detecting for the learing model
def edgeDetectAndConvertBackToRGB(image):
    segment = edgeDetect(image)
    segment = cv2.cvtColor(segment, cv2.COLOR_GRAY2RGB)
    return segment

#compute lines from image
def hough(segment):
    hough = cv2.HoughLinesP(segment, 1, np.pi / 180, 50, np.array([]), minLineLength = 40, maxLineGap = 5)
    return hough

#create a counter for lane detection, used when saving images
ldCounter = 0

#create white background
whiteBg = np.zeros([480,640,3],dtype=np.uint8)

#fill with white pixels
whiteBg.fill(255)

#filter the image for black areas and fill the remainder with a white background
def lineFilter(image):
    global whiteBg

    #define color range
    lower_color = np.array([0,0,0])
    upper_color = np.array([50,50,50])
    #create a mask for the color range
    mask = cv2.inRange(image, lower_color, upper_color)
    #create an inverted mask
    invMask = cv2.bitwise_not(mask)
    #apply the mask to the input image
    filteredImage = cv2.bitwise_and(image, image, mask= mask)
    #apply the inverted mask to the white background
    filteredBg = cv2.bitwise_and(whiteBg, whiteBg, mask= invMask)
    #add filtered image and filtered background to produce a filtered image with a white background
    filteredImage = cv2.add(filteredBg,filteredImage)
    return filteredImage

#compute a lane line from line information (slope and intercept)
def computeLane(width,heigth,line):
    slope, intercept = line

    y1 = heigth #bottom of the image
    y2 = round(heigth * 0.5) # center of image

    #compute x1 from slope, intercept and y1, as a minimum take - width and as a mximum 2*width
    x1 = max(-width, min(2*width, round((y1 - intercept)/slope)))
    #compute x2 from slope, intercept and y2, as a minimum take - width and as a mximum 2*width
    x2 = max(-width, min(2*width, round((y2 - intercept)/slope)))
    return (x1,y1,x2,y2)

#compute steering angle from image
def computeSteeringAngleWithHough(image):
    global mask
    global ldCounter
    #apply color filter
    filter = lineFilter(image)
    #edge detect to find lines
    segment = edgeDetect(filter)
    #compute lines using the Hough algorithm
    lines = hough(segment)

    height, width = segment.shape
    #select relevant lines based on boundaries (1/3 from the left and 1/3 from the right)
    boundary = 0.3
    left_boundary = width * ( 1 - boundary)
    right_boundary = width * boundary
    #store found lanes
    lanes = []
    #store found lines on the left
    left = []
    #store found lines on the right
    right = []

    #loop through the lines
    if not (lines is None) :
        for line in lines:
            x1,y1, x2, y2 = line[0]
            #skip vertical lines
            if(x1 == x2):
                continue
            #compute slope and intercept (y = x * slope + intercept)
            slope, intercept = np.polyfit((x1,x2),(y1,y2),1)
            #if the slope is negative and is within the left boundary
            if slope < 0 and x1 < left_boundary and x2 < left_boundary :
                left.append((slope, intercept))
            #if the slope is positive and in right boundary
            elif x1 > right_boundary and x2 > right_boundary :
                    right.append((slope, intercept))
            #uncomment to draw lines in image
            # cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

    #if left lines were found
    if len(left)>0 :
        left_average = np.average(left, axis = 0)
        #compute left lane
        lane = computeLane(width, height, left_average)
        #uncomment to draw lines in image
        # cv2.line(image, (lane[0], lane[1]), (lane[2], lane[3]), (255, 0, 255), thickness=2)
        lanes.append(lane)

    #if right lines were found
    if len(right)>0 :
        right_average = np.average(right, axis = 0)
        #compute right lane
        lane = computeLane(width, height, right_average)
        # cv2.line(image, (lane[0], lane[1]), (lane[2], lane[3]), (255, 0, 0), thickness=2)
        lanes.append(lane)

    #set angle to 90 (straight ahead)
    angle = 90
    #if a left and right lane was found
    if len(lanes) == 2 :
        _ , _ , lx, _ = lanes[0]
        _ , _ , rx, _ = lanes[1]

        #compute the steering angle from the center of the two lanes
        dx = (lx + rx) / 2 - (width/2)
        dy = round(height* 0.5)
        rad = math.atan(dx/dy)
        deg = math.degrees(rad)
        angle = round(deg) + 90
    #if one lane is found
    elif len(lanes) == 1 :
        x1 , _ , x2, _ = lanes[0]
        #compute the steering angle in the direction of the lane
        dx = x2 - x1
        dy = round(height * 0.4)
        rad = math.atan(dx/dy)
        deg = math.degrees(rad)
        angle = round(deg) + 90

    #uncomment to draw lines in image
    # angle_radian = angle / 180.0 * math.pi
    # x1 = int(width / 2)
    # y1 = height
    # x2 = int(x1 - (height * 0.4) / math.tan(angle_radian))
    # y2 = int(height * 0.4)
    # cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=5)

    #uncomment to write images to disk
    # cv2.imwrite('ld_' + str(ldCounter) + '_' + str(angle) + '.png', image)
    # cv2.imwrite('ld_' + str(ldCounter) + '_' + str(angle) + '_fil.png', filter)
    # cv2.imwrite('ld_' + str(ldCounter) + '_' + str(angle) + '_seg.png', segment)
    # ldCounter += 1

    return angle
