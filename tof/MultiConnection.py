from pickle import FALSE, TRUE
import sys 
sys.path.append('../../../')

from DCAM550.API.Vzense_api_550 import *
import time
import datetime
import cv2
import numpy
import argparse
import subprocess
camera = VzenseTofCam()

parser = argparse.ArgumentParser(description='tof recoder')
parser.add_argument('-t', type=int, default=20, help='time')
args = parser.parse_args()

print(f'time = {args.t}')

camera_count = camera.Ps2_GetDeviceCount()
retry_count = 100
while camera_count < 2 and retry_count > 0:
    retry_count = retry_count-1
    camera_count = camera.Ps2_GetDeviceCount()
    time.sleep(1)
    print("scaning......   ",retry_count)


if camera_count < 2: 
    print("there are no camera or only on camera")
    exit()
 
cameras = []

ret, device_infolist=camera.Ps2_GetDeviceListInfo(camera_count)
if ret==0:
    for i in range(camera_count): 
        print('cam uri:  ' + str(device_infolist[i].uri))
        cam = VzenseTofCam()
        ret = cam.Ps2_OpenDevice(device_infolist[i].uri)
        if  ret == 0:
            print(device_infolist[i].alias,"open successful")
            cameras.append(cam)
        else:
            print(device_infolist[i].alias,'Ps2_OpenDevice failed: ' + str(ret))    
else:
    print(' failed:' + ret)  
    exit()  

for i in range(camera_count): 
    ret = cameras[i].Ps2_StartStream()       
    if  ret == 0:
        print(device_infolist[i].alias,"start stream successful")
    else:
        print(device_infolist[i].alias,'Ps2_StartStream failed: ' + str(ret))

    ret = cameras[i].Ps2_SetDataMode(PsDataMode.PsDepth_30)       
    if  ret == 0:
        print(device_infolist[i].alias,"set successful")
    else:
        print(device_infolist[i].alias,'set failed: ' + str(ret))

    ret = cameras[i].Ps2_SetDepthRange(PsDepthRange.PsMidRange)       
    if  ret == 0:
        print(device_infolist[i].alias,"set successful")
    else:
        print(device_infolist[i].alias,'set failed: ' + str(ret))

    ret = cameras[i].Ps2_SetDepthDistortionCorrectionEnabled(c_bool(False))       
    if  ret == 0:
        print(device_infolist[i].alias,"set successful")
    else:
        print(device_infolist[i].alias,'set failed: ' + str(ret))

    ret = cameras[i].Ps2_SetComputeRealDepthCorrectionEnabled(c_bool(False))       
    if  ret == 0:
        print(device_infolist[i].alias,"set successful")
    else:
        print(device_infolist[i].alias,'set failed: ' + str(ret))
        

cap = [True for _ in range(camera_count)]
k = 0

res = [[] for _ in range(camera_count)]

print(datetime.datetime.now())
p = subprocess.Popen('arecord -D hw:2,0 -r 48000 -d 5 -c 8 -t wav ./final.wav', shell=True)
print(datetime.datetime.now())
while 1:
    for i in range(camera_count):
        ret, frameready = cameras[i].Ps2_ReadNextFrame()   
        if ret != 0 or not frameready.depth:
            continue

        ret, depthframe = cameras[i].Ps2_GetFrame(PsFrameType.PsDepthFrame)
        if ret != 0:
            continue

        if cap[i]:
            frametmp = numpy.ctypeslib.as_array(depthframe.pFrameData, (1, depthframe.width * depthframe.height * 2))
            frametmp.dtype = numpy.uint16
            frametmp.shape = (depthframe.height, depthframe.width)
            res[i].append(numpy.copy(frametmp))
            cap[i] = False

    if not any(cap):
        for j in range(len(cap)):
            cap[j] = True
        k += 1

    if k > 30 * args.t:
        break

print(datetime.datetime.now())

for i in range(camera_count):
    for k in range(len(res[i])):
        cv2.imwrite(f'sample/{k:06d}_{i}_dp.png', res[i][k])
        cv2.imwrite(f'sample/{k:06d}_8bits_{i}_dp.png', numpy.clip((res[i][k].astype(numpy.float32) - 500) / 3500 * 255, 0, 255).astype(numpy.uint8))

print(datetime.datetime.now())

for i in range(camera_count): 
    
    ret = cameras[i].Ps2_StopStream()       
    if  ret == 0:
        print("stop stream successful")
    else:
        print('Ps2_StopStream failed: ' + str(ret))  

    ret = cameras[i].Ps2_CloseDevice()       
    if  ret == 0:
        print("close device successful")
    else:
        print('Ps2_CloseDevice failed: ' + str(ret))  
    
           
