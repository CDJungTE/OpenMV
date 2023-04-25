# OpenMV: Low-Cost Embedded Camera Platform for Deep Learning Inspection
# TechCon note code in 2023
# By CDJung

# Device: OpenMV Cam H7
import sensor, image, time, os, tf, uos, gc, lcd, pyb
from pyb import LED, Timer, UART, Pin  # library import from pyb

sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
sensor.set_windowing([128, 114, 96, 96]) # Set window size for grab
sensor.skip_frames(time=2000)          # Let the camera adjust.
#sensor.set_auto_gain(False)           # Auto Gain off
#sensor.set_auto_whitebal(False)       # Auto Whitebalance off
lcd.init()                             # LCD Initialize

# Pin assign, led uses pin0,2,3,6,7,8
pin_Trigger = Pin('P1', Pin.IN, Pin.PULL_DOWN) # Trigger signal
# Result bit R1R2R3: 100(A), 010(B), 001(C), 110(D)
pin_R1 = Pin('P4', Pin.OUT, Pin.PULL_DOWN) # Result bit
pin_R2 = Pin('P5', Pin.OUT, Pin.PULL_DOWN) # Result bit
pin_R3 = Pin('P9', Pin.OUT, Pin.PULL_DOWN) # Result bit
pin_LED = Pin('P6', Pin.OUT, Pin.PULL_DOWN) # LED signal

# Parameters for AI
net = None
labels = None

# Load built in model
try:
    labels, net = tf.load_builtin_model('trained')
except Exception as e:
    raise Exception(e)

# LCD display size is 128x160
X_scale = 128/96
Y_scale = 160/96

Count_Total = 0
Count_A = 0
Count_B = 0
Count_C = 0
Count_D = 0

clock = time.clock()
while(True):
    clock.tick()
    v_Trigger = pin_Trigger.value()
    img = sensor.snapshot() # Camera grab
    img_for_LCD = img.copy(x_scale=X_scale, y_scale=Y_scale) # image for display at LCD

    if v_Trigger==1:
        if Count_Total <= 10:
            pin_LED.value(1)
            # default settings just do one detection... change them to search the image...
            for obj in net.classify(img, min_scale=1.0, scale_mul=0.5, x_overlap=0.5, y_overlap=0.5):
                print("**********\nPredictions at [x=%d,y=%d,w=%d,h=%d]" % obj.rect())
                # This combines the labels and confidence values into a list of tuples
                predictions_list = list(zip(labels, obj.output()))

                max_val = 0
                idx_max = 0

                for i in range(len(predictions_list)):
                    if max_val < predictions_list[i][1]:
                        max_val = predictions_list[i][1]
                        idx_max = i

                    print("%s = %f" % (predictions_list[i][0], predictions_list[i][1]))

                if idx_max == 0:
                    Count_A += 1
                elif idx_max == 1:
                    Count_B += 1
                elif idx_max == 2:
                    Count_C += 1
                elif idx_max == 3:
                    Count_D += 1

                img_for_LCD.draw_string(5, 3, predictions_list[idx_max][0] + ": {}".format(max_val), color=[255,255,0], mono_space=False, x_spacing=2,scale=1.1)

            img_for_LCD.draw_string(5, 160-14, "FPS:{}".format(round(clock.fps())),  color=[0,255,255], mono_space=False, x_spacing=2,scale=1.1)
            lcd.display(img_for_LCD)
        else:
            pin_LED.value(0)
            Result_Count = [Count_A, Count_B, Count_C, Count_D]
            Result = Result_Count.index(max(Result_Count))

            if Result == 0:
                pin_R1.on()
                pin_R2.off()
                pin_R3.off()
                img_for_LCD.draw_string(5, 3, "Result: A", color=[255,255,0], mono_space=False, x_spacing=2,scale=1.1)
            elif Result == 1:
                pin_R1.off()
                pin_R2.on()
                pin_R3.off()
                img_for_LCD.draw_string(5, 3, "Result: B", color=[255,255,0], mono_space=False, x_spacing=2,scale=1.1)
            elif Result == 2:
                pin_R1.off()
                pin_R2.off()
                pin_R3.on()
                img_for_LCD.draw_string(5, 3, "Result: C", color=[255,255,0], mono_space=False, x_spacing=2,scale=1.1)
            elif Result == 3:
                pin_R1.on()
                pin_R2.on()
                pin_R3.off()
                img_for_LCD.draw_string(5, 3, "Result: D", color=[255,255,0], mono_space=False, x_spacing=2,scale=1.1)
            lcd.display(img_for_LCD)
            Count_Total = 0
            Count_A = 0
            Count_B = 0
            Count_C = 0
            Count_D = 0

    else:
        pin_R1.off()
        pin_R2.off()
        pin_R3.off()
        Count_Total = 0
        Count_A = 0
        Count_B = 0
        Count_C = 0
        Count_D = 0
        img_for_LCD.draw_string(5, 3, "Wait trigger", color=[255,255,0], mono_space=False, x_spacing=2,scale=1.1)
        img_for_LCD.draw_string(5, 160-14, "FPS:{}".format(round(clock.fps())),  color=[0,255,255], mono_space=False, x_spacing=2,scale=1.1)
        lcd.display(img_for_LCD)

    print(clock.fps(), "fps")
