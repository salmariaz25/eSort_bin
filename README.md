# eSort - AI Powered Smart Waste Sorting Bin
*eSort is a low-cost, compact smart bin designed for small urban settings such as universities, offices, and cafes. It uses computer vision (EfficientNet-B0) on a Raspberry Pi 4 to automatically classify waste and control sorting flaps via Arduino and servo motors. The system improves waste segregation, reduces human error, and promotes sustainability.*

**Features**
1. Real-time waste classification using a USB camera and EfficientNet-B0 model.
2. 5-class model: biodegradable, recyclable, paper, nonrecyclable, hazardous.
3. Arduino-controlled servos move flaps to route waste into the correct compartment.
4. Hazardous waste mode: red LED + buzzer alert, requires clear detection before resuming.
5. Headless operation: Raspberry Pi autostarts the classification script on boot.
6. Compact design — low-cost components, suitable for small-scale deployments.

**How to start the System**
1. *Clone the repository:*
   git clone https://github.com/salmariaz25/eSort_bin.git
   cd eSort_bin

2. *Install dependencies:*
   pip install torch torchvision opencv-python pyserial numpy

3. *Connect hardware:*
   Connect Arduino to USB port
   Connect camera
   Upload Arduino sorting code to your board

4. *Configure settings:*
   Update COM_PORT in det_ser.py if needed
   Adjust MODEL_PATH to your model location

5. *Finally run the python file:*
   python det_ser.py

**How to use the system:**
1. *Normal Operation:*
    Place waste item on the sorting tray
    System automatically classifies the item
    Arduino moves item to appropriate bin
    System waits for next item

2. *Hazardous Material Handling:*
    Hazardous item detected 
    System enters "hazard mode"
    Waits for manual clearance
    Sends CLEAR command when safe
    Returns to normal operation

**Dataset:**
Training images available at:
https://drive.google.com/file/d/17teKqaoHH-SfliOGdDCh_gxPFrUg186u/view?usp=drive_link

**Demo Video link:** https://youtube.com/shorts/lXJs67SWHkw?feature=share
