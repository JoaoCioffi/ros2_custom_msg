# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# mypy: ignore-errors
# pylint: disable-all
#!/usr/bin/python

from custom_msgs.msg import Custom as CustomMsg
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from rclpy.node import Node
from rplidar import RPLidar
import numpy as np
import warnings
import rclpy
import time
import json
import os

# ignore warnings
warnings.filterwarnings("ignore")

# initialize the LiDAR object
lidar=RPLidar("/dev/ttyUSB0")
lidar.start_motor()
lidar.clean_input()

# initialize ROS2 methods
rclpy.init()
node=Node('lidar_node')
publisher=node.create_publisher(CustomMsg,'scans',10)

print(f"""
       _____ __    ___    __  _____________________
      / ___// /   /   |  /  |/  /_  __/ ____/ ____/
      \__ \/ /   / /| | / /|_/ / / / / __/ / /     
     ___/ / /___/ ___ |/ /  / / / / / /___/ /___   
    /____/_____/_/  |_/_/  /_/ /_/ /_____/\____/   
                                                RPLIDAR
                                        
    [lidar info]
        # model: {lidar.get_info()['model']}
        # firmware: {lidar.get_info()['firmware']}
        # hardware: {lidar.get_info()['hardware']}
        # serial-no: {lidar.get_info()['serialnumber']}
        # status: {lidar.get_health()}
""")

print("\n")
userParams={
    "saveInDisk":input("Save LiDAR data in disk (Y/N)? -> "),
    "renderPlot":input("Show RadarPlot (2D) for the mapped points (Y/N)? -> ")
}
print("\n")

while rclpy.ok():
    old_t = None
    valuesList=[]
    data={
            'new_scan_flag':[],
            'laser_pulse_strength':[],
            'angle_dg':[],
            'distance_mm':[],
            'frequency_hz':[],
            'rotation_rpm':[]
            }
    try:
            clock=0 #clock for limit the ROS publishing rate
            for m in lidar.iter_measures():
                
                now=time.time()
                
                # gets lidar mapping data
                data['new_scan_flag'].append(bool(m[0]))
                data['laser_pulse_strength'].append(int(m[1]))
                data['angle_dg'].append(round(m[2],3))
                data['distance_mm'].append(round(m[3],3))

                # gets frequency and velocity
                if old_t is None:
                    old_t=now
                    continue
                delta=now-old_t
                data['frequency_hz'].append(round(1/delta,3))
                data['rotation_rpm'].append(round(60/delta,3))
                old_t=now

                # publishing data with ROS
                clock+=1
                if clock==1000: #limits the publishing rate (needs to be slower than the LiDAR frequency)
                    msg=CustomMsg()
                    msg.newscanflag=data['new_scan_flag'][-1]
                    msg.laserpulsestrength=data['laser_pulse_strength'][-1]
                    msg.angledg=data['angle_dg'][-1]
                    msg.distancemm=data['distance_mm'][-1]
                    msg.frequencyhz=data['frequency_hz'][-1]
                    msg.rotationrpm=data['rotation_rpm'][-1]
                    publisher.publish(msg)
                    node.get_logger().info(f"Publishing: {msg}")
                    clock=0

    except KeyboardInterrupt:
            print('\nStoping...\n')
            try:
                node.destroy_node()
                rclpy.shutdown()
            except:
                pass
            
            lidar.stop()
            lidar.disconnect()

            if userParams['renderPlot'].lower()=='y':
                print('\nRendering Obtained Data...\n')
                DMAX,IMIN,IMAX=4000,0,100
                def update_line(num,iterator,line,ax):
                    scan=next(iterator)
                    offsets=np.array([(np.radians(meas[1]),meas[2]) for meas in scan])
                    line.set_offsets(offsets)
                    intens = np.array([meas[0] for meas in scan])
                    line.set_array(intens)
                    min_dist = np.min(intens)
                    max_dist = np.max(intens)
                    legend_text = f'Min: {min_dist:.2f}mm Max: {max_dist:.2f}mm'
                    ax.legend([legend_text],loc='lower right')
                    ax.set_xticklabels(['0°','45°','90°',
                                        '135°','180°','225°',
                                        '270°','315°'],color='lime')
                    return line,
                def plot():
                    # iterator to gather mapped points (quality,angle,distance)
                    def itt(mappedPoints):
                        minLength=5
                        scanList=[]
                        for scf,lps,ang,dst in zip(
                                                    mappedPoints['new_scan_flag'],
                                                    mappedPoints['laser_pulse_strength'],
                                                    mappedPoints['angle_dg'],
                                                    mappedPoints['distance_mm']
                                                    ):
                            if scf:
                                if len(scanList) > minLength:
                                    yield scanList
                                scanList=[]
                            if dst > 0:
                                scanList.append((lps,ang,dst))
                    fig = plt.figure()
                    fig.patch.set_facecolor('black')
                    ax = plt.subplot(111,projection='polar')
                    ax.xaxis.grid(True,color='#00CC00',linestyle='dashed')
                    ax.yaxis.grid(True,color='#00CC00',linestyle='dashed')
                    ax.set_facecolor('black')
                    ax.spines['polar'].set_visible(False)
                    line = ax.scatter(
                                        [0, 0],[0, 0],
                                        s=5,c=[IMIN, IMAX],
                                        cmap=plt.cm.Reds_r,lw=0
                                        )
                    ax.set_rmax(DMAX)
                    ax.grid(True,color='lime',linestyle='-')
                    iterator=itt(mappedPoints=data)
                    ani = animation.FuncAnimation(fig,update_line,fargs=(iterator,line,ax),interval=50)
                    ax.tick_params(axis='both', colors='#00CC00')
                    plt.show()
                plot()

            if userParams['saveInDisk'].lower()=='y':
                # converts data object to JSON format and saves into a file
                path='./output'
                os.makedirs(path,exist_ok=True)
                jsonFilePath=os.path.join(path, "lidar-data.json")
                with open(jsonFilePath,"w") as json_file:
                    json.dump(data, json_file, indent=4)
                print(f"\nJSON file saved at: {jsonFilePath}\n")
