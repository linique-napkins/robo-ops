from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

robot_config = SO101FollowerConfig(
    port="/dev/tty.usbmodem5AB90652661",
    id="my_awesome_follower_arm",
)

teleop_config = SO101LeaderConfig(
    port="/dev/tty.usbmodem5AB90657441",
    id="leader_01",
)

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)
robot.connect()
teleop_device.connect()

while True:
    action = teleop_device.get_action()
    print(action)
    robot.send_action(action)