from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

config = SO101FollowerConfig(
    port="/dev/tty.usbmodem585A0076841",
    id="my_awesome_follower_arm",
)
follower = SO101Follower(config)
follower.setup_motors()