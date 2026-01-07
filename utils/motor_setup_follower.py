from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

config = SO101FollowerConfig(
    port="/dev/tty.usbmodem5AB90652661",
    id="follower_01",
)
follower = SO101Follower(config)
follower.setup_motors()