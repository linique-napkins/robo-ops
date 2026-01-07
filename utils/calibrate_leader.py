from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader

config = SO101LeaderConfig(
    port="/dev/tty.usbmodem5AB90657441",
    id="leader_01",
)

leader = SO101Leader(config)
leader.connect(calibrate=False)
leader.calibrate()
leader.disconnect()