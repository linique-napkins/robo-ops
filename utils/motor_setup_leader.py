from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

config = SO101LeaderConfig(
    port="/dev/tty.usbmodem5AB90657441",
    id="leader_01",
)
leader = SO101Leader(config)
leader.setup_motors()