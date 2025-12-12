from franky import Affine, CartesianMotion, Robot, ReferenceType

robot = Robot("172.16.1.22")
robot.relative_dynamics_factor = 0.05

motion = CartesianMotion(Affine([0.0, 0.05, 0.0]), ReferenceType.Relative)
robot.move(motion)