
class PIDController:
    def __init__(self, target, Kp=0.1, Ki=0.0, Kd=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target = target
        self.prev_error = 0
        self.integral = 0

    def update(self, current_value, target=None):
        if target is not None:
            self.target = target
        error = self.target - current_value
        self.integral += error
        derivative = error - self.prev_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

    def reset(self, target=None):
        if target is not None:
            self.target = target
        self.prev_error = 0
        self.integral = 0

