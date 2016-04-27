import time


class PIDController:

    def __init__(self, Kp=0, Ki=0, Kd=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.current_time = time.time()
        self.previous_time = self.current_time

        self.previous_error = 0

        self.Cp = 0
        self.Ci = 0
        self.Cd = 0

    def output(self, error):
        self.current_time = time.time()
        dt = self.current_time - self.previous_time
        de = error - self.previous_error

        self.Cp = self.Kp * error
        self.Ci += error * dt

        self.Cd = 0
        if dt > 0:
            self.Cd = de / dt

        self.previous_time = self.current_time
        self.previous_error = error

        return self.Cp + (self.Ki * self.Ci) + (self.Kd * self.Cd)
