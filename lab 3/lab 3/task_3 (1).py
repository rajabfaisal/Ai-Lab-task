class ThermostatAgent:
    def __init__(self):
        self.previous_mode = None

    def decide(self, temp, occupied):
        if not occupied:
            mode = "OFF"
        elif temp < 20:
            mode = "HEAT"
        elif temp > 26:
            mode = "COOL"
        else:
            mode = "IDLE"
        send = mode != self.previous_mode
        self.previous_mode = mode
        return mode, send

tests = [
    ("Repeated percept 1 (first)", 18, True),
    ("Repeated percept 2 (same)", 18, True),
    ("Repeated percept 3 (same)", 18, True),
    ("Exact lower boundary", 20, True),
    ("Exact upper boundary", 26, True),
    ("Mode transition", 27, True),
    ("Empty room", 27, False),
    ("Empty room, cold (priority)", 15, False),
]

agent = ThermostatAgent()
for name, temp, occupied in tests:
    previous = agent.previous_mode
    mode, send = agent.decide(temp, occupied)
    if send:
        command = "SEND command"
    else:
        command = "NO command"
    print(name, "| prev =", previous, "| temp =", temp, "| occupied =", occupied,
          "| target =", mode, "|", command)
