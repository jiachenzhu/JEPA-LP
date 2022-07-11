import math

def cosine_decay(step, max_step):
    step = min(step, max_step)
    x = step / max_step
    decay_value = 0.5 * (1 + math.cos(math.pi * x))
    return decay_value

def linear_decay(step, max_step):
    step = min(step, max_step)
    x = step / max_step
    decay_value = 1 - x
    return decay_value

def square_decay(step, max_step):
    step = min(step, max_step)
    x = step / max_step
    decay_value = math.pow((x - 1), 2)
    return decay_value

def sqrt_decay(step, max_step):
    step = min(step, max_step)
    x = step / max_step
    decay_value = 1 - math.sqrt(1 - math.pow((x - 1), 2))
    return decay_value

class Scheduler:
    def __init__(self,
        name,
        num_epochs, num_steps_per_epoch,
        start_value, end_value,
        num_warmup_epochs=0,
        decay="cosine"
    ):
        self.name = name

        self.total_num_steps = num_epochs * num_steps_per_epoch
        self.num_warmup_steps = num_warmup_epochs * num_steps_per_epoch
        
        self.start_value = start_value
        self.end_value = end_value
        
        self.temp_step = 0
        self.temp_value = None

        if decay == "cosine":
            self.decay = cosine_decay
        elif decay == "linear":
            self.decay = linear_decay
        elif decay == "square":
            self.decay = square_decay
        elif decay == "sqrt":
            self.decay = sqrt_decay

    def get_value(self, step=None):
        if step is None:
            step = self.temp_step
            self.temp_step += 1
        else:
            self.temp_step = step

        if step < self.num_warmup_steps:
            value = self.start_value * step / self.num_warmup_steps
        else:
            value = self.end_value + (self.start_value - self.end_value) * self.decay(step - self.num_warmup_steps, self.total_num_steps - self.num_warmup_steps)
        
        self.temp_value = value
        return value

    def __str__(self):
        return f"{self.name}:{self.temp_value:.4f}"