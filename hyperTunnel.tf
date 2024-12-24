#musk is proposing an impossible tunnel from NYC to London - 

#import tensorflow as tf

# Define constants for distance (in kilometers) and time (in hours)
distance = tf.constant(5570.0)  # in kilometers
time = tf.constant(0.9)         # in hours

# Calculate speed
speed = distance / time

# Convert speed to Mach (speed of sound ~1235 km/h)
speed_of_sound = tf.constant(1235.0)
mach = speed / speed_of_sound

# Run the calculation in a TensorFlow session
print(f"Speed needed: {speed.numpy()} km/h")
print(f"Equivalent to Mach: {mach.numpy()}")
