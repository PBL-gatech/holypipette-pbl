import csv
import numpy as np
from scipy.signal import chirp, decimate, resample
import matplotlib.pyplot as plt
import math

class SignalGenerator:
    def __init__(self, sampling_rate=500, amplitude=10):
        """
        Initialize the SignalGenerator with default parameters.

        Args:
            sampling_rate (int): The original sampling rate in Hz.
            amplitude (float): The amplitude of the signal.
        """
        self.sampling_rate = sampling_rate
        self.amplitude = amplitude

    def generate_sinusoid(self, frequency, duration):
        """
        Generate a sinusoidal signal.

        Args:
            frequency (float): Frequency of the sinusoid in Hz.
            duration (float): Duration of the signal in seconds.

        Returns:
            t (numpy.ndarray): Time vector.
            sinusoid (numpy.ndarray): Sinusoidal signal.
        """
        t = np.linspace(0, duration, int(self.sampling_rate * duration), endpoint=False)
        sinusoid = self.amplitude * np.sin(2 * np.pi * frequency * t)
        return t, sinusoid

    def generate_chirp(self, f_start, f_end, duration, method='logarithmic'):
        """
        Generate a chirp signal.

        Args:
            f_start (float): Starting frequency in Hz.
            f_end (float): Ending frequency in Hz.
            duration (float): Duration of the signal in seconds.
            method (str): Method for frequency increase ('linear', 'logarithmic', etc.).

        Returns:
            t (numpy.ndarray): Time vector.
            chirp_signal (numpy.ndarray): Chirp signal.
        """
        t = np.linspace(0, duration, int(self.sampling_rate * duration), endpoint=False)
        chirp_signal = self.amplitude * chirp(t, f0=f_start, f1=f_end, t1=duration, method=method)
        return t, chirp_signal

    def generate_exponential(self, initial_speed, final_speed, duration):
        """
        Generate a position curve based on constant acceleration from initial to final speed.

        Args:
            initial_speed (float): Initial speed in units per second.
            final_speed (float): Final speed in units per second.
            duration (float): Duration of the signal in seconds.

        Returns:
            t (numpy.ndarray): Time vector.
            position (numpy.ndarray): Position signal with constant acceleration.
        """
        # Time vector
        t = np.linspace(0, duration, int(self.sampling_rate * duration), endpoint=False)
        
        # Calculate constant acceleration
        acceleration = (final_speed - initial_speed) / duration
        
        # Position using the equation: x(t) = v0 * t + 0.5 * a * t^2
        position = initial_speed * t + 0.5 * acceleration * t**2
        
        return t, position

    def downsample(self, t, signal, target_rate):
        """
        Downsample a signal to a target sampling rate.

        Args:
            t (numpy.ndarray): Original time vector.
            signal (numpy.ndarray): Original signal.
            target_rate (int): Target sampling rate in Hz.

        Returns:
            t_downsampled (numpy.ndarray): Downsampled time vector.
            signal_downsampled (numpy.ndarray): Downsampled signal.
        """
        if target_rate >= self.sampling_rate:
            raise ValueError("Target rate must be lower than the original sampling rate.")

        # Calculate the greatest common divisor to simplify the downsample factor
        gcd = math.gcd(self.sampling_rate, target_rate)
        downsample_factor = self.sampling_rate // gcd
        new_sampling_rate = self.sampling_rate / downsample_factor

        if new_sampling_rate != target_rate:
            print(f"Exact downsampling factor not possible. Using resampling instead.")
            return self.resample_signal(t, signal, target_rate)
        
        # Apply decimation with filtering
        signal_downsampled = decimate(signal, downsample_factor, ftype='iir', zero_phase=True)
        t_downsampled = np.linspace(t[0], t[-1], int(len(signal_downsampled)), endpoint=False)

        return t_downsampled, signal_downsampled

    def downsample_with_filter(self, t, signal, target_rate):
        """
        Downsample the signal to a target sampling rate with anti-aliasing filter.

        Args:
            t (numpy.ndarray): Original time vector.
            signal (numpy.ndarray): Original signal.
            target_rate (int): Target sampling rate in Hz.

        Returns:
            t_downsampled (numpy.ndarray): Downsampled time vector.
            signal_downsampled (numpy.ndarray): Downsampled signal.
        """
        return self.downsample(t, signal, target_rate)

    def resample_signal(self, t, signal, target_rate):
        """
        Resample the signal to a target sampling rate using interpolation.

        Args:
            t (numpy.ndarray): Original time vector.
            signal (numpy.ndarray): Original signal.
            target_rate (int): Target sampling rate in Hz.

        Returns:
            t_resampled (numpy.ndarray): Resampled time vector.
            signal_resampled (numpy.ndarray): Resampled signal.
        """
        duration = t[-1] - t[0]
        num_samples = int(target_rate * duration)
        signal_resampled = resample(signal, num_samples)
        t_resampled = np.linspace(t[0], t[-1], num_samples, endpoint=False)
        return t_resampled, signal_resampled

    def save_to_csv(self, t, signal, filename):
        """
        Save the generated signal to a CSV file in the specified format.

        Args:
            t (numpy.ndarray): Time vector (timestamps).
            signal (numpy.ndarray): Signal values (pi_z: column in CSV).
            filename (str): Name of the output CSV file.
        """
        # Fixed values for the remaining fields
        st_x, st_y, st_z = 1392.0, 39.4, -22813.9
        pi_x, pi_y = -5936.2, -7985.8

        # Write data to CSV
        with open(filename, mode='w') as file:
            for timestamp, pi_z in zip(t, signal):
                pi_z += -5321.20  # Add a small offset to pi_z
                file.write(
                    f"timestamp:{timestamp:.6f}  st_x:{st_x}  st_y:{st_y}  st_z:{st_z}  "
                    f"pi_x:{pi_x}  pi_y:{pi_y}  pi_z:{pi_z:.1f}\n"
                )

        print(f"Signal saved to {filename}")

    def plot_signal(self, t, signal, title):
        """
        Plot a signal.

        Args:
            t (numpy.ndarray): Time vector.
            signal (numpy.ndarray): Signal to plot.
            title (str): Title for the plot.
        """
        plt.figure(figsize=(10, 4))
        plt.plot(t, signal, marker='o', linestyle='-', markersize=4)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()


# Example Usage:
if __name__ == "__main__":
    # Create an instance of the SignalGenerator
    generator = SignalGenerator(sampling_rate=40, amplitude=5)

    # # Generate a chirp signal
    # t_chirp, chirp_signal = generator.generate_chirp(f_start=0.01, f_end=0.5, duration=15)
    # generator.plot_signal(t_chirp, chirp_signal, "Chirp Signal")

    # Save the chirp signal to a CSV
    # generator.save_to_csv(t_chirp, chirp_signal, r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\testing\movement\chirp_signal.csv")

    # Generate an exponential position curve
    initial_speed = 1.0  # um per second
    final_speed = 4.0    # um per second
    duration = 30        # seconds
    t_exp, exp_position = generator.generate_exponential(initial_speed, final_speed, duration)
    generator.plot_signal(t_exp, exp_position, "Exponential Position Signal")

    # Save the exponential signal to a CSV
    # generator.save_to_csv(t_exp, exp_position, r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\testing\movement\exponential_position_signal.csv")


# Example Usage:
# Create an instance of the SignalGenerator
generator = SignalGenerator(sampling_rate=40, amplitude=5)

# # Generate a sinusoidal signal
# # t_sin, sinusoid = generator.generate_sinusoid(frequency=0.5, duration=5)

# # Downsample the sinusoid to 10 Hz
# # t_down, sinusoid_down = generator.downsample(t_sin, sinusoid, target_rate=20)

# # Plot the original and downsampled signals
# # generator.plot_signal(t_sin, sinusoid, "Original Sinusoidal Signal")
# # generator.plot_signal(t_down, sinusoid_down, "Downsampled Sinusoidal Signal")


# # # Save the sinusoidal signal to a CSV
# # generator.save_to_csv(t_sin, sinusoid, r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\testing\movement\sinusoid_signal.csv")

# # # Generate a chirp signal
# t_chirp, chirp_signal = generator.generate_chirp(f_start=0.01, f_end=0.5, duration=15)
# generator.plot_signal(t_chirp, chirp_signal, "Chirp Signal")

# # Save the chirp signal to a CSV
# generator.save_to_csv(t_chirp, chirp_signal, r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\testing\movement\chirp_signal.csv")
