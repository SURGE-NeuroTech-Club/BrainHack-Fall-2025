import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import mne
import os
from scipy.signal import welch, spectrogram
from scipy.signal.windows import hann
import threading
from datetime import datetime

class FIFViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("FIF File Viewer")
        self.root.geometry("1400x900")
        
        # Data storage
        self.raw = None
        self.data = None
        self.time_vector = None
        self.current_file = None
        
        # Viewing parameters
        self.current_start_time = 0
        self.window_duration = 10  # seconds
        self.selected_channels = []
        self.filter_low = 0.5
        self.filter_high = 40
        self.filtered_data = None
        
        # Create GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the main GUI layout."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control panel (left side)
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        # Plot area (right side)
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Setup control panel
        self.setup_control_panel(control_frame)
        
        # Setup plot area
        self.setup_plot_area(plot_frame)
        
    def setup_control_panel(self, parent):
        """Setup the control panel with file loading and parameters."""
        # File operations
        file_frame = ttk.LabelFrame(parent, text="File Operations", padding="5")
        file_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(file_frame, text="Load FIF File", command=self.load_file).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Generate Demo Data", command=self.generate_demo_data).pack(fill=tk.X, pady=2)
        
        self.file_label = ttk.Label(file_frame, text="No file loaded", wraplength=200)
        self.file_label.pack(fill=tk.X, pady=2)
        
        # File info
        self.info_frame = ttk.LabelFrame(parent, text="File Information", padding="5")
        self.info_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.info_text = tk.Text(self.info_frame, height=8, width=25, font=("Courier", 9))
        info_scroll = ttk.Scrollbar(self.info_frame, orient="vertical", command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scroll.set)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Channel selection
        channel_frame = ttk.LabelFrame(parent, text="Channel Selection", padding="5")
        channel_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Channel listbox with scrollbar
        listbox_frame = ttk.Frame(channel_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True)
        
        self.channel_listbox = tk.Listbox(listbox_frame, selectmode=tk.MULTIPLE, height=6)
        channel_scroll = ttk.Scrollbar(listbox_frame, orient="vertical", command=self.channel_listbox.yview)
        self.channel_listbox.configure(yscrollcommand=channel_scroll.set)
        self.channel_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        channel_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        ttk.Button(channel_frame, text="Select All", command=self.select_all_channels).pack(fill=tk.X, pady=2)
        ttk.Button(channel_frame, text="Clear Selection", command=self.clear_channel_selection).pack(fill=tk.X, pady=2)
        
        # Time navigation
        time_frame = ttk.LabelFrame(parent, text="Time Navigation", padding="5")
        time_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(time_frame, text="Window Duration (s):").pack()
        self.window_var = tk.DoubleVar(value=self.window_duration)
        ttk.Scale(time_frame, from_=1, to=30, variable=self.window_var, 
                 orient=tk.HORIZONTAL, command=self.update_window_duration).pack(fill=tk.X)
        
        ttk.Label(time_frame, text="Start Time (s):").pack()
        self.time_var = tk.DoubleVar(value=0)
        self.time_scale = ttk.Scale(time_frame, from_=0, to=100, variable=self.time_var, 
                                   orient=tk.HORIZONTAL, command=self.update_time_window)
        self.time_scale.pack(fill=tk.X)
        
        nav_frame = ttk.Frame(time_frame)
        nav_frame.pack(fill=tk.X, pady=2)
        ttk.Button(nav_frame, text="<<", command=self.jump_backward, width=5).pack(side=tk.LEFT, padx=1)
        ttk.Button(nav_frame, text="<", command=self.step_backward, width=5).pack(side=tk.LEFT, padx=1)
        ttk.Button(nav_frame, text=">", command=self.step_forward, width=5).pack(side=tk.LEFT, padx=1)
        ttk.Button(nav_frame, text=">>", command=self.jump_forward, width=5).pack(side=tk.LEFT, padx=1)
        
        # Filtering
        filter_frame = ttk.LabelFrame(parent, text="Filtering", padding="5")
        filter_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(filter_frame, text="Low freq (Hz):").pack()
        self.low_freq_var = tk.DoubleVar(value=self.filter_low)
        ttk.Scale(filter_frame, from_=0.1, to=10, variable=self.low_freq_var, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        ttk.Label(filter_frame, text="High freq (Hz):").pack()
        self.high_freq_var = tk.DoubleVar(value=self.filter_high)
        ttk.Scale(filter_frame, from_=10, to=100, variable=self.high_freq_var, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        ttk.Button(filter_frame, text="Apply Filter", command=self.apply_filter).pack(fill=tk.X, pady=2)
        ttk.Button(filter_frame, text="Reset Filter", command=self.reset_filter).pack(fill=tk.X, pady=2)
        
        # View options
        view_frame = ttk.LabelFrame(parent, text="View Options", padding="5")
        view_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.view_type = tk.StringVar(value="Time Series")
        view_options = ["Time Series", "Power Spectral Density", "Spectrogram", "Channel Locations"]
        
        for option in view_options:
            ttk.Radiobutton(view_frame, text=option, variable=self.view_type, 
                           value=option, command=self.update_plot).pack(anchor=tk.W)
        
        # Plot controls
        ttk.Button(view_frame, text="Update Plot", command=self.update_plot).pack(fill=tk.X, pady=2)
        ttk.Button(view_frame, text="Save Plot", command=self.save_plot).pack(fill=tk.X, pady=2)
        
    def setup_plot_area(self, parent):
        """Setup the matplotlib plotting area."""
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 8), dpi=100)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, parent)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
    def load_file(self):
        """Load a .fif file."""
        file_path = filedialog.askopenfilename(
            title="Select FIF File",
            filetypes=[("FIF files", "*.fif"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Load the file
                self.raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
                self.current_file = file_path
                
                # Extract data
                self.data = self.raw.get_data()
                self.filtered_data = self.data.copy()
                self.time_vector = self.raw.times
                
                # Update GUI
                self.update_file_info()
                self.update_channel_list()
                self.update_time_scale()
                self.select_default_channels()
                self.update_plot()
                
                self.file_label.config(text=f"Loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
                
    def generate_demo_data(self):
        """Generate demonstration EEG data."""
        try:
            # Create synthetic EEG data
            sfreq = 250  # Hz
            duration = 60  # seconds
            n_channels = 8
            
            # Standard 10-20 channel names
            ch_names = ['Fp1', 'Fp2', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
            ch_types = ['eeg'] * n_channels
            
            # Generate synthetic data
            times = np.arange(0, duration, 1/sfreq)
            n_samples = len(times)
            
            # Base EEG signal (pink noise + alpha rhythm)
            np.random.seed(42)
            data = np.random.randn(n_channels, n_samples) * 20  # 20 µV noise
            
            # Add alpha rhythm (8-12 Hz) to posterior channels
            alpha_freq = 10
            alpha_signal = 15 * np.sin(2 * np.pi * alpha_freq * times)
            data[6:8] += alpha_signal  # O1, O2 channels
            
            # Add some artifacts
            # Eye blinks (low frequency, high amplitude)
            blink_times = np.random.choice(times, size=20)
            for blink_time in blink_times:
                blink_idx = np.argmin(np.abs(times - blink_time))
                blink_artifact = 100 * np.exp(-((times - blink_time) / 0.2)**2)
                data[0:2] += blink_artifact  # Fp1, Fp2
            
            # Muscle artifacts (high frequency)
            muscle_times = np.random.choice(times, size=10)
            for muscle_time in muscle_times:
                muscle_idx = np.argmin(np.abs(times - muscle_time))
                start_idx = max(0, muscle_idx - 125)  # 0.5s before
                end_idx = min(n_samples, muscle_idx + 125)  # 0.5s after
                muscle_noise = 30 * np.random.randn(n_channels, end_idx - start_idx)
                data[:, start_idx:end_idx] += muscle_noise
            
            # Create MNE Raw object
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
            self.raw = mne.io.RawArray(data, info)
            self.current_file = "Demo Data"
            
            # Set up data
            self.data = data
            self.filtered_data = self.data.copy()
            self.time_vector = times
            
            # Update GUI
            self.update_file_info()
            self.update_channel_list()
            self.update_time_scale()
            self.select_default_channels()
            self.update_plot()
            
            self.file_label.config(text="Generated: Demo EEG Data")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate demo data:\n{str(e)}")
    
    def update_file_info(self):
        """Update the file information display."""
        if self.raw is None:
            return
            
        info_text = f"File: {os.path.basename(self.current_file) if self.current_file else 'Demo'}\n"
        info_text += f"Channels: {len(self.raw.ch_names)}\n"
        info_text += f"Sample Rate: {self.raw.info['sfreq']:.1f} Hz\n"
        info_text += f"Duration: {self.raw.times[-1]:.1f} s\n"
        info_text += f"Samples: {len(self.raw.times)}\n\n"
        
        info_text += "Channel Names:\n"
        for i, ch in enumerate(self.raw.ch_names):
            info_text += f"{i+1:2d}. {ch}\n"
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)
        
    def update_channel_list(self):
        """Update the channel selection listbox."""
        if self.raw is None:
            return
            
        self.channel_listbox.delete(0, tk.END)
        for ch in self.raw.ch_names:
            self.channel_listbox.insert(tk.END, ch)
    
    def update_time_scale(self):
        """Update the time navigation scale."""
        if self.raw is None:
            return
            
        max_time = self.raw.times[-1] - self.window_duration
        self.time_scale.config(to=max_time)
        
    def select_default_channels(self):
        """Select first 4 channels by default."""
        if self.raw is None:
            return
            
        # Select first 4 channels
        for i in range(min(4, len(self.raw.ch_names))):
            self.channel_listbox.selection_set(i)
        self.update_selected_channels()
        
    def select_all_channels(self):
        """Select all channels."""
        self.channel_listbox.selection_set(0, tk.END)
        self.update_selected_channels()
        
    def clear_channel_selection(self):
        """Clear channel selection."""
        self.channel_listbox.selection_clear(0, tk.END)
        self.update_selected_channels()
        
    def update_selected_channels(self):
        """Update the list of selected channels."""
        selection = self.channel_listbox.curselection()
        self.selected_channels = [i for i in selection]
        
    def update_window_duration(self, value):
        """Update the time window duration."""
        self.window_duration = float(value)
        self.update_time_scale()
        
    def update_time_window(self, value):
        """Update the current time window."""
        self.current_start_time = float(value)
        if self.raw is not None:
            self.update_plot()
    
    def step_forward(self):
        """Step forward by half window duration."""
        if self.raw is None:
            return
        step = self.window_duration / 2
        max_time = self.raw.times[-1] - self.window_duration
        new_time = min(self.current_start_time + step, max_time)
        self.time_var.set(new_time)
        self.current_start_time = new_time
        self.update_plot()
        
    def step_backward(self):
        """Step backward by half window duration."""
        if self.raw is None:
            return
        step = self.window_duration / 2
        new_time = max(self.current_start_time - step, 0)
        self.time_var.set(new_time)
        self.current_start_time = new_time
        self.update_plot()
        
    def jump_forward(self):
        """Jump forward by full window duration."""
        if self.raw is None:
            return
        step = self.window_duration
        max_time = self.raw.times[-1] - self.window_duration
        new_time = min(self.current_start_time + step, max_time)
        self.time_var.set(new_time)
        self.current_start_time = new_time
        self.update_plot()
        
    def jump_backward(self):
        """Jump backward by full window duration."""
        if self.raw is None:
            return
        step = self.window_duration
        new_time = max(self.current_start_time - step, 0)
        self.time_var.set(new_time)
        self.current_start_time = new_time
        self.update_plot()
        
    def apply_filter(self):
        """Apply bandpass filter to the data."""
        if self.raw is None:
            return
            
        try:
            low_freq = self.low_freq_var.get()
            high_freq = self.high_freq_var.get()
            
            # Create a copy of raw data and apply filter
            raw_copy = self.raw.copy()
            raw_copy.filter(l_freq=low_freq, h_freq=high_freq, verbose=False)
            self.filtered_data = raw_copy.get_data()
            
            self.update_plot()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply filter:\n{str(e)}")
            
    def reset_filter(self):
        """Reset filter to original data."""
        if self.raw is None:
            return
        self.filtered_data = self.data.copy()
        self.update_plot()
        
    def update_plot(self):
        """Update the current plot based on selected view type."""
        if self.raw is None:
            return
            
        self.update_selected_channels()
        
        if len(self.selected_channels) == 0:
            self.fig.clear()
            self.canvas.draw()
            return
            
        view_type = self.view_type.get()
        
        # Clear previous plots
        self.fig.clear()
        
        try:
            if view_type == "Time Series":
                self.plot_time_series()
            elif view_type == "Power Spectral Density":
                self.plot_psd()
            elif view_type == "Spectrogram":
                self.plot_spectrogram()
            elif view_type == "Channel Locations":
                self.plot_channel_locations()
                
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update plot:\n{str(e)}")
            
    def plot_time_series(self):
        """Plot time series data."""
        # Get time window indices
        sfreq = self.raw.info['sfreq']
        start_sample = int(self.current_start_time * sfreq)
        end_sample = int((self.current_start_time + self.window_duration) * sfreq)
        end_sample = min(end_sample, self.filtered_data.shape[1])
        
        # Time vector for this window
        time_window = self.time_vector[start_sample:end_sample]
        
        # Create subplots
        n_channels = len(self.selected_channels)
        
        if n_channels == 1:
            ax = self.fig.add_subplot(111)
            axes = [ax]
        else:
            axes = []
            for i in range(n_channels):
                ax = self.fig.add_subplot(n_channels, 1, i+1)
                axes.append(ax)
        
        # Plot each selected channel
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for i, ch_idx in enumerate(self.selected_channels):
            data_window = self.filtered_data[ch_idx, start_sample:end_sample]
            
            axes[i].plot(time_window, data_window * 1e6, color=colors[i % 10], linewidth=0.8)  # Convert to µV
            axes[i].set_ylabel(f'{self.raw.ch_names[ch_idx]}\n(µV)', fontsize=10)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(time_window[0], time_window[-1])
            
            if i == 0:
                axes[i].set_title(f'EEG Time Series ({self.current_start_time:.1f}-{self.current_start_time+self.window_duration:.1f}s)')
            if i == len(self.selected_channels) - 1:
                axes[i].set_xlabel('Time (s)')
                
        plt.tight_layout()
        
    def plot_psd(self):
        """Plot power spectral density."""
        sfreq = self.raw.info['sfreq']
        
        ax = self.fig.add_subplot(111)
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for i, ch_idx in enumerate(self.selected_channels):
            # Calculate PSD
            freqs, psd = welch(self.filtered_data[ch_idx], fs=sfreq, nperseg=2048, 
                              window='hann', overlap=0.5)
            
            # Convert to dB
            psd_db = 10 * np.log10(psd * 1e12)  # Convert to µV²/Hz and then dB
            
            ax.semilogy(freqs, psd, color=colors[i % 10], 
                       label=self.raw.ch_names[ch_idx], linewidth=1.5)
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (µV²/Hz)')
        ax.set_title('Power Spectral Density')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, min(50, sfreq/2))
        
        plt.tight_layout()
        
    def plot_spectrogram(self):
        """Plot spectrogram of the first selected channel."""
        if len(self.selected_channels) == 0:
            return
            
        ch_idx = self.selected_channels[0]
        sfreq = self.raw.info['sfreq']
        
        # Calculate spectrogram
        freqs, times, Sxx = spectrogram(self.filtered_data[ch_idx], fs=sfreq, 
                                       nperseg=512, noverlap=256, window='hann')
        
        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx * 1e12)
        
        ax = self.fig.add_subplot(111)
        im = ax.pcolormesh(times, freqs, Sxx_db, shading='gouraud', cmap='viridis')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'Spectrogram - {self.raw.ch_names[ch_idx]}')
        ax.set_ylim(0, min(50, sfreq/2))
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB µV²/Hz)')
        
        plt.tight_layout()
        
    def plot_channel_locations(self):
        """Plot channel locations if available."""
        try:
            # Try to plot channel locations using MNE
            ax = self.fig.add_subplot(111)
            
            # Create a simple head plot if montage is available
            if hasattr(self.raw.info, 'dig') and self.raw.info['dig'] is not None:
                # Use MNE's built-in plotting
                mne.viz.plot_sensors(self.raw.info, axes=ax, show=False)
            else:
                # Create a simple circular layout for demonstration
                n_channels = len(self.raw.ch_names)
                angles = np.linspace(0, 2*np.pi, n_channels, endpoint=False)
                
                x = np.cos(angles)
                y = np.sin(angles)
                
                ax.scatter(x, y, s=100, c='red', alpha=0.7)
                
                # Add channel labels
                for i, (xi, yi, ch_name) in enumerate(zip(x, y, self.raw.ch_names)):
                    ax.annotate(ch_name, (xi, yi), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8)
                
                # Draw head circle
                circle = plt.Circle((0, 0), 1.1, fill=False, color='black', linewidth=2)
                ax.add_patch(circle)
                
                ax.set_xlim(-1.5, 1.5)
                ax.set_ylim(-1.5, 1.5)
                ax.set_aspect('equal')
                ax.set_title('Channel Locations (Schematic)')
                
            plt.tight_layout()
            
        except Exception as e:
            # Fallback to simple text display
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f'Channel locations not available\n\nChannels:\n' + 
                   '\n'.join([f'{i+1}. {ch}' for i, ch in enumerate(self.raw.ch_names)]),
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Channel Information')
            ax.axis('off')
    
    def save_plot(self):
        """Save current plot to file."""
        if self.raw is None:
            messagebox.showwarning("Warning", "No data loaded!")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Plot",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), 
                      ("SVG files", "*.svg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Plot saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plot:\n{str(e)}")

def main():
    """Main function to run the FIF viewer."""
    root = tk.Tk()
    app = FIFViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()