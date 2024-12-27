import tkinter as tk
from tkinter import filedialog, messagebox
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os

class EmotionRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech Emotion Recognition")
        self.root.geometry("800x600")

        # Title Label
        self.title_label = tk.Label(root, text="Speech Emotion Recognition", font=("Helvetica", 18))
        self.title_label.pack(pady=20)

        # Upload Button
        self.upload_button = tk.Button(root, text="Upload Audio File", command=self.upload_file, height=2, width=20)
        self.upload_button.pack(pady=10)

        # Plot Area
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(pady=20)

        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack()

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
        if file_path:
            self.process_audio(file_path)
        else:
            messagebox.showwarning("Warning", "No file selected!")

    def process_audio(self, file_path):
        try:
            audio, sr = librosa.load(file_path, sr=None)
            self.ax.clear()
            librosa.display.waveshow(audio, sr=sr, ax=self.ax)
            self.ax.set_title("Waveform of Audio")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Amplitude")
            self.canvas.draw()

            messagebox.showinfo("Success", "Audio loaded and displayed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process audio: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionRecognitionApp(root)
    root.mainloop()
