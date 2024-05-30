import tkinter as tk
from tkinter import filedialog
import csv
import pandas as pd

class CSVLabeler:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CSV Labeler")
        
        self.current_row_index = 0
        self.data = []
        self.filename = ""
        
        # Setup GUI
        self.setup_gui()
        
        # Load CSV
        self.load_csv()
        
        # Display first entry
        self.display_entry()
        
        self.root.mainloop()
    
    def setup_gui(self):
        self.text_widget = tk.Text(self.root, height=60, width=160)
        self.text_widget.pack(pady=20)
        
        self.label_frame = tk.Frame(self.root)
        self.label_frame.pack(pady=10)
        
        # Label buttons with direct labels
        button_labels = ["BYPASS", "REJECT", "UNCLEAR"]
        for label in button_labels:
            button = tk.Button(self.label_frame, text=label, command=lambda lbl=label: self.label_entry(lbl))
            button.pack(side=tk.LEFT, padx=5)
        
        # Back button
        self.back_button = tk.Button(self.root, text="Back", command=self.go_back)
        self.back_button.pack(side=tk.LEFT, padx=10)
        
        # Next button
        self.next_button = tk.Button(self.root, text="Next", command=self.go_next)
        self.next_button.pack(side=tk.RIGHT, padx=10)
    
    def load_csv(self):
        self.filename = filedialog.askopenfilename(title="Select CSV File", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        self.data = pd.read_csv(self.filename)
        
        #add a column for the labels
        if "Human label" not in self.data.columns:
            self.data.loc[:,"Human label"] = "Unlabeled"

    
    def display_entry(self):
        if self.current_row_index < len(self.data):
            self.text_widget.delete('1.0', tk.END)
            # Display with specific column names and blank line for clarity
            question = self.data["question"][self.current_row_index]
            answer = self.data["english_answer"][self.current_row_index]
            translated_answer = self.data["answers"][self.current_row_index]
            translated_question = self.data["question translation"][self.current_row_index]
            human_label = self.data["Human label"][self.current_row_index]

            display_text = f"Prompt: {question}\n\nResponse: {answer}\n\nTranslated Prompt: {translated_question}\n\nTranslated Response: {translated_answer}"

            display_text += "\n\nLabel: " + human_label
            self.text_widget.insert(tk.END, display_text)
    
    def label_entry(self, label):
        if self.current_row_index < len(self.data):
            # Update or append the direct label (BYPASS, REJECT, UNCLEAR)
            self.data.loc[self.current_row_index,"Human label"] = label
                
            self.go_next()
    
    def go_next(self):
        # Increment the index to go to the next entry, but not beyond the dataset
        if self.current_row_index < len(self.data) - 1:
            self.current_row_index += 1
            self.display_entry()
        else:
            self.save_csv()
    
    def go_back(self):
        # Decrease the index to go back, but not less than 0
        if self.current_row_index > 0:
            self.current_row_index -= 1
            self.display_entry()
    
    def save_csv(self):
        # Logic for saving the labeled CSV with correct labels
        save_filename = filedialog.asksaveasfilename(title="Save Labeled CSV File", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        self.data.to_csv(save_filename, index=False)
        
        self.root.destroy()

# Uncomment the below line to run the application
CSVLabeler()
