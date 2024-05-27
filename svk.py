import tkinter as tk
from tkinter import filedialog
fff=[]
def select_file():
    file_path = filedialog.askopenfilename()
    return file_path

def SVK():
    root = tk.Tk()
    root.title("File Selection")
    root.geometry("300x100")
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = int(screen_width * 0.5) 
    window_height = int(screen_height * 0.5)
    root.geometry(f"{window_width}x{window_height}+{int((screen_width - window_width) / 2)}+{int((screen_height - window_height) / 2)}")
    
    def on_button_click():
        selected_file = select_file()
        if selected_file:
            file_label.config(text="Selected File: " + selected_file)
            fff.append(selected_file)
            
        else:
            file_label.config(text="No file selected.")

    select_button = tk.Button(root, text="Select File", command=on_button_click)
    select_button.pack(pady=10)

    file_label = tk.Label(root, text="No file selected.")
    file_label.pack()

    root.mainloop()
    print(str(fff[0]))
    return str(fff[0])
SVK()
