import tkinter as tk
import tkinter.font as font
import ctypes
import bcM
import recc

def senderr(Strr):
    print("Sending data to receiver:", Strr)
    bcM.bcm("Vasanth","Dr.J.Jayakumar",Strr)
    recc.reciever(Strr)

def sender(Strr):
    window = tk.Tk()
    window.title("Doctor(Sender)")
    screen_width=window.winfo_screenwidth()
    screen_height=window.winfo_screenheight()
    window_width=int(screen_width * 0.5)  
    window_height=int(screen_height * 0.5)  
    window.geometry(f"{window_width}x{window_height}+{int((screen_width - window_width) / 2)}+{int((screen_height - window_height) / 2)}")
    label_font=font.Font(family='Arial', size=12, weight='bold')
    label=tk.Label(window, text="The predicted disease is:", font=label_font)
    label.pack(pady=10)
    text_font=font.Font(family='Arial', size=12)
    text=tk.Label(window, text=Strr, font=text_font)
    text.pack()
    button_font=font.Font(family='Arial', size=12)
    button=tk.Button(window, text="Send To Receiver", font=button_font, command=lambda: senderr(Strr))
    button.pack(pady=10)
    window.mainloop()



