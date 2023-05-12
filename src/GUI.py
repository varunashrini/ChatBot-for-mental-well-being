import tkinter as tk
import time
from Blend_main import get_chatbot_response
from PIL import Image, ImageTk

# Create a main window
root = tk.Tk()
root.title("ZEN")
root.geometry("600x500")  # Set window size  600x500

# Load the image and create a PhotoImage object
img = Image.open("C:/UK/UNI/Project/Group project/Final_Project/logo.png")
img = img.resize((32, 64))
photo = ImageTk.PhotoImage(img)

# Set the window icon using the PhotoImage object
root.iconphoto(False, photo)

# Create a chat window
chat_window = tk.Frame(root, bg=None)
chat_window.pack(side="top", fill="both", expand=True)

# Create a scrollbar for the chat window
scrollbar = tk.Scrollbar(chat_window)
scrollbar.pack(side="right", fill="y")

# Create a text widget for the chat messages
chat_log = tk.Text(chat_window,bg="black", fg="yellow", yscrollcommand=scrollbar.set, font=("Verdana", 12))
chat_log.pack(side="left", fill="both", padx=10, pady=10, expand=True)
scrollbar.config(command=chat_log.yview)

# Create a frame for the input field and send button
input_frame = tk.Frame(root, bg="blue")
input_frame.pack(side="bottom", fill="x", padx=10, pady=10)

# Create an input field for the user's message
user_input = tk.Entry(input_frame, font=("Verdana", 14))
user_input.pack(side="left", fill="both", expand=True)

# Create a function to handle sending messages
def send_message(event=None):
    message = user_input.get()
    response = get_chatbot_response(message)
    if message:
        if message.lower() == "quit":
            root.destroy()  # Close the GUI
            return
       
        # If the dialogue should be organized in a chat-like manner with alternating sides
        # chat_log.tag_config("user_tag", justify="right")
        # chat_log.insert("end", "User:" + message + "\n\n", "user_tag")
        # chat_log.tag_config("bot_tag", justify="left")
        chat_log.insert("end", "You: " + message + "\n\n", )        
        # Showing three jumping dots
        chat_log.insert("end", "Chatbot: ")
        for i in range(3):
            chat_log.insert("end", ".")
            chat_log.see("end")
            root.update()
            time.sleep(0.5)
        chat_log.delete("end-3c", "end") # Delete the dots

        # Remove the last character (the dot) before the response
        chat_log.delete("end-2c", "end")

        # Wait for a second before showing the response
        time.sleep(1)

        # Remove any jumping dots that were not deleted before
        end_index = chat_log.index("end-1c")
        if chat_log.get(end_index, "end") == "..." or chat_log.get(end_index, "end") == ".." or chat_log.get(end_index, "end") == "." :
            chat_log.delete(end_index, "end")

        # Show the response to the user
        chat_log.insert("end", response + "\n\n")
        chat_log.see("end")

        user_input.delete(0, "end")

send_button = tk.Button(input_frame, text="Send", font=("Verdana", 8, 'bold'), width="8", height=3,
                        bd=0, bg="#008CBA", activebackground="#3c9d9b", fg='blue',
                        command=send_message)
send_button.pack(side="right")
send_button.config(borderwidth=0, highlightthickness=0)

# Bind the Enter key to the send_message function
user_input.bind("<Return>", send_message)

# Start the main event loop
root.mainloop()