import tkinter as tk

window = tk.Tk()
window.geometry("300x200")
window.title("Speech recognition")

frame0 = tk.Frame(master=window)
frame0.pack()

frame1 = tk.Frame(master=window)
frame1.pack()

frame2 = tk.Frame(master=window)
frame2.pack()

label = tk.Label(master=frame0, text="Speech recognition")
label.pack(padx=5, pady=10)

btn_record = tk.Button(master=frame1, width=13, height=2, text="record", command=record)
btn_record.pack(side=tk.LEFT, padx=5, pady=5)

btn_playback = tk.Button(master=frame1, width=13, height=2, text="playback", command=play)
btn_playback.pack(side=tk.LEFT, padx=5, pady=5)

btn_predict = tk.Button(master=frame2, width=13, height=2, text="trim & predict", command=predict)
btn_predict.pack(side=tk.LEFT, padx=5, pady=5)

btn_playback = tk.Button(master=frame2, width=13, height=2, text="playbacktrimmed", command=playtrimmed)
btn_playback.pack(side=tk.LEFT, padx=5, pady=5)



window.mainloop()