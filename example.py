import gradio as gr

def button_click(btn_text):
    # Add your logic here to handle the button click
    print(f"Button '{btn_text}' clicked!")

# List of options representing your "buttons"
button_options = ["Button 1", "Button 2", "Button 3"]

# Create the interface using the Dropdown component and the live parameter set to True
iface = gr.Interface(button_click, "dropdown", "text", live=True, choices=button_options)

iface.launch()






