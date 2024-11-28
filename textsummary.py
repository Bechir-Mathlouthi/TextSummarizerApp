import torch
import gradio as gr 

# Use a pipeline as a high-level helper
from transformers import pipeline

model_path = "../models/models--sshleifer--distilbart-cnn-12-6/snapshots/a4f8f3ea906ed274767e9906dbaede7531d660ff"

pipe = pipeline("summarization", model=model_path,torch_dtype=torch.bfloat16)

#text="Tunisia,[a] officially the Republic of Tunisia,[b][20] is the northernmost country in Africa. It is a part of the Maghreb region of North Africa, bordered by Algeria to the west and southwest, Libya to the southeast, and the Mediterranean Sea to the north and east. Tunisia also shares maritime borders with Italy through the islands of Sicily and Sardinia to the north and Malta to the east. It features the archaeological sites of Carthage dating back to the 9th century BC, as well as the Great Mosque of Kairouan. Known for its ancient architecture, souks, and blue coasts, it covers 163,610 km2 (63,170 sq mi), and has a population of 12.1 million. It contains the eastern end of the Atlas Mountains and the northern reaches of the Sahara desert; much of its remaining territory is arable land. Its 1,300 km (810 mi) of coastline includes the African conjunction of the western and eastern parts of the Mediterranean Basin. Tunisia is home to Africa's northernmost point, Cape Angela. Located on the northeastern coast, Tunis is the capital and largest city of the country, which is itself named after Tunis. The official language of Tunisia is Modern Standard Arabic. The vast majority of Tunisia's population is Arab and Muslim. Vernacular Tunisian Arabic is the most spoken, and French also serves as an administrative and educational language in some contexts, but it has no official status"

#print(pipe(text))

def summary (input):
    output = pipe(input)
    return output[0]["summary_text"]

gr.close_all()

#demo = gr.Interface(fn=summary , inputs="text", outputs = "text") 
#demo.launch()

demo = gr.Interface(fn=summary,
                    inputs=[gr.Textbox(label="Input text to summarize",lines=6)],
                    outputs=[gr.Textbox(label="Summarized text",lines=4)],
                    title="BM Text Summarizer",
                    description="THIS APPLICATION WILL BE USED TO SUMMARIZE THE TEXT")
demo.launch(share=True)
