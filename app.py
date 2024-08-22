import pyrootutils
import gradio as gr
from apply_filter.src.ApplyFilter import apply_filter_on_image, apply_filter_on_video

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# examples
package_dir = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
example_1_dir = str("apply_filter/test/data/images/example_1.png")
example_2_dir = str("apply_filter/test/data/images/example_2.png")
example_3_dir = str("apply_filter/test/data/images/example_3.png")
example_list = [[example_1_dir], [example_2_dir], [example_3_dir]]

# for app
title = "Filter App"
description = "Using simple resnet 18 to detect landmarks and then applying filter on faces"
article = "Create by Do Canh"
filter_names = ["squid_game_front_man", "anonymous", "dog", "cat"]

# create the Gradio
image_tab = gr.Interface(fn=apply_filter_on_image,
                        inputs=[gr.Image(), gr.inputs.Radio(choices=filter_names, label="Select a filter:")],
                        outputs=gr.Image(type="pil"),
                        examples=example_list,
                        title=title,
                        description=description,
                        article=article)

video_tab = gr.Interface(fn=apply_filter_on_video,
                        inputs=[gr.Video(), gr.inputs.Radio(choices=filter_names, label="Select a filter:")],
                        outputs=gr.Video(type="pil"),
                        title=title,
                        description=description,
                        article=article)

demo = gr.TabbedInterface([image_tab, video_tab], ["Image", "Video"])

# demo.launch(debug=False, share=False)
demo.launch(server_name="0.0.0.0", server_port=4000, debug=False)
