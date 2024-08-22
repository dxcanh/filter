import torch
from ..basemodels.dlib_resnet import DlibResnet
from PIL import Image
import numpy as np
from torchvision import transforms
import pyrootutils

package_dir = str(pyrootutils.find_root(search_from=__file__, indicator=".project-root")) + "/apply_filter/"

def load_model(ckpt_path: str ="src/basemodels/dlib_resnet_state_dict.pth")->torch.nn.Module:
    """Load model from ckpt path and create a simple transform for input of model"""

    # prepare path
    ckpt_path = package_dir + ckpt_path

    # load model from pth file
    model = DlibResnet()
    model.load_state_dict(torch.load(f=ckpt_path))

    # create simple transform
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    simple_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return model, simple_transform

def get_landmarks(image: Image, model: torch.nn.Module = None, transform: transforms = None)->np.array:
    if model == None and transform == None:
        model, transform = load_model()

    # prepare input
    image = image.convert("RGB")
    input = transform(image).unsqueeze(dim=0)

    # use model to predict
    model.eval()
    with torch.inference_mode():
        output = model(input)
        output = output.squeeze()

    # denormalised output (landmarks)
    width, height = image.size
    landmarks = (output + 0.5) * np.array([width, height])

    # return landmarks (pixel corordinate)
    return landmarks

### Test detect.py
if __name__ == "__main__":
    from torchinfo import summary

    def test_load_model():
        # create model & transform
        model, simple_transform = load_model()

        # show model
        summary(model=model,
                input_size=(16, 3, 224, 224),
                col_names=["input_size", "output_size", "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"])
        
        # test input & output shape
        random_input = torch.randn([16, 3, 224, 224])
        random_input = random_input.to("cuda")
        output = model(random_input)
        print(f"\n\nINPUT SHAPE: {random_input.shape}")
        print(f"OUTPUT SHAPE: {output.shape}\n")
    
    def test_get_landmarks():
        # create model & transform & image
        model, simple_transform = load_model()
        image = Image.open("apply_filter/data/images/example_1.png")

        # get landmarks
        landmarks = get_landmarks(image=image, model=model, transform=simple_transform)
        print(f"LANDMARKS SHAPE: {landmarks.shape}")
    
    def main():
        test_load_model()
        test_get_landmarks()
    
    main()