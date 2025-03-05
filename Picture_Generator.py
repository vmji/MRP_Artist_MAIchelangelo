import torch
import gc #GPU Memory Optimierung
from diffusers import FluxPipeline, AutoPipelineForText2Image
# from accelerate import load_checkpoint_and_dispatch #manuelles verschieben Elemente auf GPU Geräte
import os
import sys
from PIL import Image
import datetime #benötigt zur Generierung von Suffixen für Speichern von Dateien
import random
#Quantisierung
# from transformers import T5EncoderModel, BitsAndBytesConfig

#Definition der Quantisierungsmethode
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# model_id = "google/t5-v1_1-xxl"
# text_encoder = T5EncoderModel.from_pretrained(
#     model_id,
#     subfolder="text_encoder_2",
#     from_tf=True,
#     quantization_config=quantization_config,
# )
#Implementation mit text_encoder_2=text_encoder in FluxPipeline
#Scheitert daran, dass Tensorflow Gewichte anders als bei Pytorch nicht mit BitsAndBytes quantisiert werden können

FILE_PATH = sys.argv[-1]

#Modellinitialisierung
model_path = "/mount/point/models/FLUX.1-schnell"
model = FluxPipeline.from_pretrained(model_path,
                                    torch_dtype=torch.bfloat16,
                                    device_map = "balanced",
                                    # max_memory={0: "16GB", 1: "16GB", 2: "16GB", 3: "16GB"} #max memory falls benötigt und andere GPUs in Nutzung
                                    # text_encoder_2 = text_encoder
                                    )

#Bildgenerierungsfunktion
def pic_gen(prompt, save_path = None, height = 1024, width = 1024,
            guidance_scale = 0.0, num_inference_steps=4, generator_device = "cpu", seed = None,
            image_num = 1, display_picture = True, return_dict=False):
 
    """Generate a picture according to your specifications"""
 
    # Check if generator device is a valid device
    assert generator_device in ["cpu", "cuda", "ipu", "xpu", "mkldnn", "opengl", "opencl", "ideep", "hip", "ve", "fpga", "ort", "xla", "lazy", "vulkan", "mps", "meta", "hpu", "mtia", "privateuseone"], "Please enter a valid generator like the following: cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone"
   
    if save_path != None:
        assert isinstance(save_path, str), "Please enter a valid string for the saving path of your image"
 
        path_test = save_path.rsplit("/", maxsplit=1)[0] #split the path by the last separator
        assert os.path.exists(path_test), "Please enter a valid path"
 
        file_format = save_path.rsplit(".", maxsplit=1)[-1] #the file format as string
        assert file_format in ["png", "jpg", "jpeg"], "Please enter a valid picture format to save the image"
    
    if seed != None:
        generator = torch.Generator(generator_device).manual_seed(seed)
    else:
        generator = torch.Generator(generator_device)

    image = model(
        prompt,
        height = height,
        width = width,
        num_images_per_prompt= image_num,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=256,
        generator=generator, #alternativ zu "cpu" "cuda" verwenden
        return_dict=return_dict,
        )
   
    if save_path != None:
 
        if image_num > 1: #saving all generated pictures
            for i, pic in enumerate(image[0]):
                img_path = save_path.rsplit(".", maxsplit=1)[0]
                img_path = f"{img_path}_{i}.{file_format}"
                pic.save(img_path)

                if display_picture == True:
                    im = Image.open(img_path)
                    im.show()
       
        else:
            img_path = save_path.rsplit(".", maxsplit=1)[0]
            img_path = f"{img_path}.{file_format}"
            image[0][0].save(img_path)

            if display_picture == True:
                im = Image.open(img_path)
                im.show()
   
    # Clean Up
    gc.collect() #Free up GPU Memory
    torch.cuda.empty_cache()
   
    return image #return all the generated data

model_name = model_path.split('/')[-1]
print(f"Hello and welcome to the {model_name} image generator!")

if __name__ == "__main__":
    while True:
        print("Please enter your picture prompt or write 'exit' to close the application:")
        prompt = str(input())
        if prompt.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        print("Do you wish to further customize the image generation by entering parameters?\n(Y/n)")
        customization = str(input())
        if customization.lower() in ["y", "yes", "ja", "ye"]:
            height, width = map(int, input("Please enter your desired picture dimensions for height and then width separated by ',': ").split(","))
            image_num = int( input("Please enter the number of images you wish to be generated (maximum of 4): ")) #Bilderanzahl ist nicht anpassbar
            if image_num > 4:
                image_num = min(image_num, 4)
            guidance_scale = float(input("Please enter the guidance scale as a positive float value: "))
            assert guidance_scale >= 0.0, "The guidance scale value must be 0.0 or above"
            file_suffix = suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S") #generieren des aktuellen Suffixes
            save_path = f"{FILE_PATH}/image_{file_suffix}.png" # f"/mount/point/veith/App_Picture_Generator/generated_images/image_{file_suffix}.png" #Speicherpfad (noch) nicht anpassbar #abspeichern als zufällige Zeichenfolgen
            num_inference_steps = int(input("Please enter the number of inference steps for the picture generation: "))
            seed = input("Please enter the seed for the picture generation as an integer or write 'random' for a random seed: ")
            if seed.lower() in ["random", "rand", "zufall"]:
                seed = random.randrange(0, 9999, 1)
            else:
                seed = int(seed)

        else: #Angabe der default Parameter
            height = 1072
            width = 1920
            guidance_scale=0.0
            file_suffix = suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S") #generieren des aktuellen Suffixes
            save_path=f"{FILE_PATH}/image_{file_suffix}.png" # f"/mount/point/veith/App_Picture_Generator/generated_images/image_{file_suffix}.png" #abspeichern als zufällige Zeichenfolgen
            image_num = 1
            num_inference_steps=4
            seed = random.randrange(0, 9999, 1)

        image = pic_gen(prompt, 
                    height = height,
                    width = width,
                    guidance_scale=guidance_scale,
                    save_path=save_path,
                    image_num = image_num,
                    num_inference_steps=num_inference_steps,
                    seed = seed, #423
                    display_picture=False
                    )

#Nach Durchführung des Skripts wird der GPU-Speicher automatisch freigegeben