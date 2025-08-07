import torch
import gc #GPU Memory Optimierung
from diffusers import FluxPipeline, AutoPipelineForText2Image, StableDiffusion3Pipeline, FluxKontextPipeline
from diffusers import BitsAndBytesConfig, PipelineQuantizationConfig, SD3Transformer2DModel # Quantisierungsoption
from diffusers.utils import load_image
# from accelerate import load_checkpoint_and_dispatch #manuelles verschieben Elemente auf GPU Geräte
import os
import sys
from PIL import Image
import datetime #benötigt zur Generierung von Suffixen für Speichern von Dateien
import random
import time
#Quantisierung
#from diffusers import BitsAndBytesConfig, SD3Transformer2DModel # Quantisierungsoption
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

FILE_PATH = sys.argv[-1] # Dateipfad, in welchem generierte Bilder abgespeichert werden
INPUT_PATH = sys.argv[-2] # Dateipfade, in welchem hochgeladene Bilder abgespeichert werden

#Modellinitialisierung
# Frage nach zu verwendendem Modell

# Fallunterscheidung, ob Bildbearbeitung oder Bildgenerierung durchgeführt werden soll
# Frage nur stellen, wenn Dateien in Ordner "/mount/point/USERNAME/input_images" vorhanden sind
if len(os.listdir(INPUT_PATH)) > 0:
    print("Do you want to edit an image or generate a new one?\n1.\tEdit an image\n2.\tGenerate a new image")
    model_question = int(input("Please answer by typing in the corresponding number: "))

    if model_question == 1:
        mode = "edit" # Variable, welche den aktuellen Operationsmodus angibt
        model_path = "/mount/point/models/FLUX.1-Kontext-dev"

        # 8-bit quantization für bottleneck transformer
        quantization_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_8bit",
            quant_kwargs={
                "load_in_8bit": True,
                "llm_int8_threshold": 6.0,
                "llm_int8_has_fp16_weight": False,
            },
            components_to_quantize=["transformer"]  # Only quantize the transformer component
        )

        model_edit = FluxKontextPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,  # Try 8-bit first
            device_map="balanced",
        )
    
    else:
        mode = "generate" # Variable, welche den aktuellen Operationsmodus angibt
        print("Which image generation model do you want to use?")
        print("1.\tFLUX.1-schnell")
        print("2.\tstable-diffusion-3.5-medium")
        model_question = int(input("Please answer by typing in the corresponding model number: "))

        model_question = int(input("Which image generation model do you want to use?\n1.\tFLUX.1-schnell\n2.\tstable-diffusion-3.5-medium\nPlease answer by typing in the corresponding model number: "))

        if model_question == 1:
            model_path = "/mount/point/models/FLUX.1-schnell"
            model = FluxPipeline.from_pretrained(model_path,
                                                torch_dtype=torch.bfloat16, #torch.bfloat32
                                                device_map = "balanced",
                                                # max_memory={0: "16GB", 1: "16GB", 2: "16GB", 3: "16GB"} #max memory falls benötigt und andere GPUs in Nutzung
                                                # text_encoder_2 = text_encoder
                                                )
        else:
            model_path = "/mount/point/models/stable-diffusion-3.5-medium"
            model = StableDiffusion3Pipeline.from_pretrained(model_path,
                                            torch_dtype=torch.float16, #torch.bfloat32
                                            device_map = "balanced",
                                            #transformer=model_nf4, # Quantisierung des Transformer-Modells
                                            )

else: # Fall, dass keine Bilder hochgeladen wurden, was nur die Nutzung von Bildgenerierungsmodellen zulässt
    mode = "generate" # Variable, welche den aktuellen Operationsmodus angibt
    print("Which image generation model do you want to use?")
    print("1.\tFLUX.1-schnell")
    print("2.\tstable-diffusion-3.5-medium")
    print("3.\tFLUX.1-Krea-dev")
    model_question = int(input("Please answer by typing in the corresponding model number: "))

    if model_question == 1:
        model_path = "/mount/point/models/FLUX.1-schnell"
        model = FluxPipeline.from_pretrained(model_path,
                                            torch_dtype=torch.bfloat16, #torch.bfloat32
                                            device_map = "balanced",
                                            # max_memory={0: "16GB", 1: "16GB", 2: "16GB", 3: "16GB"} #max memory falls benötigt und andere GPUs in Nutzung
                                            # text_encoder_2 = text_encoder
                                            )
    elif model_question == 2:
        model_path = "/mount/point/models/stable-diffusion-3.5-medium"
        model = StableDiffusion3Pipeline.from_pretrained(model_path,
                                        torch_dtype=torch.bfloat16, #torch.bfloat32
                                        device_map = "balanced",
                                        )
        
    elif model_question == 3:
        model_path = "/mount/point/models/FLUX.1-Krea-dev"
        model = FluxPipeline.from_pretrained(model_path,
                                            torch_dtype=torch.bfloat16, #torch.bfloat32
                                            device_map = "balanced",
                                            # max_memory={0: "16GB", 1: "16GB", 2: "16GB", 3: "16GB"} #max memory falls benötigt und andere GPUs in Nutzung
                                            # text_encoder_2 = text_encoder
                                            )
    else:
        model_path = "/mount/point/models/stable-diffusion-3.5-medium"
        model = StableDiffusion3Pipeline.from_pretrained(model_path,
                                        torch_dtype=torch.float16, #torch.bfloat32
                                        device_map = "balanced",
                                        #transformer=model_nf4, # Quantisierung des Transformer-Modells
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
        max_sequence_length=256 if model_question == 1 else 512,
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

# Bildeditierungsfunktion
def pic_edit(input_image, prompt: str, save_path: str = None,
            guidance_scale = 0.0, num_inference_steps=12, generator_device = "cpu", seed = None,
            image_num = 1, display_picture = False):
 
    """Edit a picture according to your specifications"""
 
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
    
    image = model_edit(
        image=input_image,
        prompt=prompt,
        height = input_image.size[1],
        width = input_image.size[0],
        max_area=input_image.size[0] * input_image.size[1],
        num_images_per_prompt= image_num,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=256, # if model_question == 1 else 512,
        generator=generator, #alternativ zu "cpu" "cuda" verwenden
        _auto_resize = False, # Do not resize the image automatically
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
time.sleep(2)
if mode == "edit":
    print(f"Hello and welcome to the {model_name} image editor!")
else:
    print(f"Hello and welcome to the {model_name} image generator!")
max_pictures = 3

if __name__ == "__main__":
    while True:
        # Frage, welches der hochgeladenen Bilder bearbeitet werden soll
        if mode == "edit":
            #Auslesen aller Bilddateien im INPUT_PATH
            input_image_files = os.listdir(INPUT_PATH)
            print("Available images for editing:")
            for i, file in enumerate(input_image_files):
                print(f'{i+1}\t{file.rsplit("/")[-1]}')
            input_image_index = int(input("Please enter the number of the image you intend to edit: ")) -1 # Korrektur Indizierung, weil Zählbeginn mit 0 unintuitiv ist
            input_image_path = INPUT_PATH + "/" + input_image_files[input_image_index]
            input_image = load_image(input_image_path).convert("RGB")
            print(f'Image {input_image_files[input_image_index].rsplit("/")} ready for editing.')
        
        # Bildgenerierung / Bildbearbeitung
        print("Please enter your picture prompt or write 'exit' to close the application:")
        prompt = str(input())
        if prompt.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        print("Do you wish to further customize the image generation by entering parameters?\n(Y/n)")
        customization = str(input())
        if customization.lower() in ["y", "yes", "ja", "ye"]:
            # Fallunterscheidung der veränderbaren Parameter abhängig vom Modus
            if mode == "generate":
                height, width = map(int, input("Please enter your desired picture dimensions for height and then width separated by ',': ").split(","))
                image_num = int( input(f"Please enter the number of images you wish to be generated (maximum of {max_pictures}): ")) #Bilderanzahl ist nicht anpassbar
                if image_num > max_pictures:
                    image_num = min(image_num, max_pictures)
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
            num_inference_steps=12 if mode == "edit" else 8
            seed = random.randrange(0, 9999, 1)

        if mode == "edit":
            image = pic_edit(
                input_image=input_image,
                prompt=prompt,
                guidance_scale=4.5,
                save_path=save_path,
                image_num=1,
                num_inference_steps=num_inference_steps,
                seed=seed,
                display_picture=False,
            )
        else:
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