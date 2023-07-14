import time
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tool.config import Cfg
from tool.translate import build_model, process_input, translate
import torch
import onnxruntime
import numpy as np

config = Cfg.load_config_from_file('./config/vgg-seq2seq.yml')
config['cnn']['pretrained']=False
config['device'] = 'cpu'
model, vocab = build_model(config)

def translate_onnx(img, session, max_seq_length=128, sos_token=1, eos_token=2):
    cnn_session, encoder_session, decoder_session = session

    # create cnn input
    cnn_input = {cnn_session.get_inputs()[0].name: img}
    src = cnn_session.run(None, cnn_input)

    # create encoder input
    encoder_input = {encoder_session.get_inputs()[0].name: src[0]}
    encoder_outputs, hidden = encoder_session.run(None, encoder_input)
    translated_sentence = [[sos_token] * len(img)]
    max_length = 0

    while max_length <= max_seq_length and not all(
            np.any(np.asarray(translated_sentence).T == eos_token, axis=1)
    ):
        tgt_inp = translated_sentence
        decoder_input = {decoder_session.get_inputs()[0].name: tgt_inp[-1],
                         decoder_session.get_inputs()[1].name: hidden,
                         decoder_session.get_inputs()[2].name: encoder_outputs}

        output, hidden, _ = decoder_session.run(None, decoder_input)
        output = np.expand_dims(output, axis=1)
        output = torch.Tensor(output)

        values, indices = torch.topk(output, 1)
        indices = indices[:, -1, 0]
        indices = indices.tolist()

        translated_sentence.append(indices)
        max_length += 1

        del output

    translated_sentence = np.asarray(translated_sentence).T

    return translated_sentence

# create inference session
cnn_session = onnxruntime.InferenceSession("./weights/cnn.onnx")
encoder_session = onnxruntime.InferenceSession("./weights/encoder.onnx")
decoder_session = onnxruntime.InferenceSession("./weights/decoder.onnx")
session = (cnn_session, encoder_session, decoder_session)

def handle(img):
    img = process_input(img, config['dataset']['image_height'], config['dataset']['image_min_width'], config['dataset']['image_max_width'])
    img = img.to(config['device'])
    return img
result_holder = []
def func_predict(img_precdict):
    # start_time = time.time()
    s = translate_onnx(np.array(img_precdict), session)[0].tolist()
    s = vocab.decode(s)
    result_holder.append(s)
    # print("Result: ", s)
    # end_time = time.time()
    # print(end_time - start_time)

start_time = time.time()


img_label1 = Image.open("images/label_1.png")
img_detec_1 = handle(img_label1.crop((48, 43, 216, 54)))
img_detec_2 = handle(img_label1.crop((62, 31, 195, 41)))
img_detec_3 = handle(img_label1.crop((48, 22, 208, 32)))
img_detec_4 = handle(img_label1.crop((64, 11, 193, 23)))
img_detec_5 = handle(img_label1.crop((227, 23, 296, 35)))
img_detec_6 = handle(img_label1.crop((226, 35, 300, 46)))
img_detec_7 = handle(img_label1.crop((4, 5, 37, 17)))
# image_demo = handle(Image.open(("./process_image/demo_detec_2.png")))

func_predict(img_detec_1)
func_predict(img_detec_2)
func_predict(img_detec_3)
func_predict(img_detec_4)
func_predict(img_detec_5)
func_predict(img_detec_6)
func_predict(img_detec_7)

positions = [
    (43, 48, 54, 216),
    (31, 62, 41, 195),
    (22, 48, 32, 208),
    (11, 64, 23, 193),
    (23, 227, 35, 296),
    (35, 226, 46, 300),
    (5, 4, 17, 37)
]


for i, pos in enumerate(positions, start=1):
    y1, x1, y2, x2 = pos
    label = result_holder[i-1]
    draw = ImageDraw.Draw(img_label1)
    draw.text((x1, y1), str(label), font=ImageFont.truetype("arial.ttf", 6), fill=(0, 0, 255), stroke_width= 1)

print(result_holder)
end_time = time.time()
print(end_time - start_time)

img_label1.show()